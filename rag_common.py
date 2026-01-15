import logging
import os
from pathlib import Path
import requests
from difflib import get_close_matches
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from config_manager import (
    config_manager,
    embed_client,
    EMBED_MODEL,
    MODEL_NAME,
    RERANKER_MODEL,
    RERANKER_URL,
)

logger = logging.getLogger(__name__)

# 统一的 LLM 选择输出
try:
    from pydantic import BaseModel
except Exception:  # pragma: no cover - pydantic 应始终可用
    BaseModel = object  # type: ignore


class SelectionOutput(BaseModel):  # type: ignore[misc]
    selected_items: List[str]


def choose_items_with_llm(
    clinical_summary: str,
    items: Sequence[str],
    label: str,
    external_client,
    cutoff: float = 0.8,
) -> List[str]:
    """使用同一模式的结构化输出选择文件/文档列表。"""
    if not items:
        return []

    options = "\n".join(f"- {name}" for name in items)
    prompt = (
        f"Given the clinical summary:\n\"\"\"\n{clinical_summary}\n\"\"\"\n"
        f"Select the relevant {label} filenames (one or multiple, or none if none are relevant). "
        f"Output only the filenames under `selected_items`.\n"
        f"Available files:\n{options}"
    )

    resp = external_client.chat.completions.parse(
        model=MODEL_NAME,
        messages=[{"role": "system", "content": prompt}],
        response_format=SelectionOutput,
    )
    selected = resp.choices[0].message.parsed.selected_items
    valid: List[str] = []
    for name in selected:
        close = get_close_matches(name, items, n=1, cutoff=cutoff)
        if close and close[0] not in valid:
            valid.append(close[0])
    return valid


def embed_query_text(base_query: str) -> List[float]:
    """单次 embedding，统一模型/格式。"""
    resp = embed_client.embeddings.create(
        model=EMBED_MODEL,
        input=[base_query],
        encoding_format="float",
    )
    return resp.data[0].embedding


def query_collection_docs(collection, embedding: List[float], where: dict, n_results: int = 20) -> Tuple[List[str], List[dict]]:
    """从 Chroma 集合中查询文档与元信息。"""
    res = collection.query(
        query_embeddings=[embedding],
        n_results=n_results,
        where=where,
        include=["documents", "metadatas"],
    )
    docs = res["documents"][0] if res.get("documents") else []
    metas = res["metadatas"][0] if res.get("metadatas") else []
    return docs, metas


def rerank_chunks(
    base_query: str,
    docs: List[str],
    metas: List[dict],
    citation_builder: Callable[[dict], str],
    top_k: int = 5,
    score_threshold: float = 0.6,
) -> List[str]:
    """调用 reranker 后拼装带 citation 的段落。"""
    if not docs or not RERANKER_URL:
        return []

    headers = {"Content-Type": "application/json"}
    # reranker 的鉴权应优先读取 reranker_config；兼容旧逻辑才 fallback 到 embedding_config
    api_key = None
    try:
        api_key = config_manager.reranker_config.get("api_key")
    except Exception:
        api_key = None
    if not api_key:
        try:
            api_key = config_manager.embedding_config.get("api_key")
        except Exception:
            api_key = None
    if api_key:
        headers["Authorization"] = f"bearer {api_key}"

    try:
        resp = requests.post(
            RERANKER_URL,
            json={"model": RERANKER_MODEL, "query": base_query, "documents": docs},
            headers=headers,
            timeout=30,
        )
        resp.raise_for_status()
        results = resp.json().get("results", [])
    except Exception:
        return []

    ranked = []
    for item in sorted(results, key=lambda x: x.get("relevance_score", 0), reverse=True):
        idx = item.get("index")
        score = item.get("relevance_score") or 0
        if idx is None or idx >= len(docs):
            continue
        if score_threshold and score < score_threshold:
            continue
        md = metas[idx] if metas else {}
        citation = citation_builder(md)
        ranked.append(f"{docs[idx]}\n— **Source:** {citation}")
        if len(ranked) >= top_k:
            break
    return ranked


def safe_tool_call(tool_name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    装饰器：统一捕获工具异常并返回可见错误信息（避免 Agent 只看到“tool failed”而不知道原因）。
    用法：
        @safe_tool_call(\"rag_tool\")
        def retrieve(...): ...
    """

    def _decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        def _wrapped(*args: Any, **kwargs: Any) -> Any:
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                msg = f"{tool_name}: {type(e).__name__}: {e}"
                logger.exception("[TOOL_ERROR] %s", msg)
                return f"⚠️ Tool execution failed: {msg}"

        return _wrapped

    return _decorator


def validate_paths(paths: Mapping[str, Path]) -> List[str]:
    """校验一组路径是否存在，返回缺失项的可读错误列表。"""
    missing: List[str] = []
    for name, p in paths.items():
        try:
            if p is None or not Path(p).exists():
                missing.append(f"{name} not found: {p}")
        except Exception:
            missing.append(f"{name} not found: {p}")
    return missing


def list_files(storage_dir: Path, suffixes: Sequence[str]) -> List[str]:
    """列出目录下指定后缀的文件名（仅文件名，不含路径）。"""
    p = Path(storage_dir)
    if not p.exists():
        return []
    suffixes_l = [s.lower() for s in suffixes]
    out: List[str] = []
    for name in os.listdir(str(p)):
        low = name.lower()
        if any(low.endswith(suf) for suf in suffixes_l):
            out.append(name)
    return out


def run_vector_rag(
    *,
    tool_name: str,
    patient_history: str,
    query: str,
    role_hint: Optional[str],
    external_client,
    sources: Mapping[
        str,
        Mapping[str, Any],
    ],
    role_templates: Optional[Mapping[str, str]] = None,
    n_results: int = 20,
    top_k: int = 5,
    score_threshold: float = 0.6,
) -> str:
    """
    统一向量检索管线：选文件 → embedding → Chroma query → rerank → 拼 citation。

    sources 每个条目需要包含：
      - label: 传给 choose_items_with_llm 的标签（如 \"ESMO\"）
      - storage_dir: Path
      - suffixes: [\".pdf\"] / [\".txt\"] / [\".docx\"]（用于枚举）
      - collection: chroma collection
      - where_key: 元信息过滤字段（\"pdf_name\"/\"txt_name\"/\"docx_name\"）
      - id_transform: Callable[[str], str]  (把选中的文件名转成 where 的值；例如去掉 .pdf)
      - citation_builder: Callable[[dict], str]
      - (可选) score_threshold/top_k/n_results 覆盖全局
    """
    try:
        base_query = query.strip() or patient_history
        if not base_query:
            logger.warning("[%s] No patient history or query provided", tool_name)
            return "No patient history or query provided."

        if role_templates and role_hint and role_hint in role_templates:
            base_query = base_query + "\n\n" + str(role_templates[role_hint])

        # 1) 枚举文件 + 路径校验
        missing_msgs: List[str] = []
        available_by_src: Dict[str, List[str]] = {}
        for src_name, cfg in sources.items():
            storage_dir = Path(cfg["storage_dir"])
            if not storage_dir.exists():
                missing_msgs.append(f"{src_name} directory not found: {storage_dir}")
                available_by_src[src_name] = []
                continue
            items = list_files(storage_dir, cfg.get("suffixes", []))
            available_by_src[src_name] = items
            logger.info("[%s] %s: %d files", tool_name, src_name, len(items))

        if all(len(v) == 0 for v in available_by_src.values()):
            details = ", ".join(f"{k}({len(v)})" for k, v in available_by_src.items())
            if missing_msgs:
                logger.error("[%s] Missing sources: %s | %s", tool_name, details, "; ".join(missing_msgs))
            return f"Missing sources/files: {details}"

        # 2) 选择相关文件（逐 source）
        summary_text = patient_history
        if query.strip():
            summary_text = f"Patient History:\n{patient_history}\n\nQuery:\n{query.strip()}"

        chosen_by_src: Dict[str, List[str]] = {}
        for src_name, cfg in sources.items():
            items = available_by_src.get(src_name, [])
            label = cfg.get("label", src_name)
            chosen = choose_items_with_llm(summary_text, items, label, external_client)
            chosen_by_src[src_name] = chosen
            logger.info("[%s] %s selected: %s", tool_name, src_name, chosen)

        if all(len(v) == 0 for v in chosen_by_src.values()):
            details = ", ".join(f"{k}({len(v)})" for k, v in chosen_by_src.items())
            logger.warning("[%s] No valid items selected: %s", tool_name, details)
            return f"No valid items selected: {details}"

        # 3) embedding
        logger.info("[%s] Embedding query...", tool_name)
        embedding = embed_query_text(base_query)

        # 4) Chroma query 聚合
        docs_all: List[str] = []
        metas_all: List[dict] = []
        for src_name, cfg in sources.items():
            chosen = chosen_by_src.get(src_name, [])
            if not chosen:
                continue

            collection = cfg["collection"]
            where_key = cfg["where_key"]
            id_transform = cfg["id_transform"]
            ids = [id_transform(x) for x in chosen]
            where = {where_key: {"$in": ids}} if len(ids) > 1 else {where_key: ids[0]}
            per_n_results = int(cfg.get("n_results", n_results))

            logger.info("[%s] Querying %s where=%s n_results=%s", tool_name, src_name, where_key, per_n_results)
            docs, metas = query_collection_docs(collection, embedding, where, n_results=per_n_results)
            docs_all.extend(docs)
            metas_all.extend(metas)

        logger.info("[%s] Retrieved %d chunks (pre-rerank)", tool_name, len(docs_all))
        if not docs_all:
            return "No relevant chunks found after reranking."

        # 5) rerank + citation
        # 这里默认复用每个 chunk 自己的 metadata；citation_builder 放在 cfg 中，
        # 但聚合后无法区分来源，因此约定：metadata 本身必须能让 citation_builder 工作
        # （例如 pdf_name/txt_name/docx_name 等字段在 md 里）。
        def _default_citation_builder(md: dict) -> str:
            # 尝试在 md 里自动识别字段
            if "pdf_name" in md:
                citation = md.get("pdf_name", "Unknown")
                section = md.get("section")
                if section:
                    citation += f', section "{section}"'
                return citation
            if "txt_name" in md:
                citation = f"{md.get('txt_name', 'Unknown')}.txt"
                section = md.get("section")
                if section:
                    citation += f', section "{section}"'
                return citation
            if "docx_name" in md:
                return f"{md.get('docx_name', 'unknown')}.docx"
            return "Unknown"

        # 优先：如果所有 sources 都提供同一个 citation_builder，则用它；否则用默认
        citation_builders = {id(cfg.get("citation_builder")): cfg.get("citation_builder") for cfg in sources.values() if cfg.get("citation_builder")}
        citation_builder = _default_citation_builder
        if len(citation_builders) == 1:
            citation_builder = list(citation_builders.values())[0]  # type: ignore[assignment]

        logger.info("[%s] Reranking chunks...", tool_name)
        outputs = rerank_chunks(
            base_query=base_query,
            docs=docs_all,
            metas=metas_all,
            citation_builder=citation_builder,
            top_k=top_k,
            score_threshold=score_threshold,
        )
        if not outputs:
            logger.warning("[%s] No chunks passed reranking threshold (or reranker unavailable)", tool_name)
            return "No relevant chunks found after reranking."
        return "\n\n".join(outputs)

    except Exception as e:
        msg = f"{tool_name}: {type(e).__name__}: {e}"
        logger.exception("[TOOL_ERROR] %s", msg)
        return f"⚠️ Tool execution failed: {msg}"

