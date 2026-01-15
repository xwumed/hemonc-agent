import json
import asyncio
import logging
import time
from typing import Any, List, Tuple, Dict
import re
import os
import httpx
from agents import (
    function_tool,
    Agent,
    Runner,
    OpenAIChatCompletionsModel,
    RunContextWrapper,
    ModelSettings,
    FunctionTool,
)
from pydantic import BaseModel, Field
# 在 pubmedv4.py 文件开头添加
from pathlib import Path

# 获取脚本所在目录
SCRIPT_DIR = Path(__file__).parent


# ──────────────────────────── Configuration ────────────────────────────

# 导入统一的配置管理器
from config_manager import (
    config_manager,
    external_client as cli_sync,
    async_external_client as cli_async,
    get_env_var,
    PUBMED_ISSN_FILE,
)

# ────────────────────────── PubMed Function Tool ──────────────────────────

# NCBI E-utilities endpoints
PUBMED_ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUBMED_ESUMMARY_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
PUBMED_EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
PUBMED_API_KEY = get_env_var("PUBMED_API_KEY")

async def pubmed_query(
    queries: List[str] = Field(..., description="List of PubMed search query strings for PubMed"),
    retmax: int = Field(5, description="Maximum number of abstracts per query to return (sorted by relevance)"),
) -> str:
    """
    • Accepts a list of PubMed search queries and retrieves up to `retmax` PMIDs for each.
    • Aggregates all IDs, removes duplicates, then filters by ISSN list before fetching abstracts.
    """
    logging.info(f"pubmed_query called with queries={queries}, retmax={retmax}")

    aggregated_ids: List[str] = []
    seen_ids: set = set()

    # Load allowed ISSNs (graceful if missing)
    allowed_issns: set = set()
    try:
        issn_file = PUBMED_ISSN_FILE
        with open(issn_file, 'r', encoding='utf-8') as f:
            for line in f:
                issn = line.strip()
                if issn:
                    allowed_issns.add(issn)
    except Exception as e:
        logging.warning(f"ISSN allowlist not loaded ({e}); skipping ISSN filter.")

    max_conn = int(os.getenv("HTTPX_MAX_CONNECTIONS", "10"))
    keepalive = int(os.getenv("HTTPX_MAX_KEEPALIVE", "5"))
    timeout = float(os.getenv("HTTPX_TIMEOUT", "30"))

    async with httpx.AsyncClient(
        timeout=timeout,
        limits=httpx.Limits(max_connections=max_conn, max_keepalive_connections=keepalive),
    ) as client:
        async def get_with_retry(url: str, params: dict, retries: int = 3, initial_delay: float = 1.0) -> httpx.Response:
            delay = initial_delay
            for attempt in range(retries):
                try:
                    start = time.perf_counter()
                    response = await client.get(url, params=params)
                    response.raise_for_status()
                    elapsed = time.perf_counter() - start
                    logging.info("PubMed request ok | url=%s | status=%s | elapsed=%.2fs", url, response.status_code, elapsed)
                    return response
                except httpx.HTTPError as exc:
                    status = getattr(exc.response, "status_code", None)
                    if status == 429:
                        wait = max(0.2, delay)
                        logging.warning("PubMed 429 Too Many Requests, sleeping %.2fs before retry", wait)
                        await asyncio.sleep(wait)
                        delay *= 2
                        continue
                    if attempt == retries - 1:
                        logging.warning("PubMed request failed after retries | url=%s", url)
                        raise
                    await asyncio.sleep(delay)
                    delay *= 2

        # 1) Loop over each query to perform ESearch
        for query in queries:
            term = query.lower()
            search_params = {
                "db": "pubmed",
                "term": term,
                "retmode": "json",
                "sort": "relevance",
                "retmax": retmax,
                "usehistory": "y",
                "datetype": "pdat",
                "reldate": 1825,
            }
            if PUBMED_API_KEY:
                search_params["api_key"] = PUBMED_API_KEY
            esearch_resp = await get_with_retry(PUBMED_ESEARCH_URL, search_params)
            esearch_data = esearch_resp.json()
            id_list = esearch_data.get("esearchresult", {}).get("idlist", [])
            logging.info("PubMed ESearch | query=%s | hits=%d", term, len(id_list))
            await asyncio.sleep(float(os.getenv("PUBMED_THROTTLE_SEC", "0.15")))

            # Collect unique IDs preserving order
            for pmid in id_list:
                if pmid not in seen_ids:
                    seen_ids.add(pmid)
                    aggregated_ids.append(pmid)

        if not aggregated_ids:
            return f"No PubMed articles found for queries: {queries}."

        # 2) ESummary: fetch ISSN metadata for aggregated PMIDs
        esummary_params = {"db": "pubmed", "id": ",".join(aggregated_ids), "retmode": "json"}
        if PUBMED_API_KEY:
            esummary_params["api_key"] = PUBMED_API_KEY
        esummary_resp = await get_with_retry(PUBMED_ESUMMARY_URL, esummary_params)
        summary_data = esummary_resp.json().get("result", {})

        # Filter PMIDs by allowed ISSNs (if loaded)
        filtered_ids: List[str] = []
        for pmid in aggregated_ids:
            entry = summary_data.get(pmid, {})
            issn_list = entry.get("issn", "")
            first_issn = issn_list.split(",")[0].strip() if issn_list else None
            if allowed_issns:
                if first_issn and first_issn in allowed_issns:
                    filtered_ids.append(pmid)
            else:
                filtered_ids.append(pmid)

        if not filtered_ids:
            return json.dumps({"articles": [], "error": "No articles match the allowed ISSNs list."}, ensure_ascii=False)

        # 3) EFetch: get abstracts for the filtered PMIDs (batched)
        def chunked(seq: List[str], size: int) -> List[List[str]]:
            return [seq[i:i + size] for i in range(0, len(seq), size)]

        abstract_map: Dict[str, str] = {}
        for batch in chunked(filtered_ids, 20):
            fetch_params = {
                "db": "pubmed",
                "id": ",".join(batch),
                "retmode": "text",
                "rettype": "abstract",
                "tool": os.getenv("PUBMED_TOOL", "hema-agent"),
                "email": os.getenv("PUBMED_EMAIL"),
            }
            if PUBMED_API_KEY:
                fetch_params["api_key"] = PUBMED_API_KEY
            efetch_resp = await get_with_retry(PUBMED_EFETCH_URL, fetch_params)
            raw_text = efetch_resp.text.strip()
            raw_text = re.sub(
                r"Conflict of interest statement:[\s\S]*?(?=\n\n\d+\.|\Z)",
                "",
                raw_text,
                flags=re.IGNORECASE | re.DOTALL
            ).strip()
            # Split abstracts by PMID markers
            parts = re.split(r"\n{2,}(?=PMID-\s*\d+|PMID:\s*\d+)", raw_text)
            for part in parts:
                m = re.search(r"PMID[-:\s]+(\d+)", part)
                if not m:
                    continue
                pmid = m.group(1)
                abstract_map[pmid] = part.strip()

        articles: List[dict] = []
        for pmid in filtered_ids:
            entry = summary_data.get(pmid, {})
            authors_raw = entry.get("authors", []) or []
            authors = [a.get("name") for a in authors_raw if isinstance(a, dict) and a.get("name")]
            journal = entry.get("fulljournalname") or entry.get("source")
            title = entry.get("title")
            pubdate = entry.get("pubdate") or ""
            year_match = re.search(r"\b(20\d{2}|19\d{2})\b", pubdate)
            year = year_match.group(1) if year_match else None
            articles.append(
                {
                    "pmid": pmid,
                    "title": title,
                    "authors": authors,
                    "journal": journal,
                    "year": year,
                    "issn": entry.get("issn"),
                    "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                    "abstract": abstract_map.get(pmid, "").strip() or "[No abstract available]",
                }
            )

        return json.dumps(
            {
                "queries": queries,
                "count": len(articles),
                "articles": articles,
            },
            ensure_ascii=False,
        )

# ───────────────────────── Test/Demo with Agent ─────────────────────────

def _demo():
    """
    Simple demonstration of pubmed_query with example queries.
    """
    # Dummy context placeholder
    class DummyCtx:
        pass

    example_queries = ['"diffuse large b-cell lymphoma" relapsed', '"diffuse large b-cell lymphoma" stage ii r-dhaox', '"diffuse large b-cell lymphoma" relapsed "post-transplant lymphoproliferative disorder" r-dhaox', '"diffuse large b-cell lymphoma" stage ii "post-transplant lymphoproliferative disorder"']

    # Run the async query and print results
    result = asyncio.run(pubmed_query(queries=example_queries, retmax=5))
    print(result)


if __name__ == "__main__":
    _demo()
