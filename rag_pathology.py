from pathlib import Path
import chromadb
from openai import OpenAI
from typing import List, Optional
from agents import function_tool, RunContextWrapper
from dataclasses import dataclass
from rag_common import run_vector_rag

# 导入统一的配置管理器
from config_manager import (
    external_client,
    PATHO_DIR,
    PATHO_DB_STORAGE,
)

# ——— Context definition ———
@dataclass
class MedicalContext:
    patient_history: str
    role_hint: Optional[str] = None  # For compatibility with MDT workflow

# ChromaDB settings for pathology
STORAGE_PATH = PATHO_DB_STORAGE
COLLECTION_NAME = "patho_collection"

class PathologyRAGToolkit:
    """
    RAG toolkit that queries a pre-embedded pathology document database using
    LLM to select relevant documents, retrieves all chunks for those documents,
    and returns them with citations.
    """
    def __init__(
        self,
        storage_path: str | Path = STORAGE_PATH,
        collection_name: str = COLLECTION_NAME,
    ):
        # 确保 storage_path 是字符串（ChromaDB 需要字符串路径）
        storage_path_str = str(storage_path) if isinstance(storage_path, Path) else storage_path
        self.client = chromadb.PersistentClient(path=storage_path_str)
        self.collection = self.client.get_or_create_collection(name=collection_name)

    @staticmethod
    @function_tool(name_override="rag_pathology")
    def retrieve_pathology(
        ctx: RunContextWrapper[MedicalContext],
        query: str = "",
        top_k: int = 5,
    ) -> str:
        return PathologyRAGToolkit._retrieve_pathology_impl(ctx, query=query, top_k=top_k, tool_name="rag_pathology")

    @staticmethod
    def _retrieve_pathology_impl(
        ctx: RunContextWrapper[MedicalContext],
        *,
        query: str,
        top_k: int,
        tool_name: str,
    ) -> str:
        toolkit = PathologyRAGToolkit()

        def build_citation(md: dict) -> str:
            return f"{md.get('docx_name', 'unknown')}.docx (chunk {md.get('chunk_index', '?')})"

        def _docx_id(name: str) -> str:
            low = name.lower()
            return name[:-5] if low.endswith(".docx") else name

        sources = {
            "PATHO": {
                "label": "pathology",
                "storage_dir": PATHO_DIR,
                "suffixes": [".docx"],
                "collection": toolkit.collection,
                "where_key": "docx_name",
                "id_transform": _docx_id,
                "citation_builder": build_citation,
                "n_results": 1000,
            }
        }

        return run_vector_rag(
            tool_name=tool_name,
            patient_history=ctx.context.patient_history,
            query=query,
            role_hint=ctx.context.role_hint,
            external_client=external_client,
            sources=sources,
            role_templates=None,
            n_results=1000,
            top_k=top_k,
            score_threshold=0.0,
        )

    def get_tools(self) -> List[callable]:
        return [self.retrieve_pathology]
