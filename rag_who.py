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
    WHO_DIR,
    WHO_DB_STORAGE,
)

# ——— Context definition ———
@dataclass
class MedicalContext:
    patient_history: str
    role_hint: Optional[str] = None

# Directory and storage for WHO text files
WHO_STORAGE = WHO_DIR

class MedicalRAGToolkitWHO:
    """
    RAG toolkit that queries a pre-populated ChromaDB for WHO text files,
    returning chunks with inline citations using Qwen embeddings and reranker.
    Allows selecting one or multiple text files.
    """
    def __init__(
        self,
        who_storage: str | Path = WHO_DB_STORAGE,
        timeout: float = 30.0,
    ):
        # 确保路径是字符串（ChromaDB 需要字符串路径）
        who_str = str(who_storage) if isinstance(who_storage, Path) else who_storage
        self.who_client = chromadb.PersistentClient(path=who_str)
        self.who_collection = self.who_client.get_or_create_collection(name="medical_collection")
        self.timeout = timeout

    @staticmethod
    @function_tool(name_override="rag_tool_who")
    def retrieve_medical_info(
        ctx: RunContextWrapper[MedicalContext],
        query: str = "",
    ) -> str:
        """
        Retrieve medical guideline chunks from the WHO database.
        Args:
            query: The query to retrieve information from medical guidelines, should contain Disease stage and histologic subtype, Disease location, Molecular markers and genetic alterations (if applicable).
            e.g. Treatment for Stage III diffuse large B-cell lymphoma, abdominal nodes, MYC/BCL2 double-expressor
        """
        toolkit = MedicalRAGToolkitWHO()

        def build_citation(md: dict) -> str:
            citation = f"{md.get('txt_name', 'Unknown file')}.txt"
            section = md.get("section")
            if section:
                citation += f', section "{section}"'
            return citation

        def _txt_id(name: str) -> str:
            low = name.lower()
            return name[:-4] if low.endswith(".txt") else name

        sources = {
            "WHO": {
                "label": "WHO",
                "storage_dir": WHO_STORAGE,
                "suffixes": [".txt"],
                "collection": toolkit.who_collection,
                "where_key": "txt_name",
                "id_transform": _txt_id,
                "citation_builder": build_citation,
                "n_results": 20,
            }
        }

        return run_vector_rag(
            tool_name="rag_tool_who",
            patient_history=ctx.context.patient_history,
            query=query,
            role_hint=ctx.context.role_hint,
            external_client=external_client,
            sources=sources,
            role_templates=None,
            n_results=20,
            top_k=5,
            score_threshold=0.6,
        )

    def get_tools(self) -> List[callable]:
        return [self.retrieve_medical_info]