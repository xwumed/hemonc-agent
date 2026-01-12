from pathlib import Path
import chromadb
from typing import Optional
from dataclasses import dataclass
from agents import function_tool, RunContextWrapper

from rag_common import run_vector_rag
from config_manager import (
    external_client,
    ESMO_STORAGE,
    NCCN_STORAGE,
    HEMA_STORAGE,
    ESMO_DB_STORAGE,
    NCCN_DB_STORAGE,
    HEMA_DB_STORAGE,
)


@dataclass
class MedicalContext:
    patient_history: str
    role_hint: Optional[str] = None


class GuidelineRAGToolkit:
    """
    Unified guideline RAG tool (auto-router).
    It will automatically select relevant documents from ESMO/NCCN/HEMA and retrieve chunks.
    """

    def __init__(
        self,
        esmo_storage: str | Path = ESMO_DB_STORAGE,
        nccn_storage: str | Path = NCCN_DB_STORAGE,
        hema_storage: str | Path = HEMA_DB_STORAGE,
        timeout: float = 30.0,
    ):
        esmo_str = str(esmo_storage) if isinstance(esmo_storage, Path) else esmo_storage
        nccn_str = str(nccn_storage) if isinstance(nccn_storage, Path) else nccn_storage
        hema_str = str(hema_storage) if isinstance(hema_storage, Path) else hema_storage
        self.esmo_client = chromadb.PersistentClient(path=esmo_str)
        self.nccn_client = chromadb.PersistentClient(path=nccn_str)
        self.hema_client = chromadb.PersistentClient(path=hema_str)
        self.esmo_collection = self.esmo_client.get_or_create_collection(name="medical_collection")
        self.nccn_collection = self.nccn_client.get_or_create_collection(name="medical_collection")
        self.hema_collection = self.hema_client.get_or_create_collection(name="medical_collection")
        self.timeout = timeout

    @staticmethod
    @function_tool(name_override="rag_guideline")
    def retrieve_medical_info(
        ctx: RunContextWrapper[MedicalContext],
        query: str = "",
    ) -> str:
        toolkit = GuidelineRAGToolkit()

        def build_citation(md: dict) -> str:
            citation = md.get("pdf_name", "Unknown")
            section = md.get("section")
            if section:
                citation += f', section "{section}"'
            return citation

        def _pdf_id(name: str) -> str:
            low = name.lower()
            return name[:-4] if low.endswith(".pdf") else name

        role_templates = {
            "surgeon": "Consider surgical perspectives.",
            "internal": "Consider medical treatment perspectives.",
            "radiation": "Consider radiotherapy perspectives.",
        }

        sources = {
            "ESMO": {
                "label": "ESMO",
                "storage_dir": ESMO_STORAGE,
                "suffixes": [".pdf"],
                "collection": toolkit.esmo_collection,
                "where_key": "pdf_name",
                "id_transform": _pdf_id,
                "citation_builder": build_citation,
                "n_results": 20,
            },
            "NCCN": {
                "label": "NCCN",
                "storage_dir": NCCN_STORAGE,
                "suffixes": [".pdf"],
                "collection": toolkit.nccn_collection,
                "where_key": "pdf_name",
                "id_transform": _pdf_id,
                "citation_builder": build_citation,
                "n_results": 20,
            },
            "HEMA": {
                "label": "HEMA",
                "storage_dir": HEMA_STORAGE,
                "suffixes": [".pdf"],
                "collection": toolkit.hema_collection,
                "where_key": "pdf_name",
                "id_transform": _pdf_id,
                "citation_builder": build_citation,
                "n_results": 20,
            },
        }

        return run_vector_rag(
            tool_name="rag_guideline",
            patient_history=ctx.context.patient_history,
            query=query,
            role_hint=ctx.context.role_hint,
            external_client=external_client,
            sources=sources,
            role_templates=role_templates,
            n_results=20,
            top_k=10,
            score_threshold=0.6,
        )





