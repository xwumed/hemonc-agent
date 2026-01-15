import os
import re
from pathlib import Path
from openai import OpenAI
from typing import List
from agents import function_tool, RunContextWrapper
from dataclasses import dataclass
from difflib import get_close_matches
import logging

logger = logging.getLogger(__name__)

from rag_common import validate_paths

# 导入统一的配置管理器
from config_manager import (
    config_manager,
    external_client,
    MODEL_NAME,
    UICC_DIR,
)

# ——— Context definition ———
@dataclass
class MedicalContext:
    patient_history: str


class UICCRAGToolkit:
    """
    Embedding‑free toolkit: uses the LLM to select which TXT files
    to return, then reads each file verbatim.
    """

    @staticmethod
    @function_tool(name_override="rag_staging_uicc")
    def retrieve_staging_uicc(
            ctx: RunContextWrapper[MedicalContext],
            query: str = "",
    ) -> str:
        """Retrieve UICC staging references (canonical tool name)."""
        return UICCRAGToolkit._retrieve_uicc_impl(ctx, query)

    @staticmethod
    def _retrieve_uicc_impl(
            ctx: RunContextWrapper[MedicalContext],
            query: str = "",
    ) -> str:
        try:
            logger.info("[RAG_STAGING_UICC] Starting retrieve_staging_uicc")
            base_query = query.strip() or ctx.context.patient_history
            if not base_query:
                logger.warning("[RAG_STAGING_UICC] No query provided")
                return "No query provided."

            uicc_dir = UICC_DIR
            if validate_paths({"UICC_DIR": Path(uicc_dir)}):
                return f"UICC directory not found: {uicc_dir}"

            files = [fn for fn in os.listdir(uicc_dir) if fn.lower().endswith(".txt")]
            doc_names = [os.path.splitext(fn)[0] for fn in files]
            if not doc_names:
                return "No source files found."

            clinical_summary = ctx.context.patient_history
            if query.strip():
                clinical_summary = (
                    f"Patient History:\n{ctx.context.patient_history}\n\n"
                    f"Query:\n{query.strip()}"
                )

            options = "\n".join(f"- {n}" for n in doc_names)
            prompt = (
                f"Given the clinical summary:\n\"\"\"\n{clinical_summary}\n\"\"\"\n\n"
                f"Select all relevant staging documents from the list below. If no documents are relevant, return an empty list.\n"
                f"Output only a comma‑separated list of filenames (without .txt):\n\n"
                f"{options}"
            )

            resp = external_client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "system", "content": prompt}],
            )
            reply = resp.choices[0].message.content or ""

            raw_choices = [part.strip() for part in re.split(r"[,\n]", reply) if part.strip()]
            valid: List[str] = []
            for choice in raw_choices:
                match = get_close_matches(choice, doc_names, n=1, cutoff=0.8)
                if match and match[0] not in valid:
                    valid.append(match[0])

            if not valid:
                logger.warning("[RAG_STAGING_UICC] No suitable staging documents found")
                return "No suitable staging documents found for the provided query or patient history."

            logger.info("[RAG_STAGING_UICC] Reading %d staging documents...", len(valid))
            outputs: List[str] = []
            for name in valid:
                path = Path(uicc_dir) / f"{name}.txt"
                try:
                    full_text = path.read_text(encoding="utf-8").strip()
                    outputs.append(f"=== {name}.txt ===\n{full_text}")
                except FileNotFoundError:
                    logger.warning("[RAG_STAGING_UICC] File not found: %s", path)
                    continue

            return "\n\n".join(outputs) if outputs else "No passages retrieved."
        except Exception as e:
            msg = f"rag_staging_uicc: {type(e).__name__}: {e}"
            logger.exception("[TOOL_ERROR] %s", msg)
            return f"⚠️ Tool execution failed: {msg}"

    def get_tools(self) -> List[callable]:
        return [self.retrieve_staging_uicc]