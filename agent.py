import os
# disable all agent tracing globally
import tomllib
import asyncio
import json
from pathlib import Path
import sys
from typing import Optional, List, Tuple, Dict, Any
from openai import OpenAI, AsyncOpenAI
from openai.types.shared import Reasoning
from pydantic import BaseModel, Field
from rag_guideline import GuidelineRAGToolkit
from agents import Agent, ItemHelpers, Runner, ModelSettings, OpenAIChatCompletionsModel, set_default_openai_client, \
    ToolCallOutputItem, set_tracing_disabled, set_trace_processors, AgentHooks, Tool
from dataclasses import dataclass
from pubmedv4 import pubmed_query
import mlflow
import logging
from logging_setup import setup_logging
from agents import RunContextWrapper
from collections import defaultdict
from gene_search import genesearch_batch_tool
from datetime import datetime
from memory_bank_store import MemoryEntry, MemoryBank
# from RAGRECISTv3 import ResponseCriteriaToolkit
from tools import no_targeted_therapy
from tools import web_search_tool

from rag_who import MedicalRAGToolkitWHO
from rag_pathology import PathologyRAGToolkit
from rag_staging_uicc import UICCRAGToolkit
from dotenv import load_dotenv
from prompts import (
    PUBMED_QUERY_SYSTEM_PROMPT,
    SURGEON_INSTRUCTION,
    INTERNAL_ONCOLOGIST_INSTRUCTION,
    RADIATION_ONCOLOGIST_INSTRUCTION,
    GP_INSTRUCTION,
    GENETICIST_INSTRUCTION,
    ADDITIONAL_ONCOLOGIST_INSTRUCTION,
    CHAIRMAN_INSTRUCTION,
    RADIOLOGIST_PRECHECK_INSTRUCTION,
    PATHOLOGIST_PRECHECK_INSTRUCTION,
)
# mlflow.openai.autolog()
set_tracing_disabled(True)
# # (Optional) If you have a remote MLflow server, you can set the tracking URI and experiment:
# mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("MDT_Workflow_0810")

setup_logging()
logger = logging.getLogger(__name__)

# ================================ HOOKS AND CONTEXT ================================ #

class ToolCountingHooks(AgentHooks):
    """
    A hook that keeps track of how many times each tool is invoked.
    You can inspect hook.tool_counts after the agent run to see the totals.
    """

    def __init__(self):
        # Use a defaultdict(int) so that every time we see a tool name we increment.
        self.tool_counts: defaultdict[str, int] = defaultdict(int)
        self.tool_failures: defaultdict[str, int] = defaultdict(int)
        self.tool_successes: defaultdict[str, int] = defaultdict(int)

    async def on_tool_start(
            self,
            context: RunContextWrapper,
            agent,  # Agent[TContext]
            tool: Tool,  # The tool that is about to be called
    ) -> None:
        # Increment the count for this tool's name (or type).
        # Here I'm just using `tool.name`, but you could use any attribute of `tool`.
        self.tool_counts[tool.name] += 1
        # (Optional) print a debug line
        print(f"[Hook] Starting tool '{tool.name}'.  Total so far: {self.tool_counts[tool.name]}")
        logger.info(f"[TOOL_START] Tool '{tool.name}' started (call #{self.tool_counts[tool.name]})")

    async def on_tool_end(
            self,
            context: RunContextWrapper,
            agent,
            tool: Tool,
            result: str,  # The result returned by the tool
    ) -> None:
        """Called after a tool completes successfully."""
        self.tool_successes[tool.name] += 1
        logger.info(f"[TOOL_SUCCESS] Tool '{tool.name}' completed successfully")
        print(f"[Hook] ✅ Tool '{tool.name}' completed successfully")

    async def on_tool_error(
            self,
            context: RunContextWrapper,
            agent,
            tool: Tool,
            error: Exception,
    ) -> None:
        """Called when a tool raises an exception."""
        self.tool_failures[tool.name] += 1
        error_msg = f"Tool '{tool.name}' failed: {type(error).__name__}: {str(error)}"
        logger.error(f"[TOOL_ERROR] {error_msg}", exc_info=True)
        print(f"[Hook] ❌ {error_msg}")
        # 可选：将错误信息写入专门的错误日志文件
        error_log_file = os.getenv("TOOL_ERROR_LOG", "tool_errors.log")
        try:
            with open(error_log_file, "a", encoding="utf-8") as f:
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"[{timestamp}] {error_msg}\n")
        except Exception as e:
            logger.warning(f"Failed to write tool error to log file: {e}")


@dataclass
class MedicalContext:
    patient_history: str
    role_hint: Optional[str] = None


# ================================ SCHEMAS ====================================== #

class Output(BaseModel):
    agent_role: str
    treatment_recommendation: str


class Precheck(BaseModel):
    is_enough: bool
    required_info: Optional[str] = None


class ReasoningMixin:
    reasoning: str = Field(...,
        description="Explain the step-by-step thought process behind the provided decision. Include key considerations, how they influenced the final decisions, and why you give the final decision."
    )

class RadOutput(BaseModel, ReasoningMixin):
    recommendation: str = Field(...,
        description="Assessment regarding tumor staging (current or newly determined), any additional imaging recommendations needed."
    )
class PathOutput(BaseModel, ReasoningMixin):
    recommendation: str = Field(
        description="The recommendation regarding the diagnosis (if required) and any additional pathological or molecular tests required to optimize the treatment plan."
    )
class PubmedQueryListOutput(BaseModel,ReasoningMixin):
    """
    Enforces that the agent returns a JSON array of PubMed query strings.
    """
    queries: List[str] = Field(
        ...,
        description="List of PubMed search query strings"
    )
class Chairman(BaseModel):
    recommendation_from_each_specialist: str
    agreement: str
    conflict: str
    unified_recommendation: str
    source: str


# ================================ CONFIGURATION ========================================== #

rag_guideline = GuidelineRAGToolkit()
patho_rag = PathologyRAGToolkit()
stage_rag= UICCRAGToolkit()
rag_who= MedicalRAGToolkitWHO()


def read_file(path: str) -> str:
    """
    Load the patient‐history text verbatim from any file,
    whether .txt or .json, returning its entire contents as a string.
    """
    if os.path.isdir(path):
        raise ValueError(f"Path '{path}' is a directory, not a file. Please provide a file path.")
    with open(path, encoding="utf-8") as f:
        return f.read().strip()

def split_filename(path: str):
    """Return (id_part, time_part) from a filename 'ID_time.txt'."""
    stem = os.path.splitext(os.path.basename(path))[0]
    if "_" in stem:
        return stem.split("_", 1)
    return stem, "unknown"

# 导入统一的配置管理器
from config_manager import (
    config_manager,
    external_client as _sync_external_client,
    async_external_client as _async_external_client,
    MODEL_NAME,
    REASONING_EFFORT,
    REASONING_VERBOSITY,
)


def get_sync_client() -> OpenAI:
    return _sync_external_client


def get_async_client() -> AsyncOpenAI:
    return _async_external_client


# ✅ 设置OpenAI client
# 异步客户端用于 Runner.run 以及所有 Agent（Runner 内部会 await）
external_client = get_async_client()
DEFAULT_REASONING = Reasoning(effort=REASONING_EFFORT) if REASONING_EFFORT else None
set_default_openai_client(external_client, use_for_tracing=False)


def build_model_settings(
    *,
    tool_choice: str | None = None,
    max_completion_tokens: int | None = None,
) -> ModelSettings:
    """
    Central helper so every agent automatically opts into GPT-5 reasoning mode
    when available, while remaining compatible with other models.
    """
    settings_kwargs: dict[str, Any] = {}

    if REASONING_VERBOSITY:
        settings_kwargs["verbosity"] = REASONING_VERBOSITY

    if max_completion_tokens is not None:
        settings_kwargs["max_completion_tokens"] = max_completion_tokens
    if tool_choice is not None:
        settings_kwargs["tool_choice"] = tool_choice
    if DEFAULT_REASONING is not None:
        settings_kwargs["reasoning"] = DEFAULT_REASONING
    return ModelSettings(**settings_kwargs)


# ==================== HELPER FUNCTIONS ====================

def serialize(obj):
    """Return obj.model_dump() if obj is a Pydantic model, else the obj itself."""
    return obj.model_dump() if isinstance(obj, BaseModel) else obj


# ==================== AGENTS ====================
counting_hooks =ToolCountingHooks()

def build_doctor_agents():
    global surgeon_agent, internal_agent, radia_agent, gp_agent, geneticist  # ⬅️ add gp_agent

    common_kwargs = dict(
        model=OpenAIChatCompletionsModel(
            model=MODEL_NAME, openai_client=external_client),
        model_settings=build_model_settings(
            tool_choice="required",
            max_completion_tokens=1024,
        ),
        tools=[rag_guideline.retrieve_medical_info, web_search_tool],  # static tools
    )

    # Surgeon Agent
    surgeon_agent = Agent[MedicalContext](
        name="Surgeon_agent",
        instructions=SURGEON_INSTRUCTION,
        **common_kwargs,
    )
    surgeon_agent.hooks = counting_hooks

    # Internal Agent (Medical Oncologist)
    internal_agent = Agent[MedicalContext](
        name="Internal_agent",
        instructions=INTERNAL_ONCOLOGIST_INSTRUCTION,
        model=OpenAIChatCompletionsModel(model=MODEL_NAME, openai_client=external_client),
        model_settings=build_model_settings(tool_choice="required", max_completion_tokens=1024),
        tools=[rag_guideline.retrieve_medical_info, genesearch_batch_tool, web_search_tool],
    )
    internal_agent.hooks = counting_hooks

    # Radiation Oncologist Agent
    radia_agent = Agent[MedicalContext](
        name="RadiationOncologist_agent",
        instructions=RADIATION_ONCOLOGIST_INSTRUCTION,
        **common_kwargs,
    )
    radia_agent.hooks = counting_hooks
    gp_agent = Agent[MedicalContext](
        name="GeneralPractitioner_agent",
        instructions=GP_INSTRUCTION,
        model=OpenAIChatCompletionsModel(model=MODEL_NAME, openai_client=external_client),
        model_settings=build_model_settings(
            tool_choice="auto",
            max_completion_tokens=500,
        ),
        tools=[web_search_tool],
    )
    gp_agent.hooks = counting_hooks

    geneticist = Agent[MedicalContext](
        name="Geneticist_agent",
        instructions=GENETICIST_INSTRUCTION,
        # no external tools needed here—everything’s already in the inputs
        tools=[genesearch_batch_tool, no_targeted_therapy],
        model=OpenAIChatCompletionsModel(model=MODEL_NAME, openai_client=external_client),
        model_settings=build_model_settings(tool_choice="required"),
    )
    geneticist.hooks = counting_hooks

async def generate_pubmed_queries(context_text: str) -> List[str]:
    user_prompt = context_text

    # 使用异步客户端进行解析
    resp = await external_client.chat.completions.parse(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": PUBMED_QUERY_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        max_completion_tokens=1024,
        response_format=PubmedQueryListOutput,
    )
    return resp.choices[0].message.parsed.queries

additional_oncologist_agent = Agent[MedicalContext](
    name="AdditionalOncologist_agent",
    instructions=ADDITIONAL_ONCOLOGIST_INSTRUCTION,
    model=OpenAIChatCompletionsModel(model=MODEL_NAME, openai_client=external_client),
    model_settings=build_model_settings(tool_choice="required", max_completion_tokens=1024),
    tools=[rag_guideline.retrieve_medical_info],
)
additional_oncologist_agent.hooks = counting_hooks

chairman = Agent[MedicalContext](
    name="Chairman_agent",
    instructions=CHAIRMAN_INSTRUCTION,
    model=OpenAIChatCompletionsModel(model=MODEL_NAME, openai_client=external_client),
    model_settings=build_model_settings(max_completion_tokens=1024),
    # output_type=Chairman
)

radiologist1_precheck_agent = Agent[MedicalContext](
    name="RadiologistPrecheck_agent",
    instructions=RADIOLOGIST_PRECHECK_INSTRUCTION,
    # output_type=RadOutput,
    model=OpenAIChatCompletionsModel(model=MODEL_NAME, openai_client=external_client),
    model_settings=build_model_settings(tool_choice="required", max_completion_tokens=1024),
    tools=[stage_rag.retrieve_staging_uicc, rag_guideline.retrieve_medical_info],
)

radiologist1_precheck_agent.hooks = counting_hooks


pathologist_precheck_agent = Agent[MedicalContext](
    name="PathologistPrecheck_agent",
    instructions=PATHOLOGIST_PRECHECK_INSTRUCTION,
    # output_type=PathOutput,
    model=OpenAIChatCompletionsModel(model=MODEL_NAME, openai_client=external_client),
    model_settings=build_model_settings(tool_choice="required", max_completion_tokens=1024),
    tools=[patho_rag.retrieve_pathology, rag_who.retrieve_medical_info, rag_guideline.retrieve_medical_info],
)
pathologist_precheck_agent.hooks = counting_hooks



# ==================== PRE-CHECK FUNCTIONS ====================


async def summarize_text(text: str, heading: str) -> str:
    prompt = f"""
You are a concise medical scribe. Compress the following {heading} pre‑check findings, preserving only the conclusions, recommendations and any "suspicious" flags.
Do not invent or add any information beyond what is explicitly stated.
If a specific diagnostic test is recommended (e.g., PET-CT, CSF analysis, CT-guided puncture),  **always name the test explicitly**. 
Output only the compressed text—no explanations, headings, numbering, or extra text. Try to make it with in 200 words.

{heading}:
{text}
"""
    # 使用异步客户端
    resp = await external_client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=200,
    )
    return resp.choices[0].message.content.strip()

async def run_prechecks(patient_history: str, memory_bank: MemoryBank, patient_id: str, timestamp_str: str):
    # 1) 用 memory 生成给预检的上下文
    precheck_ctx = memory_bank.get_context_for_prechecks(patient_history)
    ctx = MedicalContext(patient_history=precheck_ctx)

    # 2) 并发运行放射科/病理科预检
    rad_task = Runner.run(radiologist1_precheck_agent, precheck_ctx, context=ctx)
    path_task = Runner.run(pathologist_precheck_agent, precheck_ctx, context=ctx)
    res_rad, res_path = await asyncio.gather(rad_task, path_task)

    # 3) 提取原始自由文本
    rad_text = ItemHelpers.text_message_outputs(res_rad.new_items)
    pat_text = ItemHelpers.text_message_outputs(res_path.new_items)

    # 4) 精简各自建议（summary），**不写回 patient_history**
    rad_summary = await summarize_text(rad_text, "Radiology")
    pat_summary = await summarize_text(pat_text, "Pathology")

    # 返回：原始 patient_history（未改动）、两段 summary、两段原文（可选）
    return patient_history, rad_summary, pat_summary, rad_text, pat_text

# ==================== MDT CORE WITH MEMORY ====================

# ==================== MDT CORE WITH MEMORY ====================

async def run_mdt_with_memory(
    patient_history: str,
    memory_bank: MemoryBank,
    timestamp_str: str,
    rad_summary: str,
    pat_summary: str,
    rad_text: str,
    pat_text: str
) -> Any:
    """
    Run the full MDT workflow...
    """
    max_context_chars = int(os.getenv("CONTEXT_MAX_CHARS", "12000"))

    def _trim(text: str) -> str:
        if len(text) <= max_context_chars:
            return text
        return text[-max_context_chars:]

    # 1) 基础上下文（来自 memory），不包含预检 summary
    agent_context = _trim(memory_bank.get_context_for_agents(patient_history))

    # 2) 仅供医生阶段使用的上下文：在基础上下文末尾拼入两个 summary
    agent_context_with_prechecks = _trim(
        agent_context
        + "\n\n[Radiologist recommendation]:\n" + rad_summary
        + "\n\n[Pathologist recommendation]:\n" + pat_summary
    )

    # 3) 第一轮各专科调用 —— 使用 **agent_context_with_prechecks**
    context_surgeon = MedicalContext(patient_history=agent_context_with_prechecks, role_hint="surgeon")
    context_internal = MedicalContext(patient_history=agent_context_with_prechecks, role_hint="internal")
    context_radia   = MedicalContext(patient_history=agent_context_with_prechecks, role_hint="radiation")
    context_gp      = MedicalContext(patient_history=agent_context_with_prechecks, role_hint="general_practitioner")
    context_ge      = MedicalContext(patient_history=agent_context, role_hint="geneticist")

    # 并发控制，避免外部请求过载
    semaphore = asyncio.Semaphore(int(os.getenv("AGENT_MAX_CONCURRENCY", "5")))

    async def _run_with_limit(agent, text, context: MedicalContext):
        async with semaphore:
            return await Runner.run(agent, text, context=context)

    res1_surgeon, res1_internal, res1_radia, res1_gp, res1_ge = await asyncio.gather(
        _run_with_limit(surgeon_agent,  agent_context_with_prechecks, context_surgeon),
        _run_with_limit(internal_agent, agent_context_with_prechecks, context_internal),
        _run_with_limit(radia_agent,    agent_context_with_prechecks, context_radia),
        _run_with_limit(gp_agent,       agent_context_with_prechecks, context_gp),
        _run_with_limit(geneticist,     agent_context, context_ge),
    )

    msg1_surgeon  = ItemHelpers.text_message_outputs(res1_surgeon.new_items)
    msg1_internal = ItemHelpers.text_message_outputs(res1_internal.new_items)
    msg1_radia    = ItemHelpers.text_message_outputs(res1_radia.new_items)
    msg1_gp       = ItemHelpers.text_message_outputs(res1_gp.new_items)
    msg1_ge = ItemHelpers.text_message_outputs(res1_ge.new_items)
    logger.info("[AGENT_OUTPUT] surgeon_agent: %s", msg1_surgeon[:400])
    logger.info("[AGENT_OUTPUT] internal_agent: %s", msg1_internal[:400])
    logger.info("[AGENT_OUTPUT] radia_agent: %s", msg1_radia[:400])
    logger.info("[AGENT_OUTPUT] gp_agent: %s", msg1_gp[:400])
    logger.info("[AGENT_OUTPUT] geneticist: %s", msg1_ge[:400])

    # 4) PubMed 查询同样基于包含 summary 的上下文（提升相关性）
    pubmed_input = agent_context_with_prechecks
    query_list = await generate_pubmed_queries(pubmed_input)
    pm_abstracts = await pubmed_query(queries=query_list, retmax=5)
    history_with_pubmed = _trim(pubmed_input + "\n\n[Relevant PubMed Abstracts]:\n" + pm_abstracts)

    # 5) 补充肿瘤科医生（Additional Oncologist）同样用含 summary 的上下文
    context_additional = MedicalContext(patient_history=history_with_pubmed, role_hint="additional")
    res2_additional = await _run_with_limit(additional_oncologist_agent, history_with_pubmed, context_additional)
    msg2_additional = ItemHelpers.text_message_outputs(res2_additional.new_items)
    logger.info("[AGENT_OUTPUT] additional_oncologist: %s", msg2_additional[:400])

    # 6) 主席汇总输入：显式放入两段 **summary**（原文可留作追溯不必给主席）
    final_input = (
        f"Patient history:\n{agent_context}\n\n"
        "MDT recommendations:\n"
        f"- ⚙️Radiologist : {rad_text}\n"
        f"- ⚙️Pathologist : {pat_text}\n"
        f"- ⚙️Surgeon: {msg1_surgeon}\n"
        f"- ⚙️Internal Oncologist: {msg1_internal}\n"
        f"- ⚙️Radiation Oncologist: {msg1_radia}\n"
        f"- ⚙️Additional Oncologist: {msg2_additional}\n"
        f"- ⚙️General Practitioner: {msg1_gp}\n"
        f"- ⚙️Geneticist: {msg1_ge}\n"
    )
    print(final_input)

    context_final = MedicalContext(patient_history=final_input)
    final_result = await Runner.run(chairman, final_input, context=context_final)

    # 7) 写入记忆：仅保存**原始 patient_history**与最终意见，不包含 summary
    dt = datetime.strptime(timestamp_str, "%Y%m%d")
    file_date = dt.strftime("%Y-%m-%d")
    memory_bank.add_entry(
        timestamp=file_date,
        patient_info=patient_history,
        final_decision=str(final_result.final_output)
    )

    return final_result



async def initialize_or_update(input_file: str):
    counting_hooks.tool_counts = defaultdict(int)
    counting_hooks.tool_failures = defaultdict(int)
    counting_hooks.tool_successes = defaultdict(int)
    id_part, time_part = split_filename(input_file)
    patient_history = read_file(input_file)

    # 降噪 chroma telemetry 若环境未关闭
    os.environ.setdefault("CHROMA_TELEMETRY", "0")

    memory_bank = MemoryBank(id_part)
    build_doctor_agents()

    # 新：拿到 summaries 和原文，但不改 patient_history
    patient_history_unmodified, rad_summary, pat_summary, rad_text, pat_text = await run_prechecks(
        patient_history, memory_bank, patient_id=id_part, timestamp_str=time_part
    )

    # 传入 summaries 给医生阶段与主席，且 patient_history 仍为未改动版本
    final_advice = await run_mdt_with_memory(
        patient_history_unmodified,
        memory_bank,
        timestamp_str=time_part,
        rad_summary=rad_summary,
        pat_summary=pat_summary,
        rad_text=rad_text,
        pat_text=pat_text
    )

    if memory_bank.is_first_interaction():
        print(f"\n✅ Initial Assessment Complete for Patient {id_part}")
    else:
        print(f"\n✅ Follow-up Assessment Complete for Patient {id_part}")
        print(f"Previous interactions: {len(memory_bank.entries) - 1}")

    print(f"Final Recommendation:\n{final_advice.final_output}")
    
    # 打印工具调用统计报告
    print_tool_statistics(counting_hooks, id_part)
    
    return final_advice


def print_tool_statistics(hooks: ToolCountingHooks, patient_id: str):
    """打印工具调用的统计信息"""
    print("\n" + "="*70)
    print(f"📊 工具调用统计报告 - 患者 {patient_id}")
    print("="*70)
    
    all_tools = set(hooks.tool_counts.keys())
    
    if not all_tools:
        print("⚠️  没有检测到工具调用")
        return
    
    # 成功率统计
    print("\n【工具调用成功率】")
    for tool_name in sorted(all_tools):
        total = hooks.tool_counts[tool_name]
        success = hooks.tool_successes.get(tool_name, 0)
        failure = hooks.tool_failures.get(tool_name, 0)
        success_rate = (success / total * 100) if total > 0 else 0
        
        status_icon = "✅" if failure == 0 else "❌" if success == 0 else "⚠️ "
        print(f"{status_icon} {tool_name:25s}: 总计 {total:2d} | 成功 {success:2d} | 失败 {failure:2d} | 成功率 {success_rate:5.1f}%")
    
    # 失败工具列表
    failed_tools = {tool: count for tool, count in hooks.tool_failures.items() if count > 0}
    if failed_tools:
        print("\n【失败工具详情】")
        print("⚠️  以下工具调用出现失败，请检查日志文件获取详细错误信息：")
        for tool_name, count in sorted(failed_tools.items(), key=lambda x: x[1], reverse=True):
            print(f"   • {tool_name}: {count} 次失败")
        
        error_log_file = os.getenv("TOOL_ERROR_LOG", "tool_errors.log")
        print(f"\n💡 详细错误日志已保存至: {error_log_file}")
    else:
        print("\n✅ 所有工具调用均成功！")
    
    print("="*70 + "\n")




# ==================== CLI ====================

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <ID_time.txt>")
        sys.exit(1)

    file_path = sys.argv[1]
    if not os.path.exists(file_path):
        print("Error: file not found:", file_path)
        sys.exit(1)
    
    if os.path.isdir(file_path):
        print(f"Error: '{file_path}' is a directory, not a file.")
        print("Please specify a specific file, for example:")
        print(f"  python agentv8.py {file_path}\\1136600_20250502.json")
        sys.exit(1)


    async def main():
        try:
            await initialize_or_update(file_path)
        finally:
            print("✅ Cleanup complete")


    asyncio.run(main())
