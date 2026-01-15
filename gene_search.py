import requests
import json
import time
import logging
from typing import Optional
from agents import function_tool, RunContextWrapper, Agent, ItemHelpers, Runner, OpenAIChatCompletionsModel, ModelSettings, AgentHooks, Tool,set_tracing_disabled
from dataclasses import dataclass
import asyncio
from openai.types.shared import Reasoning
from typing import List
import mlflow
from urllib.parse import urljoin
import re
from typing import List, Dict, Optional
import os
from logging_setup import setup_logging

# 可选开启 mlflow autolog，默认关闭避免泄露敏感信息
if os.getenv("MLFLOW_AUTOLOG", "").lower() in ("1", "true", "yes"):
    mlflow.openai.autolog()
mlflow.set_experiment("MDT_Workflow_civic")
set_tracing_disabled(True)
# ─────────────────────────────────────────────────────────────────────────────
# CIViC v2 GraphQL Tool Definition
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MedicalContext:
    patient_history: str
    role_hint: Optional[str] = None

# 导入统一的配置管理器
from config_manager import (
    config_manager,
    async_external_client,
    REASONING_EFFORT,
    REASONING_VERBOSITY,
    get_env_var,
)

REASONING_EFFORT = config_manager.REASONING_EFFORT or "medium"
REASONING_VERBOSITY = config_manager.REASONING_VERBOSITY or "low"
DEFAULT_REASONING = Reasoning(effort=REASONING_EFFORT) if REASONING_EFFORT else None
cli_async = async_external_client
setup_logging()
logger = logging.getLogger(__name__)
# @function_tool(name_override="civic_tool")
def civic_tool(
    ctx: RunContextWrapper[MedicalContext],
    hugo_symbol: str,
    variant_partial: Optional[str] = "Mutation"
) -> str:
    if variant_partial is None:
        variant_partial = "Mutation"
    variant_partial = variant_partial.strip()

    url = "https://civicdb.org/api/graphql"
    headers = {"Content-Type": "application/json"}
    graphql_query = """
    query ConciseVariantInfo($geneSymbol: String!, $variantName: String!) {
      gene(entrezSymbol: $geneSymbol) {
        variants(name: $variantName) {
          nodes {
            name
            molecularProfiles {
              nodes {
                name
                evidenceItems {
                  nodes {
                    evidenceLevel
                    evidenceDirection
                    disease { name }
                    therapies { 
                      name
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
    """
    variables = {"geneSymbol": hugo_symbol, "variantName": variant_partial}
    print(f"[civic_tool] Query variables = {variables!r}")

    try:
        start = time.perf_counter()
        resp = requests.post(url, json={"query": graphql_query, "variables": variables}, headers=headers, timeout=10)
        resp.raise_for_status()
        elapsed = time.perf_counter() - start
        logger.info("CIViC request ok | elapsed=%.2fs | status=%s", elapsed, resp.status_code)
    except requests.RequestException as e:
        logger.warning("CIViC request failed | err=%s", e)
        if hasattr(e, "response") and e.response is not None:
            return f"CIViC GraphQL API Error {e.response.status_code}: {e.response.text}"
        return f"CIViC GraphQL API request failed: {str(e)}"

    data = resp.json()
    if "errors" in data:
        return f"CIViC GraphQL returned errors: {data['errors']}"


    result = []
    gene = (data.get("data") or {}).get("gene") or {}
    variants = (gene.get("variants") or {}).get("nodes", [])

    for var in variants:
        ve = {"variantName": var.get("name"), "profiles": []}
        for prof in (var.get("molecularProfiles") or {}).get("nodes", []):
            for ev in (prof.get("evidenceItems") or {}).get("nodes", []):
                therapies = [{"name": t.get("name")} for t in (ev.get("therapies") or [])]
                entry = {
                    "profileName":       prof.get("name"),
                    "disease":           (ev.get("disease") or {}).get("name"),
                    "therapies":         therapies,
                    "evidenceLevel":     ev.get("evidenceLevel"),
                    "evidenceDirection": ev.get("evidenceDirection")
                }
                ve["profiles"].append(entry)
        result.append(ve)
    return json.dumps(result, indent=2, ensure_ascii=False)


ONCOKB_TOKEN = get_env_var("ONCOKB_TOKEN")

# -----------------------------------------------------------------------------
# OncoKB Query Function (simplified)
# -----------------------------------------------------------------------------
def onco_kb(
    hugo_symbol: str,
    change: str,
    alteration: str
) -> dict:
    """
    Query OncoKB for a specific alteration. Returns parsed JSON as dict.
    """
    base_url = "https://www.oncokb.org/api/v1/"
    if not ONCOKB_TOKEN:
        raise EnvironmentError("Missing OncoKB token. Please set ONCOKB_TOKEN in environment.")

    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {ONCOKB_TOKEN}"
    }

    if change == "variant":
        # structural variant (fusion etc.)
        a, b = hugo_symbol.split("-", 1)
        endpoint = "annotate/structuralVariants"
        params = {
            "hugoSymbolA": a,
            "hugoSymbolB": b,
            "structuralVariantType": alteration.upper(),
            "isFunctionalFusion": "true"
        }
    elif change == "amplification":
        # copy‐number alteration
        endpoint = "annotate/copyNumberAlterations"
        params = {
            "hugoSymbol": hugo_symbol,
            "copyNameAlterationType": alteration.upper()
        }
    else:
        # default to point mutation
        endpoint = "annotate/mutations/byProteinChange"
        params = {
            "hugoSymbol": hugo_symbol,
            "alteration": alteration
        }



    url = urljoin(base_url, endpoint)
    try:
        start = time.perf_counter()
        resp = requests.get(url, params=params, headers=headers, timeout=10)
        resp.raise_for_status()
        elapsed = time.perf_counter() - start
        logger.info("OncoKB request ok | endpoint=%s | status=%s | elapsed=%.2fs", endpoint, resp.status_code, elapsed)
        return resp.json()
    except requests.RequestException as exc:
        logger.warning("OncoKB request failed | endpoint=%s | err=%s", endpoint, exc)
        return {"error": f"OncoKB request failed: {exc}"}


from typing import List, Optional
import json
from agents import function_tool, RunContextWrapper

# Assuming civic_tool and onco_kb are defined above

@function_tool(name_override="genesearch_batch_tool")
def genesearch_batch_tool(ctx: RunContextWrapper, queries: List[str]) -> List[Dict]:
    """
    Batch search OncoKB and Civic for gene alterations
    Arg:
    queries: a list of query strings, each in form of "GENE ALTERATION"
    """
    batch = []

    cnv = ["amplification", "deletion", "gain", "loss"]
    sv = ["translocation", "duplication", "insertion", "inversion", "fusion"]
    def detect_change(alteration: str) -> str:
        low = alteration.lower()
        if any(term in low for term in cnv):
            return "amplification"
        if any(term in low for term in sv):
            return "variant"
        return "mutation"

    def has_any_therapies(data):
        if not isinstance(data, list):
            return False
        return any(
            prof.get("therapies")
            for var_entry in data
            for prof in var_entry.get("profiles", [])
        )

    for q in queries:
        # 1) fusion
        m_fus = re.search(r"([\w-]+)\s+fusion$", q, flags=re.IGNORECASE)
        if m_fus:
            gene = m_fus.group(1)
            mutation_clean = "fusion"
        else:
            # 2) splice site
            m_splice = re.match(r"^(\w+)\s+splice site\s+(.+)$", q, flags=re.IGNORECASE)
            if m_splice:
                gene = m_splice.group(1)
                mutation_clean = f"c.{m_splice.group(2)}"
            else:
                parts = q.split(maxsplit=1)
                gene = parts[0]
                mutation_clean = parts[1] if len(parts) == 2 else ""

        # 3) normalize deletion/rearrangement/duplication
        lc = mutation_clean.lower()
        mutation_clean = re.sub(r'(?i)(fs)\s*\*\s*\d+\b', r'\1', mutation_clean)
        if "deletion" in lc:
            mutation_clean = "deletion"
        elif "rearrangement" in lc:
            mutation_clean = "rearrangement"
        elif "duplication" in lc:
            mutation_clean = "duplication"

        # CIViC
        try:
            raw1 = civic_tool(ctx, hugo_symbol=gene, variant_partial=mutation_clean)
            data1 = json.loads(raw1)
        except Exception:
            # skip this query entirely on error
            continue

        civic_entry = {"data": data1}
        if not has_any_therapies(data1):
            # fallback
            try:
                raw2 = civic_tool(ctx, hugo_symbol=gene, variant_partial="Mutation")
                data2 = json.loads(raw2)
                civic_entry.update({
                    "fallback_query": f"{gene} Mutation",
                    "fallback_data": data2
                })
            except Exception:
                # if even fallback fails, skip
                continue

        # OncoKB
        alteration = mutation_clean
        change = detect_change(alteration)
        try:
            onco_data = onco_kb(gene, change, alteration)
        except Exception:
            continue

        treatments = onco_data.get("treatments", []) or []
        simplified = [
            {
                "level": t.get("level"),
                "cancerType": t.get("levelAssociatedCancerType", {})
                                 .get("mainType", {})
                                 .get("name"),
                "drugs": [d.get("drugName") for d in t.get("drugs", [])]
            }
            for t in treatments
        ]

        batch.append({
            "query": q,
            "civic": civic_entry,
            "oncokb": simplified
        })

    return batch
    # return json.dumps(batch, indent=2, ensure_ascii=False)




async def demo_civic_agent():
    # counting_hooks = ToolCountingHooks()

    demo_agent = Agent[MedicalContext](
        name="Demo_CIViC_Agent",
        instructions=(
            "You are a demo agent for gene queries.\n"
            "When the user asks for clinical evidence on a variant, call genesearch_batch_tool.\n"
            "Return the JSON string containing all matched variants, molecular profiles, evidence items, and therapies."
        ),
        tools=[genesearch_batch_tool],
        model=OpenAIChatCompletionsModel(model="Llama-4-Maverick-17B-128E-Instruct-FP8", openai_client=cli_async),
        model_settings=ModelSettings(
            tool_choice="required",
            verbosity=REASONING_VERBOSITY,
            **({"reasoning": DEFAULT_REASONING} if DEFAULT_REASONING else {}),
        ),
    )
    # demo_agent.hooks = counting_hooks

    prompt = "Fetch gene data for FANCA Q1307*  TP53 splice site 375+1G>A  BCL6 M273V."

    context = MedicalContext(patient_history=prompt)

    result = await Runner.run(demo_agent, prompt, context=context)

    answer = ItemHelpers.text_message_outputs(result.new_items)
    print("\n===== Agent’s Final Output =====")
    print(answer)

    # print("\n===== Tool Usage Summary =====")
    # for name, count in counting_hooks.tool_counts.items():
    #     print(f"  • {name}: {count}")


if __name__ == "__main__":
    asyncio.run(demo_civic_agent())
