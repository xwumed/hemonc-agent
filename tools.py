import json
import logging
import time

from agents import function_tool
from tavily import TavilyClient

from config_manager import get_env_var

logger = logging.getLogger(__name__)


@function_tool(name_override="no_targeted_therapy")
async def no_targeted_therapy() -> str:
    """
    Call this when the patient has no gene alteration.
    Returns a message indicating that no gene alteration exists.
    """
    return "The patient has no gene alteration, skip Step2."


@function_tool(name_override="web_search_tool")
async def web_search_tool(user_query: str) -> str:
    """
    Search website for a given query.
    Arg:
        user_query: The query to search for.
    """
    tavily_api_key = get_env_var("TAVILY_API_KEY")
    if not tavily_api_key:
        logger.warning("Tavily API key missing")
        return "Tavily API key missing. Please set TAVILY_API_KEY in environment."

    client = TavilyClient(tavily_api_key)
    try:
        start = time.perf_counter()
        response = client.search(query=user_query)
        elapsed = time.perf_counter() - start
        logger.info("Tavily search ok | elapsed=%.2fs", elapsed)
    except Exception as e:
        logger.warning("Tavily search failed | err=%s", e)
        return f"Tavily search failed: {e}"

    if not isinstance(response, str):
        return json.dumps(response, ensure_ascii=False)
    return response





