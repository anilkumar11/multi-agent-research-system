from __future__ import annotations

import os

from dotenv import load_dotenv

from .graph import build_graph
from .live_provider import LiveResearchProvider

load_dotenv()


def build_default_graph(checkpointer=None):
    """Build the demo graph, or the live DeepSeek/Tavily graph if configured.

    Falls back to DemoResearchProvider when DEEPSEEK_API_KEY is unset so the
    system stays runnable offline with no keys, matching the original design.
    """
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("No DEEPSEEK_API_KEY found - running with the offline demo provider.")
        return build_graph(checkpointer=checkpointer)

    if not os.environ.get("TAVILY_API_KEY"):
        raise RuntimeError(
            "TAVILY_API_KEY is required when DEEPSEEK_API_KEY is set."
        )

    from langchain_deepseek import ChatDeepSeek

    llm = ChatDeepSeek(model="deepseek-chat")
    provider = LiveResearchProvider(llm)
    return build_graph(provider=provider, llm=llm, checkpointer=checkpointer)
