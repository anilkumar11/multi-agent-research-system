from __future__ import annotations

import os
import sqlite3

from dotenv import load_dotenv

from .graph import build_graph
from .live_provider import LiveResearchProvider
from .memory.store import load_persistent_store

load_dotenv()

MEMORY_DIR = ".research_memory"
STORE_PATH = os.path.join(MEMORY_DIR, "store.json")
CHECKPOINT_PATH = os.path.join(MEMORY_DIR, "checkpoints.sqlite")


def build_default_graph(checkpointer=None):
    """
    Build the demo graph, or the live DeepSeek/Tavily graph if configured.

    Falls back to DemoResearchProvider when DEEPSEEK_API_KEY is unset so the
    system stays runnable offline with no keys, matching the original design.
    Always loads persistent long-term memory and, when available, a durable
    SQLite checkpointer, regardless of which provider is used -- an offline
    demo session accumulates real memory across runs just like a live one.
    """
    store = load_persistent_store(STORE_PATH)

    if checkpointer is None:
        checkpointer = _durable_checkpointer()

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("No DEEPSEEK_API_KEY found - running with the offline demo provider.")
        return build_graph(checkpointer=checkpointer, store=store)

    if not os.environ.get("TAVILY_API_KEY"):
        raise RuntimeError(
            "TAVILY_API_KEY is required when DEEPSEEK_API_KEY is set."
        )

    from langchain_deepseek import ChatDeepSeek

    llm = ChatDeepSeek(model="deepseek-chat")
    provider = LiveResearchProvider(llm)
    print("Using live DeepSeek/Tavily provider - this will make billed API calls.")
    return build_graph(provider=provider, llm=llm, checkpointer=checkpointer, store=store)


def _durable_checkpointer():
    """
    SqliteSaver so a HITL-paused thread survives quitting and restarting the
    CLI. Returns None (letting build_graph() fall back to its InMemorySaver
    default) if the optional langgraph-checkpoint-sqlite package isn't
    installed.
    """
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
    except ImportError:
        return None

    os.makedirs(MEMORY_DIR, exist_ok=True)
    conn = sqlite3.connect(CHECKPOINT_PATH, check_same_thread=False)
    saver = SqliteSaver(conn)
    saver.setup()
    return saver
