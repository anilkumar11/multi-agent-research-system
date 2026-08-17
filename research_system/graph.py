from __future__ import annotations

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.store.memory import InMemoryStore

from .agents import (
    build_cross_agent_insights,
    detect_and_resolve_conflicts,
    specialist_node,
)
from .hitl import human_review_node, route_after_human
from .planner import build_planner_node
from .provider import DemoResearchProvider, ResearchProvider
from .quality import quality_gate_node, route_after_quality_gate
from .state import ResearchState
from .synthesis import synthesis_node
from .utils import timed_node


def _route_plan(state: dict) -> str:
    return state["execution_plan"]["mode"]


def _noop(state: dict) -> dict:
    return {"_telemetry_note": "Coordination barrier reached."}


def build_graph(
    provider: ResearchProvider | None = None,
    checkpointer=None,
    llm=None,
    store=None,
):
    provider = provider or DemoResearchProvider()
    checkpointer = checkpointer or InMemorySaver()
    store = store or InMemoryStore()

    builder = StateGraph(ResearchState)

    builder.add_node("planner", timed_node("planner", build_planner_node(store)))

    # Strategy/barrier nodes.
    builder.add_node("parallel_start", timed_node("parallel_start", _noop))
    builder.add_node("hybrid_start", timed_node("hybrid_start", _noop))
    builder.add_node("hybrid_stage2", timed_node("hybrid_stage2", _noop))

    # Parallel branch nodes.
    for name, specialty in [
        ("p_web", "web_research"),
        ("p_data", "data_analysis"),
        ("p_trend", "trend_analysis"),
        ("p_comp", "competitive_intelligence"),
    ]:
        builder.add_node(name, timed_node(name, specialist_node(provider, specialty)))

    # Sequential branch nodes.
    for name, specialty in [
        ("s_web", "web_research"),
        ("s_data", "data_analysis"),
        ("s_trend", "trend_analysis"),
        ("s_comp", "competitive_intelligence"),
    ]:
        builder.add_node(name, timed_node(name, specialist_node(provider, specialty)))

    # Hybrid branch nodes.
    for name, specialty in [
        ("h_web", "web_research"),
        ("h_data", "data_analysis"),
        ("h_trend", "trend_analysis"),
        ("h_comp", "competitive_intelligence"),
    ]:
        builder.add_node(name, timed_node(name, specialist_node(provider, specialty)))

    builder.add_node(
        "cross_agent_insights",
        timed_node("cross_agent_insights", build_cross_agent_insights(llm)),
    )
    builder.add_node(
        "conflict_resolution",
        timed_node("conflict_resolution", detect_and_resolve_conflicts(llm)),
    )
    builder.add_node(
        "quality_gate",
        timed_node("quality_gate", quality_gate_node),
    )
    builder.add_node(
        "human_review",
        timed_node("human_review", human_review_node),
    )
    builder.add_node(
        "synthesis",
        timed_node("synthesis", synthesis_node),
    )

    builder.add_edge(START, "planner")
    builder.add_conditional_edges(
        "planner",
        _route_plan,
        {
            "parallel": "parallel_start",
            "sequential": "s_web",
            "hybrid": "hybrid_start",
        },
    )

    # PARALLEL: all specialists start together.
    builder.add_edge("parallel_start", "p_web")
    builder.add_edge("parallel_start", "p_data")
    builder.add_edge("parallel_start", "p_trend")
    builder.add_edge("parallel_start", "p_comp")
    builder.add_edge(
        ["p_web", "p_data", "p_trend", "p_comp"],
        "cross_agent_insights",
    )

    # SEQUENTIAL: each downstream specialist sees upstream evidence.
    builder.add_edge("s_web", "s_data")
    builder.add_edge("s_data", "s_trend")
    builder.add_edge("s_trend", "s_comp")
    builder.add_edge("s_comp", "cross_agent_insights")

    # HYBRID: phase 1 parallel, barrier, phase 2 parallel.
    builder.add_edge("hybrid_start", "h_web")
    builder.add_edge("hybrid_start", "h_data")
    builder.add_edge(["h_web", "h_data"], "hybrid_stage2")
    builder.add_edge("hybrid_stage2", "h_trend")
    builder.add_edge("hybrid_stage2", "h_comp")
    builder.add_edge(["h_trend", "h_comp"], "cross_agent_insights")

    builder.add_edge("cross_agent_insights", "conflict_resolution")
    builder.add_edge("conflict_resolution", "quality_gate")
    builder.add_conditional_edges(
        "quality_gate",
        route_after_quality_gate,
        {
            "human_review": "human_review",
            "synthesis": "synthesis",
        },
    )
    builder.add_conditional_edges(
        "human_review",
        route_after_human,
        {
            "planner": "planner",
            "synthesis": "synthesis",
        },
    )
    builder.add_edge("synthesis", END)

    return builder.compile(checkpointer=checkpointer, store=store)
