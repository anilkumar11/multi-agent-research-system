from __future__ import annotations

import json
import sys
import time
import uuid

from langgraph.types import Command

from research_system.config import STORE_PATH, build_default_graph
from research_system.graph import build_graph
from research_system.hitl import latest_conflicts
from research_system.memory.episodic import record_episode
from research_system.memory.reflect import reflect_on_topic
from research_system.memory.store import persist_store
from research_system.memory.topic import derive_topic
from run_demo import initial_state

# Verified against the offline demo provider: any question that leaves trend_analysis
# active hits the same $8B (data_analysis) vs $14B (trend_analysis) market-size conflict
# every time, since DemoResearchProvider's evidence never varies by question -- so most
# of these fail the quality gate and need human review. Excluding trend_analysis is the
# one way to avoid it in demo mode. This mixed, empirically-real set is more useful than
# a hand-picked all-pass set: it's exactly the kind of thing an eval should surface.
DEFAULT_QUESTIONS = [
    "Quick overview and landscape scan of the Indian EV market",
    "First quantify battery-price changes, then forecast their impact on EV adoption, "
    "then determine which competitor benefits most",
    "Should an automaker enter the Indian EV market, and what competitive position should it take?",
    "Give me a market overview of the Indian EV market, excluding competitor analysis",
    "Quick overview and landscape scan of the Indian EV market, ignore trend forecasting",
]


def run_eval_question(graph, question: str, persist: bool) -> dict:
    thread_id = f"eval-{uuid.uuid4()}"
    config = {"configurable": {"thread_id": thread_id}}

    start = time.perf_counter()
    result = graph.invoke(initial_state(question), config=config)

    needed_human_review = "__interrupt__" in result
    while "__interrupt__" in result:
        # Auto-approve so the run reaches a final report even when it can't pass
        # autonomously -- the point is to measure autonomous success, not to block
        # an unattended batch run on a human decision that isn't coming.
        result = graph.invoke(
            Command(resume={
                "action": "approve",
                "reason": "Automated eval run: auto-approved to measure end-to-end completion.",
            }),
            config=config,
        )

    elapsed_ms = (time.perf_counter() - start) * 1000
    gate = result.get("quality_gate", {})

    proposed_rule_change = None
    try:
        topic = derive_topic(question)
        conflicts = latest_conflicts(result.get("conflicts", []))
        episode = {
            "question": question,
            "mode": result["execution_plan"]["mode"],
            "active_agents": result["execution_plan"]["active_agents"],
            "gate_passed": gate.get("passed", False),
            "gate_failures": gate.get("failures", []),
            "needed_human_review": needed_human_review,
            "insight_count": gate.get("cross_agent_insight_count", 0),
            "open_conflict_issues": [c["issue"] for c in conflicts if c["status"] == "open"],
        }
        record_episode(graph.store, topic, episode)
        reflection = reflect_on_topic(graph.store, topic)
        # Only the --live path persists memory to disk: the default offline path
        # is meant to stay a deterministic, repeatable baseline measurement, and
        # accumulating persisted memory across eval runs would undermine that.
        # record_episode/reflect_on_topic still run either way, in-process, so
        # the mechanism is exercised even when nothing is written to disk.
        if persist:
            persist_store(graph.store, STORE_PATH)
        proposed_rule_change = reflection["proposed_rule_change"]
    except Exception as exc:
        print(f"    (memory capture skipped due to an error: {exc})")

    return {
        "question": question,
        "mode": result["execution_plan"]["mode"],
        "active_agents": result["execution_plan"]["active_agents"],
        "gate_passed_autonomously": bool(gate.get("passed")) and not needed_human_review,
        "needed_human_review": needed_human_review,
        "gate_failures": gate.get("failures", []),
        "evidence_count": gate.get("evidence_count", 0),
        "cross_agent_insight_count": gate.get("cross_agent_insight_count", 0),
        "elapsed_ms": round(elapsed_ms, 1),
        "has_report": bool(result.get("final_report")),
        "proposed_rule_change": proposed_rule_change,
    }


def main():
    args = sys.argv[1:]
    live = "--live" in args
    questions = [a for a in args if a != "--live"] or DEFAULT_QUESTIONS

    if live:
        print("Using live DeepSeek/Tavily provider (if configured) -- this may make billed API calls.")
        graph = build_default_graph()
    else:
        print("Using the offline demo provider (deterministic, free). Pass --live to eval the live provider.")
        graph = build_graph()

    print(f"\nRunning eval over {len(questions)} question(s)...\n")
    results = []
    for i, question in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] {question[:70]}")
        r = run_eval_question(graph, question, persist=live)
        results.append(r)
        status = "PASS" if r["gate_passed_autonomously"] else ("HITL" if r["needed_human_review"] else "FAIL")
        print(
            f"    -> {status} | mode={r['mode']} | {r['elapsed_ms']:.0f}ms | "
            f"gate_failures={r['gate_failures']}"
        )

    total = len(results)
    autonomous_passes = sum(1 for r in results if r["gate_passed_autonomously"])
    rate = (autonomous_passes / total * 100) if total else 0.0
    avg_latency = (sum(r["elapsed_ms"] for r in results) / total) if total else 0.0

    print("\n=== EVAL SUMMARY ===")
    print(f"Autonomous quality-gate pass rate: {autonomous_passes}/{total} ({rate:.0f}%)")
    print(f"Needed human review: {sum(1 for r in results if r['needed_human_review'])}/{total}")
    print(f"Average latency: {avg_latency:.0f} ms")

    proposals = [r for r in results if r.get("proposed_rule_change")]
    if proposals:
        print(
            f"\n{len(proposals)} procedural rule change(s) proposed by memory reflection "
            "(not applied -- review via `python run_demo.py`):"
        )
        for r in proposals:
            print(f"  - {r['proposed_rule_change']}")

    print("\n=== PER-QUESTION DETAIL ===")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
