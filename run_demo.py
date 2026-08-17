from __future__ import annotations

import json
import uuid

from langgraph.types import Command

from research_system.config import STORE_PATH, build_default_graph
from research_system.hitl import latest_conflicts
from research_system.memory.episodic import record_episode
from research_system.memory.procedural import (
    apply_mandatory_review_override,
    is_mandatory_review_topic,
)
from research_system.memory.reflect import reflect_on_topic
from research_system.memory.store import persist_store
from research_system.memory.topic import derive_topic


def initial_state(question: str) -> dict:
    return {
        "question": question,
        "evidence": [],
        "contributions": [],
        "cross_agent_insights": [],
        "conflicts": [],
        "synthesis_threads": [],
        "telemetry": [],
        "human_decisions": [],
        "iteration_count": 0,
        "mandatory_human_review": is_mandatory_review_topic(derive_topic(question)),
        "research_complete": False,
        "final_report": "",
    }


def strongest_conflict_resolution(interrupt_payload: dict, current_state: dict) -> dict:
    conflicts = interrupt_payload.get("open_conflicts", [])
    if not conflicts:
        return {"action": "approve", "reason": "No open conflict in review packet."}

    conflict = conflicts[0]
    evidence_by_id = {
        e["evidence_id"]: e for e in current_state.get("evidence", [])
    }
    candidates = [
        evidence_by_id[eid]
        for eid in conflict["evidence_ids"]
        if eid in evidence_by_id
    ]
    chosen = max(
        candidates,
        key=lambda e: (e.get("credibility", 0), e.get("confidence", 0)),
    )
    return {
        "action": "resolve",
        "conflict_id": conflict["conflict_id"],
        "chosen_evidence_id": chosen["evidence_id"],
        "reason": (
            "Human reviewer selected the evidence with the stronger "
            "source-credibility/confidence profile."
        ),
    }


def run_one_question(graph, question: str) -> None:
    thread_id = f"research-{uuid.uuid4()}"
    config = {"configurable": {"thread_id": thread_id}}

    result = graph.invoke(initial_state(question), config=config)

    needed_human_review = "__interrupt__" in result
    while "__interrupt__" in result:
        interrupt_obj = result["__interrupt__"][0]
        payload = interrupt_obj.value

        print("\n=== HUMAN REVIEW REQUIRED ===")
        print(json.dumps(payload, indent=2))

        answer = input("\nAction [approve / more / resolve]: ").strip().lower()
        if answer == "more":
            decision = {
                "action": "more_research",
                "reason": "Reviewer requested an additional research pass.",
            }
        elif answer == "resolve":
            snapshot = graph.get_state(config).values
            decision = strongest_conflict_resolution(payload, snapshot)
        else:
            decision = {
                "action": "approve",
                "reason": "Reviewer accepts residual uncertainty and approves synthesis.",
            }

        result = graph.invoke(Command(resume=decision), config=config)

    print("\n=== EXECUTION PLAN ===")
    print(json.dumps(result["execution_plan"], indent=2))

    print("\n=== QUALITY GATE ===")
    print(json.dumps(result.get("quality_gate", {}), indent=2))

    print("\n=== FINAL REPORT ===\n")
    print(result["final_report"])

    print("\n=== TELEMETRY ===")
    for event in result.get("telemetry", []):
        print(
            f"{event['node']:<24} "
            f"{event['duration_ms']:>8.2f} ms | "
            f"{event['execution_mode']:<10} | "
            f"{event['note']}"
        )

    _remember(graph, question, result, needed_human_review)


def _remember(graph, question: str, result: dict, needed_human_review: bool) -> None:
    """
    Capture this run into long-term memory and reflect on the topic. Wrapped
    in try/except: a memory-layer failure must never take down a run whose
    report already printed successfully -- a deliberate exception to this
    codebase's usual fail-loud rule, justified because memory bookkeeping
    here cannot corrupt the research result that's already been delivered.
    """
    try:
        topic = derive_topic(question)
        gate = result.get("quality_gate", {})
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
        persist_store(graph.store, STORE_PATH)

        if reflection["proposed_rule_change"]:
            _handle_rule_proposal(reflection["proposed_rule_change"])
    except Exception as exc:
        print(f"\n(memory capture skipped due to an error: {exc})")


def _handle_rule_proposal(proposal: dict) -> None:
    print("\n=== MEMORY: PROCEDURAL RULE PROPOSED ===")
    print(json.dumps(proposal, indent=2))
    answer = input("Apply this rule change? [y/N]: ").strip().lower()
    if answer == "y":
        apply_mandatory_review_override(proposal["topic"], proposal["rationale"])
        print("Rule applied -- future runs on this topic will require human review by default.")
    else:
        print("Rule change discarded.")


def main():
    graph = build_default_graph()

    print("\nInteractive multi-agent research CLI. Type a research question, or 'quit' to exit.")
    while True:
        try:
            question = input("\nResearch question (or 'quit'): ").strip()

            if not question:
                continue
            if question.lower() in ("quit", "exit"):
                print("Exiting.")
                break

            run_one_question(graph, question)
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break


if __name__ == "__main__":
    main()
