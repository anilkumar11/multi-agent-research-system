from __future__ import annotations

import json
import uuid

from langgraph.types import Command

from research_system.config import build_default_graph


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
        "mandatory_human_review": False,
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
