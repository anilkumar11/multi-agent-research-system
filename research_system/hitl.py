from __future__ import annotations

from langgraph.types import Command, interrupt


def human_review_node(state: dict):
    """
    Exactly one interrupt call per node execution.

    The payload is JSON-serializable and gives the analyst enough provenance to
    approve, resolve uncertainty, or request another research pass.
    """
    open_conflicts = [
        c for c in state.get("conflicts", [])
        if c["status"] == "open"
    ]

    payload = {
        "type": "research_quality_review",
        "question": state["question"],
        "quality_gate": state.get("quality_gate"),
        "open_conflicts": open_conflicts,
        "instructions": {
            "approve": {"action": "approve", "reason": "Human accepts residual uncertainty."},
            "more_research": {"action": "more_research", "reason": "Gather another research pass."},
            "resolve": {
                "action": "resolve",
                "conflict_id": "<id>",
                "chosen_evidence_id": "<evidence id>",
                "reason": "<why this evidence is authoritative>",
            },
        },
    }

    decision = interrupt(payload)

    action = decision.get("action", "approve")
    reason = decision.get("reason", "")
    updates = {
        "human_decisions": [{
            "action": action,
            "reason": reason,
            "payload": decision,
        }],
        "_telemetry_note": f"Human decision: {action}",
    }

    if action == "resolve":
        # Reducer-backed conflict history is append-only; we append a resolution
        # record for auditability rather than mutating/deleting the original.
        cid = decision.get("conflict_id")
        chosen = decision.get("chosen_evidence_id")
        original = next(
            (c for c in state.get("conflicts", []) if c["conflict_id"] == cid),
            None,
        )
        if original:
            resolved = dict(original)
            resolved["status"] = "resolved"
            resolved["chosen_evidence_id"] = chosen
            resolved["rationale"] = reason or "Resolved by human reviewer."
            # A new audit record is appended. Effective-conflict helpers prefer
            # the latest record for a given conflict_id.
            updates["conflicts"] = [resolved]
        updates["research_complete"] = True

    elif action == "more_research":
        updates["iteration_count"] = state.get("iteration_count", 0) + 1
        updates["research_complete"] = False

    else:  # approve
        updates["research_complete"] = True

    return updates


def latest_conflicts(conflicts: list[dict]) -> list[dict]:
    """Collapse append-only conflict history to the latest record per ID."""
    by_id = {}
    for conflict in conflicts:
        by_id[conflict["conflict_id"]] = conflict
    return list(by_id.values())


def route_after_human(state: dict) -> str:
    decisions = state.get("human_decisions", [])
    if not decisions:
        return "synthesis"

    latest = decisions[-1]["action"]
    if latest == "more_research" and state.get("iteration_count", 0) <= 3:
        return "planner"
    return "synthesis"
