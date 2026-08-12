from __future__ import annotations

from .hitl import latest_conflicts


def synthesis_node(state: dict) -> dict:
    evidence = state.get("evidence", [])
    insights = state.get("cross_agent_insights", [])
    conflicts = latest_conflicts(state.get("conflicts", []))
    gate = state.get("quality_gate", {})

    evidence_by_id = {e["evidence_id"]: e for e in evidence}

    lines = []
    lines.append("# Market Research Report")
    lines.append("")
    lines.append(f"**Research question:** {state['question']}")
    lines.append("")
    lines.append("## Executive synthesis")

    if insights:
        for insight in insights:
            sources = [
                evidence_by_id[eid]["source_url"]
                for eid in insight["parent_evidence_ids"]
                if eid in evidence_by_id
            ]
            lines.append(f"- {insight['statement']}")
            lines.append(
                f"  - Confidence: {insight['confidence']:.2f}; "
                f"contributing agents: {', '.join(insight['contributing_agents'])}"
            )
            lines.append(f"  - Provenance: {', '.join(sources)}")
    else:
        lines.append("- No qualifying cross-agent insight was created.")

    lines.append("")
    lines.append("## Specialist contributions")
    for contribution in state.get("contributions", []):
        lines.append(
            f"- **{contribution['agent']}**: {contribution['summary']} "
            f"(used dependencies: {', '.join(contribution['depends_on_agents']) or 'none'})"
        )

    lines.append("")
    lines.append("## Conflicts and uncertainty")
    if conflicts:
        for c in conflicts:
            lines.append(
                f"- {c['issue']} — status: **{c['status']}**. "
                f"Resolution: {c['rationale']}"
            )
    else:
        lines.append("- No material conflicts were recorded.")

    lines.append("")
    lines.append("## Quality gate")
    lines.append(
        f"- Evidence: {gate.get('evidence_count', 0)}; "
        f"source types: {gate.get('source_type_count', 0)}; "
        f"average confidence: {gate.get('average_confidence', 0):.2f}; "
        f"cross-agent insights: {gate.get('cross_agent_insight_count', 0)}."
    )

    human_decisions = state.get("human_decisions", [])
    if human_decisions:
        latest = human_decisions[-1]
        lines.append(
            f"- Human review: **{latest['action']}** — {latest.get('reason', '')}"
        )

    lines.append("")
    lines.append("## Orchestration rationale")
    plan = state["execution_plan"]
    lines.append(
        f"- Mode: **{plan['mode']}**. {plan['rationale']}"
    )
    lines.append(
        "- Trade-off: parallel execution reduces discovery latency but limits "
        "same-wave dependency depth; sequential execution increases analytical "
        "depth but lengthens the critical path; hybrid execution balances both."
    )

    return {
        "final_report": "\n".join(lines),
        "research_complete": True,
        "_telemetry_note": (
            f"Synthesized {len(evidence)} evidence items, {len(insights)} "
            "cross-agent insights, and provenance links."
        ),
    }
