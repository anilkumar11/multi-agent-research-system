from __future__ import annotations

from .memory.procedural import load_rules
from .utils import average, unique_source_types


MIN_EVIDENCE = 4
MIN_SOURCE_TYPES = 2
MIN_AVG_CONFIDENCE = 0.70
MIN_CROSS_AGENT_INSIGHTS = 1
MAX_ITERATIONS = 3


def quality_gate_node(state: dict) -> dict:
    evidence = state.get("evidence", [])
    insights = state.get("cross_agent_insights", [])
    conflicts = state.get("conflicts", [])
    iteration_count = state.get("iteration_count", 0)

    rules = load_rules()["quality_gate"]
    min_evidence = rules.get("min_evidence", MIN_EVIDENCE)
    min_source_types = rules.get("min_source_types", MIN_SOURCE_TYPES)
    min_avg_confidence = rules.get("min_avg_confidence", MIN_AVG_CONFIDENCE)
    min_cross_agent_insights = rules.get("min_cross_agent_insights", MIN_CROSS_AGENT_INSIGHTS)

    unresolved_high = [
        c for c in conflicts
        if c["severity"] == "high" and c["status"] == "open"
    ]
    avg_conf = average([e["confidence"] for e in evidence])
    source_types = unique_source_types(evidence)

    failures = []
    if len(evidence) < min_evidence:
        failures.append(f"evidence_count<{min_evidence}")
    if len(source_types) < min_source_types:
        failures.append(f"source_type_count<{min_source_types}")
    if avg_conf < min_avg_confidence:
        failures.append(f"average_confidence<{min_avg_confidence}")
    if len(insights) < min_cross_agent_insights:
        failures.append(f"cross_agent_insights<{min_cross_agent_insights}")
    if unresolved_high:
        failures.append("unresolved_high_conflicts>0")
    if iteration_count > MAX_ITERATIONS:
        failures.append(f"iteration_count>{MAX_ITERATIONS}")

    gate = {
        "passed": not failures,
        "evidence_count": len(evidence),
        "source_type_count": len(source_types),
        "average_confidence": round(avg_conf, 3),
        "cross_agent_insight_count": len(insights),
        "unresolved_high_conflicts": len(unresolved_high),
        "iteration_count": iteration_count,
        "failures": failures,
    }
    return {
        "quality_gate": gate,
        "research_complete": gate["passed"],
        "_telemetry_note": f"Quality gate passed={gate['passed']}; failures={failures}",
    }


def route_after_quality_gate(state: dict) -> str:
    if state.get("mandatory_human_review"):
        return "human_review"
    return "synthesis" if state["quality_gate"]["passed"] else "human_review"
