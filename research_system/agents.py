from __future__ import annotations

from typing import Callable

from .provider import ResearchProvider
from .schemas import ConflictAnalysisSchema, InsightAnalysisSchema
from .utils import average, stable_id, unique_agents_for_evidence


def specialist_node(provider: ResearchProvider, specialty: str) -> Callable:
    def node(state: dict) -> dict:
        active_agents = state.get("execution_plan", {}).get("active_agents")
        if active_agents is not None and specialty not in active_agents:
            # planner.choose_active_agents() excluded this specialty for this
            # question -- record it as a conscious skip (visible in the report's
            # "Specialist contributions" section) without calling the provider,
            # so a skipped agent costs nothing on the live path either.
            return {
                "contributions": [{
                    "agent": specialty,
                    "summary": "Skipped: the planner determined this specialty is not relevant to the question.",
                    "evidence_ids": [],
                    "depends_on_agents": [],
                    "reasoning_note": "Excluded by choose_active_agents() based on the question's phrasing.",
                }],
                "_telemetry_note": f"{specialty}: skipped (excluded by planner, no provider call made).",
            }

        context = state.get("evidence", [])
        result = provider.research(specialty, state["question"], context)
        return {
            "evidence": result["evidence"],
            "contributions": [result["contribution"]],
            "_telemetry_note": (
                f"{specialty}: consumed {len(context)} evidence items and "
                f"produced {len(result['evidence'])}."
            ),
        }
    return node


def _hardcoded_cross_agent_insights(state: dict) -> dict:
    evidence = state.get("evidence", [])

    charging = [e for e in evidence if "charging" in e.get("tags", [])]
    agents = {e["produced_by"] for e in charging}

    insights = []
    threads = []

    if len(agents) >= 2:
        chosen = sorted(charging, key=lambda e: e["confidence"], reverse=True)[:4]
        parent_ids = [e["evidence_id"] for e in chosen]
        supporting_agents = sorted(unique_agents_for_evidence(evidence, parent_ids))
        conf = average([e["confidence"] for e in chosen])

        statement = (
            "Charging infrastructure is not merely a market-growth variable; it is "
            "also a competitive-positioning lever. Policy/infrastructure expansion "
            "can accelerate adoption while disproportionately benefiting firms that "
            "already possess strong charging partnerships."
        )
        insight_id = stable_id("insight", statement, *parent_ids)
        insights.append({
            "insight_id": insight_id,
            "statement": statement,
            "contributing_agents": supporting_agents,
            "parent_evidence_ids": parent_ids,
            "confidence": round(conf, 3),
            "why_emergent": (
                "This relationship requires combining policy/web evidence, quantitative "
                "infrastructure evidence, trend interpretation, and competitor positioning. "
                "No single specialist owns the complete inference."
            ),
        })
        threads.append({
            "thread_id": stable_id("thread", "Infrastructure as competitive leverage"),
            "title": "Infrastructure growth becomes competitive leverage",
            "insight_ids": [insight_id],
            "narrative": (
                "Infrastructure expansion improves category attractiveness, but value "
                "capture depends on which competitors are positioned to exploit it."
            ),
        })

    return {
        "cross_agent_insights": insights,
        "synthesis_threads": threads,
        "_telemetry_note": f"Created {len(insights)} emergent cross-agent insights.",
    }


def _insight_prompt(question: str, evidence: list[dict]) -> str:
    evidence_block = "\n".join(
        f"- evidence_id={e['evidence_id']} agent={e['produced_by']} claim=\"{e['claim']}\" "
        f"confidence={e['confidence']} tags={e.get('tags', [])}"
        for e in evidence
    )
    return (
        "You are the cross-agent insight builder on a multi-agent research team.\n\n"
        f"Research question: {question}\n\n"
        f"All evidence gathered so far:\n{evidence_block}\n\n"
        "Identify emergent insights: statements that require combining evidence from at "
        "least two different agents (different 'agent' values above) to see a relationship, "
        "consequence, tension, or causal hypothesis that no single agent's evidence shows "
        "alone. For each insight, list the evidence_id values (from at least two different "
        "agents) that support it. If no such insight exists, return an empty list."
    )


def _llm_cross_agent_insights(llm, state: dict) -> dict:
    evidence = state.get("evidence", [])
    if len(evidence) < 2:
        return {
            "cross_agent_insights": [],
            "synthesis_threads": [],
            "_telemetry_note": "Not enough evidence for cross-agent insight analysis.",
        }

    evidence_by_id = {e["evidence_id"]: e for e in evidence}
    prompt = _insight_prompt(state["question"], evidence)
    result = llm.with_structured_output(InsightAnalysisSchema).invoke(prompt)

    insights = []
    threads = []
    for candidate in result.insights:
        parent_ids = [eid for eid in candidate.evidence_ids if eid in evidence_by_id]
        supporting_agents = sorted(unique_agents_for_evidence(evidence, parent_ids))
        if len(supporting_agents) < 2:
            # Enforce the emergent-insight definition in code, not the model.
            continue
        conf = average([evidence_by_id[eid]["confidence"] for eid in parent_ids])
        insight_id = stable_id("insight", candidate.statement, *parent_ids)
        insights.append({
            "insight_id": insight_id,
            "statement": candidate.statement,
            "contributing_agents": supporting_agents,
            "parent_evidence_ids": parent_ids,
            "confidence": round(conf, 3),
            "why_emergent": candidate.why_emergent,
        })
        threads.append({
            "thread_id": stable_id("thread", candidate.statement),
            "title": candidate.statement[:80],
            "insight_ids": [insight_id],
            "narrative": candidate.why_emergent,
        })

    return {
        "cross_agent_insights": insights,
        "synthesis_threads": threads,
        "_telemetry_note": f"Created {len(insights)} emergent cross-agent insights (LLM-identified).",
    }


def build_cross_agent_insights(llm=None) -> Callable:
    if llm is None:
        return _hardcoded_cross_agent_insights

    def node(state: dict) -> dict:
        return _llm_cross_agent_insights(llm, state)

    return node


def _hardcoded_conflict_detection(state: dict) -> dict:
    evidence = state.get("evidence", [])
    market_size = [e for e in evidence if "market_size" in e.get("tags", [])]

    if len(market_size) < 2:
        return {"_telemetry_note": "No material conflicts detected."}

    values = []
    for e in market_size:
        claim = e["claim"].lower()
        if "$8b" in claim:
            values.append((8.0, e))
        elif "$14b" in claim:
            values.append((14.0, e))

    if len(values) < 2:
        return {"_telemetry_note": "No parseable market-size conflict detected."}

    lo = min(values, key=lambda x: x[0])
    hi = max(values, key=lambda x: x[0])
    relative_gap = (hi[0] - lo[0]) / max(lo[0], 1)
    confidence_gap = abs(hi[1]["confidence"] - lo[1]["confidence"])

    issue = (
        f"Material market-size disagreement: approximately ${lo[0]:.0f}B vs "
        f"${hi[0]:.0f}B."
    )
    conflict_id = stable_id("conflict", issue, lo[1]["evidence_id"], hi[1]["evidence_id"])

    credibility_gap = abs(hi[1]["credibility"] - lo[1]["credibility"])
    can_auto_resolve = confidence_gap >= 0.15 or credibility_gap >= 0.18

    if can_auto_resolve:
        chosen = max(
            (lo[1], hi[1]),
            key=lambda e: (e["credibility"], e["confidence"]),
        )
        conflict = {
            "conflict_id": conflict_id,
            "issue": issue,
            "evidence_ids": [lo[1]["evidence_id"], hi[1]["evidence_id"]],
            "severity": "high" if relative_gap > 0.40 else "medium",
            "status": "resolved",
            "chosen_evidence_id": chosen["evidence_id"],
            "rationale": (
                "Automatically resolved because one source had a clearly stronger "
                "credibility/confidence profile."
            ),
        }
    else:
        conflict = {
            "conflict_id": conflict_id,
            "issue": issue,
            "evidence_ids": [lo[1]["evidence_id"], hi[1]["evidence_id"]],
            "severity": "high" if relative_gap > 0.40 else "medium",
            "status": "open",
            "chosen_evidence_id": None,
            "rationale": (
                "Confidence and credibility are too close for safe automatic selection; "
                "escalate to human review."
            ),
        }

    return {
        "conflicts": [conflict],
        "_telemetry_note": f"Detected conflict {conflict_id}: {conflict['status']}.",
    }


def _conflict_prompt(question: str, evidence: list[dict]) -> str:
    evidence_block = "\n".join(
        f"- evidence_id={e['evidence_id']} agent={e['produced_by']} claim=\"{e['claim']}\" tags={e.get('tags', [])}"
        for e in evidence
    )
    return (
        "You are the conflict detector on a multi-agent research team.\n\n"
        f"Research question: {question}\n\n"
        f"All evidence gathered so far:\n{evidence_block}\n\n"
        "Identify factual conflicts: two or more evidence items that make contradicting "
        "or materially inconsistent claims about the same fact (e.g. different numbers "
        "for the same metric, opposing conclusions about the same trend). For each "
        "conflict, list the evidence_id values involved, a plain-language description "
        "of the disagreement, and a severity (low/medium/high) based on how much the "
        "claims diverge and how central they are to the research question. If there are "
        "no conflicts, return an empty list."
    )


def _resolve_conflict_candidate(evidence_by_id: dict, candidate) -> dict | None:
    items = [evidence_by_id[eid] for eid in candidate.evidence_ids if eid in evidence_by_id]
    if len(items) < 2:
        return None

    lo = min(items, key=lambda e: e["confidence"])
    hi = max(items, key=lambda e: e["confidence"])
    confidence_gap = abs(hi["confidence"] - lo["confidence"])
    credibility_gap = abs(hi["credibility"] - lo["credibility"])
    can_auto_resolve = confidence_gap >= 0.15 or credibility_gap >= 0.18

    evidence_ids = [e["evidence_id"] for e in items]
    conflict_id = stable_id("conflict", candidate.issue, *evidence_ids)
    severity = candidate.severity

    if can_auto_resolve:
        chosen = max(items, key=lambda e: (e["credibility"], e["confidence"]))
        return {
            "conflict_id": conflict_id,
            "issue": candidate.issue,
            "evidence_ids": evidence_ids,
            "severity": severity,
            "status": "resolved",
            "chosen_evidence_id": chosen["evidence_id"],
            "rationale": (
                "Automatically resolved because one source had a clearly stronger "
                "credibility/confidence profile."
            ),
        }

    return {
        "conflict_id": conflict_id,
        "issue": candidate.issue,
        "evidence_ids": evidence_ids,
        "severity": severity,
        "status": "open",
        "chosen_evidence_id": None,
        "rationale": (
            "Confidence and credibility are too close for safe automatic selection; "
            "escalate to human review."
        ),
    }


def _llm_conflict_detection(llm, state: dict) -> dict:
    evidence = state.get("evidence", [])
    if len(evidence) < 2:
        return {"_telemetry_note": "Not enough evidence for conflict analysis."}

    evidence_by_id = {e["evidence_id"]: e for e in evidence}
    prompt = _conflict_prompt(state["question"], evidence)
    result = llm.with_structured_output(ConflictAnalysisSchema).invoke(prompt)

    conflicts = [
        c for c in (
            _resolve_conflict_candidate(evidence_by_id, candidate)
            for candidate in result.conflicts
        )
        if c is not None
    ]

    return {
        "conflicts": conflicts,
        "_telemetry_note": f"Detected {len(conflicts)} conflict(s) (LLM-identified).",
    }


def detect_and_resolve_conflicts(llm=None) -> Callable:
    if llm is None:
        return _hardcoded_conflict_detection

    def node(state: dict) -> dict:
        return _llm_conflict_detection(llm, state)

    return node
