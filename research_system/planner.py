from __future__ import annotations


def choose_execution_plan(question: str) -> dict:
    """
    Explicit decision framework for balancing parallel speed vs sequential depth.
    The heuristic is deliberately transparent so it can be audited or replaced
    by a classifier/LLM later.
    """
    q = question.lower()

    speed_terms = ("quick", "overview", "scan", "landscape", "fast", "snapshot")
    dependency_terms = (
        "first ", "then ", "after ", "depends on", "causal", "because",
        "impact", "quantify", "forecast", "projection", "why", "determine which",
    )
    broad_terms = ("market", "competition", "trend", "data", "policy")

    speed_priority = min(3, sum(term in q for term in speed_terms))
    dependency_score = min(5, sum(term in q for term in dependency_terms))
    breadth = sum(term in q for term in broad_terms)

    if speed_priority >= 1 and dependency_score <= 1:
        mode = "parallel"
        rationale = (
            "The request prioritizes a broad/quick scan and has few explicit "
            "dependencies, so parallel discovery minimizes latency."
        )
        stages = [["web_research", "data_analysis", "trend_analysis", "competitive_intelligence"]]
    elif dependency_score >= 3:
        mode = "sequential"
        rationale = (
            "The request contains a strong dependency chain. Sequential execution "
            "lets downstream reasoning use validated upstream evidence."
        )
        stages = [
            ["web_research"],
            ["data_analysis"],
            ["trend_analysis"],
            ["competitive_intelligence"],
        ]
    else:
        mode = "hybrid"
        rationale = (
            "The request benefits from both breadth and dependency-aware reasoning. "
            "Web + Data run in parallel for speed; Trend + Competitive run afterward "
            "for deeper interpretation."
        )
        stages = [
            ["web_research", "data_analysis"],
            ["trend_analysis", "competitive_intelligence"],
        ]

    return {
        "mode": mode,
        "rationale": rationale,
        "dependency_score": dependency_score,
        "speed_priority": speed_priority,
        "stages": stages,
        "breadth_score": breadth,
    }


def planner_node(state: dict) -> dict:
    plan = choose_execution_plan(state["question"])
    # breadth_score is explanatory only; keep state schema compact.
    plan.pop("breadth_score", None)
    return {
        "execution_plan": plan,
        "iteration_count": state.get("iteration_count", 0),
        "_telemetry_note": plan["rationale"],
    }
