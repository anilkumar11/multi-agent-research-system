from __future__ import annotations

import operator
from typing import Annotated, Literal, TypedDict


ExecutionMode = Literal["parallel", "sequential", "hybrid"]


class Evidence(TypedDict):
    evidence_id: str
    claim: str
    source_url: str
    source_type: str
    retrieved_at: str
    produced_by: str
    confidence: float
    parent_evidence_ids: list[str]
    tags: list[str]
    credibility: float


class AgentContribution(TypedDict):
    agent: str
    summary: str
    evidence_ids: list[str]
    depends_on_agents: list[str]
    reasoning_note: str


class CrossAgentInsight(TypedDict):
    insight_id: str
    statement: str
    contributing_agents: list[str]
    parent_evidence_ids: list[str]
    confidence: float
    why_emergent: str


class Conflict(TypedDict):
    conflict_id: str
    issue: str
    evidence_ids: list[str]
    severity: Literal["low", "medium", "high"]
    status: Literal["open", "resolved", "accepted_uncertainty"]
    chosen_evidence_id: str | None
    rationale: str


class SynthesisThread(TypedDict):
    thread_id: str
    title: str
    insight_ids: list[str]
    narrative: str


class TelemetryEvent(TypedDict):
    node: str
    started_at: str
    ended_at: str
    duration_ms: float
    execution_mode: str
    evidence_before: int
    evidence_after: int
    note: str


class QualityGate(TypedDict):
    passed: bool
    evidence_count: int
    source_type_count: int
    average_confidence: float
    cross_agent_insight_count: int
    unresolved_high_conflicts: int
    iteration_count: int
    failures: list[str]


class HumanDecision(TypedDict):
    action: str
    reason: str
    payload: dict


class ExecutionPlan(TypedDict):
    mode: ExecutionMode
    rationale: str
    dependency_score: int
    speed_priority: int
    stages: list[list[str]]
    active_agents: list[str]


class ResearchState(TypedDict, total=False):
    question: str
    execution_plan: ExecutionPlan
    memory_context: dict

    # reducer-backed lists allow safe aggregation from parallel branches
    evidence: Annotated[list[Evidence], operator.add]
    contributions: Annotated[list[AgentContribution], operator.add]
    cross_agent_insights: Annotated[list[CrossAgentInsight], operator.add]
    conflicts: Annotated[list[Conflict], operator.add]
    synthesis_threads: Annotated[list[SynthesisThread], operator.add]
    telemetry: Annotated[list[TelemetryEvent], operator.add]
    human_decisions: Annotated[list[HumanDecision], operator.add]

    quality_gate: QualityGate
    iteration_count: int
    mandatory_human_review: bool
    research_complete: bool
    final_report: str
