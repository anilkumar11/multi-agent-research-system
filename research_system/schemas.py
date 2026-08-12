from __future__ import annotations

from pydantic import BaseModel, Field


class EvidenceItemSchema(BaseModel):
    claim: str = Field(description="A specific factual claim relevant to the research question.")
    source_url: str = Field(description="Must be copied exactly from one of the provided search result URLs.")
    source_type: str = Field(
        description="e.g. news_article, government_report, dataset, industry_report, company_filing, survey."
    )
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence that the claim is accurate.")
    credibility: float = Field(ge=0.0, le=1.0, description="Assessment of the source's credibility.")
    tags: list[str] = Field(description="Short topic labels for this claim.")


class SpecialistFindingsSchema(BaseModel):
    evidence: list[EvidenceItemSchema]
    summary: str = Field(description="One or two sentence summary of this specialist's findings.")


class ConflictCandidateSchema(BaseModel):
    issue: str = Field(description="Plain-language description of the disagreement.")
    evidence_ids: list[str] = Field(description="At least two evidence_id values whose claims disagree.")
    severity: str = Field(description="One of: low, medium, high.")


class ConflictAnalysisSchema(BaseModel):
    conflicts: list[ConflictCandidateSchema]


class InsightCandidateSchema(BaseModel):
    statement: str = Field(description="The emergent insight, in one or two sentences.")
    evidence_ids: list[str] = Field(
        description="evidence_id values, from at least two different agents, that support this insight."
    )
    why_emergent: str = Field(description="Why this required combining multiple specialists' evidence.")


class InsightAnalysisSchema(BaseModel):
    insights: list[InsightCandidateSchema]
