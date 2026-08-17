from __future__ import annotations

from typing import Protocol

from .utils import stable_id, utc_now


class ResearchProvider(Protocol):
    def research(
        self, specialty: str, question: str, context: list[dict], memory: dict | None = None
    ) -> dict:
        """Return a contribution and evidence for one specialist."""
        ...


class DemoResearchProvider:
    """
    Deterministic provider for an offline/runnable architecture demo.

    Replace this class with real web search, databases, filings, and LLM calls.
    The graph itself does not need to change.
    """

    def _evidence(
        self,
        specialty: str,
        claim: str,
        url: str,
        source_type: str,
        confidence: float,
        credibility: float,
        parents: list[str],
        tags: list[str],
    ) -> dict:
        return {
            "evidence_id": stable_id("ev", specialty, claim, url),
            "claim": claim,
            "source_url": url,
            "source_type": source_type,
            "retrieved_at": utc_now(),
            "produced_by": specialty,
            "confidence": confidence,
            "parent_evidence_ids": parents,
            "tags": tags,
            "credibility": credibility,
        }

    def research(
        self, specialty: str, question: str, context: list[dict], memory: dict | None = None
    ) -> dict:
        parent_ids = [e["evidence_id"] for e in context[-3:]]

        if specialty == "web_research":
            evidence = [
                self._evidence(
                    specialty,
                    "Policy support and charging-infrastructure investment are meaningful EV adoption drivers.",
                    "https://example.org/government-ev-policy",
                    "government_report",
                    0.88,
                    0.95,
                    [],
                    ["policy", "charging", "adoption"],
                ),
                self._evidence(
                    specialty,
                    "Consumer interest is increasing, but charging availability remains a purchase concern.",
                    "https://example.org/consumer-survey",
                    "survey",
                    0.78,
                    0.80,
                    [],
                    ["consumer", "charging"],
                ),
            ]
            summary = "Policy and consumer evidence suggests demand upside with infrastructure constraints."

        elif specialty == "data_analysis":
            evidence = [
                self._evidence(
                    specialty,
                    "Observed charging-point count in the sample increased about 45% year over year.",
                    "https://example.org/charging-dataset",
                    "dataset",
                    0.86,
                    0.92,
                    parent_ids,
                    ["charging", "growth"],
                ),
                self._evidence(
                    specialty,
                    "One market-sizing dataset estimates the addressable segment at about $8B.",
                    "https://example.org/market-dataset-a",
                    "dataset",
                    0.78,
                    0.84,
                    parent_ids,
                    ["market_size"],
                ),
            ]
            summary = "Quantitative indicators show fast infrastructure growth and a material market opportunity."

        elif specialty == "trend_analysis":
            evidence = [
                self._evidence(
                    specialty,
                    "If infrastructure growth persists, charging constraints should weaken as a barrier to adoption.",
                    "https://example.org/trend-model",
                    "model",
                    0.76,
                    0.75,
                    parent_ids,
                    ["charging", "forecast", "adoption"],
                ),
                self._evidence(
                    specialty,
                    "A separate forecast estimates the same addressable segment at roughly $14B.",
                    "https://example.org/market-forecast-b",
                    "industry_report",
                    0.80,
                    0.78,
                    parent_ids,
                    ["market_size", "forecast"],
                ),
            ]
            summary = "The base trend is favorable, but the market-size range contains meaningful uncertainty."

        elif specialty == "competitive_intelligence":
            evidence = [
                self._evidence(
                    specialty,
                    "Competitor A has the broadest charging-partnership footprint in the reviewed sample.",
                    "https://example.org/competitor-a-partnerships",
                    "company_filing",
                    0.84,
                    0.88,
                    parent_ids,
                    ["competitor_a", "charging", "partnerships"],
                ),
                self._evidence(
                    specialty,
                    "Competitor B is emphasizing lower acquisition price rather than charging-network differentiation.",
                    "https://example.org/competitor-b-strategy",
                    "company_report",
                    0.79,
                    0.83,
                    parent_ids,
                    ["competitor_b", "pricing", "positioning"],
                ),
            ]
            summary = "Charging partnerships and price are emerging as distinct competitive positions."

        else:
            raise ValueError(f"Unknown specialty: {specialty}")

        summary = self._augment_summary_with_memory(summary, memory)

        return {
            "evidence": evidence,
            "contribution": {
                "agent": specialty,
                "summary": summary,
                "evidence_ids": [e["evidence_id"] for e in evidence],
                "depends_on_agents": sorted({e["produced_by"] for e in context}),
                "reasoning_note": (
                    f"{specialty} used {len(context)} existing evidence items. "
                    "A non-zero context count demonstrates sequential/hybrid depth."
                ),
            },
        }

    @staticmethod
    def _augment_summary_with_memory(summary: str, memory: dict | None) -> str:
        """
        Demo evidence never varies by question, so memory can't change WHAT
        this provider finds -- but it appends a visible note when relevant
        memory exists, so the mechanism is genuinely exercised and observable
        even on the offline path.
        """
        if not memory:
            return summary

        notes = []
        facts = memory.get("semantic_facts") or []
        if facts:
            patterns = sorted({f.get("pattern", "fact") for f in facts})
            notes.append(f"{len(facts)} known fact(s) on this topic ({', '.join(patterns)})")

        episodes = memory.get("relevant_episodes") or []
        if episodes:
            notes.append(f"{len(episodes)} prior run(s) on this topic")

        if not notes:
            return summary
        return f"{summary} Related memory: {'; '.join(notes)}."
