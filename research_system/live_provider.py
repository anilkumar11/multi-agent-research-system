from __future__ import annotations

from langchain_tavily import TavilySearch

from .schemas import SpecialistFindingsSchema
from .utils import stable_id, utc_now

SPECIALTY_ROLES = {
    "web_research": (
        "a web research analyst focused on policy, consumer sentiment, and general news"
    ),
    "data_analysis": (
        "a quantitative data analyst focused on datasets, market sizing, and statistics"
    ),
    "trend_analysis": (
        "a trend/forecast analyst focused on projecting how current signals evolve"
    ),
    "competitive_intelligence": (
        "a competitive intelligence analyst focused on named competitors' strategy and positioning"
    ),
}


class LiveResearchProvider:
    """Real research provider: Tavily search grounds a ChatDeepSeek structured-output call.

    Replaces DemoResearchProvider's hardcoded fixtures with live evidence. Same
    ResearchProvider protocol shape, so the graph does not change.
    """

    def __init__(self, llm, search=None, results_per_query: int = 5):
        self._llm = llm
        self._search = search or TavilySearch(max_results=results_per_query)

    def research(
        self, specialty: str, question: str, context: list[dict], memory: dict | None = None
    ) -> dict:
        if specialty not in SPECIALTY_ROLES:
            raise ValueError(f"Unknown specialty: {specialty}")

        results = self._search_results(specialty, question)
        allowed_urls = {r["url"] for r in results if "url" in r}
        parent_ids = [e["evidence_id"] for e in context[-3:]]

        prompt = self._build_prompt(specialty, question, results, context, memory)
        findings = self._llm.with_structured_output(SpecialistFindingsSchema).invoke(prompt)

        evidence = []
        for item in findings.evidence:
            if item.source_url not in allowed_urls:
                # Only trust URLs the search actually returned; drop anything invented.
                continue
            evidence.append({
                "evidence_id": stable_id("ev", specialty, item.claim, item.source_url),
                "claim": item.claim,
                "source_url": item.source_url,
                "source_type": item.source_type,
                "retrieved_at": utc_now(),
                "produced_by": specialty,
                "confidence": item.confidence,
                "parent_evidence_ids": parent_ids,
                "tags": item.tags,
                "credibility": item.credibility,
            })

        return {
            "evidence": evidence,
            "contribution": {
                "agent": specialty,
                "summary": findings.summary,
                "evidence_ids": [e["evidence_id"] for e in evidence],
                "depends_on_agents": sorted({e["produced_by"] for e in context}),
                "reasoning_note": (
                    f"{specialty} used {len(context)} existing evidence items and "
                    f"{len(results)} live search results."
                ),
            },
        }

    def _search_results(self, specialty: str, question: str) -> list[dict]:
        query = f"{specialty.replace('_', ' ')} {question}"
        response = self._search.invoke({"query": query})
        return response.get("results", []) if isinstance(response, dict) else response

    def _build_prompt(
        self,
        specialty: str,
        question: str,
        results: list[dict],
        context: list[dict],
        memory: dict | None = None,
    ) -> str:
        role = SPECIALTY_ROLES[specialty]
        results_block = "\n".join(
            f"- url: {r.get('url')}\n  title: {r.get('title')}\n  content: {(r.get('content') or '')[:800]}"
            for r in results
        ) or "(no search results returned)"
        context_block = "\n".join(
            f"- [{e['produced_by']}] {e['claim']} (evidence_id={e['evidence_id']})"
            for e in context
        ) or "(no prior evidence yet)"
        memory_block = self._format_memory(memory)
        return (
            f"You are {role} on a multi-agent research team.\n\n"
            f"Research question: {question}\n\n"
            f"Live search results:\n{results_block}\n\n"
            f"Evidence already gathered by other specialists:\n{context_block}\n\n"
            f"Relevant memory from prior research on this topic:\n{memory_block}\n\n"
            "Produce 2 to 4 evidence items strictly grounded in the search results above. "
            "Each evidence item's source_url MUST be copied exactly, character for "
            "character, from one of the search result URLs above -- never invent a URL "
            "or modify one. Also produce a one- or two-sentence summary of your findings."
        )

    @staticmethod
    def _format_memory(memory: dict | None) -> str:
        if not memory:
            return "(no relevant memory for this topic yet)"

        lines = []
        for fact in memory.get("semantic_facts") or []:
            lines.append(f"- known pattern: {fact}")
        for episode in memory.get("relevant_episodes") or []:
            lines.append(
                f"- prior run: mode={episode.get('mode')} "
                f"gate_passed={episode.get('gate_passed')} "
                f"question=\"{episode.get('question')}\""
            )
        return "\n".join(lines) if lines else "(no relevant memory for this topic yet)"
