# Interactive DeepSeek/Tavily Research Provider Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the offline-only demo with a real, DeepSeek-and-Tavily-backed research provider, generalize conflict/insight detection beyond the EV-market fixtures, wire up LangSmith tracing, and turn `run_demo.py` into an interactive CLI — while keeping `DemoResearchProvider` and all existing tests working unchanged when no API keys are configured.

**Architecture:** A new `LiveResearchProvider` (Tavily search + `ChatDeepSeek` structured output) implements the existing `ResearchProvider` protocol so `graph.py`'s topology is untouched. `agents.py`'s `build_cross_agent_insights`/`detect_and_resolve_conflicts` become factories taking an optional `llm`; `llm=None` preserves today's hardcoded behavior exactly, `llm` set swaps in an LLM-identified-conflicts/insights path with resolution/filtering logic still enforced in code. `research_system/config.py` picks demo vs. live based on `DEEPSEEK_API_KEY` presence and loads `.env` via `python-dotenv`. `run_demo.py` loops on a question prompt instead of taking one CLI arg.

**Tech Stack:** Python 3.9 (existing repo target), `langgraph`, `langchain-deepseek` (`ChatDeepSeek`), `langchain-tavily` (`TavilySearch`), `pydantic` v2 (structured output schemas), `python-dotenv`, `unittest` (existing test runner).

## Global Constraints

- Existing tests (`tests/test_planner.py`, `tests/test_quality.py`, `tests/test_provenance.py`) must keep passing unmodified — verified by running `python3 -m unittest discover -s tests -v` after every task.
- `DemoResearchProvider` and the hardcoded (`llm=None`) branches of `agents.py` must produce byte-identical behavior to what exists today.
- No real network/API calls in any automated test — DeepSeek/Tavily object construction may be tested with fake env-var keys (confirmed safe: constructing `ChatDeepSeek`/`TavilySearch` does not make network calls), but no test may call `.invoke()` on a real or fake LLM/search client.
- `.env` must never be committed. Only `.env.example` (with empty values) is tracked.
- New packages confirmed installable and importable in this environment: `langchain-deepseek==0.1.4`, `langchain-tavily==0.2.11` (exact pins not required in `requirements.txt`, use `>=` as the rest of the file does).
- `ChatDeepSeek(model="deepseek-chat")` reads `DEEPSEEK_API_KEY` from the environment automatically (via `secret_from_env`) — no need to pass `api_key=` explicitly. `TavilySearch(max_results=N)` likewise reads `TAVILY_API_KEY` from the environment automatically.
- `TavilySearch().invoke({"query": "..."})` returns a dict shaped `{"query": ..., "results": [{"title": ..., "url": ..., "content": ..., "score": ..., "raw_content": ...}, ...], ...}`.

---

### Task 1: Initialize git, dependencies, `.gitignore`, and `.env.example`

**Files:**
- Modify: `requirements.txt`
- Modify: `.gitignore`
- Create: `.env.example`

**Interfaces:**
- Produces: three new importable packages (`langchain_deepseek`, `langchain_tavily`, `dotenv`) available to every later task; a documented set of env var names (`DEEPSEEK_API_KEY`, `TAVILY_API_KEY`, `LANGSMITH_TRACING`, `LANGSMITH_API_KEY`, `LANGSMITH_PROJECT`) that Task 6's `config.py` and Task 5's `live_provider.py` will read; an initialized git repo on branch `main` so every later task's commit step has something to commit to.

- [ ] **Step 0: Initialize git**

This directory is not yet a git repository. Run:
```bash
git status 2>&1 | head -1
```
If it prints `fatal: not a git repository...`, run:
```bash
git init
git branch -m main
```
If it's already a git repo (e.g. this step already ran), skip.

- [ ] **Step 1: Update `requirements.txt`**

Replace the full file contents with:

```text
langgraph>=0.6
typing-extensions>=4.12
langchain-deepseek>=0.1
langchain-tavily>=0.2
python-dotenv>=1.0
pydantic>=2.7
```

- [ ] **Step 2: Update `.gitignore`**

Replace the full file contents with:

```text
.venv/
__pycache__/
*.pyc
.env
```

- [ ] **Step 3: Create `.env.example`**

```text
# Copy this file to .env and fill in your own keys.
# If DEEPSEEK_API_KEY is unset, the system falls back to the offline demo provider.

DEEPSEEK_API_KEY=
TAVILY_API_KEY=

# Optional: LangSmith tracing (https://smith.langchain.com)
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=
LANGSMITH_PROJECT=multi-agent-research-system
```

- [ ] **Step 4: Verify dependencies install**

Run:
```bash
pip3 install --user -r requirements.txt
python3 -c "import langchain_deepseek, langchain_tavily, dotenv, pydantic; print('ok')"
```
Expected: prints `ok` with no `ModuleNotFoundError`.

(`.gitignore`'s `.env` entry is verified later, in Task 9 Step 2, once the repo actually exists — `git check-ignore` needs a git repo to run against.)

- [ ] **Step 5: Commit**

```bash
git add requirements.txt .gitignore .env.example
git commit -m "chore: add DeepSeek/Tavily/dotenv dependencies and env template"
```

---

### Task 2: Structured-output schemas

**Files:**
- Create: `research_system/schemas.py`
- Test: `tests/test_schemas.py`

**Interfaces:**
- Consumes: nothing (pure Pydantic models).
- Produces: `EvidenceItemSchema`, `SpecialistFindingsSchema` (consumed by Task 5's `live_provider.py`), `ConflictCandidateSchema`, `ConflictAnalysisSchema`, `InsightCandidateSchema`, `InsightAnalysisSchema` (consumed by Task 3's `agents.py`).

- [ ] **Step 1: Write the failing test**

Create `tests/test_schemas.py`:

```python
import unittest

from pydantic import ValidationError

from research_system.schemas import EvidenceItemSchema, SpecialistFindingsSchema


class SchemaTests(unittest.TestCase):
    def test_evidence_item_accepts_valid_payload(self):
        item = EvidenceItemSchema(
            claim="Adoption is rising.",
            source_url="https://example.org/a",
            source_type="news_article",
            confidence=0.8,
            credibility=0.7,
            tags=["adoption"],
        )
        self.assertEqual(item.confidence, 0.8)

    def test_evidence_item_rejects_out_of_range_confidence(self):
        with self.assertRaises(ValidationError):
            EvidenceItemSchema(
                claim="x",
                source_url="https://example.org/a",
                source_type="news_article",
                confidence=1.5,
                credibility=0.7,
                tags=[],
            )

    def test_specialist_findings_wraps_evidence_list(self):
        findings = SpecialistFindingsSchema(
            evidence=[
                EvidenceItemSchema(
                    claim="x",
                    source_url="https://example.org/a",
                    source_type="news_article",
                    confidence=0.5,
                    credibility=0.5,
                    tags=[],
                )
            ],
            summary="Short summary.",
        )
        self.assertEqual(len(findings.evidence), 1)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest tests.test_schemas -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'research_system.schemas'`

- [ ] **Step 3: Write the implementation**

Create `research_system/schemas.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m unittest tests.test_schemas -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add research_system/schemas.py tests/test_schemas.py
git commit -m "feat: add pydantic schemas for LLM structured output"
```

---

### Task 3: Generalize `agents.py` conflict/insight detection

**Files:**
- Modify: `research_system/agents.py`
- Test: `tests/test_agents.py`

**Interfaces:**
- Consumes: `research_system.schemas.ConflictAnalysisSchema`, `InsightAnalysisSchema` (Task 2). An `llm` object passed in must support `llm.with_structured_output(SchemaClass).invoke(prompt: str) -> SchemaClass instance` (this is the standard LangChain chat model interface — satisfied by `ChatDeepSeek` in Task 5/6, and by a hand-written fake object in this task's tests).
- Produces: `build_cross_agent_insights(llm=None) -> Callable[[dict], dict]` and `detect_and_resolve_conflicts(llm=None) -> Callable[[dict], dict]` — both now **factories** (breaking change from today's plain functions). Task 4's `graph.py` must call them as `build_cross_agent_insights(llm)` / `detect_and_resolve_conflicts(llm)` when wiring nodes, not pass the bare name.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_agents.py`:

```python
import unittest

from research_system.agents import build_cross_agent_insights, detect_and_resolve_conflicts


def _evidence(evidence_id, produced_by, claim, confidence, credibility, tags):
    return {
        "evidence_id": evidence_id,
        "claim": claim,
        "source_url": f"https://example.org/{evidence_id}",
        "source_type": "report",
        "retrieved_at": "2026-01-01T00:00:00+00:00",
        "produced_by": produced_by,
        "confidence": confidence,
        "parent_evidence_ids": [],
        "tags": tags,
        "credibility": credibility,
    }


class HardcodedCrossAgentInsightsTests(unittest.TestCase):
    def test_llm_none_matches_current_behavior_two_agents(self):
        state = {
            "evidence": [
                _evidence("e1", "web_research", "Charging is growing", 0.8, 0.9, ["charging"]),
                _evidence("e2", "data_analysis", "Charging points up 45%", 0.85, 0.88, ["charging", "growth"]),
            ]
        }
        node = build_cross_agent_insights()  # llm=None -> hardcoded path
        result = node(state)
        self.assertEqual(len(result["cross_agent_insights"]), 1)
        insight = result["cross_agent_insights"][0]
        self.assertEqual(set(insight["contributing_agents"]), {"web_research", "data_analysis"})
        self.assertEqual(len(result["synthesis_threads"]), 1)

    def test_llm_none_no_insight_with_single_agent(self):
        state = {
            "evidence": [
                _evidence("e1", "web_research", "Charging is growing", 0.8, 0.9, ["charging"]),
            ]
        }
        node = build_cross_agent_insights()
        result = node(state)
        self.assertEqual(result["cross_agent_insights"], [])


class HardcodedConflictDetectionTests(unittest.TestCase):
    def test_llm_none_detects_market_size_conflict(self):
        state = {
            "evidence": [
                _evidence("e1", "data_analysis", "Segment is about $8B", 0.78, 0.84, ["market_size"]),
                _evidence("e2", "trend_analysis", "Segment is about $14B", 0.80, 0.78, ["market_size"]),
            ]
        }
        node = detect_and_resolve_conflicts()  # llm=None -> hardcoded path
        result = node(state)
        self.assertEqual(len(result["conflicts"]), 1)
        self.assertEqual(result["conflicts"][0]["status"], "open")

    def test_llm_none_no_conflict_below_two_market_size_items(self):
        state = {"evidence": [_evidence("e1", "data_analysis", "Segment is about $8B", 0.78, 0.84, ["market_size"])]}
        node = detect_and_resolve_conflicts()
        result = node(state)
        self.assertNotIn("conflicts", result)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m unittest tests.test_agents -v`
Expected: FAIL — `build_cross_agent_insights()` currently requires a `state` positional arg and isn't callable with zero args (`TypeError: build_cross_agent_insights() missing 1 required positional argument: 'state'`).

- [ ] **Step 3: Rewrite `research_system/agents.py`**

Replace the full file contents with:

```python
from __future__ import annotations

from typing import Callable

from .provider import ResearchProvider
from .schemas import ConflictAnalysisSchema, InsightAnalysisSchema
from .utils import average, stable_id, unique_agents_for_evidence


def specialist_node(provider: ResearchProvider, specialty: str) -> Callable:
    def node(state: dict) -> dict:
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
    severity = candidate.severity if candidate.severity in ("low", "medium", "high") else "medium"

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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m unittest tests.test_agents -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Run the full existing test suite to confirm no regressions**

Run: `python3 -m unittest discover -s tests -v`
Expected: PASS — all tests including `test_planner`, `test_quality`, `test_provenance`, `test_schemas`, `test_agents`.

- [ ] **Step 6: Commit**

```bash
git add research_system/agents.py tests/test_agents.py
git commit -m "feat: generalize conflict/insight detection with optional LLM path"
```

---

### Task 4: Wire `llm` through `graph.py`

**Files:**
- Modify: `research_system/graph.py`

**Interfaces:**
- Consumes: `build_cross_agent_insights(llm=None)`, `detect_and_resolve_conflicts(llm=None)` from Task 3 (now factories, not plain functions).
- Produces: `build_graph(provider=None, checkpointer=None, llm=None)` — new `llm` keyword param. Task 6's `config.py` calls this with `llm=<ChatDeepSeek instance>` for the live path.

- [ ] **Step 1: Modify `build_graph`'s signature and the two node registrations**

In `research_system/graph.py`, change:

```python
def build_graph(
    provider: ResearchProvider | None = None,
    checkpointer=None,
):
```

to:

```python
def build_graph(
    provider: ResearchProvider | None = None,
    checkpointer=None,
    llm=None,
):
```

And change:

```python
    builder.add_node(
        "cross_agent_insights",
        timed_node("cross_agent_insights", build_cross_agent_insights),
    )
    builder.add_node(
        "conflict_resolution",
        timed_node("conflict_resolution", detect_and_resolve_conflicts),
    )
```

to:

```python
    builder.add_node(
        "cross_agent_insights",
        timed_node("cross_agent_insights", build_cross_agent_insights(llm)),
    )
    builder.add_node(
        "conflict_resolution",
        timed_node("conflict_resolution", detect_and_resolve_conflicts(llm)),
    )
```

No other lines in `graph.py` change.

- [ ] **Step 2: Verify the demo path still builds and runs end-to-end**

Run:
```bash
python3 -c "
from research_system.graph import build_graph
graph = build_graph()
result = graph.invoke(
    {
        'question': 'Quick overview and landscape scan of the Indian EV market',
        'evidence': [], 'contributions': [], 'cross_agent_insights': [],
        'conflicts': [], 'synthesis_threads': [], 'telemetry': [],
        'human_decisions': [], 'iteration_count': 0,
        'mandatory_human_review': False, 'research_complete': False, 'final_report': '',
    },
    config={'configurable': {'thread_id': 't1'}},
)
print('mode:', result['execution_plan']['mode'])
print('has_report:', bool(result.get('final_report')))
"
```
Expected: prints `mode: parallel` and `has_report: True` (a quick-scan question triggers parallel mode and, since `llm=None` here, the original hardcoded conflict/insight logic runs — the demo evidence contains the `charging`/`market_size` fixtures so this should pass the quality gate and synthesize without hitting human review).

- [ ] **Step 3: Run the full test suite**

Run: `python3 -m unittest discover -s tests -v`
Expected: PASS, no regressions.

- [ ] **Step 4: Commit**

```bash
git add research_system/graph.py
git commit -m "feat: thread optional llm client through build_graph"
```

---

### Task 5: `LiveResearchProvider`

**Files:**
- Create: `research_system/live_provider.py`

**Interfaces:**
- Consumes: `research_system.schemas.SpecialistFindingsSchema` (Task 2); `langchain_tavily.TavilySearch`; an `llm` object supporting `.with_structured_output(Schema).invoke(prompt) -> Schema instance`.
- Produces: `LiveResearchProvider(llm, search=None, results_per_query=5)` implementing `ResearchProvider.research(specialty, question, context) -> {"evidence": [...], "contribution": {...}}` — same return shape as `DemoResearchProvider.research()`, so `specialist_node` (Task 3, unchanged) works with either provider. Consumed by Task 6's `config.py`.

Per the approved spec, this task has no automated tests (no CI credentials to exercise real Tavily/DeepSeek calls). Correctness is checked via import/construction smoke tests using fake keys, not `.invoke()` calls.

- [ ] **Step 1: Create `research_system/live_provider.py`**

```python
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

    def research(self, specialty: str, question: str, context: list[dict]) -> dict:
        if specialty not in SPECIALTY_ROLES:
            raise ValueError(f"Unknown specialty: {specialty}")

        results = self._search_results(specialty, question)
        allowed_urls = {r["url"] for r in results if "url" in r}
        parent_ids = [e["evidence_id"] for e in context[-3:]]

        prompt = self._build_prompt(specialty, question, results, context)
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
        self, specialty: str, question: str, results: list[dict], context: list[dict]
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
        return (
            f"You are {role} on a multi-agent research team.\n\n"
            f"Research question: {question}\n\n"
            f"Live search results:\n{results_block}\n\n"
            f"Evidence already gathered by other specialists:\n{context_block}\n\n"
            "Produce 2 to 4 evidence items strictly grounded in the search results above. "
            "Each evidence item's source_url MUST be copied exactly, character for "
            "character, from one of the search result URLs above -- never invent a URL "
            "or modify one. Also produce a one- or two-sentence summary of your findings."
        )
```

- [ ] **Step 2: Smoke-test the module imports and constructs with fake keys**

Run:
```bash
TAVILY_API_KEY=fake-tavily-key python3 -c "
from research_system.live_provider import LiveResearchProvider

class FakeLLM:
    pass

provider = LiveResearchProvider(FakeLLM())
print('constructed ok:', provider)
try:
    provider.research('not_a_specialty', 'q', [])
except ValueError as exc:
    print('raises on unknown specialty:', exc)
"
```
Expected: prints `constructed ok: <...>` then `raises on unknown specialty: Unknown specialty: not_a_specialty`.

- [ ] **Step 3: Run the full test suite to confirm no regressions**

Run: `python3 -m unittest discover -s tests -v`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add research_system/live_provider.py
git commit -m "feat: add Tavily+DeepSeek-backed LiveResearchProvider"
```

---

### Task 6: `config.py` — provider selection and `.env` loading

**Files:**
- Create: `research_system/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: `research_system.graph.build_graph(provider=None, checkpointer=None, llm=None)` (Task 4); `research_system.live_provider.LiveResearchProvider` (Task 5); `langchain_deepseek.ChatDeepSeek`.
- Produces: `build_default_graph(checkpointer=None)` — consumed by Task 7's `run_demo.py`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_config.py`:

```python
import os
import unittest
from unittest import mock

from research_system import config


class BuildDefaultGraphTests(unittest.TestCase):
    def test_falls_back_to_demo_without_deepseek_key(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with mock.patch("research_system.config.build_graph") as mock_build:
                mock_build.return_value = "demo-graph"
                result = config.build_default_graph()
        mock_build.assert_called_once_with(checkpointer=None)
        self.assertEqual(result, "demo-graph")

    def test_raises_without_tavily_key(self):
        env = {"DEEPSEEK_API_KEY": "fake-deepseek-key"}
        with mock.patch.dict(os.environ, env, clear=True):
            with self.assertRaises(RuntimeError):
                config.build_default_graph()

    def test_builds_live_graph_when_both_keys_present(self):
        env = {
            "DEEPSEEK_API_KEY": "fake-deepseek-key",
            "TAVILY_API_KEY": "fake-tavily-key",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            with mock.patch("research_system.config.build_graph") as mock_build:
                mock_build.return_value = "live-graph"
                result = config.build_default_graph()
        self.assertEqual(result, "live-graph")
        _, kwargs = mock_build.call_args
        self.assertIsNotNone(kwargs.get("provider"))
        self.assertIsNotNone(kwargs.get("llm"))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m unittest tests.test_config -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'research_system.config'`

- [ ] **Step 3: Create `research_system/config.py`**

```python
from __future__ import annotations

import os

from dotenv import load_dotenv

from .graph import build_graph
from .live_provider import LiveResearchProvider

load_dotenv()


def build_default_graph(checkpointer=None):
    """Build the demo graph, or the live DeepSeek/Tavily graph if configured.

    Falls back to DemoResearchProvider when DEEPSEEK_API_KEY is unset so the
    system stays runnable offline with no keys, matching the original design.
    """
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("No DEEPSEEK_API_KEY found - running with the offline demo provider.")
        return build_graph(checkpointer=checkpointer)

    if not os.environ.get("TAVILY_API_KEY"):
        raise RuntimeError(
            "TAVILY_API_KEY is required when DEEPSEEK_API_KEY is set."
        )

    from langchain_deepseek import ChatDeepSeek

    llm = ChatDeepSeek(model="deepseek-chat")
    provider = LiveResearchProvider(llm)
    return build_graph(provider=provider, llm=llm, checkpointer=checkpointer)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m unittest tests.test_config -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Run the full test suite**

Run: `python3 -m unittest discover -s tests -v`
Expected: PASS, no regressions.

- [ ] **Step 6: Commit**

```bash
git add research_system/config.py tests/test_config.py
git commit -m "feat: add config.build_default_graph with demo/live provider selection"
```

---

### Task 7: Interactive `run_demo.py`

**Files:**
- Modify: `run_demo.py`

**Interfaces:**
- Consumes: `research_system.config.build_default_graph()` (Task 6).
- Produces: no importable interface (entry-point script only).

- [ ] **Step 1: Replace `run_demo.py`'s full contents**

```python
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
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit"):
            print("Exiting.")
            break

        run_one_question(graph, question)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Manually verify the offline demo path still works interactively**

Run:
```bash
printf 'Quick overview and landscape scan of the Indian EV market\nquit\n' | python3 run_demo.py
```
Expected: prints `No DEEPSEEK_API_KEY found - running with the offline demo provider.`, then the interactive prompt banner, then the full execution plan / quality gate / final report / telemetry for the piped-in question, then exits cleanly on `quit` with no traceback.

- [ ] **Step 3: Run the full test suite**

Run: `python3 -m unittest discover -s tests -v`
Expected: PASS, no regressions (`run_demo.py` has no unit tests — it's an entry-point script, matching its treatment before this change).

- [ ] **Step 4: Commit**

```bash
git add run_demo.py
git commit -m "feat: make run_demo.py an interactive multi-question CLI"
```

---

### Task 8: Update `CLAUDE.md`

**Files:**
- Modify: `CLAUDE.md`

**Interfaces:**
- Consumes: nothing (documentation only).
- Produces: nothing (documentation only).

- [ ] **Step 1: Update the Commands section**

In `CLAUDE.md`, replace the `## Commands` section's fenced code block with:

```markdown
## Commands

\`\`\`bash
# Setup
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Optional: enable the live DeepSeek/Tavily provider + LangSmith tracing
cp .env.example .env   # then fill in DEEPSEEK_API_KEY / TAVILY_API_KEY / LANGSMITH_*

# Run the interactive CLI (prompts for a research question; 'quit' to exit)
python run_demo.py

# Run all tests
python -m unittest discover -s tests -v

# Run a single test file / test case
python -m unittest tests.test_planner -v
python -m unittest tests.test_planner.PlannerTests.test_hybrid_for_complex_general_question -v
\`\`\`

Without a \`DEEPSEEK_API_KEY\` in the environment, \`run_demo.py\` prints a notice and falls
back to the deterministic offline \`DemoResearchProvider\` — no keys are required to run or
test the system. With \`DEEPSEEK_API_KEY\` and \`TAVILY_API_KEY\` set, it uses
\`LiveResearchProvider\` (real Tavily search + \`ChatDeepSeek\` structured-output calls)
instead. \`LANGSMITH_*\` vars (see \`.env.example\`) enable tracing automatically with no
code changes — \`langchain-core\` picks them up from the environment.

When the graph pauses for human review, it prints a JSON review packet and expects one of
\`approve\`, \`more\`, or \`resolve\` on stdin. \`resolve\` auto-picks the strongest evidence by
credibility/confidence (see \`strongest_conflict_resolution\` in \`run_demo.py\`) — it's a
stand-in for what a human analyst would decide in a real UI. After a question's report
prints, the CLI loops back to prompt for another question (each question is an independent
research session/thread — there is no cross-question conversational memory).
```

(Escape the literal `\`\`\`` fences correctly when editing — the above is written with `\`` escapes for inclusion inside this plan's own fenced block; write plain triple-backtick fences in the actual file.)

- [ ] **Step 2: Update the Architecture section's provider description**

Find this paragraph in `CLAUDE.md`:

```markdown
It is an **architecture demonstration, not a production research tool**. `DemoResearchProvider` (`research_system/provider.py`) returns deterministic, hardcoded evidence per specialty so the graph runs fully offline with no API keys. Real search/LLM/data providers should implement the `ResearchProvider` protocol and be swapped in via `build_graph(provider=...)` without touching the graph itself.
```

Replace it with:

```markdown
`DemoResearchProvider` (`research_system/provider.py`) returns deterministic, hardcoded evidence per specialty so the graph runs fully offline with no API keys — this remains the default when `DEEPSEEK_API_KEY` is unset. `LiveResearchProvider` (`research_system/live_provider.py`) is a real implementation: it grounds evidence in live Tavily search results and uses `ChatDeepSeek` (`langchain-deepseek`) structured output to turn those results into `Evidence` items, enforcing that every `source_url` is copied verbatim from an actual search result (never invented by the model). `research_system/config.py`'s `build_default_graph()` picks between the two based on which API keys are present — see the Commands section above. Both providers implement the same `ResearchProvider` protocol, so `graph.py`'s topology never needs to change.
```

- [ ] **Step 3: Add a note about the generalized conflict/insight detection**

In the `agents.py` bullet of the "Key pieces" list in `CLAUDE.md`, after the existing sentence ending "...a real provider would need matching tag conventions or these functions would need generalizing.", add:

```markdown
  `build_cross_agent_insights(llm=None)` and `detect_and_resolve_conflicts(llm=None)` are factories: with `llm=None` (the demo path) they run the exact hardcoded logic described above; with an `llm` (the live path, wired in by `config.py`) they instead make one `ChatDeepSeek` structured-output call over all accumulated evidence to *identify* candidate conflicts/insights, while conflict auto-resolution thresholds and the "≥2 distinct agents" emergent-insight rule remain enforced in code either way — the LLM finds candidates, code decides what qualifies.
```

- [ ] **Step 4: Verify the file renders sensibly**

Run: `python3 -c "print(open('CLAUDE.md').read()[:200])"` (or just visually re-read the file) to confirm no broken markdown fences were introduced.

- [ ] **Step 5: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: document live provider, interactive CLI, and env config in CLAUDE.md"
```

---

### Task 9: Create the GitHub repository and push

By this point git was initialized in Task 1 Step 0, and Tasks 1–8 each committed their own
changes incrementally. This task confirms nothing was missed, double-checks `.env` is
excluded, then creates and pushes to GitHub.

**Files:** none (repo-level operations only)

**Interfaces:** none

- [ ] **Step 1: Confirm `.env` is ignored**

```bash
touch .env
git check-ignore -v .env
```
Expected: prints a line showing `.gitignore:4:.env	.env` (or similar) confirming `.env` is excluded. If this prints nothing, STOP and fix `.gitignore` before proceeding — do not stage or commit until this passes.

- [ ] **Step 2: Stage and commit anything left over**

Pre-existing project files (`README.md`, `DESIGN_MAPPING.md`, and any others never touched by
an earlier task's explicit `git add`) are still untracked at this point — this step catches
them.

```bash
git add -A
git status
```
Review the output: confirm `.env` and `.venv/`/`__pycache__/` do **not** appear in the list of
files to be committed. Then, only if `git status` shows staged changes:
```bash
git commit -m "chore: add remaining pre-existing project files"
```
(If nothing is staged, skip the commit.)

- [ ] **Step 3: Create the GitHub repository and push**

```bash
gh repo create multi-agent-research-system --public --source=. --remote=origin --push
```
Expected: creates a public GitHub repo named `multi-agent-research-system` under the
authenticated `gh` account, adds it as the `origin` remote, and pushes the current branch.

- [ ] **Step 4: Verify**

```bash
git remote -v
gh repo view multi-agent-research-system --web=false
```
Expected: `origin` points at the new GitHub repo; `gh repo view` prints its description/URL
with no errors.
