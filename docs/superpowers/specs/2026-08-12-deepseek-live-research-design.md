# Interactive DeepSeek/Tavily research provider + LangSmith tracing

**Date:** 2026-08-12
**Status:** Approved for implementation

## Purpose

Today the system is a fully offline architecture demo: `DemoResearchProvider` returns
hardcoded evidence, and `run_demo.py` runs exactly one question passed as a CLI arg.
This spec adds a real, LLM-and-search-backed research path (DeepSeek for reasoning,
Tavily for web search, LangSmith for tracing) and turns `run_demo.py` into an
interactive, repeatable CLI — while keeping the existing offline demo path and its
tests working unchanged.

## Non-goals

- No conversational memory across questions. Each question runs as an independent
  research session (fresh graph thread/state). The research-state model in
  `state.py` represents one research session's evidence/conflicts/insights, not a
  chat history, and stretching it to also carry chat turns is out of scope.
- No new automated tests exercising real DeepSeek/Tavily calls (no CI credentials).
  New code is written to be dependency-injectable/mockable, but writing the mocks
  is left for a future pass if desired.
- No credibility-scoring heuristic based on domain authority. Source `credibility`
  continues to be a 0–1 number attached to each evidence item, just LLM-assigned
  instead of hardcoded, with the same known limitation the demo already has
  (unverified scoring) — documented, not solved, here.

## Architecture

### Provider layer

`research_system/live_provider.py` adds `LiveResearchProvider`, implementing the
existing `ResearchProvider` protocol (`research(specialty, question, context) -> dict`)
so `specialist_node()` and the graph don't change shape:

1. Build a specialty-scoped query (`f"{specialty} {question}"`) and call
   `langchain_tavily.TavilySearch` for ~5 results (title, url, content snippet).
2. Call `ChatDeepSeek` (from `langchain-deepseek`) with `.with_structured_output()`
   against a Pydantic schema constraining `source_url` to one of the returned
   result URLs, plus `claim`, `source_type`, `confidence`, `credibility`, `tags`.
   The prompt includes the specialty's role, the question, the search results, and
   existing `context` evidence (so sequential/hybrid branches still get richer
   context, matching current behavior).
3. `evidence_id` (`stable_id`) and `retrieved_at` (`utc_now`) are still generated
   in code, exactly as `DemoResearchProvider` does today — only the *content* is
   LLM-generated, not the bookkeeping fields.
4. Returns `{"evidence": [...], "contribution": {...}}` in the same shape
   `DemoResearchProvider.research()` returns today.

`DemoResearchProvider` is unchanged and remains the default when no DeepSeek key is
configured.

### Conflict / insight detection generalization

`agents.py`'s `build_cross_agent_insights` and `detect_and_resolve_conflicts`
currently pattern-match the EV demo's fixed tags (`"charging"`) and literal claim
substrings (`"$8b"`/`"$14b"`). That only fires for the canned demo data. Both become
factory functions parameterized by an optional LLM client:

- `build_cross_agent_insights(llm=None)` / `detect_and_resolve_conflicts(llm=None)`.
- `llm=None` → current hardcoded behavior (unchanged), so `tests/test_quality.py`,
  `tests/test_provenance.py`, and the offline demo keep working exactly as today.
- `llm` set → one `ChatDeepSeek` structured-output call over all accumulated
  evidence returns candidate conflicts (`issue`, `evidence_ids`, `severity`) and
  candidate insights (`statement`, `evidence_ids`, `why_emergent`).
- Code still owns the objective/deterministic parts, matching the current design's
  "explicit, auditable" stop/resolution logic:
  - Insight `confidence` = `average()` of parent evidence confidences (existing
    helper), not LLM-reported.
  - Conflict auto-resolution keeps the existing `confidence_gap >= 0.15 or
    credibility_gap >= 0.18` thresholds from `agents.py`; the LLM only identifies
    *which* evidence conflicts, not whether it's auto-resolvable.
  - Any LLM-proposed insight with fewer than 2 distinct `contributing_agents` is
    filtered out in code (the "emergent insight" definition is enforced by code,
    not trusted from the model).

### Graph wiring

`build_graph(provider=None, checkpointer=None, llm=None)` gains an `llm` parameter,
threaded into `build_cross_agent_insights(llm)` / `detect_and_resolve_conflicts(llm)`
node construction. Everything else in `graph.py` (the parallel/sequential/hybrid
branch topology) is unchanged.

### Config / provider selection

New `research_system/config.py`:
- Loads `.env` via `python-dotenv` (`load_dotenv()`), if present.
- `build_default_graph(checkpointer=None)`: if `DEEPSEEK_API_KEY` is set,
  constructs `ChatDeepSeek` + `LiveResearchProvider` (which itself requires
  `TAVILY_API_KEY`) and calls `build_graph(provider=live_provider, llm=llm,
  checkpointer=checkpointer)`. Otherwise prints a one-line notice
  ("No DEEPSEEK_API_KEY found — running with the offline demo provider.") and
  calls `build_graph(checkpointer=checkpointer)` (existing demo path, `llm=None`).
- `run_demo.py` calls `build_default_graph()` instead of `build_graph()` directly.

LangSmith tracing requires no code changes — `langchain-core` (a transitive
dependency via `langgraph`/`langchain-deepseek`) picks up `LANGSMITH_TRACING`,
`LANGSMITH_API_KEY`, and `LANGSMITH_PROJECT` from the environment automatically
once `.env` is loaded.

### Interactive CLI

`run_demo.py`'s `main()` becomes a loop:

1. Prompt: `Research question (or 'quit'): `.
2. On `quit`/`exit`/EOF (Ctrl-D) or `KeyboardInterrupt` (Ctrl-C), exit cleanly.
3. Otherwise run the existing HITL-aware `graph.invoke` / `interrupt` /
   `Command(resume=...)` flow to completion (unchanged from today, just wrapped in
   the loop) with a **fresh `thread_id`** and fresh `initial_state(question)` per
   question — no state carries over between questions.
4. Print the report/quality gate/telemetry as today, then loop back to the prompt.

### New dependencies

Added to `requirements.txt`: `langchain-deepseek`, `langchain-tavily`,
`python-dotenv`. (`langchain-core` arrives transitively via these.)

A new `.env.example` documents:
```
DEEPSEEK_API_KEY=
TAVILY_API_KEY=
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=
LANGSMITH_PROJECT=multi-agent-research-system
```

## Error handling

- Missing `DEEPSEEK_API_KEY` → silent fallback to demo provider (not an error;
  this is the existing offline-first behavior, just now automatic instead of the
  only option).
- `DEEPSEEK_API_KEY` set but `TAVILY_API_KEY` missing → `config.py` raises a clear
  `RuntimeError` at startup ("TAVILY_API_KEY is required when DEEPSEEK_API_KEY is
  set") rather than failing deep inside a graph node mid-run.
- Search/LLM call failures inside `LiveResearchProvider.research()` or the
  LLM-driven conflict/insight functions propagate as exceptions (no silent
  swallowing/fallback evidence) — consistent with "no error handling for
  scenarios that can't be meaningfully recovered from" and keeping failures
  visible instead of polluting evidence with fabricated fallback content.

## Testing

- No changes to `tests/test_planner.py`, `tests/test_quality.py`,
  `tests/test_provenance.py` — all exercise pure functions untouched by this work.
- `build_cross_agent_insights`/`detect_and_resolve_conflicts` keep their `llm=None`
  path as the tested, deterministic default.
- No new tests are added for `LiveResearchProvider` or the LLM-driven
  conflict/insight path in this pass (no CI credentials to exercise them); the
  code is structured with injectable clients so mocked tests can be added later
  without refactoring.
