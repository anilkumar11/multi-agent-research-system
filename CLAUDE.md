# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A runnable reference implementation for a "Designing a Multi-Agent Research System" assignment, built on LangGraph. It demonstrates: shared collaborative state, adaptive (parallel/sequential/hybrid) orchestration, provenance-tracked evidence, conflict resolution, an explicit quality-gate stop condition, and human-in-the-loop review via LangGraph `interrupt()`.

`DemoResearchProvider` (`research_system/provider.py`) returns deterministic, hardcoded evidence per specialty so the graph runs fully offline with no API keys — this remains the default when `DEEPSEEK_API_KEY` is unset. `LiveResearchProvider` (`research_system/live_provider.py`) is a real implementation: it grounds evidence in live Tavily search results and uses `ChatDeepSeek` (`langchain-deepseek`) structured output to turn those results into `Evidence` items, enforcing that every `source_url` is copied verbatim from an actual search result (never invented by the model). `research_system/config.py`'s `build_default_graph()` picks between the two based on which API keys are present — see the Commands section above. Both providers implement the same `ResearchProvider` protocol, so `graph.py`'s topology never needs to change.

## Commands

```bash
# Setup
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Optional: enable the live DeepSeek/Tavily provider + LangSmith tracing
cp .env.example .env   # then fill in DEEPSEEK_API_KEY / TAVILY_API_KEY / LANGSMITH_*

# Run the interactive CLI (prompts for a research question; 'quit' to exit)
python run_demo.py

# Run the batch eval (autonomous quality-gate pass rate across a question set)
python run_eval.py                    # offline demo provider, default 5-question set, free
python run_eval.py --live             # live provider if configured (bills API calls)
python run_eval.py "custom question"  # override the question set

# Run all tests
python -m unittest discover -s tests -v

# Run a single test file / test case
python -m unittest tests.test_planner -v
python -m unittest tests.test_planner.PlannerTests.test_hybrid_for_complex_general_question -v
```

Without a `DEEPSEEK_API_KEY` in the environment, `run_demo.py` prints a notice and falls
back to the deterministic offline `DemoResearchProvider` — no keys are required to run or
test the system. With `DEEPSEEK_API_KEY` and `TAVILY_API_KEY` set, it uses
`LiveResearchProvider` (real Tavily search + `ChatDeepSeek` structured-output calls)
instead. `LANGSMITH_*` vars (see `.env.example`) enable tracing automatically with no
code changes — `langchain-core` picks them up from the environment.

`run_eval.py` runs a fixed (or custom) question set through the graph and reports the
**autonomous quality-gate pass rate**: the fraction of questions where `quality_gate.passed`
was `True` on the first pass, with no human review needed. Any question that pauses for HITL
is auto-approved (so the run still completes and the summary reflects true end-to-end
behavior) but counted as not-autonomous. This is the only "success rate" the system
currently reports anywhere — there is no LangSmith Evaluations pipeline wired up (only
passive tracing via `LANGSMITH_TRACING`, which shows individual run traces, not an
aggregate pass-rate metric).

When the graph pauses for human review, it prints a JSON review packet and expects one of
`approve`, `more`, or `resolve` on stdin. `resolve` auto-picks the strongest evidence by
credibility/confidence (see `strongest_conflict_resolution` in `run_demo.py`) — it's a
stand-in for what a human analyst would decide in a real UI. After a question's report
prints, the CLI loops back to prompt for another question (each question is an independent
research session/thread — there is no cross-question conversational memory).

## Architecture

Everything is a LangGraph `StateGraph` over a single shared `ResearchState` (`research_system/state.py`). List-valued fields (`evidence`, `contributions`, `cross_agent_insights`, `conflicts`, `synthesis_threads`, `telemetry`, `human_decisions`) use `Annotated[..., operator.add]` reducers so parallel branches can append to shared state concurrently without clobbering each other.

Node execution flow (`research_system/graph.py`):

```
START -> planner -> {parallel_start | s_web | hybrid_start}   (routed by execution_plan.mode)
      -> [specialist nodes for the chosen mode]
      -> cross_agent_insights -> conflict_resolution -> quality_gate
      -> {human_review -> {planner | synthesis}} | synthesis -> END
```

Key pieces, each in its own module:

- **`planner.py`** — `choose_execution_plan()` is a transparent keyword-heuristic that picks `parallel`, `sequential`, or `hybrid` based on speed vs. dependency language in the question. The decision (mode + rationale + stages) is written into `state["execution_plan"]` so it's observable/auditable, not hidden in an LLM's head. On resume-after-`more_research`, control returns here to replan. `choose_active_agents(question)` is a separate decision (also written into `execution_plan["active_agents"]`): `web_research`/`data_analysis` always run; `trend_analysis`/`competitive_intelligence` are excluded only when the question explicitly signals that lens is out of scope (e.g. "excluding competitor analysis") — deliberately opt-out-on-signal rather than opt-in-on-topic, so the no-signal default still runs all four specialists and existing behavior is unaffected.
- **`graph.py`** — Builds three parallel node families (`p_*`, `s_*`, `h_*`) for the same four specialties (`web_research`, `data_analysis`, `trend_analysis`, `competitive_intelligence`), wired differently per mode:
  - **parallel**: all four specialists fan out from `parallel_start` simultaneously.
  - **sequential**: `s_web -> s_data -> s_trend -> s_comp`, each downstream agent sees all upstream evidence via `state["evidence"]`.
  - **hybrid** (default for mixed-dependency questions): `hybrid_start` fans out `h_web`/`h_data` in parallel, both join at the `hybrid_stage2` barrier node, then `h_trend`/`h_comp` fan out in parallel using that stage's evidence.
  All three branches converge on `cross_agent_insights`.
- **`agents.py`** — `specialist_node(provider, specialty)` is a factory producing the actual node function that calls `provider.research(...)` with all evidence accumulated so far as context. If `state["execution_plan"]["active_agents"]` exists and excludes this `specialty` (see `planner.py`), the node returns a "Skipped: ..." contribution and makes **no provider call at all** — visible in the report's "Specialist contributions" section, and on the live path this means an excluded agent costs nothing. Also holds `build_cross_agent_insights()` (finds evidence tagged `"charging"` contributed by >=2 different agents and synthesizes an emergent insight — a finding no single specialist could have produced alone) and `detect_and_resolve_conflicts()` (specifically looks for `"market_size"`-tagged evidence with conflicting `$8B`/`$14B` claims; auto-resolves via credibility/confidence gap, otherwise leaves the conflict `open` for HITL).
  `build_cross_agent_insights(llm=None)` and `detect_and_resolve_conflicts(llm=None)` are factories: with `llm=None` (the demo path) they run the exact hardcoded logic described above; with an `llm` (the live path, wired in by `config.py`) they instead make one `ChatDeepSeek` structured-output call over all accumulated evidence to *identify* candidate conflicts/insights, while conflict auto-resolution thresholds and the "≥2 distinct agents" emergent-insight rule remain enforced in code either way — the LLM finds candidates, code decides what qualifies.
- **`quality.py`** — `quality_gate_node()` is the explicit stop condition: minimum evidence count, minimum distinct source types, minimum average confidence, minimum cross-agent insights, zero unresolved high-severity conflicts, and a max iteration cap. Thresholds are module-level constants at the top of the file. `route_after_quality_gate()` sends state to `human_review` if the gate fails (or `mandatory_human_review` is set) else to `synthesis`.
- **`hitl.py`** — `human_review_node()` calls LangGraph's `interrupt()` exactly once per invocation with a JSON-serializable payload (question, quality gate, open conflicts, and an `instructions` block describing the three possible actions). On resume, `Command(resume=decision)` is passed back in with `action` one of `approve` / `resolve` / `more_research`. Conflict resolutions are **appended**, not mutated in place — `latest_conflicts()` collapses the append-only history to the latest record per `conflict_id` when synthesis reads conflicts (`research_system/quality.py`'s `quality_gate_node` does NOT do this — it counts the raw conflicts list, so a resolved conflict's earlier "open" record still counts toward `unresolved_high_conflicts` if both records exist in state). `route_after_human()` sends `more_research` back to `planner` (bounded by `iteration_count <= 3`), everything else to `synthesis`.
- **`synthesis.py`** — `synthesis_node()` renders the final markdown report by walking accumulated state: executive synthesis from cross-agent insights (with provenance URLs resolved via `evidence_by_id`), per-specialist contributions, conflicts/uncertainty (via `latest_conflicts`), the quality-gate scorecard, the human decision if any, and the orchestration rationale.
- **`utils.py`** — `timed_node()` wraps every node (used throughout `graph.py`) to emit a `TelemetryEvent` (duration, evidence before/after, execution mode, and a `_telemetry_note` the wrapped node can set) into the reducer-backed `telemetry` list. `stable_id()` produces deterministic SHA1-based IDs so evidence/insight/conflict IDs are reproducible given the same inputs — useful for tests and dedup.
- **`provider.py`** — `ResearchProvider` is a `Protocol` with one method: `research(specialty, question, context) -> {"evidence": [...], "contribution": {...}}`. `DemoResearchProvider` (this file) returns fixed evidence per specialty (with tags like `charging`, `market_size` that `agents.py`'s hardcoded `llm=None` path specifically looks for) so the demo is fully deterministic and offline; `LiveResearchProvider` (`research_system/live_provider.py`, described above) is the real second implementation.

### Provenance model

Every `Evidence` item carries `parent_evidence_ids` pointing at upstream evidence it was derived from, and `CrossAgentInsight` items carry `parent_evidence_ids` pointing at the evidence that produced them — so a final report claim can be traced back through insights to original evidence and its source. See `DESIGN_MAPPING.md` for the full rubric-requirement-to-code mapping and `README.md` for worked examples of provenance, conflict resolution, and emergent insights.

### Checkpointing

`build_graph()` defaults to `InMemorySaver` (in-process, non-durable) as the checkpointer, keyed by `thread_id` in the invoke config. Swap in a durable LangGraph checkpointer for production use; the graph resumes an interrupted run by reusing the same `thread_id` with `Command(resume=decision)`.

### Known limitation: `more_research` accumulation

Each `more_research` HITL decision re-runs the specialist nodes, and their output is
appended (never deduped) to the reducer-backed evidence/conflict lists via
`operator.add`. On the demo path this is mostly harmless (deterministic IDs mean
re-runs mostly re-add identical items). On the live path, DeepSeek's non-deterministic
phrasing means re-runs produce *different* `stable_id` values for the same underlying
fact, so evidence and conflicts can grow unbounded across iterations and the quality
gate's `evidence_count`/conflict counts become unreliable signals after repeated
`more_research` passes. This is a pre-existing characteristic of the reducer-based
state model (not introduced by the live provider), documented here rather than fixed,
since a real fix means deciding whether `more_research` means "augment" or "replace"
state — a design decision that touches the accumulation model shared by both providers.

### Memory management

`research_system/memory/` implements LangGraph's standard short-term/long-term
memory split, end to end:

- **Short-term (thread-scoped)** — the existing checkpointer (`InMemorySaver` by
  default in `build_graph()`) already provides this. `config.py`'s
  `build_default_graph()` upgrades it to a `SqliteSaver` pointed at
  `.research_memory/checkpoints.sqlite` when `langgraph-checkpoint-sqlite` is
  installed, so a HITL-paused thread survives quitting and restarting the CLI.
- **Semantic long-term memory** (`memory/semantic.py`) — durable facts about a
  topic (`upsert_fact`/`relevant_facts`), namespaced `(topic, "semantic")` in a
  real `langgraph.store.memory.InMemoryStore`.
- **Episodic long-term memory** (`memory/episodic.py`) — a log of past runs on
  a topic (`record_episode`/`recent_episodes`), namespaced `(topic, "episodic")`.
- **Procedural long-term memory** (`memory/procedural.py`,
  `memory/procedural_rules.json`) — a plain, git-versioned JSON file (not a
  store entry), holding quality-gate thresholds and per-topic
  `mandatory_human_review` overrides. Ships with defaults that exactly match
  `quality.py`'s original hardcoded constants. "Versioning/rollback" is this
  repo's own git history — `git checkout <rev> -- research_system/memory/procedural_rules.json`
  to roll back, git tags/branches to distinguish versions.

The store is threaded through explicitly (`build_graph(store=...)` →
`build_planner_node(store)` → `state["memory_context"]` → `specialist_node` →
`provider.research(..., memory=...)`), matching this codebase's existing
factory/closure dependency-injection pattern (`specialist_node`,
`build_cross_agent_insights(llm=None)`) rather than LangGraph's implicit
`get_store()`.

**The management loop**, per `research_system/memory/reflect.py`:
1. **Capture** — already existed (`telemetry` state + optional LangSmith
   tracing); `run_demo.py`/`run_eval.py` additionally call
   `episodic.record_episode()` after every completed run.
2. **Analyze and refine** — `reflect.reflect_on_topic()`, pure deterministic
   Python pattern analysis (no LLM, on either provider path, in this
   iteration): upserts a semantic fact when a failure pattern recurs across
   ≥2 recent episodes for a topic, and *proposes* (never auto-applies) forcing
   `mandatory_human_review` for a topic when the same failure recurs across
   all of the last 3 episodes.
3. **Store and version** — `store.persist_store()` writes the long-term
   memory store to `.research_memory/store.json` (gitignored, same treatment
   as `.env`) after every run; procedural rule changes go through
   `run_demo.py`'s approve/reject prompt before
   `procedural.apply_mandatory_review_override()` writes them to the
   committed `procedural_rules.json`.
4. **Apply and consolidate** — `build_planner_node()` reads relevant semantic
   facts + episodes back into `state["memory_context"]` on every new run;
   `DemoResearchProvider` surfaces them as a "Related memory: ..." note (demo
   evidence itself can't change, so this is how the mechanism stays visible
   offline); `LiveResearchProvider` folds them into the DeepSeek prompt, where
   they can genuinely influence research; `quality_gate_node` reads
   procedural thresholds via `load_rules()`.

`run_eval.py`'s default (non-`--live`) path exercises `record_episode`/
`reflect_on_topic` in-process but does **not** persist to disk, so it stays a
deterministic, repeatable baseline measurement across separate invocations;
pass `--live` to accumulate real persisted memory.
