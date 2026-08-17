# Memory management: short-term + long-term (semantic/episodic/procedural)

**Date:** 2026-08-17
**Status:** Approved for implementation

## Purpose

Today every research question is a fully independent run: nothing learned on one
question carries over to the next, even on the same topic. This adds a real,
working memory subsystem across LangGraph's two standard memory categories:

- **Short-term (thread-scoped)** — already exists via the checkpointer, just not
  durable across CLI process restarts. Upgraded to survive a HITL pause even if
  you quit and re-launch the CLI.
- **Long-term (cross-session)**, split into the three sub-types from the
  assignment source material:
  - **Semantic** — discrete facts about a topic, namespaced per topic.
  - **Episodic** — a log of past runs on a topic, used to inform future ones.
  - **Procedural** — versioned operational rules (thresholds, HITL defaults)
    that actually govern runtime behavior, analogous to an `AGENTS.md` bundle.

And the four-step management loop from the source material — capture, analyze
and refine, store and version, apply and consolidate — implemented as real,
runnable code rather than described in prose.

## What already exists vs. what's new

| Pillar | Status |
|---|---|
| Capture traces | **Already exists** — `telemetry` state list + optional LangSmith tracing. No new work beyond documentation. |
| Short-term memory | **Exists, gets upgraded** — checkpointer already provides this; swap `InMemorySaver` for a durable `SqliteSaver` on the real CLI path. |
| Semantic long-term memory | **New** |
| Episodic long-term memory | **New** |
| Procedural long-term memory | **New** |
| Analyze and refine | **New** |
| Store and version context | **New** — implemented via this repo's own git history, not an external product |
| Apply and consolidate | **New** |

## Architecture

### Storage substrate

Long-term memory uses the *real* `langgraph.store.memory.InMemoryStore` — the
actual LangGraph primitive nodes are meant to use — rather than a hand-rolled
key/value store. There is no offline-friendly official *persistent* Store
package (the durable options are Postgres-backed, which would break this
project's no-external-services design), so `research_system/memory/store.py`
adds a thin JSON-file read/write wrapper around an in-process `InMemoryStore`:
`load_persistent_store(path)` rehydrates one from disk at startup via
`store.put()` per record; `persist_store(store, path)` dumps it back via
`store.list_namespaces()` + `store.search()`. The Store itself is never
reimplemented — only persisted around.

Procedural memory is deliberately **not** in the Store. Per the source
material's own framing ("AGENTS.md file bundles"), it's a plain versioned file
— `research_system/memory/procedural_rules.json`, committed to git. "Tags
(dev/prod) and rollback" from the source material map onto this repo's actual
git tooling: git branches/tags distinguish rule-file versions, and rollback is
`git checkout <rev> -- research_system/memory/procedural_rules.json` or
`git revert`. This repo already uses git as its real version control; there's
no need for (and I don't have confidence there's a matching real product
called) a separate external "Context Hub" to integrate against.

### Dependency injection, not `get_store()`

LangGraph offers an implicit `get_store()`/contextvar mechanism for nodes to
reach the store compiled into the graph. This repo instead already has an
established, explicit pattern for exactly this kind of dependency:
`specialist_node(provider, specialty)`, `build_cross_agent_insights(llm=None)`,
and `detect_and_resolve_conflicts(llm=None)` are all factories that close over
their dependency and return a plain `state -> dict` function. Memory-aware
nodes follow the same idiom — the store is passed explicitly into a factory,
not fetched implicitly at runtime. This keeps every node trivially unit
-testable with a plain `InMemoryStore()` and a plain state dict, exactly like
the rest of the codebase, and avoids introducing a second, inconsistent way of
wiring dependencies. (`build_graph()` still compiles with `store=store` so the
store is available to any future code that does want `get_store()`.)

### Topic derivation

Long-term memory is namespaced by **topic**, auto-derived from the question
text (`research_system/memory/topic.py`, `derive_topic(question) -> str`):
lowercase, tokenize, drop a stopword list (English filler words *and*
`planner.py`'s own `speed_terms`/`dependency_terms`, since those describe *how*
to research, not *what* the topic is), then sort and join *all* remaining
significant words with no truncation/cap — a longer question just produces a
longer (still stable, still deterministic) topic key. Sorting (not positional
order) makes the key stable
regardless of phrasing — "quick overview of the Indian EV market" and "give me
an Indian EV market overview" land on the same topic key.

**Known limitation, stated up front:** this is a simple deterministic
heuristic, not semantic topic modeling. Two real-world-same-topic questions
phrased with little vocabulary overlap may land on different topic keys, and
unrelated questions that happen to share enough vocabulary could collide. A
more robust version (embedding similarity search, which `InMemoryStore`
actually supports via its `index_config` parameter) is a natural future
improvement, not built here — consistent with this repo's practice of
documenting real limitations rather than overclaiming.

### Data flow through the graph

1. `research_system/planner.py`'s `planner_node` becomes a factory,
   `build_planner_node(store=None)`. The returned node computes `topic`,
   reads `semantic.relevant_facts(store, topic)` and
   `episodic.recent_episodes(store, topic)`, and writes both into a new
   `state["memory_context"]` field (`{"topic": ..., "semantic_facts": [...],
   "relevant_episodes": [...]}`) alongside the existing `execution_plan`.
2. `agents.py`'s `specialist_node` reads `state.get("memory_context")` and
   passes it to `provider.research(specialty, question, context, memory=...)`
   — a new optional parameter on the `ResearchProvider` protocol.
   - **`DemoResearchProvider`**: evidence content is fixed regardless of
     input (that's the whole point of the demo provider), so memory can't
     change *what* it finds — but it appends a visible "Related memory: ..."
     line to the contribution summary when facts/episodes exist, so the
     mechanism is genuinely exercised and observable even offline.
   - **`LiveResearchProvider`**: folds `memory` into `_build_prompt()` as an
     extra context block, so it genuinely influences the DeepSeek call.
3. `research_system/quality.py`'s `quality_gate_node` reads its thresholds via
   `procedural.load_rules()` instead of the module-level constants directly.
   `load_rules()` returns those exact constants as defaults when no rules file
   override exists, so existing behavior and existing tests are unchanged
   by default.
4. After a run completes, `run_demo.py` (and `run_eval.py`, minus the
   interactive prompt — see Error Handling) automatically:
   - `episodic.record_episode(store, topic, {...})` — question, mode,
     active_agents, quality-gate result, whether HITL was needed, insight
     count, timestamp.
   - `reflect.reflect_on_topic(store, topic)` — the analyze-and-refine step
     (see below).
   - `persist_store(store, path)` — writes the updated store back to disk.

### The analyze-and-refine step (`reflect.py`)

Deterministic pure-Python pattern analysis over recent episodes for a topic —
**not** an LLM call, on either provider path. This mirrors the existing
`llm=None` hardcoded-logic pattern already used for
`build_cross_agent_insights`/`detect_and_resolve_conflicts`: keep it free,
offline, and fully unit-testable without network access. (An LLM-powered
reflection pass that writes richer, nuanced facts is a reasonable future
extension, following the same `llm=None`/`llm=<client>` factory split already
established elsewhere — out of scope here to keep this addition bounded.)

Two concrete rules for v1:

1. **Semantic fact upsert (always applied automatically):** if a
   `quality_gate.failures` reason recurs across the topic's recent episodes,
   or a conflict's `issue` text recurs with `status: "open"`, upsert a
   semantic fact recording the pattern and occurrence count. These are what
   step 2 above surfaces to specialists on the *next* run for that topic.
2. **Procedural rule proposal (never auto-applied):** if the *same*
   `quality_gate.failures` reason recurs in the last 3 consecutive episodes
   for a topic, propose flipping `mandatory_human_review` on for future runs
   on that topic — a real, already-existing lever
   (`research_system/state.py`'s `mandatory_human_review: bool`,
   already read by `route_after_quality_gate()`). `run_demo.py` prints the
   proposal and asks for a plain approve/reject (not a new LangGraph
   `interrupt()` — this happens *after* the graph run completes, so it's
   implemented as a CLI-layer post-step, avoiding a new node/interrupt type
   in the state machine for something this infrequent). Applying a proposal
   means `derive_topic()`-keyed override is written into
   `procedural_rules.json`, and `run_demo.py`'s `initial_state(question)`
   (which already hardcodes `mandatory_human_review: False` today) checks for
   a matching topic override going forward, using the same `derive_topic()`
   helper.

The `procedural_rules.json` file format also supports manually-editable
numeric threshold overrides (`MIN_EVIDENCE`, `MIN_SOURCE_TYPES`,
`MIN_AVG_CONFIDENCE`, `MIN_CROSS_AGENT_INSIGHTS`) for a human to curate
directly — `reflect.py`'s v1 auto-detection only ever proposes the
`mandatory_human_review` rule type, to keep the auto-detection logic small and
safe; the file format is intentionally slightly ahead of what's auto-proposed,
for a human to use directly.

### Short-term memory durability

`build_graph()` gains a `store=None` parameter (defaults to a fresh
`InMemoryStore()`, so existing callers/tests are unaffected) and compiles with
`store=store`. Its checkpointer default stays `InMemorySaver` — unchanged, no
new hard dependency for library/test callers. `config.py`'s
`build_default_graph()` (the real CLI path) additionally constructs a
`SqliteSaver` pointed at `.research_memory/checkpoints.sqlite` when the
`langgraph-checkpoint-sqlite` package is available, so a HITL-paused thread
survives quitting and restarting the CLI — a concrete, demonstrable upgrade to
the short-term/thread-scoped pillar.

### Local memory storage location

`.research_memory/` (store JSON + sqlite checkpoints) is created on first use
and **gitignored**, same treatment as `.env` — it can accumulate real research
content across sessions and has no place in a public repo.
`research_system/memory/procedural_rules.json` is the opposite: committed,
versioned, ships with defaults that exactly match today's hardcoded
`quality.py` constants, so a fresh clone behaves identically to today until
memory actually proposes and you approve a change.

## Error handling

- Corrupted or missing `.research_memory/store.json` →
  `load_persistent_store()` starts a fresh empty store (printed note, not a
  crash) — memory is an enhancement layer, not a hard requirement for
  research to work.
- Missing or malformed `procedural_rules.json` → `load_rules()` returns the
  hardcoded defaults, never raises.
- **Deliberate exception to this codebase's usual "fail loudly, never swallow
  errors" rule:** `run_demo.py`/`run_eval.py` wrap the post-run episode
  recording + reflection call in a try/except that prints a warning and
  continues, rather than letting a memory-layer exception take down a run
  that already successfully produced a report. This is justified because
  memory failures here cannot corrupt the *returned research result* the way
  e.g. a live-provider evidence-fabrication bug could — the report is already
  final by the time memory bookkeeping runs. Everywhere else (provider calls,
  structured-output parsing, the actual research path) keeps the existing
  fail-loud behavior unchanged.
- `run_eval.py` never prompts interactively for a procedural rule proposal
  (would hang an unattended batch run) — it reports "procedural rule change
  proposed — review via run_demo.py" in its summary instead of applying or
  discarding it silently.

## Testing

- `tests/test_memory_topic.py` — `derive_topic()` stability across rephrased
  but same-topic questions; distinctness across genuinely different topics.
- `tests/test_memory_store.py` — persist/load round-trip against a temp file;
  corrupted-file fallback to an empty store.
- `tests/test_memory_episodic_semantic.py` — record/retrieve against a plain
  `InMemoryStore()`, no persistence needed for these — fast, pure logic.
- `tests/test_memory_procedural.py` — default-fallback and override-applied
  cases for `load_rules()`, against a temp file.
- `tests/test_memory_reflect.py` — the two v1 pattern-detection rules, against
  constructed episode fixtures — no LLM, no network.
- `tests/test_planner.py` — extended for `build_planner_node(store)`, mirroring
  how `tests/test_agents.py` already tests `build_cross_agent_insights(llm=None)`
  as a factory.
- `tests/test_agents.py` — extended: `specialist_node` passes `memory_context`
  through to the provider (fake-provider capture, matching the existing
  `FakeProvider` test pattern).
- `tests/test_quality.py` — confirm `quality_gate_node` with no rules file
  present produces byte-identical results to today (regression guard for the
  "existing tests unaffected" claim).
- End-to-end manual verification (offline, free): run two questions on the
  same topic back-to-back via the demo provider, confirm run 2's contribution
  summary shows a "Related memory" note referencing run 1.

## Non-goals

- No LLM-powered reflection in this pass (see above) — pure deterministic
  pattern analysis only.
- No real "Context Hub" product integration — versioning is this repo's own
  git history.
- No embedding-based topic similarity — simple keyword-heuristic topic keys
  only, limitation documented above.
- No change to the existing `more_research` accumulation behavior/limitation
  (already documented in `CLAUDE.md`) — orthogonal to this work.
