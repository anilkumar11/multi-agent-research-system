# Memory Management Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add real short-term (durable, thread-scoped) and long-term (semantic, episodic, procedural) memory to the research system, plus the capture → analyze-and-refine → store-and-version → apply-and-consolidate loop, all as working, tested code.

**Architecture:** A new `research_system/memory/` package wraps LangGraph's real `InMemoryStore` with local JSON persistence (no offline-friendly durable Store package exists). Memory-aware nodes (`planner`, `specialist_node`) receive the store via the same explicit factory/closure dependency-injection pattern already used throughout this codebase (`specialist_node(provider, specialty)`, `build_cross_agent_insights(llm=None)`) — not LangGraph's implicit `get_store()`. Procedural memory is a plain git-versioned JSON file, not a store entry. Short-term memory gets a durability upgrade via `SqliteSaver`.

**Tech Stack:** `langgraph.store.memory.InMemoryStore` (already installed), `langgraph-checkpoint-sqlite` (new dependency), Python stdlib `json`/`sqlite3`/`uuid` — no new required external services.

**Spec:** `docs/superpowers/specs/2026-08-17-memory-management-design.md`

## Global Constraints

- Existing tests' *numeric outcomes* must stay identical where the spec claims "unaffected": `PlannerTests` (3 original tests in `tests/test_planner.py`), `tests/test_provenance.py`, and the original 2 `QualityGateTests` in `tests/test_quality.py`. Several of these test *files* are extended with new tests/imports in this plan — that's expected — but no existing assertion's expected value changes.
- `reflect.py`'s analyze-and-refine step is pure deterministic Python pattern analysis — no LLM call, on either provider path, in this iteration (per spec's Non-goals).
- No real network/API calls in any automated test.
- `.research_memory/` (persisted store JSON + sqlite checkpoints) is gitignored. `research_system/memory/procedural_rules.json` is committed, with defaults that exactly match today's hardcoded `quality.py` constants: `min_evidence=4`, `min_source_types=2`, `min_avg_confidence=0.70`, `min_cross_agent_insights=1`.
- Topic derivation (`derive_topic()`) has no length cap on the resulting key, and is documented as a simple keyword heuristic, not semantic topic modeling.
- Memory-layer exceptions in `run_demo.py`/`run_eval.py`'s post-run capture step must never propagate past a run whose report already printed successfully (deliberate, documented exception to this codebase's usual fail-loud rule — justified because memory bookkeeping here cannot corrupt an already-delivered research result).
- The Store is accessed via explicit factory/closure injection (matching `specialist_node`/`build_cross_agent_insights`/`detect_and_resolve_conflicts`), never via LangGraph's `get_store()`/contextvar mechanism.
- `InMemoryStore.search()`/`.list_namespaces()` default to `limit=10` — every bulk read in this codebase passes an explicit `MAX_RECORDS=10_000` instead of relying on that default.
- `SqliteSaver(conn)` requires an explicit `.setup()` call before first use (verified directly against the installed `langgraph-checkpoint-sqlite==2.0.11`) — omitting it is a bug, not an oversight to fix later.
- A compiled graph exposes its store back via `compiled_graph.store` (verified directly against the installed `langgraph==0.6.11`) — this is how `run_demo.py`/`run_eval.py` reach the store after `build_default_graph()`/`build_graph()` return, without changing either function's return type.

---

### Task 1: `research_system/memory/topic.py` — topic derivation

**Files:**
- Create: `research_system/memory/__init__.py`
- Create: `research_system/memory/topic.py`
- Test: `tests/test_memory_topic.py`

**Interfaces:**
- Produces: `derive_topic(question: str) -> str` — consumed by Tasks 5, 6, 11, 12.

- [ ] **Step 1: Write the failing test**

Create `tests/test_memory_topic.py`:

```python
import unittest

from research_system.memory.topic import derive_topic


class DeriveTopicTests(unittest.TestCase):
    def test_stable_across_rephrasing(self):
        a = derive_topic("Quick overview of the Indian EV market")
        b = derive_topic("Indian EV market overview, give me a quick scan")
        self.assertEqual(a, b)
        self.assertEqual(a, "ev_indian_market")

    def test_distinct_for_different_topics(self):
        ev = derive_topic("Quick overview of the Indian EV market")
        chips = derive_topic("What should we know about semiconductor supply chains?")
        self.assertNotEqual(ev, chips)

    def test_all_stopword_question_falls_back_to_general(self):
        self.assertEqual(derive_topic("Should it do this?"), "general")
        self.assertEqual(derive_topic(""), "general")
        self.assertEqual(derive_topic("Please give me a quick overview."), "general")

    def test_case_and_punctuation_insensitive(self):
        a = derive_topic("INDIAN EV MARKET!!")
        b = derive_topic("indian, ev, market")
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest tests.test_memory_topic -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'research_system.memory'`

- [ ] **Step 3: Write the implementation**

Create `research_system/memory/__init__.py` (empty file — marks the package).

Create `research_system/memory/topic.py`:

```python
from __future__ import annotations

import re

_STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "of", "in", "on", "for", "to", "is", "are",
    "what", "which", "that", "this", "it", "its", "into", "with", "should",
    "would", "could", "will", "do", "does", "did", "take", "give", "me",
    "please", "so", "just", "there",
    "quick", "overview", "scan", "landscape", "fast", "snapshot",
    "first", "then", "after", "depends", "because", "impact",
    "quantify", "forecast", "projection", "why", "determine", "benefits", "most",
    "excluding", "exclude", "ignore", "skip", "without", "no", "not",
})


def derive_topic(question: str) -> str:
    """
    Deterministic, offline topic key for namespacing long-term memory. A simple
    keyword heuristic, not semantic topic modeling: strips filler words and
    meta-instruction words (words about HOW to research, not WHAT the topic
    is), then sorts and joins whatever's left so the key is stable regardless
    of phrasing/word order. No length cap -- a longer question just produces a
    longer (still stable, still deterministic) key.

    Known limitation: two real-world-same-topic questions phrased with little
    vocabulary overlap may land on different topic keys. A more robust version
    (embedding similarity, which InMemoryStore actually supports via
    index_config) is a reasonable future improvement, not built here.
    """
    words = re.findall(r"[a-z0-9]+", question.lower())
    significant = sorted({w for w in words if w not in _STOPWORDS and len(w) > 1})
    return "_".join(significant) if significant else "general"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m unittest tests.test_memory_topic -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add research_system/memory/__init__.py research_system/memory/topic.py tests/test_memory_topic.py
git commit -m "feat: add derive_topic() for long-term memory namespacing"
```

---

### Task 2: `research_system/memory/store.py` — persistent Store wrapper

**Files:**
- Create: `research_system/memory/store.py`
- Test: `tests/test_memory_store.py`

**Interfaces:**
- Consumes: `langgraph.store.memory.InMemoryStore` (already installed).
- Produces: `load_persistent_store(path: str) -> InMemoryStore`, `persist_store(store: InMemoryStore, path: str) -> None`, `MAX_RECORDS = 10_000` — consumed by Tasks 3, 11, 12.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_memory_store.py`:

```python
import os
import tempfile
import unittest

from langgraph.store.memory import InMemoryStore

from research_system.memory.store import load_persistent_store, persist_store


class StorePersistenceTests(unittest.TestCase):
    def test_round_trip_persists_and_reloads_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "store.json")

            store = InMemoryStore()
            store.put(("topic-a", "semantic"), "fact-1", {"pattern": "x"})
            persist_store(store, path)

            reloaded = load_persistent_store(path)
            item = reloaded.get(("topic-a", "semantic"), "fact-1")
            self.assertIsNotNone(item)
            self.assertEqual(item.value, {"pattern": "x"})

    def test_missing_file_returns_empty_store(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "does-not-exist.json")
            store = load_persistent_store(path)
            self.assertEqual(store.list_namespaces(), [])

    def test_corrupted_file_falls_back_to_empty_store(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "corrupt.json")
            with open(path, "w") as f:
                f.write("{not valid json")

            store = load_persistent_store(path)
            self.assertEqual(store.list_namespaces(), [])

    def test_persist_creates_parent_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "nested", "dir", "store.json")
            store = InMemoryStore()
            store.put(("t", "episodic"), "e1", {"question": "q"})
            persist_store(store, path)
            self.assertTrue(os.path.exists(path))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m unittest tests.test_memory_store -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'research_system.memory.store'`

- [ ] **Step 3: Write the implementation**

Create `research_system/memory/store.py`:

```python
from __future__ import annotations

import json
import os

from langgraph.store.memory import InMemoryStore

MAX_RECORDS = 10_000  # InMemoryStore's search()/list_namespaces() default to
                       # limit=10; every bulk read in this package passes this
                       # explicit limit instead, to actually fetch everything.


def load_persistent_store(path: str) -> InMemoryStore:
    """
    Rehydrate an InMemoryStore from a JSON file. Starts fresh (with a printed
    warning) if the file is missing, unreadable, or malformed -- long-term
    memory is an enhancement layer, not a hard requirement for research to
    work.
    """
    store = InMemoryStore()
    if not os.path.exists(path):
        return store

    try:
        with open(path, "r") as f:
            records = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"Warning: could not read memory store at {path} ({exc}); starting fresh.")
        return store

    for record in records:
        store.put(tuple(record["namespace"]), record["key"], record["value"])
    return store


def persist_store(store: InMemoryStore, path: str) -> None:
    """Dump every namespace/key/value in the store to a JSON file."""
    records = []
    for namespace in store.list_namespaces(limit=MAX_RECORDS):
        for item in store.search(namespace, limit=MAX_RECORDS):
            records.append({
                "namespace": list(item.namespace),
                "key": item.key,
                "value": item.value,
            })

    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w") as f:
        json.dump(records, f, indent=2)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m unittest tests.test_memory_store -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Run the full test suite to confirm no regressions**

Run: `python3 -m unittest discover -s tests -v`
Expected: PASS — all existing tests plus the new ones.

- [ ] **Step 6: Commit**

```bash
git add research_system/memory/store.py tests/test_memory_store.py
git commit -m "feat: add persist/load JSON wrapper around InMemoryStore"
```

---

### Task 3: `research_system/memory/episodic.py` + `semantic.py`

**Files:**
- Create: `research_system/memory/episodic.py`
- Create: `research_system/memory/semantic.py`
- Test: `tests/test_memory_episodic_semantic.py`

**Interfaces:**
- Consumes: `research_system.memory.store.MAX_RECORDS` (Task 2); `research_system.utils.utc_now()` (existing).
- Produces: `record_episode(store, topic: str, episode: dict) -> None`, `recent_episodes(store, topic: str, limit: int = 5) -> list[dict]`, `upsert_fact(store, topic: str, key: str, fact: dict) -> None`, `relevant_facts(store, topic: str) -> list[dict]` — all consumed by Tasks 5, 6, 12.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_memory_episodic_semantic.py`:

```python
import unittest

from langgraph.store.memory import InMemoryStore

from research_system.memory.episodic import record_episode, recent_episodes
from research_system.memory.semantic import relevant_facts, upsert_fact


class EpisodicMemoryTests(unittest.TestCase):
    def test_record_and_retrieve_single_episode(self):
        store = InMemoryStore()
        record_episode(store, "topic-a", {"question": "q1", "mode": "parallel"})

        episodes = recent_episodes(store, "topic-a")
        self.assertEqual(len(episodes), 1)
        self.assertEqual(episodes[0]["question"], "q1")
        self.assertIn("recorded_at", episodes[0])

    def test_recent_episodes_respects_limit_and_recency_order(self):
        store = InMemoryStore()
        for i in range(5):
            record_episode(store, "topic-b", {"question": f"q{i}"})

        episodes = recent_episodes(store, "topic-b", limit=2)
        self.assertEqual(len(episodes), 2)
        # newest-first: the last-recorded question (q4) must come before q0
        questions = [e["question"] for e in episodes]
        self.assertIn("q4", questions)
        self.assertNotIn("q0", questions)

    def test_episodes_are_scoped_per_topic(self):
        store = InMemoryStore()
        record_episode(store, "topic-c", {"question": "c"})
        record_episode(store, "topic-d", {"question": "d"})

        self.assertEqual(len(recent_episodes(store, "topic-c")), 1)
        self.assertEqual(len(recent_episodes(store, "topic-d")), 1)

    def test_no_episodes_returns_empty_list(self):
        store = InMemoryStore()
        self.assertEqual(recent_episodes(store, "topic-empty"), [])


class SemanticMemoryTests(unittest.TestCase):
    def test_upsert_and_retrieve_fact(self):
        store = InMemoryStore()
        upsert_fact(store, "topic-a", "fact-1", {"pattern": "recurring_gate_failure"})

        facts = relevant_facts(store, "topic-a")
        self.assertEqual(len(facts), 1)
        self.assertEqual(facts[0]["pattern"], "recurring_gate_failure")
        self.assertIn("updated_at", facts[0])

    def test_upsert_same_key_replaces_not_duplicates(self):
        store = InMemoryStore()
        upsert_fact(store, "topic-a", "fact-1", {"occurrences": 1})
        upsert_fact(store, "topic-a", "fact-1", {"occurrences": 2})

        facts = relevant_facts(store, "topic-a")
        self.assertEqual(len(facts), 1)
        self.assertEqual(facts[0]["occurrences"], 2)

    def test_facts_are_scoped_per_topic(self):
        store = InMemoryStore()
        upsert_fact(store, "topic-a", "f1", {"x": 1})
        upsert_fact(store, "topic-b", "f1", {"x": 2})

        self.assertEqual(len(relevant_facts(store, "topic-a")), 1)
        self.assertEqual(len(relevant_facts(store, "topic-b")), 1)

    def test_no_facts_returns_empty_list(self):
        store = InMemoryStore()
        self.assertEqual(relevant_facts(store, "topic-empty"), [])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m unittest tests.test_memory_episodic_semantic -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'research_system.memory.episodic'`

- [ ] **Step 3: Write the implementation**

Create `research_system/memory/episodic.py`:

```python
from __future__ import annotations

import uuid

from .store import MAX_RECORDS
from ..utils import utc_now


def record_episode(store, topic: str, episode: dict) -> None:
    """Append one completed run's summary to a topic's episodic memory."""
    recorded_at = utc_now()
    payload = {**episode, "recorded_at": recorded_at}
    store.put((topic, "episodic"), str(uuid.uuid4()), payload)


def recent_episodes(store, topic: str, limit: int = 5) -> list[dict]:
    """Return up to `limit` most recent episodes for a topic, newest first."""
    items = store.search((topic, "episodic"), limit=MAX_RECORDS)
    ordered = sorted(items, key=lambda item: item.value["recorded_at"], reverse=True)
    return [item.value for item in ordered[:limit]]
```

Create `research_system/memory/semantic.py`:

```python
from __future__ import annotations

from .store import MAX_RECORDS
from ..utils import utc_now


def upsert_fact(store, topic: str, key: str, fact: dict) -> None:
    """
    Write or update a durable fact about a topic under a stable key -- calling
    this again with the same key (e.g. an updated occurrence count) replaces
    the old value rather than duplicating it.
    """
    payload = {**fact, "updated_at": utc_now()}
    store.put((topic, "semantic"), key, payload)


def relevant_facts(store, topic: str) -> list[dict]:
    """Return all semantic facts stored for a topic."""
    items = store.search((topic, "semantic"), limit=MAX_RECORDS)
    return [item.value for item in items]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m unittest tests.test_memory_episodic_semantic -v`
Expected: PASS (8 tests)

- [ ] **Step 5: Run the full test suite**

Run: `python3 -m unittest discover -s tests -v`
Expected: PASS, no regressions.

- [ ] **Step 6: Commit**

```bash
git add research_system/memory/episodic.py research_system/memory/semantic.py tests/test_memory_episodic_semantic.py
git commit -m "feat: add episodic and semantic long-term memory primitives"
```

---

### Task 4: `research_system/memory/procedural.py` + default rules file

**Files:**
- Create: `research_system/memory/procedural.py`
- Create: `research_system/memory/procedural_rules.json`
- Test: `tests/test_memory_procedural.py`

**Interfaces:**
- Produces: `load_rules(path: str = DEFAULT_PATH) -> dict`, `apply_mandatory_review_override(topic: str, rationale: str, path: str = DEFAULT_PATH) -> None`, `is_mandatory_review_topic(topic: str, path: str = DEFAULT_PATH) -> bool` — consumed by Tasks 5, 10, 12.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_memory_procedural.py`:

```python
import os
import tempfile
import unittest

from research_system.memory import procedural


class ProceduralRulesTests(unittest.TestCase):
    def test_load_rules_returns_defaults_when_file_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "missing.json")
            rules = procedural.load_rules(path)
            self.assertEqual(rules["quality_gate"]["min_evidence"], 4)
            self.assertEqual(rules["quality_gate"]["min_source_types"], 2)
            self.assertEqual(rules["quality_gate"]["min_avg_confidence"], 0.70)
            self.assertEqual(rules["quality_gate"]["min_cross_agent_insights"], 1)
            self.assertEqual(rules["mandatory_human_review_topics"], {})

    def test_load_rules_falls_back_on_malformed_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "bad.json")
            with open(path, "w") as f:
                f.write("not json")
            rules = procedural.load_rules(path)
            self.assertEqual(rules["quality_gate"]["min_evidence"], 4)

    def test_committed_default_file_matches_hardcoded_fallback(self):
        rules = procedural.load_rules()  # real committed procedural_rules.json
        self.assertEqual(rules["quality_gate"]["min_evidence"], 4)
        self.assertEqual(rules["quality_gate"]["min_source_types"], 2)
        self.assertEqual(rules["quality_gate"]["min_avg_confidence"], 0.70)
        self.assertEqual(rules["quality_gate"]["min_cross_agent_insights"], 1)

    def test_apply_mandatory_review_override_persists(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "rules.json")
            procedural.apply_mandatory_review_override("topic-x", "recurring conflict", path)

            rules = procedural.load_rules(path)
            self.assertIn("topic-x", rules["mandatory_human_review_topics"])

    def test_is_mandatory_review_topic(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "rules.json")
            self.assertFalse(procedural.is_mandatory_review_topic("topic-x", path))
            procedural.apply_mandatory_review_override("topic-x", "reason", path)
            self.assertTrue(procedural.is_mandatory_review_topic("topic-x", path))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m unittest tests.test_memory_procedural -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'research_system.memory.procedural'`

- [ ] **Step 3: Write the implementation**

Create `research_system/memory/procedural.py`:

```python
from __future__ import annotations

import json
import os

DEFAULT_PATH = os.path.join(os.path.dirname(__file__), "procedural_rules.json")

DEFAULT_RULES = {
    "quality_gate": {
        "min_evidence": 4,
        "min_source_types": 2,
        "min_avg_confidence": 0.70,
        "min_cross_agent_insights": 1,
    },
    "mandatory_human_review_topics": {},
}


def _deep_copy_defaults() -> dict:
    return json.loads(json.dumps(DEFAULT_RULES))


def load_rules(path: str = DEFAULT_PATH) -> dict:
    """
    Load procedural rules, falling back to DEFAULT_RULES (which exactly match
    quality.py's original hardcoded constants) on any missing or malformed
    file -- procedural memory is an enhancement layer, never a hard
    requirement, matching this package's other memory-layer error handling.
    """
    if not os.path.exists(path):
        return _deep_copy_defaults()

    try:
        with open(path, "r") as f:
            rules = json.load(f)
    except (json.JSONDecodeError, OSError):
        return _deep_copy_defaults()

    merged = _deep_copy_defaults()
    merged["quality_gate"].update(rules.get("quality_gate", {}))
    merged["mandatory_human_review_topics"].update(rules.get("mandatory_human_review_topics", {}))
    return merged


def apply_mandatory_review_override(topic: str, rationale: str, path: str = DEFAULT_PATH) -> None:
    """Persist a human-approved procedural rule change: force HITL for this topic."""
    rules = load_rules(path)
    rules["mandatory_human_review_topics"][topic] = rationale

    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w") as f:
        json.dump(rules, f, indent=2)


def is_mandatory_review_topic(topic: str, path: str = DEFAULT_PATH) -> bool:
    rules = load_rules(path)
    return topic in rules["mandatory_human_review_topics"]
```

Create `research_system/memory/procedural_rules.json` (committed defaults, exactly matching `DEFAULT_RULES` above):

```json
{
  "quality_gate": {
    "min_evidence": 4,
    "min_source_types": 2,
    "min_avg_confidence": 0.70,
    "min_cross_agent_insights": 1
  },
  "mandatory_human_review_topics": {}
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m unittest tests.test_memory_procedural -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Run the full test suite**

Run: `python3 -m unittest discover -s tests -v`
Expected: PASS, no regressions.

- [ ] **Step 6: Commit**

```bash
git add research_system/memory/procedural.py research_system/memory/procedural_rules.json tests/test_memory_procedural.py
git commit -m "feat: add procedural memory (versioned, git-tracked operational rules)"
```

---

### Task 5: `research_system/memory/reflect.py` — analyze-and-refine step

**Files:**
- Create: `research_system/memory/reflect.py`
- Test: `tests/test_memory_reflect.py`

**Interfaces:**
- Consumes: `episodic.recent_episodes` (Task 3), `semantic.upsert_fact` (Task 3).
- Produces: `reflect_on_topic(store, topic: str) -> dict` returning `{"updated_facts": list[dict], "proposed_rule_change": dict | None}` — consumed by Task 12.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_memory_reflect.py`:

```python
import unittest

from langgraph.store.memory import InMemoryStore

from research_system.memory import episodic, reflect, semantic


def _episode(gate_failures, open_conflict_issues=None):
    return {
        "question": "q",
        "mode": "parallel",
        "gate_failures": gate_failures,
        "open_conflict_issues": open_conflict_issues or [],
    }


class ReflectOnTopicTests(unittest.TestCase):
    def test_no_facts_or_proposal_with_no_episodes(self):
        store = InMemoryStore()
        result = reflect.reflect_on_topic(store, "topic-empty")
        self.assertEqual(result["updated_facts"], [])
        self.assertIsNone(result["proposed_rule_change"])

    def test_single_episode_does_not_trigger_a_fact(self):
        store = InMemoryStore()
        episodic.record_episode(store, "topic-b", _episode(["unresolved_high_conflicts>0"]))

        result = reflect.reflect_on_topic(store, "topic-b")
        self.assertEqual(result["updated_facts"], [])

    def test_recurring_failure_upserts_semantic_fact(self):
        store = InMemoryStore()
        episodic.record_episode(store, "topic-a", _episode(["unresolved_high_conflicts>0"]))
        episodic.record_episode(store, "topic-a", _episode(["unresolved_high_conflicts>0"]))

        result = reflect.reflect_on_topic(store, "topic-a")

        self.assertEqual(len(result["updated_facts"]), 1)
        self.assertEqual(result["updated_facts"][0]["reason"], "unresolved_high_conflicts>0")
        self.assertEqual(len(semantic.relevant_facts(store, "topic-a")), 1)

    def test_recurring_open_conflict_upserts_semantic_fact(self):
        store = InMemoryStore()
        episodic.record_episode(store, "topic-e", _episode([], ["Material market-size disagreement"]))
        episodic.record_episode(store, "topic-e", _episode([], ["Material market-size disagreement"]))

        result = reflect.reflect_on_topic(store, "topic-e")

        patterns = [f["pattern"] for f in result["updated_facts"]]
        self.assertIn("recurring_open_conflict", patterns)

    def test_three_consecutive_identical_failures_propose_mandatory_review(self):
        store = InMemoryStore()
        for _ in range(3):
            episodic.record_episode(store, "topic-c", _episode(["unresolved_high_conflicts>0"]))

        result = reflect.reflect_on_topic(store, "topic-c")

        self.assertIsNotNone(result["proposed_rule_change"])
        self.assertEqual(result["proposed_rule_change"]["rule"], "mandatory_human_review_topics")
        self.assertEqual(result["proposed_rule_change"]["topic"], "topic-c")

    def test_two_of_three_failures_does_not_propose_rule_change(self):
        store = InMemoryStore()
        episodic.record_episode(store, "topic-d", _episode(["unresolved_high_conflicts>0"]))
        episodic.record_episode(store, "topic-d", _episode(["unresolved_high_conflicts>0"]))
        episodic.record_episode(store, "topic-d", _episode([]))

        result = reflect.reflect_on_topic(store, "topic-d")
        self.assertIsNone(result["proposed_rule_change"])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m unittest tests.test_memory_reflect -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'research_system.memory.reflect'`

- [ ] **Step 3: Write the implementation**

Create `research_system/memory/reflect.py`:

```python
from __future__ import annotations

from . import episodic, semantic
from ..utils import stable_id

RECENT_WINDOW = 3  # how many of the most recent episodes reflection considers


def _count_recurrences(episodes: list[dict], extract) -> dict[str, int]:
    counts: dict[str, int] = {}
    for ep in episodes:
        for value in extract(ep):
            counts[value] = counts.get(value, 0) + 1
    return counts


def reflect_on_topic(store, topic: str) -> dict:
    """
    Analyze-and-refine step: reads the topic's most recent episodes, upserts
    semantic facts for recurring patterns, and proposes -- but never
    auto-applies -- a procedural rule change when one failure pattern recurs
    across every one of the recent episodes considered.
    """
    episodes = episodic.recent_episodes(store, topic, limit=RECENT_WINDOW)

    failure_counts = _count_recurrences(episodes, lambda ep: ep.get("gate_failures", []))
    conflict_counts = _count_recurrences(episodes, lambda ep: ep.get("open_conflict_issues", []))

    updated_facts = []
    for reason, count in failure_counts.items():
        if count >= 2:
            fact = {
                "pattern": "recurring_gate_failure",
                "reason": reason,
                "occurrences": count,
                "checked_over": len(episodes),
            }
            semantic.upsert_fact(store, topic, f"gate_failure:{reason}", fact)
            updated_facts.append(fact)

    for issue, count in conflict_counts.items():
        if count >= 2:
            fact = {
                "pattern": "recurring_open_conflict",
                "issue": issue,
                "occurrences": count,
                "checked_over": len(episodes),
            }
            semantic.upsert_fact(store, topic, f"conflict:{stable_id('conflict-issue', issue)}", fact)
            updated_facts.append(fact)

    proposed_rule_change = None
    if len(episodes) >= RECENT_WINDOW:
        for reason, count in failure_counts.items():
            if count >= RECENT_WINDOW:
                proposed_rule_change = {
                    "rule": "mandatory_human_review_topics",
                    "topic": topic,
                    "rationale": (
                        f"The same quality-gate failure ({reason}) occurred in all of "
                        f"the last {RECENT_WINDOW} runs on this topic."
                    ),
                }
                break

    return {
        "updated_facts": updated_facts,
        "proposed_rule_change": proposed_rule_change,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m unittest tests.test_memory_reflect -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Run the full test suite**

Run: `python3 -m unittest discover -s tests -v`
Expected: PASS, no regressions.

- [ ] **Step 6: Commit**

```bash
git add research_system/memory/reflect.py tests/test_memory_reflect.py
git commit -m "feat: add reflect_on_topic() analyze-and-refine step"
```

---

### Task 6: `state.py` + `planner.py` — `memory_context` and `build_planner_node`

**Files:**
- Modify: `research_system/state.py`
- Modify: `research_system/planner.py` (full replacement)
- Modify: `tests/test_planner.py`

**Interfaces:**
- Consumes: `research_system.memory.topic.derive_topic` (Task 1), `research_system.memory.episodic.recent_episodes` + `research_system.memory.semantic.relevant_facts` (Task 3).
- Produces: `build_planner_node(store=None) -> Callable[[dict], dict]` (factory, replacing the old plain `planner_node` function) — consumed by Task 7. State gains `memory_context: dict`.

- [ ] **Step 1: Write the failing tests**

Modify `tests/test_planner.py` — change the import line and append a new test class. Full new file content:

```python
import unittest

from research_system.planner import build_planner_node, choose_active_agents, choose_execution_plan

ALL_FOUR = {"web_research", "data_analysis", "trend_analysis", "competitive_intelligence"}


class PlannerTests(unittest.TestCase):
    def test_parallel_for_quick_scan(self):
        plan = choose_execution_plan("Quick overview and landscape scan of the EV market")
        self.assertEqual(plan["mode"], "parallel")

    def test_sequential_for_dependency_chain(self):
        plan = choose_execution_plan(
            "First quantify battery prices, then forecast the impact, "
            "then determine which competitor benefits and why"
        )
        self.assertEqual(plan["mode"], "sequential")

    def test_hybrid_for_complex_general_question(self):
        plan = choose_execution_plan(
            "Should a new automaker enter the EV market and what competitive position should it take?"
        )
        self.assertEqual(plan["mode"], "hybrid")


class ActiveAgentSelectionTests(unittest.TestCase):
    def test_defaults_to_all_four_specialists(self):
        self.assertEqual(set(choose_active_agents("Give me a market overview")), ALL_FOUR)

    def test_execution_plan_includes_active_agents_by_default(self):
        plan = choose_execution_plan(
            "Should a new automaker enter the EV market and what competitive position should it take?"
        )
        self.assertEqual(set(plan["active_agents"]), ALL_FOUR)

    def test_excludes_competitive_intelligence_when_explicitly_out_of_scope(self):
        active = choose_active_agents("Give me a market overview, excluding competitor analysis")
        self.assertNotIn("competitive_intelligence", active)
        self.assertIn("web_research", active)
        self.assertIn("data_analysis", active)

    def test_excludes_trend_analysis_when_explicitly_out_of_scope(self):
        active = choose_active_agents("Research the EV market, ignore trend forecasting")
        self.assertNotIn("trend_analysis", active)
        self.assertIn("web_research", active)
        self.assertIn("data_analysis", active)

    def test_web_research_and_data_analysis_are_never_excluded(self):
        active = choose_active_agents(
            "excluding competitor analysis, ignore trend forecasting, just give me the basics"
        )
        self.assertIn("web_research", active)
        self.assertIn("data_analysis", active)


class BuildPlannerNodeTests(unittest.TestCase):
    def test_memory_context_empty_without_store(self):
        node = build_planner_node()  # store=None
        state = {"question": "Quick overview and landscape scan of the Indian EV market"}
        result = node(state)
        self.assertEqual(result["memory_context"]["semantic_facts"], [])
        self.assertEqual(result["memory_context"]["relevant_episodes"], [])
        self.assertEqual(result["memory_context"]["topic"], "ev_indian_market")

    def test_memory_context_populated_from_store(self):
        from langgraph.store.memory import InMemoryStore

        from research_system.memory.episodic import record_episode
        from research_system.memory.semantic import upsert_fact

        store = InMemoryStore()
        topic = "ev_indian_market"
        upsert_fact(store, topic, "gate_failure:x", {"pattern": "recurring_gate_failure", "reason": "x"})
        record_episode(store, topic, {"question": "prior question", "mode": "parallel"})

        node = build_planner_node(store)
        state = {"question": "Quick overview and landscape scan of the Indian EV market"}
        result = node(state)

        self.assertEqual(len(result["memory_context"]["semantic_facts"]), 1)
        self.assertEqual(len(result["memory_context"]["relevant_episodes"]), 1)

    def test_still_returns_execution_plan_and_iteration_count(self):
        node = build_planner_node()
        state = {"question": "Quick overview of the Indian EV market", "iteration_count": 2}
        result = node(state)
        self.assertEqual(result["execution_plan"]["mode"], "parallel")
        self.assertEqual(result["iteration_count"], 2)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m unittest tests.test_planner -v`
Expected: FAIL with `ImportError: cannot import name 'build_planner_node' from 'research_system.planner'`

- [ ] **Step 3: Modify `research_system/state.py`**

Add one field to `ResearchState` (leave everything else in the file unchanged):

```python
class ResearchState(TypedDict, total=False):
    question: str
    execution_plan: ExecutionPlan
    memory_context: dict
```

(Insert `memory_context: dict` immediately after `execution_plan: ExecutionPlan` at `research_system/state.py:96`.)

- [ ] **Step 4: Replace `research_system/planner.py` in full**

```python
from __future__ import annotations

from .memory.episodic import recent_episodes
from .memory.semantic import relevant_facts
from .memory.topic import derive_topic

ALL_SPECIALTIES = ("web_research", "data_analysis", "trend_analysis", "competitive_intelligence")

# web_research and data_analysis are foundational (baseline sourcing + quantitative
# grounding) and are never excluded. trend_analysis and competitive_intelligence are
# opt-out only, via an explicit signal that the lens is out of scope -- this keeps the
# no-signal default identical to running all four specialists, so existing behavior is
# unaffected unless the question actually says so.
EXCLUSION_TERMS = {
    "trend_analysis": (
        "no trend", "not trend", "excluding trend", "exclude trend",
        "ignore trend", "skip trend", "without forecast", "no forecast",
    ),
    "competitive_intelligence": (
        "no competitor", "not competitor", "excluding competitor", "exclude competitor",
        "ignore competitor", "skip competitive", "without competitive", "no competitive",
    ),
}


def choose_active_agents(question: str) -> list[str]:
    """
    Explicit "which agents activate" decision, separate from the parallel/sequential/
    hybrid coordination-pattern decision. Deliberately conservative (opt-out on an
    explicit signal, not opt-in on a topic keyword) so it can't silently starve the
    quality gate or regress an existing broad question.
    """
    q = question.lower()
    active = list(ALL_SPECIALTIES)
    for specialty, exclusion_terms in EXCLUSION_TERMS.items():
        if any(term in q for term in exclusion_terms):
            active.remove(specialty)
    return active


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
        "active_agents": choose_active_agents(question),
        "breadth_score": breadth,
    }


def build_planner_node(store=None):
    """
    Factory producing the planner node, matching the same explicit-dependency-
    injection pattern already used by specialist_node/build_cross_agent_insights/
    detect_and_resolve_conflicts. With store=None, memory_context is always
    empty (safe default for callers/tests that don't care about memory).
    """
    def planner_node(state: dict) -> dict:
        plan = choose_execution_plan(state["question"])
        # breadth_score is explanatory only; keep state schema compact.
        plan.pop("breadth_score", None)

        topic = derive_topic(state["question"])
        memory_context = {"topic": topic, "semantic_facts": [], "relevant_episodes": []}
        if store is not None:
            memory_context["semantic_facts"] = relevant_facts(store, topic)
            memory_context["relevant_episodes"] = recent_episodes(store, topic, limit=3)

        return {
            "execution_plan": plan,
            "memory_context": memory_context,
            "iteration_count": state.get("iteration_count", 0),
            "_telemetry_note": plan["rationale"],
        }
    return planner_node
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python3 -m unittest tests.test_planner -v`
Expected: PASS (11 tests)

- [ ] **Step 6: Run the full test suite**

Run: `python3 -m unittest discover -s tests -v`
Expected: PASS. (`graph.py` still imports the now-removed `planner_node` name at this point in the plan — that's fixed in Task 7, next. If this task is executed standalone, `python3 -m unittest discover -s tests -v` will fail at collection due to `graph.py`'s stale import; note this in your report as expected and resolved by Task 7, not a regression in this task's own files.)

- [ ] **Step 7: Commit**

```bash
git add research_system/state.py research_system/planner.py tests/test_planner.py
git commit -m "feat: planner writes memory_context; planner_node becomes build_planner_node(store)"
```

---

### Task 7: `graph.py` — wire the store through `build_graph()`

**Files:**
- Modify: `research_system/graph.py` (full replacement)

**Interfaces:**
- Consumes: `build_planner_node(store=None)` (Task 6).
- Produces: `build_graph(provider=None, checkpointer=None, llm=None, store=None)` — new `store` parameter, defaults to a fresh `InMemoryStore()`. Compiles with `store=store`, so `compiled_graph.store` is reachable afterward (consumed by Task 12). Consumed by Task 11.

- [ ] **Step 1: Replace `research_system/graph.py` in full**

```python
from __future__ import annotations

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.store.memory import InMemoryStore

from .agents import (
    build_cross_agent_insights,
    detect_and_resolve_conflicts,
    specialist_node,
)
from .hitl import human_review_node, route_after_human
from .planner import build_planner_node
from .provider import DemoResearchProvider, ResearchProvider
from .quality import quality_gate_node, route_after_quality_gate
from .state import ResearchState
from .synthesis import synthesis_node
from .utils import timed_node


def _route_plan(state: dict) -> str:
    return state["execution_plan"]["mode"]


def _noop(state: dict) -> dict:
    return {"_telemetry_note": "Coordination barrier reached."}


def build_graph(
    provider: ResearchProvider | None = None,
    checkpointer=None,
    llm=None,
    store=None,
):
    provider = provider or DemoResearchProvider()
    checkpointer = checkpointer or InMemorySaver()
    store = store or InMemoryStore()

    builder = StateGraph(ResearchState)

    builder.add_node("planner", timed_node("planner", build_planner_node(store)))

    # Strategy/barrier nodes.
    builder.add_node("parallel_start", timed_node("parallel_start", _noop))
    builder.add_node("hybrid_start", timed_node("hybrid_start", _noop))
    builder.add_node("hybrid_stage2", timed_node("hybrid_stage2", _noop))

    # Parallel branch nodes.
    for name, specialty in [
        ("p_web", "web_research"),
        ("p_data", "data_analysis"),
        ("p_trend", "trend_analysis"),
        ("p_comp", "competitive_intelligence"),
    ]:
        builder.add_node(name, timed_node(name, specialist_node(provider, specialty)))

    # Sequential branch nodes.
    for name, specialty in [
        ("s_web", "web_research"),
        ("s_data", "data_analysis"),
        ("s_trend", "trend_analysis"),
        ("s_comp", "competitive_intelligence"),
    ]:
        builder.add_node(name, timed_node(name, specialist_node(provider, specialty)))

    # Hybrid branch nodes.
    for name, specialty in [
        ("h_web", "web_research"),
        ("h_data", "data_analysis"),
        ("h_trend", "trend_analysis"),
        ("h_comp", "competitive_intelligence"),
    ]:
        builder.add_node(name, timed_node(name, specialist_node(provider, specialty)))

    builder.add_node(
        "cross_agent_insights",
        timed_node("cross_agent_insights", build_cross_agent_insights(llm)),
    )
    builder.add_node(
        "conflict_resolution",
        timed_node("conflict_resolution", detect_and_resolve_conflicts(llm)),
    )
    builder.add_node(
        "quality_gate",
        timed_node("quality_gate", quality_gate_node),
    )
    builder.add_node(
        "human_review",
        timed_node("human_review", human_review_node),
    )
    builder.add_node(
        "synthesis",
        timed_node("synthesis", synthesis_node),
    )

    builder.add_edge(START, "planner")
    builder.add_conditional_edges(
        "planner",
        _route_plan,
        {
            "parallel": "parallel_start",
            "sequential": "s_web",
            "hybrid": "hybrid_start",
        },
    )

    # PARALLEL: all specialists start together.
    builder.add_edge("parallel_start", "p_web")
    builder.add_edge("parallel_start", "p_data")
    builder.add_edge("parallel_start", "p_trend")
    builder.add_edge("parallel_start", "p_comp")
    builder.add_edge(
        ["p_web", "p_data", "p_trend", "p_comp"],
        "cross_agent_insights",
    )

    # SEQUENTIAL: each downstream specialist sees upstream evidence.
    builder.add_edge("s_web", "s_data")
    builder.add_edge("s_data", "s_trend")
    builder.add_edge("s_trend", "s_comp")
    builder.add_edge("s_comp", "cross_agent_insights")

    # HYBRID: phase 1 parallel, barrier, phase 2 parallel.
    builder.add_edge("hybrid_start", "h_web")
    builder.add_edge("hybrid_start", "h_data")
    builder.add_edge(["h_web", "h_data"], "hybrid_stage2")
    builder.add_edge("hybrid_stage2", "h_trend")
    builder.add_edge("hybrid_stage2", "h_comp")
    builder.add_edge(["h_trend", "h_comp"], "cross_agent_insights")

    builder.add_edge("cross_agent_insights", "conflict_resolution")
    builder.add_edge("conflict_resolution", "quality_gate")
    builder.add_conditional_edges(
        "quality_gate",
        route_after_quality_gate,
        {
            "human_review": "human_review",
            "synthesis": "synthesis",
        },
    )
    builder.add_conditional_edges(
        "human_review",
        route_after_human,
        {
            "planner": "planner",
            "synthesis": "synthesis",
        },
    )
    builder.add_edge("synthesis", END)

    return builder.compile(checkpointer=checkpointer, store=store)
```

- [ ] **Step 2: Verify the demo path still builds and runs end-to-end, and the store is reachable**

Run:
```bash
python3 -c "
from research_system.graph import build_graph
graph = build_graph()
print('store reachable:', hasattr(graph, 'store'))
result = graph.invoke(
    {
        'question': 'Quick overview and landscape scan of the Indian EV market, ignore trend forecasting',
        'evidence': [], 'contributions': [], 'cross_agent_insights': [],
        'conflicts': [], 'synthesis_threads': [], 'telemetry': [],
        'human_decisions': [], 'iteration_count': 0,
        'mandatory_human_review': False, 'research_complete': False, 'final_report': '',
    },
    config={'configurable': {'thread_id': 'store-wiring-test'}},
)
print('mode:', result['execution_plan']['mode'])
print('memory_context topic:', result['memory_context']['topic'])
print('gate passed:', result.get('quality_gate', {}).get('passed'))
"
```
Expected: `store reachable: True`, `mode: parallel`, `memory_context topic: ev_forecasting_indian_market_trend` (topic includes "forecasting"/"trend" since the question mentions "ignore trend forecasting" — those words aren't in the stopword list), `gate passed: True` (matches the empirically-verified pass case from the earlier eval work).

- [ ] **Step 3: Run the full test suite**

Run: `python3 -m unittest discover -s tests -v`
Expected: PASS, no regressions.

- [ ] **Step 4: Commit**

```bash
git add research_system/graph.py
git commit -m "feat: thread optional store through build_graph, wire build_planner_node"
```

---

### Task 8: `provider.py` + `agents.py` — thread `memory` into specialists

**Files:**
- Modify: `research_system/provider.py` (full replacement)
- Modify: `research_system/agents.py:10-39` (replace the `specialist_node` function only)
- Modify: `tests/test_agents.py` (update `FakeProvider` and add a memory-pass-through test)

**Interfaces:**
- Produces: `ResearchProvider.research(specialty, question, context, memory=None) -> dict` — new optional `memory` parameter on the protocol, implemented by `DemoResearchProvider` here and `LiveResearchProvider` in Task 9. `specialist_node` now reads `state.get("memory_context")` and passes it through.

- [ ] **Step 1: Write the failing test**

Modify `tests/test_agents.py`: update `FakeProvider.research()`'s signature to accept `memory=None` (otherwise Step 3's `specialist_node` change will raise `TypeError: research() got an unexpected keyword argument 'memory'` against the old signature), and add one new test. Full new file content:

```python
import unittest

from research_system.agents import (
    build_cross_agent_insights,
    detect_and_resolve_conflicts,
    specialist_node,
)


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


class FakeProvider:
    def __init__(self):
        self.calls = []
        self.received_memory = []

    def research(self, specialty, question, context, memory=None):
        self.calls.append(specialty)
        self.received_memory.append(memory)
        return {
            "evidence": [_evidence("e1", specialty, "some claim", 0.8, 0.8, [])],
            "contribution": {
                "agent": specialty,
                "summary": "did research",
                "evidence_ids": ["e1"],
                "depends_on_agents": [],
                "reasoning_note": "",
            },
        }


class SpecialistNodeActivationTests(unittest.TestCase):
    def test_runs_provider_when_specialty_is_active(self):
        provider = FakeProvider()
        node = specialist_node(provider, "web_research")
        state = {
            "question": "q",
            "evidence": [],
            "execution_plan": {"active_agents": ["web_research", "data_analysis"]},
        }
        result = node(state)
        self.assertEqual(provider.calls, ["web_research"])
        self.assertEqual(len(result["evidence"]), 1)

    def test_skips_provider_when_specialty_excluded(self):
        provider = FakeProvider()
        node = specialist_node(provider, "competitive_intelligence")
        state = {
            "question": "q",
            "evidence": [],
            "execution_plan": {"active_agents": ["web_research", "data_analysis"]},
        }
        result = node(state)
        self.assertEqual(provider.calls, [])
        self.assertEqual(result.get("evidence", []), [])
        self.assertEqual(len(result["contributions"]), 1)
        self.assertEqual(result["contributions"][0]["agent"], "competitive_intelligence")

    def test_runs_provider_when_no_active_agents_key(self):
        provider = FakeProvider()
        node = specialist_node(provider, "trend_analysis")
        state = {"question": "q", "evidence": [], "execution_plan": {}}
        result = node(state)
        self.assertEqual(provider.calls, ["trend_analysis"])

    def test_passes_memory_context_through_to_provider(self):
        provider = FakeProvider()
        node = specialist_node(provider, "web_research")
        memory_context = {"topic": "ev_indian_market", "semantic_facts": [{"x": 1}], "relevant_episodes": []}
        state = {
            "question": "q",
            "evidence": [],
            "execution_plan": {"active_agents": ["web_research"]},
            "memory_context": memory_context,
        }
        node(state)
        self.assertEqual(provider.received_memory, [memory_context])

    def test_memory_is_none_when_state_has_no_memory_context(self):
        provider = FakeProvider()
        node = specialist_node(provider, "web_research")
        state = {"question": "q", "evidence": [], "execution_plan": {"active_agents": ["web_research"]}}
        node(state)
        self.assertEqual(provider.received_memory, [None])


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
Expected: FAIL — `test_passes_memory_context_through_to_provider` and `test_memory_is_none_when_state_has_no_memory_context` fail because `specialist_node` doesn't read `memory_context` or pass `memory=` yet.

- [ ] **Step 3: Replace `research_system/agents.py:10-39`** (the `specialist_node` function only — every other function in the file is unchanged)

```python
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
        memory_context = state.get("memory_context")
        result = provider.research(specialty, state["question"], context, memory=memory_context)
        return {
            "evidence": result["evidence"],
            "contributions": [result["contribution"]],
            "_telemetry_note": (
                f"{specialty}: consumed {len(context)} evidence items and "
                f"produced {len(result['evidence'])}."
            ),
        }
    return node
```

- [ ] **Step 4: Replace `research_system/provider.py` in full**

```python
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
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python3 -m unittest tests.test_agents -v`
Expected: PASS (9 tests)

- [ ] **Step 6: Run the full test suite**

Run: `python3 -m unittest discover -s tests -v`
Expected: PASS, no regressions.

- [ ] **Step 7: Commit**

```bash
git add research_system/provider.py research_system/agents.py tests/test_agents.py
git commit -m "feat: thread memory_context from specialist_node into ResearchProvider.research()"
```

---

### Task 9: `live_provider.py` — fold memory into the DeepSeek prompt

**Files:**
- Modify: `research_system/live_provider.py` (full replacement)
- Test: `tests/test_live_provider.py` (new — tests the pure, offline-safe `_format_memory` helper only; `research()` itself stays untested per the existing project convention of no network-dependent tests)

**Interfaces:**
- Produces: `LiveResearchProvider.research(specialty, question, context, memory=None)` matching the updated `ResearchProvider` protocol from Task 8. `_format_memory(memory) -> str` (static method) is the new testable piece.

- [ ] **Step 1: Write the failing test**

Create `tests/test_live_provider.py`:

```python
import unittest

from research_system.live_provider import LiveResearchProvider


class FormatMemoryTests(unittest.TestCase):
    def test_no_memory_returns_placeholder(self):
        self.assertIn("no relevant memory", LiveResearchProvider._format_memory(None))
        self.assertIn("no relevant memory", LiveResearchProvider._format_memory({}))

    def test_formats_facts_and_episodes(self):
        memory = {
            "semantic_facts": [{"pattern": "recurring_gate_failure", "reason": "x"}],
            "relevant_episodes": [{"mode": "parallel", "gate_passed": False, "question": "q"}],
        }
        formatted = LiveResearchProvider._format_memory(memory)
        self.assertIn("known pattern", formatted)
        self.assertIn("prior run", formatted)
        self.assertIn("parallel", formatted)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest tests.test_live_provider -v`
Expected: FAIL with `AttributeError: type object 'LiveResearchProvider' has no attribute '_format_memory'`

- [ ] **Step 3: Replace `research_system/live_provider.py` in full**

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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m unittest tests.test_live_provider -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Run the full test suite**

Run: `python3 -m unittest discover -s tests -v`
Expected: PASS, no regressions.

- [ ] **Step 6: Commit**

```bash
git add research_system/live_provider.py tests/test_live_provider.py
git commit -m "feat: fold memory context into LiveResearchProvider's DeepSeek prompt"
```

---

### Task 10: `quality.py` — read thresholds from procedural memory

**Files:**
- Modify: `research_system/quality.py` (full replacement)
- Modify: `tests/test_quality.py`

**Interfaces:**
- Consumes: `research_system.memory.procedural.load_rules()` (Task 4).
- Produces: `quality_gate_node` behavior unchanged by default (verified via regression test); reads `min_evidence`/`min_source_types`/`min_avg_confidence`/`min_cross_agent_insights` from `load_rules()["quality_gate"]` instead of only the module constants.

- [ ] **Step 1: Write the failing test**

Modify `tests/test_quality.py` — add one new test class. Full new file content:

```python
import unittest

from research_system.quality import quality_gate_node


class QualityGateTests(unittest.TestCase):
    def test_gate_rejects_weak_state(self):
        state = {
            "evidence": [{
                "confidence": 0.5,
                "source_type": "blog",
            }],
            "cross_agent_insights": [],
            "conflicts": [],
            "iteration_count": 0,
        }
        result = quality_gate_node(state)
        self.assertFalse(result["quality_gate"]["passed"])
        self.assertIn("evidence_count<4", result["quality_gate"]["failures"])

    def test_gate_blocks_unresolved_high_conflict(self):
        evidence = [
            {"confidence": 0.8, "source_type": "dataset"},
            {"confidence": 0.8, "source_type": "report"},
            {"confidence": 0.8, "source_type": "dataset"},
            {"confidence": 0.8, "source_type": "report"},
        ]
        state = {
            "evidence": evidence,
            "cross_agent_insights": [{"insight_id": "i1"}],
            "conflicts": [{
                "severity": "high",
                "status": "open",
            }],
            "iteration_count": 0,
        }
        result = quality_gate_node(state)
        self.assertFalse(result["quality_gate"]["passed"])
        self.assertIn(
            "unresolved_high_conflicts>0",
            result["quality_gate"]["failures"],
        )


class QualityGateProceduralDefaultsTests(unittest.TestCase):
    def test_default_rules_match_hardcoded_constants(self):
        # Regression guard: with no override file present, load_rules() must
        # return exactly the original hardcoded thresholds -- this is what
        # keeps QualityGateTests above passing unchanged.
        from research_system.memory.procedural import load_rules
        rules = load_rules()["quality_gate"]
        self.assertEqual(rules["min_evidence"], 4)
        self.assertEqual(rules["min_source_types"], 2)
        self.assertEqual(rules["min_avg_confidence"], 0.70)
        self.assertEqual(rules["min_cross_agent_insights"], 1)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests to verify current state**

Run: `python3 -m unittest tests.test_quality -v`
Expected: PASS (3 tests) — this task's test additions don't require new production code to pass (they check the already-built `procedural.py` from Task 4 plus the still-unmodified `quality.py`), so this step establishes the baseline before Step 3's refactor. Confirm no failures here before proceeding.

- [ ] **Step 3: Replace `research_system/quality.py` in full**

```python
from __future__ import annotations

from .memory.procedural import load_rules
from .utils import average, unique_source_types


MIN_EVIDENCE = 4
MIN_SOURCE_TYPES = 2
MIN_AVG_CONFIDENCE = 0.70
MIN_CROSS_AGENT_INSIGHTS = 1
MAX_ITERATIONS = 3


def quality_gate_node(state: dict) -> dict:
    evidence = state.get("evidence", [])
    insights = state.get("cross_agent_insights", [])
    conflicts = state.get("conflicts", [])
    iteration_count = state.get("iteration_count", 0)

    rules = load_rules()["quality_gate"]
    min_evidence = rules.get("min_evidence", MIN_EVIDENCE)
    min_source_types = rules.get("min_source_types", MIN_SOURCE_TYPES)
    min_avg_confidence = rules.get("min_avg_confidence", MIN_AVG_CONFIDENCE)
    min_cross_agent_insights = rules.get("min_cross_agent_insights", MIN_CROSS_AGENT_INSIGHTS)

    unresolved_high = [
        c for c in conflicts
        if c["severity"] == "high" and c["status"] == "open"
    ]
    avg_conf = average([e["confidence"] for e in evidence])
    source_types = unique_source_types(evidence)

    failures = []
    if len(evidence) < min_evidence:
        failures.append(f"evidence_count<{min_evidence}")
    if len(source_types) < min_source_types:
        failures.append(f"source_type_count<{min_source_types}")
    if avg_conf < min_avg_confidence:
        failures.append(f"average_confidence<{min_avg_confidence}")
    if len(insights) < min_cross_agent_insights:
        failures.append(f"cross_agent_insights<{min_cross_agent_insights}")
    if unresolved_high:
        failures.append("unresolved_high_conflicts>0")
    if iteration_count > MAX_ITERATIONS:
        failures.append(f"iteration_count>{MAX_ITERATIONS}")

    gate = {
        "passed": not failures,
        "evidence_count": len(evidence),
        "source_type_count": len(source_types),
        "average_confidence": round(avg_conf, 3),
        "cross_agent_insight_count": len(insights),
        "unresolved_high_conflicts": len(unresolved_high),
        "iteration_count": iteration_count,
        "failures": failures,
    }
    return {
        "quality_gate": gate,
        "research_complete": gate["passed"],
        "_telemetry_note": f"Quality gate passed={gate['passed']}; failures={failures}",
    }


def route_after_quality_gate(state: dict) -> str:
    if state.get("mandatory_human_review"):
        return "human_review"
    return "synthesis" if state["quality_gate"]["passed"] else "human_review"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m unittest tests.test_quality -v`
Expected: PASS (3 tests) — confirms the refactor produced byte-identical numeric behavior.

- [ ] **Step 5: Run the full test suite**

Run: `python3 -m unittest discover -s tests -v`
Expected: PASS, no regressions.

- [ ] **Step 6: Commit**

```bash
git add research_system/quality.py tests/test_quality.py
git commit -m "feat: quality_gate_node reads thresholds via procedural memory (unchanged defaults)"
```

---

### Task 11: `config.py` — persistent store + durable checkpointer

**Files:**
- Modify: `research_system/config.py` (full replacement)
- Modify: `tests/test_config.py` (full replacement)
- Modify: `requirements.txt`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: `research_system.memory.store.load_persistent_store` (Task 2); `build_graph(store=None)` (Task 7).
- Produces: `build_default_graph(checkpointer=None)` — same public signature/return type as before (still returns a compiled graph; `graph.store` is how callers reach the store, per the Global Constraints note). New module constants `MEMORY_DIR`, `STORE_PATH`, `CHECKPOINT_PATH` and helper `_durable_checkpointer()` — consumed by Task 12.

- [ ] **Step 1: Write the failing tests**

Replace `tests/test_config.py` in full:

```python
import os
import tempfile
import unittest
from unittest import mock

from research_system import config


class BuildDefaultGraphTests(unittest.TestCase):
    def test_falls_back_to_demo_without_deepseek_key(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with mock.patch("research_system.config.build_graph") as mock_build, \
                 mock.patch("research_system.config.load_persistent_store", return_value="fake-store"), \
                 mock.patch("research_system.config._durable_checkpointer", return_value=None):
                mock_build.return_value = "demo-graph"
                result = config.build_default_graph()
        mock_build.assert_called_once_with(checkpointer=None, store="fake-store")
        self.assertEqual(result, "demo-graph")

    def test_raises_without_tavily_key(self):
        env = {"DEEPSEEK_API_KEY": "fake-deepseek-key"}
        with mock.patch.dict(os.environ, env, clear=True):
            with mock.patch("research_system.config.load_persistent_store", return_value="fake-store"), \
                 mock.patch("research_system.config._durable_checkpointer", return_value=None):
                with self.assertRaises(RuntimeError):
                    config.build_default_graph()

    def test_builds_live_graph_when_both_keys_present(self):
        env = {
            "DEEPSEEK_API_KEY": "fake-deepseek-key",
            "TAVILY_API_KEY": "fake-tavily-key",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            with mock.patch("research_system.config.build_graph") as mock_build, \
                 mock.patch("research_system.config.load_persistent_store", return_value="fake-store"), \
                 mock.patch("research_system.config._durable_checkpointer", return_value=None):
                mock_build.return_value = "live-graph"
                result = config.build_default_graph()
        self.assertEqual(result, "live-graph")
        _, kwargs = mock_build.call_args
        self.assertIsNotNone(kwargs.get("provider"))
        self.assertIsNotNone(kwargs.get("llm"))
        self.assertEqual(kwargs.get("store"), "fake-store")


class DurableCheckpointerTests(unittest.TestCase):
    def test_returns_a_working_sqlite_checkpointer(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_path = os.path.join(tmp, "checkpoints.sqlite")
            with mock.patch("research_system.config.MEMORY_DIR", tmp), \
                 mock.patch("research_system.config.CHECKPOINT_PATH", checkpoint_path):
                saver = config._durable_checkpointer()
        self.assertIsNotNone(saver)
        self.assertTrue(os.path.exists(checkpoint_path))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m unittest tests.test_config -v`
Expected: FAIL — `test_falls_back_to_demo_without_deepseek_key` and `test_builds_live_graph_when_both_keys_present` fail on the `mock_build.assert_called_once_with(...)` assertion (current code doesn't pass `store=`); `AttributeError` on `research_system.config.load_persistent_store`/`_durable_checkpointer` (don't exist yet) for the others.

- [ ] **Step 3: Add the new dependency to `requirements.txt`**

Replace the full file contents with:

```text
langgraph>=0.6
typing-extensions>=4.12
langchain-deepseek>=0.1
langchain-tavily>=0.2
python-dotenv>=1.0
pydantic>=2.7
langgraph-checkpoint-sqlite>=2.0
```

(Added as a normal, always-installed dependency rather than truly optional: it's small — verified as `langgraph-checkpoint-sqlite==2.0.11` plus `aiosqlite`/`sqlite-vec` transitively, no external service — and the durability feature should work out of the box for anyone following the Setup instructions. `config.py`'s `try/except ImportError` around it stays as defensive programming, not the primary expected path.)

Run: `pip3 install --user -r requirements.txt` and confirm it completes with no errors.

- [ ] **Step 4: Add the memory directory to `.gitignore`**

Replace the full file contents with:

```text
.venv/
__pycache__/
*.pyc
.env
.research_memory/
```

- [ ] **Step 5: Replace `research_system/config.py` in full**

```python
from __future__ import annotations

import os
import sqlite3

from dotenv import load_dotenv

from .graph import build_graph
from .live_provider import LiveResearchProvider
from .memory.store import load_persistent_store

load_dotenv()

MEMORY_DIR = ".research_memory"
STORE_PATH = os.path.join(MEMORY_DIR, "store.json")
CHECKPOINT_PATH = os.path.join(MEMORY_DIR, "checkpoints.sqlite")


def build_default_graph(checkpointer=None):
    """
    Build the demo graph, or the live DeepSeek/Tavily graph if configured.

    Falls back to DemoResearchProvider when DEEPSEEK_API_KEY is unset so the
    system stays runnable offline with no keys, matching the original design.
    Always loads persistent long-term memory and, when available, a durable
    SQLite checkpointer, regardless of which provider is used -- an offline
    demo session accumulates real memory across runs just like a live one.
    """
    store = load_persistent_store(STORE_PATH)

    if checkpointer is None:
        checkpointer = _durable_checkpointer()

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("No DEEPSEEK_API_KEY found - running with the offline demo provider.")
        return build_graph(checkpointer=checkpointer, store=store)

    if not os.environ.get("TAVILY_API_KEY"):
        raise RuntimeError(
            "TAVILY_API_KEY is required when DEEPSEEK_API_KEY is set."
        )

    from langchain_deepseek import ChatDeepSeek

    llm = ChatDeepSeek(model="deepseek-chat")
    provider = LiveResearchProvider(llm)
    print("Using live DeepSeek/Tavily provider - this will make billed API calls.")
    return build_graph(provider=provider, llm=llm, checkpointer=checkpointer, store=store)


def _durable_checkpointer():
    """
    SqliteSaver so a HITL-paused thread survives quitting and restarting the
    CLI. Returns None (letting build_graph() fall back to its InMemorySaver
    default) if the optional langgraph-checkpoint-sqlite package isn't
    installed.
    """
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
    except ImportError:
        return None

    os.makedirs(MEMORY_DIR, exist_ok=True)
    conn = sqlite3.connect(CHECKPOINT_PATH, check_same_thread=False)
    saver = SqliteSaver(conn)
    saver.setup()
    return saver
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `python3 -m unittest tests.test_config -v`
Expected: PASS (4 tests)

- [ ] **Step 7: Run the full test suite**

Run: `python3 -m unittest discover -s tests -v`
Expected: PASS, no regressions.

- [ ] **Step 8: Manual verification — real durability, offline, free**

Run, from the repo root:
```bash
DEEPSEEK_API_KEY= TAVILY_API_KEY= python3 -c "
from research_system.config import build_default_graph, CHECKPOINT_PATH, STORE_PATH
import os
graph = build_default_graph()
print('checkpoint file exists:', os.path.exists(CHECKPOINT_PATH))
print('store will be at:', STORE_PATH)
"
ls -la .research_memory/
git status --short .research_memory/  # must be empty -- confirms gitignore works
```
Expected: `checkpoint file exists: True`; `.research_memory/checkpoints.sqlite` present on disk; `git status --short .research_memory/` prints nothing (directory is gitignored).

- [ ] **Step 9: Commit**

```bash
git add research_system/config.py tests/test_config.py requirements.txt .gitignore
git commit -m "feat: persistent long-term memory store + durable SQLite checkpointer in config.py"
```

---

### Task 12: `run_demo.py` + `run_eval.py` — capture episodes, reflect, apply

**Files:**
- Modify: `run_demo.py` (full replacement)
- Modify: `run_eval.py` (full replacement)

**Interfaces:**
- Consumes: `graph.store` (Task 7, via `compiled_graph.store`); `episodic.record_episode`, `reflect.reflect_on_topic`, `store.persist_store` (Tasks 2, 3, 5); `procedural.apply_mandatory_review_override`, `procedural.is_mandatory_review_topic` (Task 4); `topic.derive_topic` (Task 1); `hitl.latest_conflicts` (existing).
- Produces: no importable interface (entry-point scripts only). `run_demo.py`'s `initial_state()` is still imported by `run_eval.py`, unchanged import path.

This task has no dedicated unit tests, matching the existing project convention for these two entry-point scripts (neither had tests before this plan) — verification is manual, end-to-end, offline, free.

- [ ] **Step 1: Replace `run_demo.py` in full**

```python
from __future__ import annotations

import json
import uuid

from langgraph.types import Command

from research_system.config import STORE_PATH, build_default_graph
from research_system.hitl import latest_conflicts
from research_system.memory.episodic import record_episode
from research_system.memory.procedural import (
    apply_mandatory_review_override,
    is_mandatory_review_topic,
)
from research_system.memory.reflect import reflect_on_topic
from research_system.memory.store import persist_store
from research_system.memory.topic import derive_topic


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
        "mandatory_human_review": is_mandatory_review_topic(derive_topic(question)),
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

    needed_human_review = "__interrupt__" in result
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

    _remember(graph, question, result, needed_human_review)


def _remember(graph, question: str, result: dict, needed_human_review: bool) -> None:
    """
    Capture this run into long-term memory and reflect on the topic. Wrapped
    in try/except: a memory-layer failure must never take down a run whose
    report already printed successfully -- a deliberate exception to this
    codebase's usual fail-loud rule, justified because memory bookkeeping
    here cannot corrupt the research result that's already been delivered.
    """
    try:
        topic = derive_topic(question)
        gate = result.get("quality_gate", {})
        conflicts = latest_conflicts(result.get("conflicts", []))
        episode = {
            "question": question,
            "mode": result["execution_plan"]["mode"],
            "active_agents": result["execution_plan"]["active_agents"],
            "gate_passed": gate.get("passed", False),
            "gate_failures": gate.get("failures", []),
            "needed_human_review": needed_human_review,
            "insight_count": gate.get("cross_agent_insight_count", 0),
            "open_conflict_issues": [c["issue"] for c in conflicts if c["status"] == "open"],
        }
        record_episode(graph.store, topic, episode)
        reflection = reflect_on_topic(graph.store, topic)
        persist_store(graph.store, STORE_PATH)

        if reflection["proposed_rule_change"]:
            _handle_rule_proposal(reflection["proposed_rule_change"])
    except Exception as exc:
        print(f"\n(memory capture skipped due to an error: {exc})")


def _handle_rule_proposal(proposal: dict) -> None:
    print("\n=== MEMORY: PROCEDURAL RULE PROPOSED ===")
    print(json.dumps(proposal, indent=2))
    answer = input("Apply this rule change? [y/N]: ").strip().lower()
    if answer == "y":
        apply_mandatory_review_override(proposal["topic"], proposal["rationale"])
        print("Rule applied -- future runs on this topic will require human review by default.")
    else:
        print("Rule change discarded.")


def main():
    graph = build_default_graph()

    print("\nInteractive multi-agent research CLI. Type a research question, or 'quit' to exit.")
    while True:
        try:
            question = input("\nResearch question (or 'quit'): ").strip()

            if not question:
                continue
            if question.lower() in ("quit", "exit"):
                print("Exiting.")
                break

            run_one_question(graph, question)
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Replace `run_eval.py` in full**

```python
from __future__ import annotations

import json
import sys
import time
import uuid

from langgraph.types import Command

from research_system.config import STORE_PATH, build_default_graph
from research_system.graph import build_graph
from research_system.hitl import latest_conflicts
from research_system.memory.episodic import record_episode
from research_system.memory.reflect import reflect_on_topic
from research_system.memory.store import persist_store
from research_system.memory.topic import derive_topic
from run_demo import initial_state

# Verified against the offline demo provider: any question that leaves trend_analysis
# active hits the same $8B (data_analysis) vs $14B (trend_analysis) market-size conflict
# every time, since DemoResearchProvider's evidence never varies by question -- so most
# of these fail the quality gate and need human review. Excluding trend_analysis is the
# one way to avoid it in demo mode. This mixed, empirically-real set is more useful than
# a hand-picked all-pass set: it's exactly the kind of thing an eval should surface.
DEFAULT_QUESTIONS = [
    "Quick overview and landscape scan of the Indian EV market",
    "First quantify battery-price changes, then forecast their impact on EV adoption, "
    "then determine which competitor benefits most",
    "Should an automaker enter the Indian EV market, and what competitive position should it take?",
    "Give me a market overview of the Indian EV market, excluding competitor analysis",
    "Quick overview and landscape scan of the Indian EV market, ignore trend forecasting",
]


def run_eval_question(graph, question: str, persist: bool) -> dict:
    thread_id = f"eval-{uuid.uuid4()}"
    config = {"configurable": {"thread_id": thread_id}}

    start = time.perf_counter()
    result = graph.invoke(initial_state(question), config=config)

    needed_human_review = "__interrupt__" in result
    while "__interrupt__" in result:
        # Auto-approve so the run reaches a final report even when it can't pass
        # autonomously -- the point is to measure autonomous success, not to block
        # an unattended batch run on a human decision that isn't coming.
        result = graph.invoke(
            Command(resume={
                "action": "approve",
                "reason": "Automated eval run: auto-approved to measure end-to-end completion.",
            }),
            config=config,
        )

    elapsed_ms = (time.perf_counter() - start) * 1000
    gate = result.get("quality_gate", {})

    proposed_rule_change = None
    try:
        topic = derive_topic(question)
        conflicts = latest_conflicts(result.get("conflicts", []))
        episode = {
            "question": question,
            "mode": result["execution_plan"]["mode"],
            "active_agents": result["execution_plan"]["active_agents"],
            "gate_passed": gate.get("passed", False),
            "gate_failures": gate.get("failures", []),
            "needed_human_review": needed_human_review,
            "insight_count": gate.get("cross_agent_insight_count", 0),
            "open_conflict_issues": [c["issue"] for c in conflicts if c["status"] == "open"],
        }
        record_episode(graph.store, topic, episode)
        reflection = reflect_on_topic(graph.store, topic)
        # Only the --live path persists memory to disk: the default offline path
        # is meant to stay a deterministic, repeatable baseline measurement, and
        # accumulating persisted memory across eval runs would undermine that.
        # record_episode/reflect_on_topic still run either way, in-process, so
        # the mechanism is exercised even when nothing is written to disk.
        if persist:
            persist_store(graph.store, STORE_PATH)
        proposed_rule_change = reflection["proposed_rule_change"]
    except Exception as exc:
        print(f"    (memory capture skipped due to an error: {exc})")

    return {
        "question": question,
        "mode": result["execution_plan"]["mode"],
        "active_agents": result["execution_plan"]["active_agents"],
        "gate_passed_autonomously": bool(gate.get("passed")) and not needed_human_review,
        "needed_human_review": needed_human_review,
        "gate_failures": gate.get("failures", []),
        "evidence_count": gate.get("evidence_count", 0),
        "cross_agent_insight_count": gate.get("cross_agent_insight_count", 0),
        "elapsed_ms": round(elapsed_ms, 1),
        "has_report": bool(result.get("final_report")),
        "proposed_rule_change": proposed_rule_change,
    }


def main():
    args = sys.argv[1:]
    live = "--live" in args
    questions = [a for a in args if a != "--live"] or DEFAULT_QUESTIONS

    if live:
        print("Using live DeepSeek/Tavily provider (if configured) -- this may make billed API calls.")
        graph = build_default_graph()
    else:
        print("Using the offline demo provider (deterministic, free). Pass --live to eval the live provider.")
        graph = build_graph()

    print(f"\nRunning eval over {len(questions)} question(s)...\n")
    results = []
    for i, question in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] {question[:70]}")
        r = run_eval_question(graph, question, persist=live)
        results.append(r)
        status = "PASS" if r["gate_passed_autonomously"] else ("HITL" if r["needed_human_review"] else "FAIL")
        print(
            f"    -> {status} | mode={r['mode']} | {r['elapsed_ms']:.0f}ms | "
            f"gate_failures={r['gate_failures']}"
        )

    total = len(results)
    autonomous_passes = sum(1 for r in results if r["gate_passed_autonomously"])
    rate = (autonomous_passes / total * 100) if total else 0.0
    avg_latency = (sum(r["elapsed_ms"] for r in results) / total) if total else 0.0

    print("\n=== EVAL SUMMARY ===")
    print(f"Autonomous quality-gate pass rate: {autonomous_passes}/{total} ({rate:.0f}%)")
    print(f"Needed human review: {sum(1 for r in results if r['needed_human_review'])}/{total}")
    print(f"Average latency: {avg_latency:.0f} ms")

    proposals = [r for r in results if r.get("proposed_rule_change")]
    if proposals:
        print(
            f"\n{len(proposals)} procedural rule change(s) proposed by memory reflection "
            "(not applied -- review via `python run_demo.py`):"
        )
        for r in proposals:
            print(f"  - {r['proposed_rule_change']}")

    print("\n=== PER-QUESTION DETAIL ===")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run the full test suite**

Run: `python3 -m unittest discover -s tests -v`
Expected: PASS, no regressions (these two files have no dedicated tests, but the full suite exercises `initial_state`/`build_default_graph`/`build_graph` indirectly).

- [ ] **Step 4: Manual verification — `run_eval.py` still works and shows the eval summary**

Run:
```bash
python3 run_eval.py
```
Expected: same shape of output as before this plan (per-question PASS/HITL lines, `=== EVAL SUMMARY ===`, `=== PER-QUESTION DETAIL ===`) — the autonomous pass rate should still be 1/5 (20%), matching the empirically-verified baseline, since the default (non-`--live`) path doesn't persist memory across separate `python run_eval.py` invocations and each of the 5 `DEFAULT_QUESTIONS` maps to a different-enough topic that in-process reflection doesn't change any single question's own outcome.

- [ ] **Step 5: Manual verification — cross-question memory sharing within one `run_demo.py` session, offline, free**

Run:
```bash
rm -f .research_memory/store.json
printf 'Quick overview of the Indian EV market\nIndian EV market overview, give me a quick scan\nquit\n' | python3 run_demo.py
```
Expected: both questions derive the same topic (`ev_indian_market`, verified in Task 1). The **second** question's `=== FINAL REPORT ===` output, under "## Specialist contributions", must show at least one contribution summary containing the text `Related memory:` and `1 prior run(s) on this topic` — because the first question's episode was recorded into the same in-process `graph.store` before the second question's `planner` node ran. The first question's report must NOT contain this text (no prior episodes existed yet when it ran).

- [ ] **Step 6: Manual verification — `.research_memory/store.json` now exists and is gitignored**

Run:
```bash
cat .research_memory/store.json | python3 -m json.tool | head -20
git status --short .research_memory/
```
Expected: valid JSON with at least 2 records (both episodes from Step 5); `git status --short` prints nothing.

- [ ] **Step 7: Commit**

```bash
git add run_demo.py run_eval.py
git commit -m "feat: capture episodic memory, reflect, and offer procedural rule approval after each run"
```

---

### Task 13: Documentation — CLAUDE.md, README.md, DESIGN_MAPPING.md

**Files:**
- Modify: `CLAUDE.md`
- Modify: `README.md`
- Modify: `DESIGN_MAPPING.md`

**Interfaces:** none (documentation only).

- [ ] **Step 1: Add a new subsection to `CLAUDE.md`**

Append this new subsection at the end of `CLAUDE.md`, after the existing "### Known limitation: `more_research` accumulation" section:

```markdown

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
```

- [ ] **Step 2: Add a new section to `README.md`**

Append this new section at the end of `README.md`, after the existing "## Important implementation note" section:

```markdown

---

## 8. Memory Management

Beyond a single research session's shared state (section 1 above), the system
distinguishes **short-term** (one research thread) from **long-term**
(cross-session) memory, split into the three standard long-term sub-types:

- **Semantic** — durable facts about a topic (e.g. "this topic's quality gate
  has failed on `unresolved_high_conflicts>0` in 2 of its last 3 runs").
- **Episodic** — a log of past runs on a topic (question, mode, whether the
  gate passed, whether HITL was needed).
- **Procedural** — versioned operational rules (quality-gate thresholds,
  per-topic `mandatory_human_review` overrides) that actually govern runtime
  behavior, analogous to an `AGENTS.md` bundle. Versioning is this repo's own
  git history, not an external product.

Memory is namespaced by **topic**, a keyword heuristic derived from the
question text (`research_system/memory/topic.py`) — not a real topic model,
documented as a known limitation there.

Example of the loop actually mattering: ask the same (or similarly-phrased)
question on the offline demo provider three times in a row, and by the third
run `reflect_on_topic()` will propose forcing human review by default for that
topic going forward, because the same quality-gate failure recurred every
time — a genuine "the system learned from experience" moment, not simulated.

See `CLAUDE.md`'s "Memory management" section for the full architecture and
the capture → analyze-and-refine → store-and-version → apply-and-consolidate
loop mapped to code.
```

- [ ] **Step 3: Update `DESIGN_MAPPING.md`**

Add these four rows to the existing table, after the `Checkpoints` row:

```markdown
| Short-term memory durability | `SqliteSaver` via `config.py`'s `_durable_checkpointer()` |
| Semantic long-term memory | `research_system/memory/semantic.py` |
| Episodic long-term memory | `research_system/memory/episodic.py` |
| Procedural long-term memory | `research_system/memory/procedural.py` + `procedural_rules.json`, versioned via this repo's git history |
```

- [ ] **Step 4: Verify no broken markdown**

Run: `python3 -c "print(open('CLAUDE.md').read()[-500:]); print('---'); print(open('README.md').read()[-500:])"` and visually confirm both files end cleanly with no unclosed code fences.

- [ ] **Step 5: Run the full test suite one final time**

Run: `python3 -m unittest discover -s tests -v`
Expected: PASS, no regressions.

- [ ] **Step 6: Commit**

```bash
git add CLAUDE.md README.md DESIGN_MAPPING.md
git commit -m "docs: document the memory management subsystem"
```
