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
