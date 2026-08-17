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
