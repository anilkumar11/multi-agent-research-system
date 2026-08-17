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
