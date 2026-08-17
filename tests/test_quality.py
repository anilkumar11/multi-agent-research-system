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
