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


if __name__ == "__main__":
    unittest.main()
