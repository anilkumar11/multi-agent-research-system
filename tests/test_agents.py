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
