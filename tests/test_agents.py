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
