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
