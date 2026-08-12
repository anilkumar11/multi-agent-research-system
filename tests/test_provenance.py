import unittest

from research_system.provider import DemoResearchProvider


class ProvenanceTests(unittest.TestCase):
    def test_derived_agent_evidence_keeps_parent_lineage(self):
        provider = DemoResearchProvider()
        web = provider.research("web_research", "EV market", [])
        context = web["evidence"]
        data = provider.research("data_analysis", "EV market", context)

        self.assertTrue(data["evidence"][0]["parent_evidence_ids"])
        self.assertEqual(data["evidence"][0]["produced_by"], "data_analysis")
        self.assertTrue(data["evidence"][0]["source_url"])


if __name__ == "__main__":
    unittest.main()
