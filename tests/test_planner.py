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
