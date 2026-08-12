import unittest

from research_system.planner import choose_execution_plan


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


if __name__ == "__main__":
    unittest.main()
