import os
import unittest
from unittest import mock

from research_system import config


class BuildDefaultGraphTests(unittest.TestCase):
    def test_falls_back_to_demo_without_deepseek_key(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with mock.patch("research_system.config.build_graph") as mock_build:
                mock_build.return_value = "demo-graph"
                result = config.build_default_graph()
        mock_build.assert_called_once_with(checkpointer=None)
        self.assertEqual(result, "demo-graph")

    def test_raises_without_tavily_key(self):
        env = {"DEEPSEEK_API_KEY": "fake-deepseek-key"}
        with mock.patch.dict(os.environ, env, clear=True):
            with self.assertRaises(RuntimeError):
                config.build_default_graph()

    def test_builds_live_graph_when_both_keys_present(self):
        env = {
            "DEEPSEEK_API_KEY": "fake-deepseek-key",
            "TAVILY_API_KEY": "fake-tavily-key",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            with mock.patch("research_system.config.build_graph") as mock_build:
                mock_build.return_value = "live-graph"
                result = config.build_default_graph()
        self.assertEqual(result, "live-graph")
        _, kwargs = mock_build.call_args
        self.assertIsNotNone(kwargs.get("provider"))
        self.assertIsNotNone(kwargs.get("llm"))


if __name__ == "__main__":
    unittest.main()
