import unittest

from research_system.memory.topic import derive_topic


class DeriveTopicTests(unittest.TestCase):
    def test_stable_across_rephrasing(self):
        a = derive_topic("Quick overview of the Indian EV market")
        b = derive_topic("Indian EV market overview, give me a quick scan")
        self.assertEqual(a, b)
        self.assertEqual(a, "ev_indian_market")

    def test_distinct_for_different_topics(self):
        ev = derive_topic("Quick overview of the Indian EV market")
        chips = derive_topic("What should we know about semiconductor supply chains?")
        self.assertNotEqual(ev, chips)

    def test_all_stopword_question_falls_back_to_general(self):
        self.assertEqual(derive_topic("Should it do this?"), "general")
        self.assertEqual(derive_topic(""), "general")
        self.assertEqual(derive_topic("Please give me a quick overview."), "general")

    def test_case_and_punctuation_insensitive(self):
        a = derive_topic("INDIAN EV MARKET!!")
        b = derive_topic("indian, ev, market")
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
