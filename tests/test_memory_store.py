import os
import tempfile
import unittest

from langgraph.store.memory import InMemoryStore

from research_system.memory.store import load_persistent_store, persist_store


class StorePersistenceTests(unittest.TestCase):
    def test_round_trip_persists_and_reloads_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "store.json")

            store = InMemoryStore()
            store.put(("topic-a", "semantic"), "fact-1", {"pattern": "x"})
            persist_store(store, path)

            reloaded = load_persistent_store(path)
            item = reloaded.get(("topic-a", "semantic"), "fact-1")
            self.assertIsNotNone(item)
            self.assertEqual(item.value, {"pattern": "x"})

    def test_missing_file_returns_empty_store(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "does-not-exist.json")
            store = load_persistent_store(path)
            self.assertEqual(store.list_namespaces(), [])

    def test_corrupted_file_falls_back_to_empty_store(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "corrupt.json")
            with open(path, "w") as f:
                f.write("{not valid json")

            store = load_persistent_store(path)
            self.assertEqual(store.list_namespaces(), [])

    def test_persist_creates_parent_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "nested", "dir", "store.json")
            store = InMemoryStore()
            store.put(("t", "episodic"), "e1", {"question": "q"})
            persist_store(store, path)
            self.assertTrue(os.path.exists(path))


if __name__ == "__main__":
    unittest.main()
