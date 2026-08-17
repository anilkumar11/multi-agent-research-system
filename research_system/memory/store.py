from __future__ import annotations

import json
import os

from langgraph.store.memory import InMemoryStore

MAX_RECORDS = 10_000  # InMemoryStore's search()/list_namespaces() default to
                       # limit=10; every bulk read in this package passes this
                       # explicit limit instead, to actually fetch everything.


def load_persistent_store(path: str) -> InMemoryStore:
    """
    Rehydrate an InMemoryStore from a JSON file. Starts fresh (with a printed
    warning) if the file is missing, unreadable, or malformed -- long-term
    memory is an enhancement layer, not a hard requirement for research to
    work.
    """
    store = InMemoryStore()
    if not os.path.exists(path):
        return store

    try:
        with open(path, "r") as f:
            records = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"Warning: could not read memory store at {path} ({exc}); starting fresh.")
        return store

    for record in records:
        store.put(tuple(record["namespace"]), record["key"], record["value"])
    return store


def persist_store(store: InMemoryStore, path: str) -> None:
    """Dump every namespace/key/value in the store to a JSON file."""
    records = []
    for namespace in store.list_namespaces(limit=MAX_RECORDS):
        for item in store.search(namespace, limit=MAX_RECORDS):
            records.append({
                "namespace": list(item.namespace),
                "key": item.key,
                "value": item.value,
            })

    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w") as f:
        json.dump(records, f, indent=2)
