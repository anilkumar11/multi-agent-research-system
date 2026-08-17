from __future__ import annotations

from .store import MAX_RECORDS
from ..utils import utc_now


def upsert_fact(store, topic: str, key: str, fact: dict) -> None:
    """
    Write or update a durable fact about a topic under a stable key -- calling
    this again with the same key (e.g. an updated occurrence count) replaces
    the old value rather than duplicating it.
    """
    payload = {**fact, "updated_at": utc_now()}
    store.put((topic, "semantic"), key, payload)


def relevant_facts(store, topic: str) -> list[dict]:
    """Return all semantic facts stored for a topic."""
    items = store.search((topic, "semantic"), limit=MAX_RECORDS)
    return [item.value for item in items]
