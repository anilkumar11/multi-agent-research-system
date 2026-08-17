from __future__ import annotations

import uuid

from .store import MAX_RECORDS
from ..utils import utc_now


def record_episode(store, topic: str, episode: dict) -> None:
    """Append one completed run's summary to a topic's episodic memory."""
    recorded_at = utc_now()
    payload = {**episode, "recorded_at": recorded_at}
    store.put((topic, "episodic"), str(uuid.uuid4()), payload)


def recent_episodes(store, topic: str, limit: int = 5) -> list[dict]:
    """Return up to `limit` most recent episodes for a topic, newest first."""
    items = store.search((topic, "episodic"), limit=MAX_RECORDS)
    ordered = sorted(items, key=lambda item: (item.value["recorded_at"], item.key), reverse=True)
    return [item.value for item in ordered[:limit]]
