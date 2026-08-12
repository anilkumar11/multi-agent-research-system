from __future__ import annotations

import hashlib
import time
from datetime import datetime, timezone
from typing import Any, Callable


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_id(prefix: str, *parts: str) -> str:
    payload = "|".join(parts).encode("utf-8")
    digest = hashlib.sha1(payload).hexdigest()[:10]
    return f"{prefix}-{digest}"


def timed_node(node_name: str, fn: Callable):
    """Wrap a LangGraph node and emit append-only telemetry."""
    def wrapper(state, *args, **kwargs):
        start = time.perf_counter()
        started_at = utc_now()
        before = len(state.get("evidence", []))
        result = fn(state, *args, **kwargs) or {}
        after = before + len(result.get("evidence", []))
        ended_at = utc_now()
        elapsed_ms = (time.perf_counter() - start) * 1000
        mode = state.get("execution_plan", {}).get("mode", "unknown")
        event = {
            "node": node_name,
            "started_at": started_at,
            "ended_at": ended_at,
            "duration_ms": round(elapsed_ms, 2),
            "execution_mode": mode,
            "evidence_before": before,
            "evidence_after": after,
            "note": result.pop("_telemetry_note", ""),
        }
        result["telemetry"] = [event]
        return result
    wrapper.__name__ = node_name
    return wrapper


def average(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def unique_source_types(evidence: list[dict]) -> set[str]:
    return {e.get("source_type", "unknown") for e in evidence}


def unique_agents_for_evidence(evidence: list[dict], evidence_ids: list[str]) -> set[str]:
    wanted = set(evidence_ids)
    return {e["produced_by"] for e in evidence if e["evidence_id"] in wanted}
