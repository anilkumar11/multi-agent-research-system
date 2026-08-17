from __future__ import annotations

from . import episodic, semantic
from ..utils import stable_id

RECENT_WINDOW = 3  # how many of the most recent episodes reflection considers


def _count_recurrences(episodes: list[dict], extract) -> dict[str, int]:
    counts: dict[str, int] = {}
    for ep in episodes:
        for value in extract(ep):
            counts[value] = counts.get(value, 0) + 1
    return counts


def reflect_on_topic(store, topic: str) -> dict:
    """
    Analyze-and-refine step: reads the topic's most recent episodes, upserts
    semantic facts for recurring patterns, and proposes -- but never
    auto-applies -- a procedural rule change when one failure pattern recurs
    across every one of the recent episodes considered.
    """
    episodes = episodic.recent_episodes(store, topic, limit=RECENT_WINDOW)

    failure_counts = _count_recurrences(episodes, lambda ep: ep.get("gate_failures", []))
    conflict_counts = _count_recurrences(episodes, lambda ep: ep.get("open_conflict_issues", []))

    updated_facts = []
    for reason, count in failure_counts.items():
        if count >= 2:
            fact = {
                "pattern": "recurring_gate_failure",
                "reason": reason,
                "occurrences": count,
                "checked_over": len(episodes),
            }
            semantic.upsert_fact(store, topic, f"gate_failure:{reason}", fact)
            updated_facts.append(fact)

    for issue, count in conflict_counts.items():
        if count >= 2:
            fact = {
                "pattern": "recurring_open_conflict",
                "issue": issue,
                "occurrences": count,
                "checked_over": len(episodes),
            }
            semantic.upsert_fact(store, topic, f"conflict:{stable_id('conflict-issue', issue)}", fact)
            updated_facts.append(fact)

    proposed_rule_change = None
    if len(episodes) >= RECENT_WINDOW:
        for reason, count in failure_counts.items():
            if count >= RECENT_WINDOW:
                proposed_rule_change = {
                    "rule": "mandatory_human_review_topics",
                    "topic": topic,
                    "rationale": (
                        f"The same quality-gate failure ({reason}) occurred in all of "
                        f"the last {RECENT_WINDOW} runs on this topic."
                    ),
                }
                break

    return {
        "updated_facts": updated_facts,
        "proposed_rule_change": proposed_rule_change,
    }
