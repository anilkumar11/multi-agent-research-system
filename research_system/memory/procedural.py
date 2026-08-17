from __future__ import annotations

import json
import os

DEFAULT_PATH = os.path.join(os.path.dirname(__file__), "procedural_rules.json")

DEFAULT_RULES = {
    "quality_gate": {
        "min_evidence": 4,
        "min_source_types": 2,
        "min_avg_confidence": 0.70,
        "min_cross_agent_insights": 1,
    },
    "mandatory_human_review_topics": {},
}


def _deep_copy_defaults() -> dict:
    return json.loads(json.dumps(DEFAULT_RULES))


def load_rules(path: str = DEFAULT_PATH) -> dict:
    """
    Load procedural rules, falling back to DEFAULT_RULES (which exactly match
    quality.py's original hardcoded constants) on any missing or malformed
    file -- procedural memory is an enhancement layer, never a hard
    requirement, matching this package's other memory-layer error handling.
    """
    if not os.path.exists(path):
        return _deep_copy_defaults()

    try:
        with open(path, "r") as f:
            rules = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"Warning: could not read procedural rules at {path} ({exc}); using defaults.")
        return _deep_copy_defaults()

    merged = _deep_copy_defaults()
    merged["quality_gate"].update(rules.get("quality_gate", {}))
    merged["mandatory_human_review_topics"].update(rules.get("mandatory_human_review_topics", {}))
    return merged


def apply_mandatory_review_override(topic: str, rationale: str, path: str = DEFAULT_PATH) -> None:
    """Persist a human-approved procedural rule change: force HITL for this topic."""
    rules = load_rules(path)
    rules["mandatory_human_review_topics"][topic] = rationale

    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w") as f:
        json.dump(rules, f, indent=2)
    os.replace(tmp_path, path)


def is_mandatory_review_topic(topic: str, path: str = DEFAULT_PATH) -> bool:
    rules = load_rules(path)
    return topic in rules["mandatory_human_review_topics"]
