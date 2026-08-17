"""
Read-only CLI for browsing the long-term memory store built up by
run_demo.py / run_eval.py --live.

    python inspect_memory.py                     # list every topic on record
    python inspect_memory.py <topic-or-question>  # episodes + facts for a topic
    python inspect_memory.py <topic-or-question> --raw   # raw JSON instead

A bare word doesn't need to be the exact stored topic key -- if it isn't
found verbatim, it's run through the same derive_topic() heuristic the
graph itself uses, so you can pass the original question text.

Never writes anything; safe to run at any time, including mid-research-session.
"""
from __future__ import annotations

import json
import sys

from research_system.config import STORE_PATH
from research_system.memory.episodic import recent_episodes
from research_system.memory.procedural import DEFAULT_PATH, load_rules
from research_system.memory.semantic import relevant_facts
from research_system.memory.store import MAX_RECORDS, load_persistent_store
from research_system.memory.topic import derive_topic

EPISODE_LIMIT = 20  # generous ceiling for a single topic's inspection, not a hard cap


def topics_in_store(store) -> dict:
    """Map topic -> {'episodic': n, 'semantic': n} from every namespace on record."""
    topics: dict = {}
    for namespace in store.list_namespaces(limit=MAX_RECORDS):
        if len(namespace) != 2:
            continue
        topic, kind = namespace
        if kind not in ("episodic", "semantic"):
            continue
        counts = topics.setdefault(topic, {"episodic": 0, "semantic": 0})
        counts[kind] = len(store.search(namespace, limit=MAX_RECORDS))
    return topics


def resolve_topic(store, raw: str) -> str:
    """Exact topic key if it's already on record, else derive one from free text."""
    if raw in topics_in_store(store):
        return raw
    derived = derive_topic(raw)
    if derived != raw:
        print(f"(no topic '{raw}' on record -- treating it as a question, derived topic: '{derived}')")
    return derived


def print_summary(store, rules: dict) -> None:
    topics = topics_in_store(store)
    if not topics:
        print("No topics in long-term memory yet -- run run_demo.py or run_eval.py --live first.")
    else:
        print(f"=== TOPICS ({len(topics)}) ===")
        mandatory = rules["mandatory_human_review_topics"]
        for topic in sorted(topics):
            counts = topics[topic]
            flag = "  [mandatory human review]" if topic in mandatory else ""
            print(f"  {topic:<40} episodic={counts['episodic']:<3} semantic={counts['semantic']:<3}{flag}")

    print("\n=== PROCEDURAL RULES ===")
    print(f"(source: {DEFAULT_PATH})")
    print(json.dumps(rules, indent=2))

    print(f"\nInspect one topic with: python inspect_memory.py <topic-or-question>")


def print_episode(episode: dict) -> None:
    status = "PASSED" if episode.get("gate_passed") else "FAILED"
    print(f"  [{episode.get('recorded_at', '?')}] gate={status} mode={episode.get('mode', '?')}")
    print(f"    question: {episode.get('question', '?')}")
    print(f"    active_agents: {', '.join(episode.get('active_agents', [])) or '(none)'}")
    if episode.get("gate_failures"):
        print(f"    gate_failures: {', '.join(episode['gate_failures'])}")
    print(f"    insight_count: {episode.get('insight_count', 0)}  "
          f"needed_human_review: {episode.get('needed_human_review', False)}")
    if episode.get("open_conflict_issues"):
        print(f"    open_conflict_issues: {', '.join(episode['open_conflict_issues'])}")


def print_fact(key: str, fact: dict) -> None:
    print(f"  [{fact.get('updated_at', '?')}] {key}")
    for field, value in fact.items():
        if field == "updated_at":
            continue
        print(f"    {field}: {value}")


def print_topic(store, rules: dict, topic: str, raw_mode: bool) -> None:
    episodes = recent_episodes(store, topic, limit=EPISODE_LIMIT)
    facts = relevant_facts(store, topic)
    mandatory = rules["mandatory_human_review_topics"].get(topic)

    if raw_mode:
        print(json.dumps({"topic": topic, "episodes": episodes, "facts": facts}, indent=2))
        return

    print(f"=== TOPIC: {topic} ===")
    if mandatory:
        print(f"  mandatory human review: {mandatory}")

    print(f"\n--- Episodes ({len(episodes)}, newest first) ---")
    if not episodes:
        print("  (none)")
    for episode in episodes:
        print_episode(episode)
        print()

    print(f"--- Semantic facts ({len(facts)}) ---")
    if not facts:
        print("  (none)")
    for i, fact in enumerate(facts):
        print_fact(f"fact_{i}", fact)
        print()


def main() -> None:
    args = [a for a in sys.argv[1:] if a != "--raw"]
    raw_mode = "--raw" in sys.argv[1:]

    store = load_persistent_store(STORE_PATH)
    rules = load_rules()

    if not args:
        print_summary(store, rules)
        return

    topic = resolve_topic(store, " ".join(args))
    print_topic(store, rules, topic, raw_mode)


if __name__ == "__main__":
    main()
