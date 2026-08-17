from __future__ import annotations

import re

_STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "of", "in", "on", "for", "to", "is", "are",
    "what", "which", "that", "this", "it", "its", "into", "with", "should",
    "would", "could", "will", "do", "does", "did", "take", "give", "me",
    "please", "so", "just", "there",
    "quick", "overview", "scan", "landscape", "fast", "snapshot",
    "first", "then", "after", "depends", "because", "impact",
    "quantify", "forecast", "projection", "why", "determine", "benefits", "most",
    "excluding", "exclude", "ignore", "skip", "without", "no", "not",
})


def derive_topic(question: str) -> str:
    """
    Deterministic, offline topic key for namespacing long-term memory. A simple
    keyword heuristic, not semantic topic modeling: strips filler words and
    meta-instruction words (words about HOW to research, not WHAT the topic
    is), then sorts and joins whatever's left so the key is stable regardless
    of phrasing/word order. No length cap -- a longer question just produces a
    longer (still stable, still deterministic) key.

    Known limitation: two real-world-same-topic questions phrased with little
    vocabulary overlap may land on different topic keys. A more robust version
    (embedding similarity, which InMemoryStore actually supports via
    index_config) is a reasonable future improvement, not built here.
    """
    words = re.findall(r"[a-z0-9]+", question.lower())
    significant = sorted({w for w in words if w not in _STOPWORDS and len(w) > 1})
    return "_".join(significant) if significant else "general"
