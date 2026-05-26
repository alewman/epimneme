"""Keyword reranking — boost search results that match query terms.

Two-stage reranking inspired by MemPalace's hybrid retrieval pipeline:

Stage 1: Query-term matching with stop-word filtering
  - Tokenize query and result content
  - Remove common stop words
  - Count exact and fuzzy term matches
  - Boost scores proportional to match ratio

Stage 2: Phrase matching
  - Detect 2-3 word phrases from query
  - Bonus for contiguous phrase matches in content

This is the "keyword rerank" stage that sits between semantic search
and (optional) LLM reranking.  MemPalace shows this adds +1-3pp on
retrieval benchmarks by catching exact terminology that embeddings miss.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Common English stop words — filtered from both query and content
# during term matching to focus on meaningful terms
STOP_WORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "it", "as", "be", "was", "were",
    "been", "are", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "can", "shall", "not",
    "no", "so", "if", "then", "than", "that", "this", "these", "those",
    "i", "me", "my", "we", "us", "our", "you", "your", "he", "him",
    "his", "she", "her", "they", "them", "their", "its", "what", "which",
    "who", "whom", "how", "when", "where", "why", "all", "each", "every",
    "both", "few", "more", "most", "some", "any", "also", "just", "about",
    "up", "out", "into", "over", "after", "before", "between", "under",
    "again", "very", "too", "here", "there", "now", "only",
})

# Regex to split on non-alphanumeric (preserving hyphens in identifiers)
_SPLIT_RE = re.compile(r"[^a-z0-9_]+")  # hyphens are splitters; "art-related" → ["art", "related"]


@dataclass
class RerankResult:
    """Result of keyword reranking a single memory."""

    memory_id: str
    original_score: float
    keyword_boost: float
    final_score: float


def _tokenize(text: str) -> list[str]:
    """Lowercase, split on non-alphanumeric, remove stop words."""
    tokens = _SPLIT_RE.split(text.lower())
    return [t for t in tokens if t and t not in STOP_WORDS and len(t) > 1]


def _extract_phrases(tokens: list[str], max_len: int = 3) -> list[str]:
    """Extract 2 and 3 word phrases from token list."""
    phrases = []
    for n in range(2, min(max_len + 1, len(tokens) + 1)):
        for i in range(len(tokens) - n + 1):
            phrases.append(" ".join(tokens[i : i + n]))
    return phrases


def keyword_rerank(
    query: str,
    results: list[tuple[str, float, str]],
    *,
    term_weight: float = 0.15,
    phrase_weight: float = 0.10,
) -> list[RerankResult]:
    """Rerank search results by keyword/phrase match with the query.

    Args:
        query: The original search query.
        results: List of (memory_id, semantic_score, content) tuples.
        term_weight: Max score boost for term matches (0.0-1.0).
        phrase_weight: Max score boost for phrase matches (0.0-1.0).

    Returns:
        List of RerankResult sorted by final_score descending.
    """
    query_tokens = _tokenize(query)
    if not query_tokens:
        return [
            RerankResult(
                memory_id=mid,
                original_score=score,
                keyword_boost=0.0,
                final_score=score,
            )
            for mid, score, _ in results
        ]

    query_phrases = _extract_phrases(query_tokens)
    query_token_set = set(query_tokens)

    reranked: list[RerankResult] = []

    for memory_id, original_score, content in results:
        content_tokens = _tokenize(content)
        content_token_set = set(content_tokens)

        # TF-weighted term match: count occurrences of each query term in content.
        # Cap controls how much repetition is rewarded:
        #   cap=1 → binary presence (original pre-v120 behavior)
        #   cap=2 → mild TF boost (v208 test)
        #   cap=3 → original v120 TF (caused 25 regressions)
        _TF_CAP = 2
        if query_token_set:
            from collections import Counter
            content_freq = Counter(content_tokens)
            tf_sum = sum(min(content_freq.get(t, 0), _TF_CAP) for t in query_token_set)
            term_ratio = tf_sum / (len(query_token_set) * _TF_CAP)
        else:
            term_ratio = 0.0

        # Phrase matching: check if query phrases appear in content
        phrase_ratio = 0.0
        if query_phrases:
            content_lower = content.lower()
            phrase_hits = sum(1 for p in query_phrases if p in content_lower)
            phrase_ratio = phrase_hits / len(query_phrases)

        boost = (term_ratio * term_weight) + (phrase_ratio * phrase_weight)

        reranked.append(
            RerankResult(
                memory_id=memory_id,
                original_score=original_score,
                keyword_boost=boost,
                final_score=min(1.0, original_score + boost),
            )
        )

    reranked.sort(key=lambda r: r.final_score, reverse=True)
    return reranked
