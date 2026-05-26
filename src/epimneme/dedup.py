"""SimHash near-duplicate detection for memories.

Locality-sensitive hashing detects near-duplicate content before embedding,
saving compute and preventing memory bloat.  Memories with Hamming
distance <= threshold (default 3) are considered near-duplicates.
"""

from __future__ import annotations

import hashlib
import re
import struct


def _token_hash(token: str) -> int:
    """Hash a single token to a 64-bit integer."""
    digest = hashlib.md5(token.encode("utf-8")).digest()
    return struct.unpack("<Q", digest[:8])[0]


def compute_simhash(text: str, hashbits: int = 64) -> int:
    """Compute a SimHash fingerprint for text content.

    Args:
        text: Input text to hash
        hashbits: Number of hash bits (default 64)

    Returns:
        Integer fingerprint
    """
    tokens = text.lower().split()
    if not tokens:
        return 0

    v = [0] * hashbits
    for token in tokens:
        h = _token_hash(token)
        for i in range(hashbits):
            if h & (1 << i):
                v[i] += 1
            else:
                v[i] -= 1

    fingerprint = 0
    for i in range(hashbits):
        if v[i] > 0:
            fingerprint |= 1 << i

    # Convert to signed 64-bit so it fits in PostgreSQL BIGINT (-2^63 to 2^63-1)
    if fingerprint >= (1 << 63):
        fingerprint -= 1 << 64

    return fingerprint


def hamming_distance(a: int, b: int) -> int:
    """Count the number of differing bits between two integers."""
    return bin(a ^ b).count("1")


def is_near_duplicate(hash1: int, hash2: int, threshold: int = 3) -> bool:
    """Check if two SimHash fingerprints indicate near-duplicate content."""
    return hamming_distance(hash1, hash2) <= threshold


# ── Entity-aware dedup ──────────────────────────────────────────────────

# Words that are too common to be distinctive entities
_ENTITY_STOP = frozenset({
    "i", "me", "my", "we", "our", "you", "your", "it", "its", "they",
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "is", "am", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "can", "may", "might", "not", "no", "so", "if",
    "that", "this", "these", "those", "some", "any", "all", "more",
    "much", "very", "just", "also", "too", "than", "then", "now",
    "here", "there", "when", "where", "how", "what", "which", "who",
    "with", "from", "about", "into", "by", "as", "up", "out", "new",
    "really", "because", "something", "things", "think", "know",
    "want", "need", "like", "prefer", "enjoy", "love", "good", "great",
    "best", "better", "one", "ones", "get", "got", "going", "make",
    "time", "way", "thing", "lot", "many", "few", "other", "well",
})

_ENTITY_WORD_RE = re.compile(r"\b([A-Za-z0-9][\w\-]{2,})\b")


def extract_distinctive_nouns(text: str) -> frozenset[str]:
    """Extract distinctive content words for entity-aware dedup.

    Returns lowercased nouns/entities that characterize the specific
    "what" of a memory — used to prevent merging memories about
    different things that have similar structure.

    E.g. "I like the red one" vs "I like the blue one" should yield
    {"red"} and {"blue"} respectively.
    """
    words = _ENTITY_WORD_RE.findall(text)
    return frozenset(
        w.lower() for w in words
        if w.lower() not in _ENTITY_STOP and len(w) >= 3
    )


def entities_diverge(text_a: str, text_b: str, min_jaccard: float = 0.3) -> bool:
    """Check if two texts have sufficiently different entities.

    Returns True if the distinctive nouns diverge enough that they
    should NOT be considered duplicates, even if overall similarity
    is high.  Uses Jaccard similarity on distinctive noun sets.

    A min_jaccard of 0.3 means: if fewer than 30% of the distinctive
    words overlap, the entities are considered divergent.  This catches
    "I like red" vs "I like blue" (Jaccard=0) while allowing
    "PostgreSQL on port 5432" vs "database uses port 5432" (Jaccard=0.33).
    """
    nouns_a = extract_distinctive_nouns(text_a)
    nouns_b = extract_distinctive_nouns(text_b)

    if not nouns_a or not nouns_b:
        return False  # can't determine, default to "same"

    intersection = nouns_a & nouns_b
    union = nouns_a | nouns_b

    jaccard = len(intersection) / len(union) if union else 1.0
    return jaccard < min_jaccard
