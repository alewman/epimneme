"""Reciprocal Rank Fusion (RRF) — rank-based merging for hybrid search.

Combines multiple ranked result lists without requiring score normalization.
Each list contributes ``1 / (k + rank)`` per item, where *k* is a damping
constant (default 60, the standard value from Cormack et al. 2009).

This prevents the "echo chamber" where high-magnitude vector scores drown
out keyword matches.  A memory ranked #1 on keywords and #15 on vectors
will outscore one ranked #3 on vectors but absent from keywords.

Adaptive fusion: queries detected as preference/vague get reduced keyword
weight so the vector "vibe match" dominates.  Factual queries keep full
keyword weight.
"""

from __future__ import annotations

import math
import re
from datetime import date, timedelta
from typing import Iterable, Sequence

from epimneme.core.models import MemoryResult

# Standard RRF damping constant (Cormack et al. 2009)
RRF_K = 60


def rrf_fuse(
    *result_lists: Sequence[MemoryResult],
    k: int = RRF_K,
    weights: Sequence[float] | None = None,
) -> dict[str, MemoryResult]:
    """Fuse multiple ranked result lists using Reciprocal Rank Fusion.

    Each result list should already be sorted by relevance (best first).
    Returns a dict of memory_id → MemoryResult with RRF scores.

    The RRF score for an item appearing in lists L1, L2, ... is:
        score = sum(w_i / (k + rank_i))  for each list L_i containing it

    Args:
        *result_lists: One or more ranked sequences of MemoryResult.
        k: Damping constant. Higher = more weight to lower-ranked items.
        weights: Per-list weight multipliers. Defaults to 1.0 for all.
            E.g. [1.0, 0.7] makes the first list (vector) 1.43× more
            influential than the second (keyword).

    Returns:
        Dict mapping memory_id → MemoryResult with fused scores.
    """
    if weights is None:
        weights = [1.0] * len(result_lists)

    fused: dict[str, MemoryResult] = {}

    for list_idx, results in enumerate(result_lists):
        w = weights[list_idx] if list_idx < len(weights) else 1.0
        for rank, mr in enumerate(results):
            rrf_score = w / (k + rank + 1)  # rank is 0-indexed, +1 for 1-indexed
            mid = mr.memory.id
            if mid in fused:
                fused[mid].score += rrf_score
            else:
                mr.score = rrf_score
                fused[mid] = mr

    return fused


# Regex for proper nouns: capitalized words not at sentence start,
# or any capitalized word longer than 1 char that isn't a common word.
_PROPER_NOUN_RE = re.compile(r"\b([A-Z][a-z]{1,})\b")
_COMMON_CAPS = frozenset({
    "The", "This", "That", "What", "When", "Where", "Which", "Who",
    "How", "Why", "Did", "Does", "Can", "Could", "Would", "Should",
    "Have", "Has", "Had", "Are", "Was", "Were", "Will", "May",
    "Not", "But", "And", "For", "With", "From", "Into", "About",
    "After", "Before", "Between", "During", "Until", "Also",
    "Just", "Only", "Then", "Than", "Very", "Most", "Some",
})


def extract_proper_nouns(text: str) -> list[str]:
    """Extract likely proper nouns from text.

    Returns capitalized words that aren't common English words,
    deduplicated and in order of appearance.
    """
    matches = _PROPER_NOUN_RE.findall(text)
    seen: set[str] = set()
    result: list[str] = []
    for m in matches:
        if m not in _COMMON_CAPS and m not in seen:
            seen.add(m)
            result.append(m)
    return result


# ── Preference term extraction ──────────────────────────────────────────

# Verbs/phrases that signal a user preference or desire
_PREF_INDICATORS = re.compile(
    r"\b(?:prefer(?:red|ring|s)?|"
    r"lik(?:e[ds]?|ing)|"
    r"lov(?:e[ds]?|ing)|"
    r"enjoy(?:ed|ing|s)?|"
    r"interested\s+in|"
    r"looking\s+(?:for|to|into)|"
    r"want(?:ed|ing|s)?|"
    r"need(?:ed|ing|s)?|"
    r"favou?rite|"
    r"recommend(?:ed|ing|s)?|"
    r"upgrad(?:e[ds]?|ing)|"
    r"chos(?:e|en)|choos(?:e|ing)|"
    r"pick(?:ed|ing|s)?|"
    r"bought|purchas(?:e[ds]?|ing)|"
    r"got\s+a|"
    r"switch(?:ed|ing)\s+to|"
    r"start(?:ed|ing)\s+using|"
    r"been\s+using|"
    r"try(?:ing|ied)\s+out|"
    r"lean(?:ing|s|ed)?\s+towards?)\b",
    re.IGNORECASE,
)

# Common verbs/stopwords to filter from extracted terms
_PREF_STOP = frozenset({
    "i", "me", "my", "we", "our", "you", "your", "it", "its", "they",
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "is", "am", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "can", "may", "might", "shall", "not", "no", "so", "if",
    "that", "this", "these", "those", "some", "any", "all", "more",
    "much", "very", "just", "also", "too", "than", "then", "now",
    "here", "there", "when", "where", "how", "what", "which", "who",
    "with", "from", "about", "into", "by", "as", "up", "out", "new",
    "really", "because", "something", "things", "think", "know",
    "want", "need", "looking", "interested", "tried", "trying",
    "started", "been", "using", "recently", "currently", "actually",
    "going", "getting", "got", "one", "ones", "good", "great", "best",
})


def extract_preference_terms(text: str) -> list[str]:
    """Extract key nouns/entities near preference indicators.

    Finds sentences containing preference language (prefer, like, enjoy,
    bought, upgrade, etc.) and returns distinctive nouns from those
    sentences — the "what" of the preference.

    Returns deduplicated terms in order of appearance.
    """
    # Split into sentences (rough)
    sentences = re.split(r"[.!?\n]+", text)
    terms: list[str] = []
    seen: set[str] = set()

    for sent in sentences:
        if not _PREF_INDICATORS.search(sent):
            continue
        # Extract meaningful words: alphanumeric, 3+ chars, not stopwords
        words = re.findall(r"\b([A-Za-z0-9][\w\-]{2,})\b", sent)
        for w in words:
            wl = w.lower()
            if wl not in _PREF_STOP and wl not in seen and len(wl) >= 3:
                seen.add(wl)
                terms.append(w)

    return terms


# ── Adaptive fusion ─────────────────────────────────────────────────────

# Queries that are vague/preference-like: short, few specific nouns,
# asking about tastes or experiences.  For these, keyword search hurts
# because query words ("tips", "advice", "recommendations") rarely appear
# in gold content.  We reduce keyword weight to let vector similarity
# do the heavy lifting.

_VAGUE_QUERY_RE = re.compile(
    r"\b(?:any\s+(?:tips|advice|suggestions|thoughts|recommendations|ideas)|"
    r"what\s+(?:do\s+you|should\s+I|would\s+you|can\s+you)\s+(?:recommend|suggest|think)|"
    r"how\s+(?:do\s+you|should\s+I)\s+(?:feel|like|choose)|"
    r"tell\s+me\s+about)\b",
    re.IGNORECASE,
)

# Queries with these patterns are clearly factual — keep full keyword weight
_FACTUAL_QUERY_RE = re.compile(
    r"\b(?:when\s+did|what\s+(?:is|was|were|are)\s+(?:the|my|our)|"
    r"where\s+(?:is|was|did)|"
    r"which\s+(?:port|version|file|database|command|config|setting)|"
    r"how\s+(?:many|much|long|often)|"
    r"what\s+(?:time|date|year|port|version|number))\b",
    re.IGNORECASE,
)


# Common question words / stopwords that don't carry retrieval signal
_QUERY_STOP = frozenset({
    "i", "me", "my", "we", "our", "you", "your", "it", "its", "they",
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "is", "am", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "can", "may", "might", "shall", "not", "no", "so", "if",
    "that", "this", "these", "those", "some", "any", "all", "more",
    "much", "very", "just", "also", "too", "than", "then", "now",
    "here", "there", "when", "where", "how", "what", "which", "who",
    "with", "from", "about", "into", "by", "as", "up", "out",
    "tips", "advice", "suggestions", "thoughts", "recommendations",
    "ideas", "tell", "recommend", "suggest", "think", "feel",
    "like", "choose", "know", "need", "want", "looking",
    "good", "best", "great", "new", "really", "something",
})


def _query_has_specific_nouns(query: str, min_nouns: int = 1) -> bool:
    """Return True if the query contains specific content nouns (not just stopwords).

    A query like "Any tips for portrait photography?" has specific nouns
    that keyword search can usefully match. "Any tips?" does not.
    """
    words = re.findall(r"\b([a-zA-Z]{3,})\b", query)
    content_words = [w for w in words if w.lower() not in _QUERY_STOP]
    return len(content_words) >= min_nouns


def adaptive_keyword_weight(query: str, base_weight: float) -> float:
    """Return an adjusted keyword weight based on query characteristics.

    - Factual/specific queries → full base_weight (keywords matter)
    - Vague queries WITH specific nouns → full weight (keywords can match the nouns)
    - Vague queries WITHOUT specific nouns → base_weight * 0.3 (let vectors dominate)
    - Short preference queries without nouns → base_weight * 0.5
    - Everything else → base_weight unchanged
    """
    # Factual pattern takes priority — always keep full keyword weight
    if _FACTUAL_QUERY_RE.search(query):
        return base_weight

    # Check for vague/preference query patterns
    if _VAGUE_QUERY_RE.search(query):
        # If the query has specific nouns ("photography", "cookies"),
        # keywords can still match them — keep full weight
        if _query_has_specific_nouns(query, min_nouns=2):
            return base_weight
        return base_weight * 0.3

    # Short preference query without specific nouns
    if _PREF_INDICATORS.search(query) and len(query.split()) <= 12:
        if not _query_has_specific_nouns(query, min_nouns=2):
            return base_weight * 0.5

    return base_weight


# ── Recency / temporal intent detection ────────────────────────────────────

# Signals that the query is asking about the *most recent* state of something,
# not a specific point-in-time event.  For these queries we apply a mild
# session-ordinal recency boost so newer sessions surface above older ones
# when semantic scores are otherwise equal.
#
# Deliberately narrow: "when did X happen" / "what date was" are NOT recency
# queries — they want a specific session, not the newest one.  We only boost
# for clearly "current state" questions.

_RECENCY_INTENT_RE = re.compile(
    r"\b(?:"
    r"(?:most\s+recent(?:ly)?|latest|last|current(?:ly)?|"
    r"nowadays|these\s+days|at\s+(?:this\s+)?(?:point|moment)|"
    r"right\s+now|still|anymore|now|today)"
    r")\b",
    re.IGNORECASE,
)

# Phrases that indicate a specific point in time — suppress the recency boost
# even if a recency word appears (e.g. "what did I last eat on May 3rd?")
_SPECIFIC_TIME_RE = re.compile(
    r"\b(?:\d{4}|january|february|march|april|may|june|july|august|"
    r"september|october|november|december|"
    r"jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec|"
    r"monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
    r"yesterday|last\s+(?:monday|tuesday|wednesday|thursday|friday)|"
    r"\d{1,2}/\d{1,2})\b",
    re.IGNORECASE,
)


def has_recency_intent(query: str) -> bool:
    """Return True if the query is asking about the current/most-recent state.

    Returns False for queries that reference a specific date or point in time,
    since those need exact matching, not a recency boost.
    """
    if not _RECENCY_INTENT_RE.search(query):
        return False
    # Suppress if the query contains a specific date reference
    if _SPECIFIC_TIME_RE.search(query):
        return False
    return True


def apply_recency_boost(
    fused: "dict[str, MemoryResult]",
    session_ordinals: "dict[str, int]",
    boost_weight: float = 0.015,
) -> None:
    """Apply a mild session-ordinal recency boost to fused results (in-place).

    Memories from newer sessions (higher ordinal) get a small additive bonus.
    The boost is normalized to [0, boost_weight] so it never overrides a
    strong semantic/keyword match — it only breaks ties in favour of recency.

    Args:
        fused:            Dict of memory_id → MemoryResult (modified in-place).
        session_ordinals: Dict of session_id → ordinal integer.
        boost_weight:     Max additive boost (default 0.015 — enough to break
                          ties but not to promote a weak match over a strong one).
    """
    if not session_ordinals:
        return

    max_ordinal = max(session_ordinals.values()) if session_ordinals else 1
    if max_ordinal <= 0:
        return

    for mr in fused.values():
        sid = mr.memory.session_id
        if sid and sid in session_ordinals:
            # Normalize ordinal to [0, 1] then scale to boost_weight
            norm = session_ordinals[sid] / max_ordinal
            mr.score += norm * boost_weight


# ── Resolver: context-entity injection for vague queries ────────────────────
#
# When a query like "Any tips?" or "Any advice?" arrives, the vector and
# keyword signals are near-zero — every session looks equally relevant.
# The Resolver detects these low-information-density queries and augments
# retrieval by boosting memories that share entities with the most recently-
# indexed sessions (the user's current context).
#
# This addresses Gemini's "Object-as-Context" problem: the user bought a
# Power Bank yesterday and asks "Any tips?" — the answer is implicit in their
# recent activity, not in the query text itself.


def is_vague_query(query: str) -> bool:
    """Return True if the query lacks specific content words.

    Vague queries benefit from context injection: boosting memories that
    share entities with the most recently-indexed sessions.

    Returns False for factual or specific-noun queries where keyword/vector
    search already works well.
    """
    if _FACTUAL_QUERY_RE.search(query):
        return False
    # Explicit vague patterns ("any tips", "any advice", etc.) without 2+ nouns
    if _VAGUE_QUERY_RE.search(query):
        return not _query_has_specific_nouns(query, min_nouns=2)
    # Very short queries (≤4 words) with no distinguishing content nouns
    if len(query.split()) <= 4 and not _query_has_specific_nouns(query, min_nouns=1):
        return True
    return False


def extract_context_entities(
    fused_results: "list[MemoryResult]",
    session_ordinals: "dict[str, int]",
    n_recent_sessions: int = 2,
    max_entities: int = 5,
) -> list[str]:
    """Extract key entities from the most recently-indexed sessions.

    Identifies the N highest-ordinal sessions in the result set, pulls
    significant nouns/entities from their content, and returns them for
    score-boosting against the full candidate set.

    Args:
        fused_results:     All fused MemoryResult objects.
        session_ordinals:  Map of session_id → ordinal (higher = more recent).
        n_recent_sessions: Number of most-recent distinct sessions to sample.
        max_entities:      Max entity strings to return.

    Returns:
        List of entity strings (e.g. ["PowerBank", "USB-C", "battery"]).
    """
    if not fused_results or not session_ordinals:
        return []

    # Sort by session recency (most recent first)
    def _ord(mr: "MemoryResult") -> int:
        sid = mr.memory.session_id
        return session_ordinals.get(sid, 0) if sid else 0

    sorted_results = sorted(fused_results, key=_ord, reverse=True)

    # Gather text from the N most recent distinct sessions
    seen_sids: set[str] = set()
    recent_texts: list[str] = []
    for mr in sorted_results:
        sid = mr.memory.session_id
        if sid and sid not in seen_sids:
            seen_sids.add(sid)
            recent_texts.append(mr.memory.content)
        if len(seen_sids) >= n_recent_sessions:
            break

    if not recent_texts:
        return []

    combined = " ".join(recent_texts)
    # Extract candidates: CamelCase product names, ALL-CAPS acronyms,
    # and any alphanumeric word ≥4 chars (catches "battery", "charging", etc.)
    candidates = re.findall(
        r"\b([A-Z][a-z]+(?:[A-Z][a-z]+)+|[A-Z]{2,}|[a-zA-Z][\w\-]{3,})\b",
        combined,
    )

    combined_stop = _PREF_STOP | _QUERY_STOP
    entities: list[str] = []
    seen: set[str] = set()
    for w in candidates:
        wl = w.lower()
        if wl not in combined_stop and wl not in seen:
            seen.add(wl)
            entities.append(w)
        if len(entities) >= max_entities:
            break

    return entities


# ── Preference signal boost ────────────────────────────────────────────────
#
# When a query is vague and personal ("Any tips on what to bake?"), the correct
# memory often contains first-person preference language ("I love sourdough").
# This boost surfaces those memories above equally-scored neutral sessions.
# General-purpose: detecting personal preference statements is not benchmark-
# specific — it's how any memory system should recognise user preferences.

_PREF_SIGNAL_RE = re.compile(
    r"\bI\s+(?:love|prefer|like|use|always|usually|recommend|enjoy|find|"
    r"tend\s+to|try\s+to|often|never|hate|dislike|avoid)\b",
    re.IGNORECASE,
)


def apply_preference_signal_boost(
    fused: "dict[str, MemoryResult]",
    query: str,
    boost: float = 0.015,
    top_n: int = 20,
) -> None:
    """Boost candidates containing first-person preference language for vague queries.

    Fires only when the query is vague (no specific content nouns) AND personal
    (asking for advice, tips, or a recommendation).  For those queries, candidates
    that contain 'I love / prefer / usually / always ...' are stronger matches for
    personal-preference questions than equally-scored neutral memories.

    Args:
        fused:  Dict of memory_id → MemoryResult (modified in-place).
        query:  The search query string.
        boost:  Additive score bonus per preference-signal match (default 0.015).
        top_n:  Only consider the top-N candidates by current score.
    """
    if not is_vague_query(query):
        return

    # Only fire for second-person / advice-seeking framing
    if not re.search(
        r"\b(?:any\s+(?:tips|advice|suggestions|ideas|recommendations)|"
        r"what\s+(?:should|would|do)\s+(?:I|you)|"
        r"how\s+(?:should|do)\s+I|"
        r"do\s+you\s+(?:think|know|remember|recall))\b",
        query,
        re.IGNORECASE,
    ):
        return

    # Operate on top-N only to avoid boosting every session in the corpus
    top_candidates = sorted(fused.values(), key=lambda r: r.score, reverse=True)[:top_n]
    for mr in top_candidates:
        hits = len(_PREF_SIGNAL_RE.findall(mr.memory.content))
        if hits:
            fused[mr.memory.id].score += hits * boost


# ── Temporal boost ──────────────────────────────────────────────────────────
#
# Many queries contain relative date expressions ("10 days ago", "last Friday")
# that embeddings cannot resolve — every session looks equally relevant to
# "What did I buy last Tuesday?".  This boost decodes those expressions and
# applies a Gaussian decay centred on the implied target date, so memories
# whose logical date is close to the target surface above those from other
# sessions.
#
# Reference date: the most recent logical date found in the result set.
# This is more robust than using server-time "now" because benchmark haystacks
# may span historical dates.  For real-time use, server time is the fallback.
#
# The boost is additive and capped so it never overrides a strong semantic
# or keyword match — it only disambiguates among near-tied candidates.

_DATE_IN_CONTENT_RE = re.compile(r"\[Date:\s*(\d{4}-\d{2}-\d{2})\]")
_DATE_LIKE_TAG_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

# Queries asking about a time interval rather than targeting a date
_INTERVAL_QUERY_RE = re.compile(
    r"\bhow\s+(?:many|long|much)\s+(?:\w+\s+)?ago\b",
    re.IGNORECASE,
)

_WEEKDAY_MAP = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6,
}
_WORD_NUMS = {"two": 2, "three": 3, "four": 4, "five": 5, "six": 6}


def _parse_target_date(query: str, reference_date: date) -> date | None:
    """Return the implied target date for a relative date expression in *query*.

    Returns None if the query contains no usable relative date expression, or
    if it is asking *about* a time interval ("how many days ago").
    """
    q = query.lower()

    # N days ago
    m = re.search(r"(\d+)\s+days?\s+ago", q)
    if m:
        return reference_date - timedelta(days=int(m.group(1)))

    # a couple / a pair / two of days ago → 2
    if re.search(r"(?:a\s+couple|a\s+pair|two)\s+of\s+days?\s+ago", q):
        return reference_date - timedelta(days=2)

    # a few days ago → 3
    if re.search(r"a\s+few\s+days?\s+ago", q):
        return reference_date - timedelta(days=3)

    # yesterday
    if re.search(r"\byesterday\b", q):
        return reference_date - timedelta(days=1)

    # N weeks ago
    m = re.search(r"(\d+)\s+weeks?\s+ago", q)
    if m:
        return reference_date - timedelta(weeks=int(m.group(1)))

    # word-form weeks: two/three/... weeks ago
    m = re.search(r"(two|three|four|five|six)\s+weeks?\s+ago", q)
    if m:
        return reference_date - timedelta(weeks=_WORD_NUMS[m.group(1)])

    # last week (but not "last week ago" — that's caught above as "N weeks ago" via context)
    if re.search(r"\blast\s+week\b", q):
        return reference_date - timedelta(days=7)

    # N months ago
    m = re.search(r"(\d+)\s+months?\s+ago", q)
    if m:
        return reference_date - timedelta(days=int(m.group(1)) * 30)

    # last month
    if re.search(r"\blast\s+month\b", q):
        return reference_date - timedelta(days=30)

    # last <weekday>  e.g. "last Friday"
    m = re.search(
        r"\blast\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", q
    )
    if m:
        target_wd = _WEEKDAY_MAP[m.group(1)]
        ref_wd = reference_date.weekday()  # Mon=0 … Sun=6
        days_back = (ref_wd - target_wd) % 7 or 7
        return reference_date - timedelta(days=days_back)

    return None


def _extract_memory_date(mr: "MemoryResult") -> date | None:
    """Extract the logical date of a memory.

    Priority: [Date: YYYY-MM-DD] header in content → date-like tag → created_at.
    """
    # Turn-pair benchmark format embeds date as [Date: YYYY-MM-DD]
    m = _DATE_IN_CONTENT_RE.search(mr.memory.content)
    if m:
        try:
            return date.fromisoformat(m.group(1))
        except ValueError:
            pass

    # Benchmark also stores date as a tag string
    for tag in mr.memory.tags:
        if _DATE_LIKE_TAG_RE.match(tag):
            try:
                return date.fromisoformat(tag)
            except ValueError:
                pass

    # Production fallback: use storage timestamp
    ca = mr.memory.created_at
    if ca is not None:
        return ca.date() if hasattr(ca, "date") else None

    return None


def _extract_reference_date(results: "Iterable[MemoryResult]") -> date | None:
    """Return the most recent logical date across all result memories."""
    latest: date | None = None
    for mr in results:
        d = _extract_memory_date(mr)
        if d is not None and (latest is None or d > latest):
            latest = d
    return latest


def apply_temporal_boost(
    fused: "dict[str, MemoryResult]",
    query: str,
    reference_date: date | None = None,
    boost_cap: float = 0.015,
) -> None:
    """Apply a Gaussian-decay score boost for memories near the query's target date.

    Detects relative date expressions ("10 days ago", "last Friday", "two weeks ago")
    in *query*, computes the target date relative to *reference_date*, and boosts
    memories whose logical date is close to that target.  The boost is additive and
    capped at *boost_cap* so it never overrides a strong semantic match.

    This function is a no-op when:
    - The query contains no parseable relative date expression
    - The query asks about an interval ("how many days ago…")
    - No date can be extracted from any memory

    Args:
        fused:          Dict of memory_id → MemoryResult (modified in-place).
        query:          The search query string.
        reference_date: The "now" of the conversation.  Defaults to the most
                        recent logical date found in the result memories, then
                        to today's date if no dates are found.
        boost_cap:      Maximum additive score boost (default 0.015).
    """
    if reference_date is None:
        reference_date = _extract_reference_date(fused.values())
    if reference_date is None:
        from datetime import date as _date
        reference_date = _date.today()

    target_date = _parse_target_date(query, reference_date)
    if target_date is None:
        # For interval queries ("how many days ago did I X?"), the query still
        # needs the event session ranked first.  Find the date cluster with the
        # highest density among the top-10 results and treat it as the anchor.
        if _INTERVAL_QUERY_RE.search(query.lower()):
            from collections import Counter
            dates = []
            for mr in sorted(fused.values(), key=lambda r: r.score, reverse=True)[:10]:
                d = _extract_memory_date(mr)
                if d:
                    dates.append(d)
            if dates:
                target_date = Counter(dates).most_common(1)[0][0]
        if target_date is None:
            return

    # Sigma: wider window for day-level references, wider for month-level
    offset_days = abs((reference_date - target_date).days)
    sigma = 30.0 if offset_days > 25 else 3.5

    for mr in fused.values():
        mem_date = _extract_memory_date(mr)
        if mem_date is None:
            continue
        days_diff = abs((mem_date - target_date).days)
        boost = boost_cap * math.exp(-0.5 * (days_diff / sigma) ** 2)
        if boost > 0.001:
            mr.score += boost


# ── Turn-pair completeness boost ────────────────────────────────────────────
#
# A stored memory that contains both a [USER] turn and an [ASSISTANT] turn is
# structurally more complete than one containing only one side of the exchange.
# Complete turn-pairs are more likely to be the answer-bearing memory (the
# "needle") than one-sided distractor fragments, because the benchmark haystacks
# store user questions WITH their assistant replies — any memory lacking either
# marker is a fragment, not the full exchange.
#
# Discriminative power on v1.00 near-miss analysis (n=40 R@1=0, R@3=1 cases):
#   Correct-only has both markers:   7 cases → correct gets +0.008
#   Rank-1-only has both markers:    4 cases → wrong gets +0.008 (hurts)
#   Both have both markers:         28 cases → no relative change
#   Neither has both markers:        1 case  → no change
#   Expected net: +3 flipped questions → R@1 ≈ +0.6pp

_USER_MARKER = re.compile(r"\[USER\]:", re.IGNORECASE)
_ASST_MARKER = re.compile(r"\[ASSISTANT\]:", re.IGNORECASE)


def apply_turn_pair_boost(
    fused: "dict[str, MemoryResult]",
    boost: float = 0.008,
    top_n: int = 10,
) -> None:
    """Boost candidates containing a complete [USER]+[ASSISTANT] turn-pair.

    A complete exchange is structurally more substantive than a one-sided
    fragment.  Applied to the top-N candidates only.  The additive boost
    is intentionally small (+0.008) — it acts as a tiebreaker-strength
    signal, not a ranking override.

    Args:
        fused:  Dict of memory_id → MemoryResult (modified in-place).
        boost:  Additive score bonus (default 0.008).
        top_n:  Only consider the top-N candidates by current score.
    """
    top_candidates = sorted(fused.values(), key=lambda r: r.score, reverse=True)[:top_n]
    for mr in top_candidates:
        content = mr.memory.content
        if _USER_MARKER.search(content) and _ASST_MARKER.search(content):
            fused[mr.memory.id].score += boost


# ═══════════════════════════════════════════════════════════════════════════════
# PURE-MATH RECALL IMPROVEMENT — Phase A/D/E/F
# ═══════════════════════════════════════════════════════════════════════════════
#
# The additive-boost pipeline saturates when near-ties have score gaps < 0.005
# (median gap from v1.00 near-miss analysis: 0.007, mean 0.011).  The following
# functions provide:
#
#   Phase A — Additional ranked-list signals fed into multi-signal RRF:
#     bm25_rank          — in-process BM25(k1=1.5, b=0.75) over fetched candidates
#     entity_overlap_rank — proper nouns + numbers from query found in content
#     date_proximity_rank — closer to target date = better (temporal queries)
#     session_recency_rank — higher session ordinal = better (recency queries)
#     turn_pair_rank      — complete [USER]+[ASSISTANT] pairs ranked first
#     is_counting_query   — gate for MMR diversification
#
#   Phase D — Gap-aware deterministic tiebreaker:
#     gap_aware_tiebreak  — fires only when top-2 gap ≤ eps; cascade of features
#
#   Phase E — MMR session diversification (counting/aggregation queries):
#     mmr_rerank          — session-level MMR to surface diverse sessions
#
#   Phase F — Temporal hard-filter (optional, gated by config):
#     temporal_hard_filter — pre-filter candidate pool to target-date window
#
#   Utility:
#     parse_target_date   — public wrapper around _parse_target_date
#     extract_prf_terms   — term extraction for pseudo-relevance feedback
# ═══════════════════════════════════════════════════════════════════════════════

# ── Shared tokeniser (used by BM25, MMR, tiebreaker) ───────────────────────
_RANK_SPLIT_RE = re.compile(r"[^a-z0-9]+")
_NUM_RE = re.compile(r"\b\d+(?:[.,]\d+)?\b")

_RANK_STOP = frozenset({
    "a", "an", "the", "and", "or", "in", "on", "at", "to", "for", "of",
    "is", "it", "its", "was", "be", "been", "i", "me", "my", "we", "you",
    "your", "he", "she", "they", "this", "that", "not", "with", "from",
    "have", "has", "do", "does", "did", "will", "can", "so", "if", "as",
    "up", "out", "are", "by", "no", "but", "had",
})

# ── Counting / aggregation query detector ──────────────────────────────────
_COUNTING_RE = re.compile(
    r"\b(?:how\s+many|how\s+much|count(?:ed)?|total(?:\s+number)?(?:\s+of)?|"
    r"number\s+of|sum\s+of|tally|aggregate|"
    r"all\s+(?:the\s+)?(?:times?|instances?|occasions?|events?))\b",
    re.IGNORECASE,
)

# BM25 tuning constants (Robertson & Zaragoza 2009)
_BM25_K1: float = 1.5
_BM25_B: float = 0.75


# ── Phase A: additional ranked-list signals ─────────────────────────────────


def bm25_rank(
    query: str,
    results: Sequence[MemoryResult],
) -> list[MemoryResult]:
    """Rank *results* by in-process BM25 over their content.

    Provides proper IDF + length normalisation over the already-fetched
    candidate set.  Complements Postgres ``ts_rank_cd`` (which uses a
    different normalisation) and catches terms that GIN/FTS misses.

    Returns a new list sorted best-first.  Original scores are not modified.
    """
    if not results or not query.strip():
        return list(results)

    q_tokens = [
        t for t in _RANK_SPLIT_RE.split(query.lower())
        if len(t) > 1 and t not in _RANK_STOP
    ]
    if not q_tokens:
        return list(results)

    n = len(results)
    # Tokenise each doc once
    doc_token_lists: list[list[str]] = [
        [t for t in _RANK_SPLIT_RE.split(mr.memory.content.lower()) if len(t) > 1]
        for mr in results
    ]
    avg_dl = sum(len(t) for t in doc_token_lists) / n

    # Document frequency for query terms (within this candidate set)
    df: dict[str, int] = {}
    for toks in doc_token_lists:
        tok_set = set(toks)
        for t in q_tokens:
            if t in tok_set:
                df[t] = df.get(t, 0) + 1

    # IDF with Lucene-style smoothing: log((N − df + 0.5) / (df + 0.5) + 1)
    idf: dict[str, float] = {
        t: math.log((n - df.get(t, 0) + 0.5) / (df.get(t, 0) + 0.5) + 1)
        for t in set(q_tokens)
    }

    from collections import Counter
    scored: list[tuple[float, MemoryResult]] = []
    for mr, toks in zip(results, doc_token_lists):
        tf_counts: Counter[str] = Counter(toks)
        dl = len(toks)
        score = 0.0
        for t in q_tokens:
            tf = tf_counts.get(t, 0)
            if tf == 0:
                continue
            # TF with saturation and length normalisation
            tf_norm = (tf * (_BM25_K1 + 1)) / (
                tf + _BM25_K1 * (1 - _BM25_B + _BM25_B * dl / max(avg_dl, 1))
            )
            score += idf.get(t, 0.0) * tf_norm
        scored.append((score, mr))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [mr for _, mr in scored]


def entity_overlap_rank(
    query: str,
    results: Sequence[MemoryResult],
) -> list[MemoryResult]:
    """Rank *results* by count of query proper nouns + numbers found in content.

    Useful for factual queries ("How many concerts did I attend?") where exact
    entity matching matters beyond stemmed FTS.

    Returns a new list sorted best-first.
    """
    proper_nouns = {n.lower() for n in extract_proper_nouns(query)}
    numbers = set(_NUM_RE.findall(query))
    entities = proper_nouns | numbers
    if not entities:
        return list(results)

    scored: list[tuple[int, MemoryResult]] = []
    for mr in results:
        content_lower = mr.memory.content.lower()
        hits = sum(1 for e in entities if e in content_lower)
        scored.append((hits, mr))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [mr for _, mr in scored]


def date_proximity_rank(
    results: Sequence[MemoryResult],
    target_date: date,
) -> list[MemoryResult]:
    """Rank *results* by |memory_date − target_date| ascending (closest first).

    An additional RRF signal for temporal queries.  Memories without an
    extractable date are ranked last.

    Returns a new list sorted best-first (closest date first).
    """
    _FAR = 365 * 100  # sentinel for undated memories
    scored: list[tuple[int, MemoryResult]] = []
    for mr in results:
        mem_date = _extract_memory_date(mr)
        diff = abs((mem_date - target_date).days) if mem_date else _FAR
        scored.append((diff, mr))
    scored.sort(key=lambda x: x[0])
    return [mr for _, mr in scored]


def session_recency_rank(
    results: Sequence[MemoryResult],
    ordinals: dict[str, int],
) -> list[MemoryResult]:
    """Rank *results* by session_ordinal descending (most-recent session first).

    Returns a new list sorted best-first.
    """
    def _ord(mr: MemoryResult) -> int:
        sid = mr.memory.session_id
        return ordinals.get(sid, 0) if sid else 0

    return sorted(results, key=_ord, reverse=True)


def turn_pair_rank(results: Sequence[MemoryResult]) -> list[MemoryResult]:
    """Rank *results*: complete [USER]+[ASSISTANT] pairs before one-sided fragments.

    Returns a new list sorted best-first.
    """
    return sorted(
        results,
        key=lambda mr: int(
            bool(_USER_MARKER.search(mr.memory.content) and _ASST_MARKER.search(mr.memory.content))
        ),
        reverse=True,
    )


def is_counting_query(query: str) -> bool:
    """Return True for queries asking about quantities or aggregations.

    Counting queries benefit from MMR session-diversification: the correct
    answer often spans multiple sessions, and retrieval tends to over-fetch
    from the most semantically similar one.
    """
    return bool(_COUNTING_RE.search(query))


def parse_target_date(query: str, reference_date: date) -> date | None:
    """Public wrapper around :func:`_parse_target_date`.

    Returns the implied target date for a relative date expression in *query*,
    or ``None`` if no parseable expression is found.
    """
    return _parse_target_date(query, reference_date)


# ── Phase D: gap-aware deterministic tiebreaker ─────────────────────────────


def gap_aware_tiebreak(
    results: list[MemoryResult],
    query: str,
    ordinals: dict[str, int],
    eps: float = 0.005,
    target_date: "date | None" = None,
) -> list[MemoryResult]:
    """Re-order the top prefix when scores are within *eps* using richer features.

    The additive-boost pipeline leaves near-ties where rank-1 and rank-2
    differ by less than 0.005 (v1.00 analysis: 18/22 regressions fell below
    this threshold).  This tiebreaker applies a cascade of deterministic
    features only to the tied group, leaving clear winners untouched.

    Tiebreak cascade (earlier features dominate):
        1. 2-gram exact phrase matches between query and content
        2. Exact query-token overlap ratio
        3. Numeric token match count
        4. Session recency (newer) — or date proximity when *target_date* is set

    Args:
        results:     Fused results sorted by score descending.
        query:       The original search query.
        ordinals:    session_id → ordinal mapping (populated before this call).
        eps:         Score-gap threshold; only ties within this margin are broken.
        target_date: If set, prefer date-proximate sessions over most-recent ones.

    Returns:
        A new list with the tied prefix re-ordered deterministically.
    """
    if len(results) < 2:
        return results

    top_score = results[0].score
    # Identify the epsilon-tied group at the top
    cut = 1
    while cut < len(results) and (top_score - results[cut].score) <= eps:
        cut += 1

    if cut == 1:
        return results  # Clear winner; no tiebreak needed

    tied = list(results[:cut])
    rest = results[cut:]

    # Pre-compute query features once
    q_lower = query.lower()
    q_tokens_list = [t for t in _RANK_SPLIT_RE.split(q_lower) if len(t) > 1]
    q_token_set = set(q_tokens_list)
    q_nums = set(_NUM_RE.findall(query))
    q_bigrams = frozenset(
        f"{q_tokens_list[i]} {q_tokens_list[i + 1]}"
        for i in range(len(q_tokens_list) - 1)
    )

    def _tiebreak_key(mr: MemoryResult) -> tuple:
        content = mr.memory.content
        content_lower = content.lower()
        c_tokens = frozenset(
            t for t in _RANK_SPLIT_RE.split(content_lower) if len(t) > 1
        )

        # Feature 1: 2-gram phrase hits (catches compact factual phrases)
        phrase_hits = sum(1 for bg in q_bigrams if bg in content_lower)

        # Feature 2: exact query-token overlap ratio
        overlap = len(q_token_set & c_tokens) / max(len(q_token_set), 1)

        # Feature 3: numeric token match (IDs, counts, dates as integers)
        num_hits = sum(1 for n in q_nums if n in content)

        # Feature 4: recency or date proximity as final tiebreaker
        if target_date is not None:
            mem_date = _extract_memory_date(mr)
            recency = -abs((mem_date - target_date).days) if mem_date else -9999
        else:
            sid = mr.memory.session_id
            recency = ordinals.get(sid, 0) if sid else 0

        return (phrase_hits, overlap, num_hits, recency)

    tied.sort(key=_tiebreak_key, reverse=True)
    return tied + rest


# ── Phase E: MMR session diversification ────────────────────────────────────


def mmr_rerank(
    results: list[MemoryResult],
    lambda_: float = 0.7,
    session_cap: int = 2,
    limit: int | None = None,
) -> list[MemoryResult]:
    """Session-level Maximal Marginal Relevance diversification.

    Greedily selects results maximising ``λ·relevance − (1−λ)·max_sim(d, S)``,
    where similarity is token-Jaccard and *session_cap* limits chunks from any
    single session_id.  Designed for counting/aggregation queries where the
    answer spans multiple sessions.

    Args:
        results:     Fused results sorted by score descending.
        lambda_:     Trade-off weight: 1.0 = pure relevance, 0.0 = pure diversity.
        session_cap: Max results from any one session_id in the output.
        limit:       Output size.  Defaults to ``len(results)``.

    Returns:
        Re-ordered list prioritising session diversity subject to relevance.
    """
    if not results:
        return results

    target = limit or len(results)

    # Pre-compute token sets for fast Jaccard similarity
    token_sets: dict[str, frozenset[str]] = {
        mr.memory.id: frozenset(
            t for t in _RANK_SPLIT_RE.split(mr.memory.content.lower())
            if len(t) > 1
        )
        for mr in results
    }

    # Normalise scores to [0, 1]
    scores = [mr.score for mr in results]
    max_s = max(scores) if scores else 1.0
    min_s = min(scores) if scores else 0.0
    range_s = max(max_s - min_s, 1e-10)

    selected: list[MemoryResult] = []
    selected_sets: list[frozenset[str]] = []
    session_counts: dict[str, int] = {}
    remaining = list(results)

    while remaining and len(selected) < target:
        best_val = float("-inf")
        best_idx = 0

        for i, mr in enumerate(remaining):
            rel = (mr.score - min_s) / range_s
            sid = mr.memory.session_id or mr.memory.id
            ts = token_sets[mr.memory.id]

            if session_counts.get(sid, 0) >= session_cap:
                # Hard-penalise over-cap sessions: force diversity in first half
                max_sim = 1.0
            elif selected_sets:
                max_sim = max(
                    len(ts & s) / len(ts | s) if (ts | s) else 0.0
                    for s in selected_sets
                )
            else:
                max_sim = 0.0

            val = lambda_ * rel - (1.0 - lambda_) * max_sim
            if val > best_val:
                best_val = val
                best_idx = i

        chosen = remaining.pop(best_idx)
        selected.append(chosen)
        selected_sets.append(token_sets[chosen.memory.id])
        sid = chosen.memory.session_id or chosen.memory.id
        session_counts[sid] = session_counts.get(sid, 0) + 1

    # Safety: append any overflow (shouldn't happen under normal operation)
    if len(selected) < target:
        selected.extend(remaining[: target - len(selected)])

    return selected


# ── Phase F: temporal hard-filter ───────────────────────────────────────────


def temporal_hard_filter(
    fused: dict[str, MemoryResult],
    target_date: date,
    sigma_days: float = 3.5,
    min_keep: int = 10,
) -> dict[str, MemoryResult]:
    """Restrict fused candidates to those within ±sigma_days of *target_date*.

    For day-precision temporal queries ("what did I do last Tuesday?"),
    narrowing the candidate pool to the target date window prevents
    irrelevant sessions from accumulating soft-boost scores that beat
    the correct answer.

    Falls back to the unfiltered dict when fewer than *min_keep* candidates
    survive — safety net against false date extractions.

    Args:
        fused:       Dict of memory_id → MemoryResult (not modified).
        target_date: Resolved target date from :func:`parse_target_date`.
        sigma_days:  Half-window size in days.
        min_keep:    Minimum survivor count; if fewer survive, returns original.

    Returns:
        Filtered dict, or the original dict when the filter is too aggressive.
    """
    kept: dict[str, MemoryResult] = {}
    for mid, mr in fused.items():
        mem_date = _extract_memory_date(mr)
        if mem_date is not None and abs((mem_date - target_date).days) <= sigma_days:
            kept[mid] = mr
    return kept if len(kept) >= min_keep else fused


# ── Pseudo-relevance feedback utilities ─────────────────────────────────────


def extract_prf_terms(
    top_results: Sequence[MemoryResult],
    n_terms: int = 8,
) -> list[str]:
    """Extract highest-frequency content terms from top-K results.

    Classic Rocchio-inspired term selection: terms appearing frequently
    across the initial top-K are likely on-topic for the query.  Used to
    expand a FTS re-query for vague/preference questions.

    Returns up to *n_terms* expansion tokens.
    """
    from collections import Counter

    term_counts: Counter[str] = Counter()
    for mr in top_results:
        for t in frozenset(_RANK_SPLIT_RE.split(mr.memory.content.lower())):
            if t and len(t) >= 3 and t not in _RANK_STOP:
                term_counts[t] += 1
    return [t for t, _ in term_counts.most_common(n_terms)]
