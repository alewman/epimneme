"""Context assembly — deterministic post-retrieval formatting for the reader.

Retrieval on LongMemEval-S is nearly saturated (R@10 ~98%), but the end-to-end
reader (an LLM synthesizing an answer from retrieved chunks) loses ~38pp
relative to that ceiling. Diagnosed root causes are all *presentation*
problems, not retrieval problems: the reader is asked to do date arithmetic
it can't do, stale and current facts are shown with no supersession cue, long
chunks get truncated before the reader sees the answer-bearing sentence, and
counting questions are starved by a fixed top-K.

This module fixes the presentation, not the retrieval. It is $0/query,
fully deterministic, and operates on plain `(text, score, metadata)` triples
so both the live server and the offline benchmark harness can call it without
a round trip through the database.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Callable, Collection, Sequence

from epimneme.dedup import compute_simhash, entities_diverge, is_near_duplicate
from epimneme.fusion import extract_logical_date, is_counting_query, parse_target_date

# ── Config defaults (mirrors EngramConfig.assembly_* — see core/config.py) ───

DEFAULT_BUDGET_CHARS = 12_000
DEFAULT_K_SINGLE = 5
DEFAULT_K_DEFAULT = 10
DEFAULT_K_COUNTING = 20

# Presentation steps that `assemble_context(skip=...)` can leave out.
# "dates" is the whole temporal annotation; it splits into "deltas" (the per-header
# "— N days before the question" suffix) and "anchor" (the one-line preamble
# stating the absolute date a relative expression in the query resolves to).
ASSEMBLY_STEPS = frozenset({"prune", "group", "chrono", "dates", "deltas", "anchor"})

# Default pipeline exclusions (measured, LongMemEval-S, qwen3.8:27b reader, judged,
# paired per question; see benchmarks/BENCHMARK_RESULTS.md "Phase 3"):
#   chrono — chronological re-ordering after session grouping never paid: neutral
#            on multi-session (0.526 → 0.519 without it) and −5 questions on
#            temporal-reasoning (0.609 → 0.647 without it, McNemar p=0.011 vs raw).
#            Relevance order (what grouping already preserves) is what the reader
#            wants. Still available: pass default_skip=().
DEFAULT_SKIP = frozenset({"chrono"})

RECENCY_NOTE = (
    "Note: if the excerpts give different values for the same fact, "
    "the value from the excerpt with the latest date is the current one."
)

# The per-header "— N days before the question" suffix ("deltas") is date
# arithmetic precomputed for the reader. Measured per question with the suffix on
# vs off: temporal-reasoning 81 → 75 hits (10 hurt, 4 helped), multi-session
# 70 → 85 (3 hurt, 18 helped). The questions it helps ask for elapsed time or
# ordering ("how many days passed between…", "how many weeks ago…", "which
# happened first…"); on everything else it is noise. So the suffix is emitted only
# when the query shows a date-arithmetic cue or contains a parseable relative
# date. Gate it differently with `date_delta_gate=`, or always emit with `None`.
_DATE_ARITHMETIC_CUE_RE = re.compile(
    r"\b(days?|weeks?|months?|years?|hours?)\b.*\b(passed|between|elapsed|ago|since|until|before|after)\b"
    r"|\b(ago|since|passed)\b"
    r"|\bthe order of\b"
    r"|\bwhich .* (first|last|earlier|later|most recently)\b"
    r"|\b(first|last|earliest|latest|most recent)\b",
    re.IGNORECASE,
)


_DATE_HEADER_RE = re.compile(r"^\[Date:\s*([^\]]*)\]\n?")
_SUPERSEDED_TAG_RE = re.compile(r"^\[SUPERSEDED[^\]]*\]\n?")


def needs_date_arithmetic(query: str, reference_date: date | None = None) -> bool:
    """True when the reader will have to compute elapsed time or order events."""
    if _DATE_ARITHMETIC_CUE_RE.search(query):
        return True
    if reference_date is not None and parse_target_date(query, reference_date) is not None:
        return True
    return False


@dataclass
class Excerpt:
    """One retrieved unit of context, before or after assembly transforms.

    `metadata` carries whatever the caller has available — recognized keys:
      - `memory_id`: str, stable identifier (used for supersession links)
      - `session_id`: str, groups excerpts from the same conversation/session
      - `turn_index`: int, order within a session (falls back to input order)
      - `tags`: list[str], may include a date-like tag
      - `created_at`: datetime-like, fallback date source
      - `supersedes`: memory_id this excerpt explicitly replaces
      - `version_of`: original memory_id this is a new version of
    All keys are optional; every function degrades gracefully when absent.
    """

    text: str
    score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AssembledContext:
    """Final result of `assemble_context`."""

    text: str
    excerpt_count: int
    char_count: int
    truncated: bool


# ── 1. Date extraction ───────────────────────────────────────────────────────


def excerpt_date(excerpt: Excerpt) -> date | None:
    """Return the logical date of an excerpt: header → tag → created_at."""
    md = excerpt.metadata
    return extract_logical_date(
        excerpt.text,
        md.get("tags") or (),
        md.get("created_at"),
    )


def _reference_date(excerpts: Sequence[Excerpt]) -> date | None:
    """Most recent logical date across excerpts — the conversation's "now"."""
    latest: date | None = None
    for ex in excerpts:
        d = excerpt_date(ex)
        if d is not None and (latest is None or d > latest):
            latest = d
    return latest


# ── 2. Temporal scaffolding ──────────────────────────────────────────────────


def _rewrite_date_header(text: str, suffix: str) -> str:
    """Append `suffix` inside the leading `[Date: …]` header, if present."""
    m = _DATE_HEADER_RE.match(text)
    if not m:
        return text
    header_body = m.group(1)
    new_header = f"[Date: {header_body} — {suffix}]\n"
    return new_header + text[m.end():]


def _slash_date(d: date) -> str:
    """Format a date to match the source convention (`YYYY/MM/DD`)."""
    return f"{d.year:04d}/{d.month:02d}/{d.day:02d}"


def _delta_phrase(reference_date: date, excerpt_dt: date) -> str:
    days = (reference_date - excerpt_dt).days
    if days == 0:
        return "on the day of the question"
    if days > 0:
        return f"{days} day{'s' if days != 1 else ''} before the question"
    return f"{-days} day{'s' if -days != 1 else ''} after the question"


def annotate_temporal(
    excerpts: Sequence[Excerpt],
    query: str,
    reference_date: date | None = None,
    *,
    deltas: bool = True,
    anchor: bool = True,
) -> tuple[list[Excerpt], str | None]:
    """Precompute date arithmetic so the reader never has to do it.

    All date arithmetic happens here, in code — never delegated to the reader.
    Returns (annotated_excerpts, preamble) where `preamble` is a one-line
    statement of the resolved target date (or None if the query has no
    parseable relative-date expression).
    """
    if reference_date is None:
        reference_date = _reference_date(excerpts)

    preamble: str | None = None
    if anchor and reference_date is not None:
        target_date = parse_target_date(query, reference_date)
        if target_date is not None:
            preamble = f"The question refers to approximately {_slash_date(target_date)}."

    if reference_date is None or not deltas:
        return list(excerpts), preamble

    annotated = []
    for ex in excerpts:
        d = excerpt_date(ex)
        if d is None:
            annotated.append(ex)
            continue
        new_text = _rewrite_date_header(ex.text, _delta_phrase(reference_date, d))
        annotated.append(Excerpt(text=new_text, score=ex.score, metadata=ex.metadata))
    return annotated, preamble


# ── 3. Chronological presentation ────────────────────────────────────────────


def chronological_order(excerpts: Sequence[Excerpt]) -> list[Excerpt]:
    """Reorder for presentation by logical date; undated excerpts sort last.

    Selection (which excerpts made the cut) stays rank-based — this only
    changes the order they're *shown* in, with a stable tiebreak on the
    original rank so equal/missing dates don't reshuffle arbitrarily.
    """
    indexed = list(enumerate(excerpts))

    def key(item: tuple[int, Excerpt]) -> tuple[int, date, int]:
        idx, ex = item
        d = excerpt_date(ex)
        return (0, d, idx) if d is not None else (1, date.max, idx)

    indexed.sort(key=key)
    return [ex for _, ex in indexed]


# ── 4. Supersession pruning ──────────────────────────────────────────────────


def prune_superseded(excerpts: Sequence[Excerpt]) -> list[Excerpt]:
    """Drop explicitly-superseded excerpts; flag likely-stale duplicates.

    Two passes:
      1. Explicit links (`supersedes` / `version_of` in metadata) are a
         confirmed same-fact replacement — the superseded excerpt is dropped
         outright. This directly removes the knowledge-update failure mode
         (reader picking the stale value) and saves tokens.
      2. Near-duplicate content with different dates but no explicit link is
         a heuristic guess, so it's ANNOTATED (`[SUPERSEDED …]`) rather than
         dropped — the reader keeps the information but is told to prefer
         the newer excerpt.
    """
    by_id: dict[str, Excerpt] = {}
    for ex in excerpts:
        mid = ex.metadata.get("memory_id")
        if mid:
            by_id[mid] = ex

    dropped_ids: set[str] = set()
    for ex in excerpts:
        replaces = ex.metadata.get("supersedes") or ex.metadata.get("version_of")
        if replaces and replaces in by_id:
            dropped_ids.add(replaces)

    survivors = [
        ex for ex in excerpts
        if not (ex.metadata.get("memory_id") in dropped_ids)
    ]

    # Pass 2: implicit near-duplicates among survivors with resolvable dates.
    # Hash the body only — the date header necessarily differs between an
    # older and newer excerpt of the same fact, which would otherwise push
    # the Hamming distance past the near-duplicate threshold.
    dated = [(ex, excerpt_date(ex)) for ex in survivors]
    simhashes = [compute_simhash(_DATE_HEADER_RE.sub("", ex.text, count=1)) for ex, _ in dated]
    flagged_older: set[int] = set()
    n = len(dated)
    for i in range(n):
        ex_i, date_i = dated[i]
        if date_i is None or i in flagged_older:
            continue
        for j in range(i + 1, n):
            ex_j, date_j = dated[j]
            if date_j is None or date_i == date_j:
                continue
            if not is_near_duplicate(simhashes[i], simhashes[j]):
                continue
            if entities_diverge(ex_i.text, ex_j.text):
                continue
            older_idx, older_date = (i, date_i) if date_i < date_j else (j, date_j)
            flagged_older.add(older_idx)

    result = []
    for idx, (ex, d) in enumerate(dated):
        if idx in flagged_older and not _SUPERSEDED_TAG_RE.match(ex.text.lstrip()):
            tag = f"[SUPERSEDED {d.isoformat()}]\n" if d else "[SUPERSEDED]\n"
            result.append(Excerpt(text=tag + ex.text, score=ex.score, metadata=ex.metadata))
        else:
            result.append(ex)
    return result


# ── 5. Adaptive K / token budgeting ──────────────────────────────────────────


def select_k(
    excerpts: Sequence[Excerpt],
    query: str,
    *,
    k_single: int = DEFAULT_K_SINGLE,
    k_default: int = DEFAULT_K_DEFAULT,
    k_counting: int = DEFAULT_K_COUNTING,
) -> list[Excerpt]:
    """Cut the rank-ordered input to an adaptive K based on query shape.

    Counting/aggregation queries need to see every contributing session, so
    they get a wider K; simple single-fact lookups need very little context.
    Selection stays rank-based — this assumes `excerpts` is already sorted by
    relevance (best first).
    """
    if is_counting_query(query):
        k = k_counting
    elif len(query.split()) <= 6:
        k = k_single
    else:
        k = k_default
    return list(excerpts[:k])


def budget_by_chars(
    excerpts: Sequence[Excerpt],
    budget_chars: int = DEFAULT_BUDGET_CHARS,
) -> tuple[list[Excerpt], bool]:
    """Greedily keep excerpts (in given order) until the char budget is hit.

    Always keeps at least the first excerpt, even if it alone exceeds budget
    (an empty context is worse than one over-budget excerpt). Returns
    (kept, truncated) where `truncated` is True iff any excerpt was dropped.
    """
    kept: list[Excerpt] = []
    total = 0
    for i, ex in enumerate(excerpts):
        cost = len(ex.text) + 4  # separator overhead
        if kept and total + cost > budget_chars:
            return kept, True
        kept.append(ex)
        total += cost
    return kept, False


# ── 6. Session grouping ──────────────────────────────────────────────────────


def group_by_session(excerpts: Sequence[Excerpt]) -> list[Excerpt]:
    """Merge same-session excerpts under one date header, ordered by turn.

    Excerpts without a `session_id` pass through untouched. This both saves
    tokens (repeated `[Date: …]` headers collapse to one) and gives the
    reader the full session in reading order instead of scattered fragments.
    """
    groups: dict[str, list[tuple[int, Excerpt]]] = {}
    order: list[str] = []
    passthrough: list[tuple[int, Excerpt]] = []

    for idx, ex in enumerate(excerpts):
        sid = ex.metadata.get("session_id")
        if not sid:
            passthrough.append((idx, ex))
            continue
        if sid not in groups:
            groups[sid] = []
            order.append(sid)
        groups[sid].append((idx, ex))

    result: list[tuple[int, Excerpt]] = list(passthrough)
    for sid in order:
        members = groups[sid]
        if len(members) == 1:
            result.append(members[0])
            continue
        members.sort(key=lambda item: item[1].metadata.get("turn_index", item[0]))
        first_idx = members[0][0]
        header_date = None
        bodies = []
        best_score = 0.0
        for _, ex in members:
            m = _DATE_HEADER_RE.match(ex.text)
            if m and header_date is None:
                header_date = m.group(1)
            body = _DATE_HEADER_RE.sub("", ex.text, count=1)
            bodies.append(body.rstrip("\n"))
            best_score = max(best_score, ex.score)
        header = f"[Date: {header_date}]\n" if header_date else ""
        merged_text = header + "\n".join(bodies)
        merged = Excerpt(
            text=merged_text,
            score=best_score,
            metadata={**members[0][1].metadata, "_merged_session": sid},
        )
        result.append((first_idx, merged))

    result.sort(key=lambda item: item[0])
    return [ex for _, ex in result]


# ── Parent-document (small-to-big) expansion ─────────────────────────────────

# Expansion is only worthwhile when the query is narrowly focused on one or a
# few sessions — a broad multi-session query would blow the char budget for
# little benefit, and counting/aggregation queries specifically want breadth
# (many sessions), not depth (more of the same session).
_MAX_SESSIONS_FOR_EXPANSION = 3


def expand_parents(
    excerpts: Sequence[Excerpt],
    fetch_neighbors: Callable[[Excerpt], Sequence[Excerpt]] | None,
    query: str,
) -> list[Excerpt]:
    """Fetch sibling chunks adjacent to each hit and splice them in.

    `fetch_neighbors` is injected so this stays pure and testable without a
    database: the server implementation queries the store for same-session
    neighbors by `created_at`; the bench harness looks up `corpus_id`
    `..._turn_{n±1}` in the staged corpus. A `None` fetcher (or a query that
    doesn't qualify) is a no-op.

    Newly fetched neighbors are tagged with the same `session_id` they came
    from, so the later `group_by_session` step merges them with the original
    hit automatically — this function only decides *which* neighbors to add.
    """
    if fetch_neighbors is None or is_counting_query(query):
        return list(excerpts)

    session_ids = {sid for ex in excerpts if (sid := ex.metadata.get("session_id"))}
    if not session_ids or len(session_ids) > _MAX_SESSIONS_FOR_EXPANSION:
        return list(excerpts)

    existing_ids = {mid for ex in excerpts if (mid := ex.metadata.get("memory_id"))}
    expanded = list(excerpts)
    for ex in excerpts:
        if not ex.metadata.get("session_id"):
            continue
        for neighbor in fetch_neighbors(ex):
            nid = neighbor.metadata.get("memory_id")
            if nid and nid in existing_ids:
                continue
            if nid:
                existing_ids.add(nid)
            expanded.append(neighbor)
    return expanded


# ── Composition ───────────────────────────────────────────────────────────────


def assemble_context(
    excerpts: Sequence[Excerpt],
    query: str,
    *,
    reference_date: date | None = None,
    budget_chars: int = DEFAULT_BUDGET_CHARS,
    k_single: int = DEFAULT_K_SINGLE,
    k_default: int = DEFAULT_K_DEFAULT,
    k_counting: int = DEFAULT_K_COUNTING,
    fetch_neighbors: Callable[[Excerpt], Sequence[Excerpt]] | None = None,
    enable_parent_expansion: bool = True,
    skip: Collection[str] = (),
    default_skip: Collection[str] = DEFAULT_SKIP,
    date_delta_gate: Callable[[str, date | None], bool] | None = needs_date_arithmetic,
    recency_note: bool = False,
) -> AssembledContext:
    """Run the full assembly pipeline: select → prune → budget → present.

    `excerpts` must already be relevance-ranked (best first) — assembly only
    reorders for *presentation*, never for relevance. Parent-document
    expansion (`fetch_neighbors`) is opt-in via a caller-supplied fetcher —
    without one, `enable_parent_expansion` has no effect.

    `skip` names presentation steps to leave out (for ablation): any of
    ``"prune"`` (supersession pruning), ``"group"`` (session grouping),
    ``"chrono"`` (chronological order), ``"dates"`` (all temporal annotation),
    or its parts ``"deltas"`` (per-header day-delta suffix) and ``"anchor"``
    (resolved-target-date preamble).
    Unknown names raise ``ValueError``. `default_skip` (default: `DEFAULT_SKIP`)
    is merged in — pass ``()`` for the full pipeline. `date_delta_gate` decides per
    query whether the per-header day-delta suffix is emitted (default:
    `needs_date_arithmetic`); ``None`` always emits it. `recency_note` prepends
    one line telling the reader that where excerpts disagree the latest-dated
    value is current (experimental: the knowledge-update failure mode is the
    reader answering from the top-ranked, older excerpt when a newer one is
    present; supersession pruning cannot catch it without explicit links).
    """
    skip = set(skip) | set(default_skip)
    if date_delta_gate is not None and not date_delta_gate(query, reference_date):
        skip.add("deltas")
    unknown = skip - ASSEMBLY_STEPS
    if unknown:
        raise ValueError(f"unknown assembly step(s) to skip: {sorted(unknown)}")
    selected = select_k(excerpts, query, k_single=k_single, k_default=k_default, k_counting=k_counting)
    pruned = prune_superseded(selected) if "prune" not in skip else list(selected)
    budgeted, truncated = budget_by_chars(pruned, budget_chars)

    if enable_parent_expansion and fetch_neighbors is not None:
        expanded = expand_parents(budgeted, fetch_neighbors, query)
        budgeted, truncated_2 = budget_by_chars(expanded, budget_chars)
        truncated = truncated or truncated_2

    grouped = group_by_session(budgeted) if "group" not in skip else list(budgeted)
    ordered = chronological_order(grouped) if "chrono" not in skip else list(grouped)
    if "dates" not in skip:
        annotated, preamble = annotate_temporal(
            ordered, query, reference_date,
            deltas="deltas" not in skip, anchor="anchor" not in skip,
        )
    else:
        annotated, preamble = list(ordered), None

    parts = [preamble] if preamble else []
    if recency_note and sum(1 for ex in annotated if excerpt_date(ex) is not None) >= 2:
        parts.insert(0, RECENCY_NOTE)
    parts.extend(ex.text for ex in annotated)
    text = "\n---\n".join(parts)

    if len(text) > budget_chars:
        text = text[:budget_chars]
        truncated = True

    return AssembledContext(
        text=text,
        excerpt_count=len(annotated),
        char_count=len(text),
        truncated=truncated,
    )
