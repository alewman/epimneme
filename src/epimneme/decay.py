"""Memory decay — EXPONENTIAL retrievability scoring.

FSRS-inspired in shape, but the curve is an exponential, not the power law FSRS
uses. This file previously described itself as power-law while implementing
`e^(-t/S)`; the name was wrong, not the code.

- storage_strength grows with each access (never decays)
- retrieval_strength decays exponentially since last access
- retrievability = blend of both, used to boost/penalise search results

On access:
    retrieval_strength resets to 1.0
    storage_strength grows with diminishing returns

Over time:
    retrievability = e^(-t / S)
    where t = days since last access
    and S = base_stability * (1 + storage_strength)

WHY THE CURVE MATTERS OPERATIONALLY
-----------------------------------
`reflection._phase_gc` soft-deletes any non-pinned, non-exempt memory whose
retrievability falls below `gc_retrievability_threshold` (0.05), so this
function is the retention policy, not just a ranking signal.

    exponential (this file), S = base_stability * 2   -> R<0.05 at  ~6x base_stability days
    FSRS power law, R = (1 + t/(9S))^-1               -> R<0.05 at ~342x base_stability days

At the old default `EPIMNEME_DECAY_STABILITY=30` that meant anything not
*recalled* within ~180 days was obsoleted, which had marked 63% of one
production store obsolete by 2026-09-26. The deployed fix was to raise
`EPIMNEME_DECAY_STABILITY` to 180 (~3-year horizon) rather than change the
curve, because a power law here is effectively "never GC" and that is a
different policy decision. Adjust the horizon with that env var; switching to a
true power law is a deliberate change, not a bug fix.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone


def calculate_retrievability(
    storage_strength: float,
    last_accessed: datetime | None,
    base_stability: float = 1.0,
    now: datetime | None = None,
) -> float:
    """Calculate current retrievability (0.0 to 1.0).

    Args:
        storage_strength: Accumulated strength from repeated access (0+)
        last_accessed: When the memory was last retrieved
        base_stability: Base half-life in days (configurable)
        now: Current time (defaults to UTC now)
    """
    if last_accessed is None:
        return 1.0  # Freshly created

    now = now or datetime.now(timezone.utc)
    elapsed_days = max(0.0, (now - last_accessed).total_seconds() / 86400)

    if elapsed_days < 0.001:
        return 1.0

    # Stability grows with storage strength
    stability = base_stability * (1.0 + storage_strength)

    # Exponential decay (NOT a power law — see the module docstring; this
    # also sets the GC horizon via reflection._phase_gc).
    retrievability = math.exp(-elapsed_days / stability)

    return max(0.0, min(1.0, retrievability))


def update_on_access(
    storage_strength: float,
    access_count: int,
    growth_factor: float = 0.5,
) -> tuple[float, float, int]:
    """Update memory strengths after a successful recall.

    Returns:
        (new_storage_strength, new_retrieval_strength, new_access_count)
    """
    new_access_count = access_count + 1
    # Diminishing returns — early accesses matter more
    new_storage_strength = storage_strength + growth_factor / (1 + storage_strength * 0.5)
    new_retrieval_strength = 1.0  # Reset on access

    return new_storage_strength, new_retrieval_strength, new_access_count


def decay_score_boost(retrievability: float) -> float:
    """Convert retrievability to a search score multiplier.

    Fresh/frequently-accessed memories get a small boost.
    Old/stale memories get penalized but never zeroed out.

    Returns multiplier in range [0.3, 1.2]
    """
    return 0.3 + 0.9 * retrievability
