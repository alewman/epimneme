"""Memory decay — power-law retrievability scoring.

Simplified FSRS-inspired model:
- storage_strength grows with each access (never decays)
- retrieval_strength decays via power law since last access
- retrievability = blend of both, used to boost/penalize search results

On access:
    retrieval_strength resets to 1.0
    storage_strength grows with diminishing returns

Over time:
    retrievability = e^(-t / S)
    where t = days since last access
    and S = base_stability * (1 + storage_strength)
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

    # Power-law decay: R = e^(-t/S)
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
