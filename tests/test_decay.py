"""Tests for engram.decay — pure unit tests for power-law retrievability scoring."""

import math
from datetime import datetime, timedelta, timezone

import pytest

from epimneme.decay import calculate_retrievability, decay_score_boost, update_on_access


# ── calculate_retrievability ─────────────────────────────────────────────────


class TestCalculateRetrievability:
    def test_freshly_created_no_last_accessed(self):
        """A memory with no last_accessed should return 1.0."""
        r = calculate_retrievability(storage_strength=0.0, last_accessed=None)
        assert r == 1.0

    def test_just_accessed(self):
        """A memory accessed right now should return ~1.0."""
        now = datetime.now(timezone.utc)
        r = calculate_retrievability(
            storage_strength=0.0, last_accessed=now, now=now
        )
        assert r == 1.0

    def test_very_recent_access(self):
        """Access within a second should still be ~1.0."""
        now = datetime.now(timezone.utc)
        r = calculate_retrievability(
            storage_strength=0.0,
            last_accessed=now - timedelta(seconds=0.05),
            now=now,
        )
        assert r == 1.0  # elapsed < 0.001 days threshold

    def test_decays_over_time(self):
        """Retrievability should decrease as time passes."""
        now = datetime.now(timezone.utc)
        r_1day = calculate_retrievability(
            storage_strength=0.0,
            last_accessed=now - timedelta(days=1),
            now=now,
        )
        r_7days = calculate_retrievability(
            storage_strength=0.0,
            last_accessed=now - timedelta(days=7),
            now=now,
        )
        assert 0.0 < r_7days < r_1day < 1.0

    def test_storage_strength_slows_decay(self):
        """Higher storage_strength should slow decay (higher retrievability)."""
        now = datetime.now(timezone.utc)
        last = now - timedelta(days=5)

        r_low = calculate_retrievability(
            storage_strength=0.0, last_accessed=last, now=now
        )
        r_high = calculate_retrievability(
            storage_strength=5.0, last_accessed=last, now=now
        )
        assert r_high > r_low

    def test_base_stability_parameter(self):
        """Higher base_stability should slow decay."""
        now = datetime.now(timezone.utc)
        last = now - timedelta(days=3)

        r_low_stab = calculate_retrievability(
            storage_strength=0.0, last_accessed=last, base_stability=0.5, now=now
        )
        r_high_stab = calculate_retrievability(
            storage_strength=0.0, last_accessed=last, base_stability=5.0, now=now
        )
        assert r_high_stab > r_low_stab

    def test_never_goes_below_zero(self):
        """Even after years, retrievability should be >= 0.0."""
        now = datetime.now(timezone.utc)
        r = calculate_retrievability(
            storage_strength=0.0,
            last_accessed=now - timedelta(days=365 * 10),
            now=now,
        )
        assert r >= 0.0

    def test_never_exceeds_one(self):
        """Retrievability should never exceed 1.0."""
        now = datetime.now(timezone.utc)
        r = calculate_retrievability(
            storage_strength=100.0,
            last_accessed=now - timedelta(seconds=1),
            now=now,
        )
        assert r <= 1.0

    def test_known_value_one_day_base_stability_one(self):
        """With storage_strength=0 and base_stability=1, after 1 day: e^(-1) ≈ 0.368."""
        now = datetime.now(timezone.utc)
        r = calculate_retrievability(
            storage_strength=0.0,
            last_accessed=now - timedelta(days=1),
            base_stability=1.0,
            now=now,
        )
        assert abs(r - math.exp(-1.0)) < 0.001


# ── update_on_access ─────────────────────────────────────────────────────────


class TestUpdateOnAccess:
    def test_returns_three_values(self):
        result = update_on_access(storage_strength=0.0, access_count=0)
        assert len(result) == 3

    def test_access_count_increments(self):
        _, _, new_count = update_on_access(storage_strength=0.0, access_count=5)
        assert new_count == 6

    def test_retrieval_strength_resets_to_one(self):
        _, new_retrieval, _ = update_on_access(storage_strength=2.0, access_count=3)
        assert new_retrieval == 1.0

    def test_storage_strength_increases(self):
        new_storage, _, _ = update_on_access(storage_strength=0.0, access_count=0)
        assert new_storage > 0.0

    def test_diminishing_returns(self):
        """Each subsequent access should add less to storage_strength."""
        s1, _, _ = update_on_access(storage_strength=0.0, access_count=0)
        s2, _, _ = update_on_access(storage_strength=s1, access_count=1)
        delta_1 = s1 - 0.0
        delta_2 = s2 - s1
        assert delta_2 < delta_1

    def test_custom_growth_factor(self):
        """Higher growth_factor should increase storage_strength faster."""
        s_low, _, _ = update_on_access(
            storage_strength=0.0, access_count=0, growth_factor=0.1
        )
        s_high, _, _ = update_on_access(
            storage_strength=0.0, access_count=0, growth_factor=2.0
        )
        assert s_high > s_low

    def test_many_accesses(self):
        """Run many accesses and confirm storage keeps growing."""
        s = 0.0
        count = 0
        for _ in range(100):
            s, _, count = update_on_access(s, count)
        assert s > 5.0
        assert count == 100


# ── decay_score_boost ────────────────────────────────────────────────────────


class TestDecayScoreBoost:
    def test_range_at_full_retrievability(self):
        """retrievability=1.0 → max boost = 0.3 + 0.9 = 1.2."""
        assert decay_score_boost(1.0) == pytest.approx(1.2)

    def test_range_at_zero_retrievability(self):
        """retrievability=0.0 → min boost = 0.3."""
        assert decay_score_boost(0.0) == pytest.approx(0.3)

    def test_mid_retrievability(self):
        """retrievability=0.5 → 0.3 + 0.45 = 0.75."""
        assert decay_score_boost(0.5) == pytest.approx(0.75)

    def test_monotonically_increasing(self):
        """Boost should increase with retrievability."""
        values = [decay_score_boost(r / 10) for r in range(11)]
        for i in range(1, len(values)):
            assert values[i] >= values[i - 1]
