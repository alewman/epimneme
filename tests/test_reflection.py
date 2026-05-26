"""Tests for engram.reflection — ReflectionEngine 3-phase cycle.

All store/manager methods are mocked. No DB required.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from epimneme.core.models import Memory, MemoryKind, Project, RememberResult
from epimneme.reflection import ReflectionConfig, ReflectionEngine, ReflectionResult


# ── Helpers ───────────────────────────────────────────────────────────────────

NOW = datetime(2026, 3, 27, 12, 0, 0, tzinfo=timezone.utc)


def _old_mem(
    content="old memory",
    kind=MemoryKind.FACT,
    days_old=30,
    access_count=0,
    storage_strength=0.0,
    confidence=1.0,
    **kw,
) -> Memory:
    """Create a memory dated `days_old` days before NOW."""
    t = NOW - timedelta(days=days_old)
    return Memory(
        kind=kind,
        content=content,
        created_at=t,
        last_accessed=t,
        access_count=access_count,
        storage_strength=storage_strength,
        confidence=confidence,
        **kw,
    )


def _project(name="test-proj") -> Project:
    return Project(name=name, description="test", path="/tmp")


def _make_engine(
    mock_manager: AsyncMock,
    config: ReflectionConfig | None = None,
) -> ReflectionEngine:
    return ReflectionEngine(mock_manager, config or ReflectionConfig())


# ── ReflectionResult ──────────────────────────────────────────────────────────


class TestReflectionResult:
    def test_to_dict_defaults(self):
        r = ReflectionResult()
        d = r.to_dict()
        assert d["gc_obsoleted"] == 0
        assert d["consolidated"] == 0
        assert d["conflicts_resolved"] == 0
        assert d["error"] is None
        assert d["started_at"] is None

    def test_to_dict_with_values(self):
        r = ReflectionResult(
            gc_obsoleted=3,
            consolidated=1,
            conflicts_resolved=2,
            started_at=NOW,
            finished_at=NOW + timedelta(seconds=5),
            duration_seconds=5.0,
            log="done",
        )
        d = r.to_dict()
        assert d["gc_obsoleted"] == 3
        assert d["consolidated"] == 1
        assert d["conflicts_resolved"] == 2
        assert d["duration_seconds"] == 5.0
        assert d["started_at"] == NOW.isoformat()
        assert "done" in d["log"]


# ── ReflectionConfig ──────────────────────────────────────────────────────────


class TestReflectionConfig:
    def test_defaults(self):
        c = ReflectionConfig()
        assert c.gc_retrievability_threshold == 0.05
        assert c.gc_min_age_days == 7.0
        assert "decision" in c.gc_exempt_kinds
        assert "procedure" in c.gc_exempt_kinds
        assert c.consolidation_similarity == 0.88
        assert c.min_cluster_size == 3
        assert c.conflict_similarity == 0.85

    def test_custom(self):
        c = ReflectionConfig(gc_retrievability_threshold=0.1, min_cluster_size=5)
        assert c.gc_retrievability_threshold == 0.1
        assert c.min_cluster_size == 5


# ── Phase 1: Garbage Collection ──────────────────────────────────────────────


class TestPhaseGC:
    @pytest.mark.asyncio
    async def test_gc_obsoletes_faded_memories(self):
        """Memories with very low retrievability should be marked obsolete."""
        mgr = AsyncMock()
        mgr.config = MagicMock()
        mgr.config.decay_base_stability = 1.0

        faded = _old_mem("forgotten thing", days_old=90, storage_strength=0.0)
        mgr.store.get_memories_for_gc.return_value = [faded]
        mgr.store.mark_obsolete = AsyncMock()

        engine = _make_engine(mgr)
        count = await engine._phase_gc()

        assert count == 1
        mgr.store.mark_obsolete.assert_awaited_once_with(faded.id)

    @pytest.mark.asyncio
    async def test_gc_skips_exempt_kinds(self):
        """Decisions and procedures should not be GC'd regardless of age."""
        mgr = AsyncMock()
        mgr.config = MagicMock()
        mgr.config.decay_base_stability = 1.0

        decision = _old_mem("chose postgres", kind=MemoryKind.DECISION, days_old=90)
        procedure = _old_mem("how to deploy", kind=MemoryKind.PROCEDURE, days_old=90)
        mgr.store.get_memories_for_gc.return_value = [decision, procedure]
        mgr.store.mark_obsolete = AsyncMock()

        engine = _make_engine(mgr)
        count = await engine._phase_gc()

        assert count == 0
        mgr.store.mark_obsolete.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_gc_keeps_recently_accessed(self):
        """A recently accessed memory should have high retrievability and survive."""
        mgr = AsyncMock()
        mgr.config = MagicMock()
        mgr.config.decay_base_stability = 1.0

        recent = _old_mem("recent thing", days_old=30)
        # Override last_accessed to be very recent
        recent.last_accessed = NOW - timedelta(hours=1)
        recent.storage_strength = 5.0
        mgr.store.get_memories_for_gc.return_value = [recent]
        mgr.store.mark_obsolete = AsyncMock()

        engine = _make_engine(mgr)
        # Pin the engine's notion of "now" to NOW so retrievability is high.
        with patch("epimneme.reflection.datetime") as mock_dt:
            mock_dt.now.return_value = NOW
            mock_dt.side_effect = datetime  # passthrough for other calls
            count = await engine._phase_gc()

        assert count == 0
        mgr.store.mark_obsolete.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_gc_empty_returns_zero(self):
        """No candidate memories → 0 obsoleted."""
        mgr = AsyncMock()
        mgr.config = MagicMock()
        mgr.config.decay_base_stability = 1.0
        mgr.store.get_memories_for_gc.return_value = []

        engine = _make_engine(mgr)
        count = await engine._phase_gc()
        assert count == 0


# ── Phase 2: Consolidation ───────────────────────────────────────────────────


class TestPhaseConsolidate:
    @pytest.mark.asyncio
    async def test_consolidation_merges_cluster(self):
        """A cluster of similar memories should be merged into one."""
        mgr = AsyncMock()
        mgr.config = MagicMock()

        proj = _project()
        mgr.store.list_projects.return_value = [proj]

        m1 = _old_mem("postgres runs on 5432", days_old=10)
        m2 = _old_mem("PostgreSQL port is 5432", days_old=8)
        m3 = _old_mem("PG default port: 5432", days_old=5)

        # First call (for the project) returns the cluster;
        # second call (for project_id=None) returns nothing.
        mgr.store.find_similar_clusters.side_effect = [
            [[m1, m2, m3]],  # project cluster
            [],               # null-project cluster
        ]
        mgr.store.mark_obsolete = AsyncMock()
        mgr.store.set_memory_project = AsyncMock()

        # Simulate remember() returning a RememberResult (not blocked by dedup)
        new_mem = _old_mem("consolidated", days_old=0)
        mgr.remember.return_value = RememberResult(
            memory=new_mem, conflicts=[]
        )

        engine = _make_engine(mgr)
        count = await engine._phase_consolidate()

        assert count == 1
        mgr.remember.assert_awaited_once()
        # Originals should be marked obsolete
        assert mgr.store.mark_obsolete.await_count >= 2

    @pytest.mark.asyncio
    async def test_consolidation_identical_content(self):
        """Cluster with identical content should just obsolete duplicates."""
        mgr = AsyncMock()
        mgr.config = MagicMock()

        proj = _project()
        mgr.store.list_projects.return_value = [proj]

        m1 = _old_mem("exact same thing", days_old=10)
        m2 = _old_mem("exact same thing", days_old=5)

        mgr.store.find_similar_clusters.side_effect = [
            [[m1, m2]],  # project cluster
            [],           # null-project cluster
        ]
        mgr.store.mark_obsolete = AsyncMock()

        engine = _make_engine(mgr)
        count = await engine._phase_consolidate()

        assert count == 1
        # _merge_cluster sorts newest first — m2 (days_old=5) is newest, so m1 gets obsoleted
        mgr.store.mark_obsolete.assert_awaited_once_with(m1.id)
        mgr.remember.assert_not_awaited()  # no new memory needed

    @pytest.mark.asyncio
    async def test_consolidation_dedup_blocks_merge(self):
        """If remember() returns a dedup string, originals should still be obsoleted."""
        mgr = AsyncMock()
        mgr.config = MagicMock()

        proj = _project()
        mgr.store.list_projects.return_value = [proj]

        m1 = _old_mem("thing A", days_old=10)
        m2 = _old_mem("thing B", days_old=8)
        m3 = _old_mem("thing C", days_old=5)
        mgr.store.find_similar_clusters.side_effect = [
            [[m1, m2, m3]],  # project clusters
            [],               # null-project clusters
        ]
        mgr.store.mark_obsolete = AsyncMock()

        # Simulate dedup blocking the consolidated remember
        mgr.remember.return_value = "Near-duplicate of memory abcd1234…"

        engine = _make_engine(mgr)
        count = await engine._phase_consolidate()

        assert count == 1
        # The "rest" (m2, m3) should be obsoleted even though merge was dedup'd
        assert mgr.store.mark_obsolete.await_count >= 2

    @pytest.mark.asyncio
    async def test_consolidation_no_clusters(self):
        """No clusters found → 0 consolidated."""
        mgr = AsyncMock()
        mgr.config = MagicMock()

        mgr.store.list_projects.return_value = [_project()]
        mgr.store.find_similar_clusters.side_effect = [
            [],  # project clusters
            [],  # null-project clusters
        ]

        engine = _make_engine(mgr)
        count = await engine._phase_consolidate()
        assert count == 0

    @pytest.mark.asyncio
    async def test_consolidation_respects_max_limit(self):
        """Only consolidate up to max_consolidations_per_run."""
        mgr = AsyncMock()
        mgr.config = MagicMock()

        proj = _project()
        mgr.store.list_projects.return_value = [proj]

        # Create 5 clusters but limit to 2
        clusters = []
        for i in range(5):
            m1 = _old_mem(f"cluster {i} a", days_old=10)
            m2 = _old_mem(f"cluster {i} b", days_old=5)
            clusters.append([m1, m2])

        # First call returns the clusters; null-project call shouldn't be
        # reached since we hit the limit first, but provide empty just in case.
        mgr.store.find_similar_clusters.side_effect = [
            clusters,  # project clusters
            [],        # null-project clusters (may not be called)
        ]
        mgr.store.mark_obsolete = AsyncMock()

        config = ReflectionConfig(max_consolidations_per_run=2)
        engine = _make_engine(mgr, config)
        count = await engine._phase_consolidate()

        assert count == 2


# ── Phase 3: Conflict Resolution ─────────────────────────────────────────────


class TestPhaseConflictResolution:
    @pytest.mark.asyncio
    async def test_resolves_stale_conflict(self):
        """Older memory should be obsoleted when a newer one has equal confidence."""
        mgr = AsyncMock()

        newer = _old_mem("new fact", days_old=1, confidence=1.0)
        older = _old_mem("old fact", days_old=30, confidence=0.8)
        mgr.store.find_conflicting_pairs.side_effect = [
            [(newer, older, 0.90)],  # FACT kind
            [],                       # DECISION kind
        ]
        mgr.store.mark_obsolete = AsyncMock()

        engine = _make_engine(mgr)
        count = await engine._phase_resolve_conflicts()

        assert count == 1
        mgr.store.mark_obsolete.assert_awaited_once_with(older.id)

    @pytest.mark.asyncio
    async def test_skips_when_older_has_higher_confidence(self):
        """Should NOT obsolete older memory if it has higher confidence."""
        mgr = AsyncMock()

        newer = _old_mem("new fact", days_old=1, confidence=0.5)
        older = _old_mem("old fact", days_old=30, confidence=1.0)
        mgr.store.find_conflicting_pairs.side_effect = [
            [(newer, older, 0.90)],  # FACT kind
            [],                       # DECISION kind
        ]
        mgr.store.mark_obsolete = AsyncMock()

        engine = _make_engine(mgr)
        count = await engine._phase_resolve_conflicts()

        assert count == 0
        mgr.store.mark_obsolete.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_processes_both_facts_and_decisions(self):
        """Conflict resolution should check both FACT and DECISION kinds."""
        mgr = AsyncMock()

        newer_fact = _old_mem("new fact", days_old=1, confidence=1.0)
        older_fact = _old_mem("old fact", days_old=30, confidence=0.8)
        newer_decision = _old_mem("new decision", kind=MemoryKind.DECISION, days_old=1, confidence=1.0)
        older_decision = _old_mem("old decision", kind=MemoryKind.DECISION, days_old=30, confidence=0.9)

        mgr.store.find_conflicting_pairs.side_effect = [
            [(newer_fact, older_fact, 0.90)],       # for FACT kind
            [(newer_decision, older_decision, 0.88)],  # for DECISION kind
        ]
        mgr.store.mark_obsolete = AsyncMock()

        engine = _make_engine(mgr)
        count = await engine._phase_resolve_conflicts()

        assert count == 2
        assert mgr.store.mark_obsolete.await_count == 2

    @pytest.mark.asyncio
    async def test_no_conflicts_returns_zero(self):
        """No conflicting pairs → 0 resolved."""
        mgr = AsyncMock()
        mgr.store.find_conflicting_pairs.side_effect = [
            [],  # FACT kind
            [],  # DECISION kind
        ]

        engine = _make_engine(mgr)
        count = await engine._phase_resolve_conflicts()
        assert count == 0


# ── Full Cycle (run) ──────────────────────────────────────────────────────────


class TestFullCycle:
    @pytest.mark.asyncio
    async def test_run_returns_result(self):
        """A full run should return a ReflectionResult with timing."""
        mgr = AsyncMock()
        mgr.config = MagicMock()
        mgr.config.decay_base_stability = 1.0
        mgr.store.get_memories_for_gc.return_value = []
        mgr.store.list_projects.return_value = []
        mgr.store.find_similar_clusters.return_value = []
        mgr.store.find_conflicting_pairs.return_value = []

        engine = _make_engine(mgr)
        result = await engine.run()

        assert isinstance(result, ReflectionResult)
        assert result.error is None
        assert result.started_at is not None
        assert result.finished_at is not None
        assert result.duration_seconds >= 0
        assert "Reflection cycle complete" in result.log

    @pytest.mark.asyncio
    async def test_run_captures_exception(self):
        """If any phase throws, the error should be captured, not re-raised."""
        mgr = AsyncMock()
        mgr.config = MagicMock()
        mgr.config.decay_base_stability = 1.0
        mgr.store.get_memories_for_gc.side_effect = RuntimeError("db gone")

        engine = _make_engine(mgr)
        result = await engine.run()

        assert result.error is not None
        assert "db gone" in result.error
        assert result.finished_at is not None

    @pytest.mark.asyncio
    async def test_run_aggregates_all_phases(self):
        """Full run should sum up results from all three phases."""
        mgr = AsyncMock()
        mgr.config = MagicMock()
        mgr.config.decay_base_stability = 1.0

        # Phase 1: 2 GC'd
        faded1 = _old_mem("faded1", days_old=90)
        faded2 = _old_mem("faded2", days_old=90)
        mgr.store.get_memories_for_gc.return_value = [faded1, faded2]
        mgr.store.mark_obsolete = AsyncMock()

        # Phase 2: no clusters
        mgr.store.list_projects.return_value = []
        mgr.store.find_similar_clusters.return_value = []

        # Phase 3: 1 conflict
        newer = _old_mem("new", days_old=1, confidence=1.0)
        older = _old_mem("old", days_old=30, confidence=0.5)
        mgr.store.find_conflicting_pairs.side_effect = [
            [(newer, older, 0.90)],  # FACT
            [],                       # DECISION
        ]

        engine = _make_engine(mgr)
        result = await engine.run()

        assert result.gc_obsoleted == 2
        assert result.consolidated == 0
        assert result.conflicts_resolved == 1
        assert result.error is None
