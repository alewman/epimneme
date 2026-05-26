"""Tests for engram.scheduler — ReflectionScheduler background loop.

Uses mocked MemoryManager + ReflectionEngine so no DB is needed.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from epimneme.core.config import EngramConfig
from epimneme.reflection import ReflectionResult
from epimneme.scheduler import ReflectionScheduler


# ── Helpers ───────────────────────────────────────────────────────────────────


def _mock_manager(
    reflection_enabled: bool = True,
    interval_hours: float = 24.0,
) -> MagicMock:
    """Build a mocked MemoryManager with a real-ish config."""
    mgr = MagicMock()
    mgr.config = EngramConfig(
        pg_host="localhost",
        pg_user="test",
        pg_password="test",
        pg_database="test",
        embeddings_enabled=False,
        reflection_enabled=reflection_enabled,
        reflection_interval_hours=interval_hours,
    )
    mgr.store = AsyncMock()
    return mgr


# ── Scheduler basics ─────────────────────────────────────────────────────────


class TestSchedulerBasics:
    def test_init_reads_config(self):
        mgr = _mock_manager(reflection_enabled=False, interval_hours=12.0)
        sched = ReflectionScheduler(mgr)
        assert sched._enabled is False
        assert sched._interval_hours == 12.0
        assert sched._last_result is None
        assert sched._history == []

    def test_get_status_initial(self):
        mgr = _mock_manager()
        sched = ReflectionScheduler(mgr)
        status = sched.get_status()

        assert status["running"] is False
        assert status["scheduler"]["enabled"] is True
        assert status["scheduler"]["interval_hours"] == 24.0
        assert status["scheduler"]["next_run"] is None
        assert status["last_run"] == {}
        assert status["history"] == []
        assert "gc_threshold" in status["config"]

    def test_get_status_reflects_config(self):
        mgr = _mock_manager(reflection_enabled=False, interval_hours=6.0)
        sched = ReflectionScheduler(mgr)
        status = sched.get_status()
        assert status["scheduler"]["enabled"] is False
        assert status["scheduler"]["interval_hours"] == 6.0


# ── run_now ───────────────────────────────────────────────────────────────────


class TestRunNow:
    @pytest.mark.asyncio
    async def test_run_now_executes_reflection(self):
        """run_now() should execute a reflection cycle and record the result."""
        mgr = _mock_manager()

        # Mock the ReflectionEngine.run() to return a clean result
        good_result = ReflectionResult(
            gc_obsoleted=2,
            consolidated=1,
            conflicts_resolved=0,
            started_at=datetime.now(timezone.utc),
            finished_at=datetime.now(timezone.utc),
            duration_seconds=0.5,
        )

        with patch("epimneme.scheduler.ReflectionEngine") as MockEngine:
            instance = AsyncMock()
            instance.run.return_value = good_result
            MockEngine.return_value = instance

            sched = ReflectionScheduler(mgr)
            result = await sched.run_now()

        assert result.gc_obsoleted == 2
        assert result.consolidated == 1
        assert result.error is None
        assert sched._last_result is result
        assert len(sched._history) == 1

    @pytest.mark.asyncio
    async def test_run_now_rejects_concurrent(self):
        """If a cycle is already running, run_now() should return error."""
        mgr = _mock_manager()
        sched = ReflectionScheduler(mgr)
        sched._running = True

        result = await sched.run_now()
        assert result.error is not None
        assert "already running" in result.error

    @pytest.mark.asyncio
    async def test_run_now_captures_exception(self):
        """If reflection throws, run_now() should capture it as error."""
        mgr = _mock_manager()

        with patch("epimneme.scheduler.ReflectionEngine") as MockEngine:
            instance = AsyncMock()
            instance.run.side_effect = RuntimeError("boom")
            MockEngine.return_value = instance

            sched = ReflectionScheduler(mgr)
            result = await sched.run_now()

        assert result.error is not None
        assert "boom" in result.error
        assert sched._running is False  # should be reset even on error


# ── History management ────────────────────────────────────────────────────────


class TestHistory:
    @pytest.mark.asyncio
    async def test_history_capped_at_max(self):
        """History should not grow beyond _MAX_HISTORY (50)."""
        mgr = _mock_manager()

        result = ReflectionResult(
            started_at=datetime.now(timezone.utc),
            finished_at=datetime.now(timezone.utc),
        )

        with patch("epimneme.scheduler.ReflectionEngine") as MockEngine:
            instance = AsyncMock()
            instance.run.return_value = result
            MockEngine.return_value = instance

            sched = ReflectionScheduler(mgr)
            for _ in range(60):
                await sched.run_now()

        assert len(sched._history) == 50

    @pytest.mark.asyncio
    async def test_status_shows_recent_history(self):
        """get_status() should show up to 20 most recent runs, reversed."""
        mgr = _mock_manager()
        result = ReflectionResult(
            gc_obsoleted=1,
            started_at=datetime.now(timezone.utc),
            finished_at=datetime.now(timezone.utc),
        )

        with patch("epimneme.scheduler.ReflectionEngine") as MockEngine:
            instance = AsyncMock()
            instance.run.return_value = result
            MockEngine.return_value = instance

            sched = ReflectionScheduler(mgr)
            for _ in range(5):
                await sched.run_now()

        status = sched.get_status()
        assert len(status["history"]) == 5


# ── Start / Stop ─────────────────────────────────────────────────────────────


class TestStartStop:
    @pytest.mark.asyncio
    async def test_start_creates_task(self):
        """start() should create an asyncio task."""
        mgr = _mock_manager()
        sched = ReflectionScheduler(mgr)
        sched.start()

        assert sched._task is not None
        assert not sched._task.done()

        await sched.stop()
        # After stop, task should be cleaned up
        assert sched._task is None

    @pytest.mark.asyncio
    async def test_stop_idempotent(self):
        """stop() on an already-stopped scheduler should not raise."""
        mgr = _mock_manager()
        sched = ReflectionScheduler(mgr)
        await sched.stop()  # Already stopped — should be a no-op

    @pytest.mark.asyncio
    async def test_start_idempotent(self):
        """Starting twice should not create duplicate tasks."""
        mgr = _mock_manager()
        sched = ReflectionScheduler(mgr)
        sched.start()
        first_task = sched._task
        sched.start()  # second start

        assert sched._task is first_task  # same task object
        await sched.stop()


# ── _build_reflection_config ────────────────────────────────────────────────


class TestBuildConfig:
    def test_builds_from_engram_config(self):
        mgr = _mock_manager()
        mgr.config.reflection_gc_threshold = 0.1
        mgr.config.reflection_consolidation_similarity = 0.9
        mgr.config.reflection_conflict_similarity = 0.8

        sched = ReflectionScheduler(mgr)
        cfg = sched._build_reflection_config()

        assert cfg.gc_retrievability_threshold == 0.1
        assert cfg.consolidation_similarity == 0.9
        assert cfg.conflict_similarity == 0.8
