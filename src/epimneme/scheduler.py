"""Background scheduler for periodic reflection cycles.

Runs as an asyncio task inside the FastAPI lifespan — no external cron needed.
Stores run history in memory (survives across cycles but not container restarts).
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional, TYPE_CHECKING

from epimneme.reflection import ReflectionConfig, ReflectionEngine, ReflectionResult

if TYPE_CHECKING:
    from epimneme.manager import MemoryManager

logger = logging.getLogger("engram.scheduler")

# Maximum run history entries kept in memory
_MAX_HISTORY = 50


class ReflectionScheduler:
    """Manages the periodic reflection background task."""

    def __init__(self, manager: "MemoryManager"):
        self.manager = manager
        self._task: Optional[asyncio.Task] = None
        self._running = False
        self._enabled: bool = manager.config.reflection_enabled
        self._interval_hours: float = manager.config.reflection_interval_hours
        self._history: list[dict] = []
        self._last_result: Optional[ReflectionResult] = None
        self._next_run: Optional[datetime] = None

    def _build_reflection_config(self) -> ReflectionConfig:
        """Build ReflectionConfig from the EngramConfig."""
        cfg = self.manager.config
        return ReflectionConfig(
            gc_retrievability_threshold=cfg.reflection_gc_threshold,
            gc_min_age_days=cfg.reflection_gc_min_age_days,
            consolidation_similarity=cfg.reflection_consolidation_similarity,
            min_cluster_size=cfg.reflection_min_cluster_size,
            max_consolidations_per_run=cfg.reflection_max_consolidations,
            conflict_similarity=cfg.reflection_conflict_similarity,
            conflict_age_gap_days=cfg.reflection_conflict_age_gap_days,
        )

    def start(self) -> None:
        """Start the background scheduler task."""
        if self._task and not self._task.done():
            return
        self._task = asyncio.create_task(self._loop())
        logger.info(
            f"Reflection scheduler started — "
            f"interval={self._interval_hours}h, "
            f"enabled={self._enabled}"
        )

    async def stop(self) -> None:
        """Stop the background scheduler."""
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("Reflection scheduler stopped")

    async def run_now(self) -> ReflectionResult:
        """Trigger an immediate reflection cycle (manual or API-driven)."""
        if self._running:
            return ReflectionResult(
                error="A reflection cycle is already running",
                started_at=datetime.now(timezone.utc),
                finished_at=datetime.now(timezone.utc),
            )

        return await self._execute_cycle()

    async def _execute_cycle(self) -> ReflectionResult:
        """Run one reflection cycle and record the result."""
        self._running = True
        try:
            config = self._build_reflection_config()
            engine = ReflectionEngine(self.manager, config)
            result = await engine.run()

            self._last_result = result
            self._history.append(result.to_dict())
            if len(self._history) > _MAX_HISTORY:
                self._history = self._history[-_MAX_HISTORY:]

            logger.info(
                f"Reflection cycle complete — "
                f"gc={result.gc_obsoleted}, "
                f"consolidated={result.consolidated}, "
                f"conflicts={result.conflicts_resolved}, "
                f"duration={result.duration_seconds:.1f}s"
            )
            return result
        except Exception as e:
            logger.exception("Reflection cycle failed")
            result = ReflectionResult(
                error=str(e),
                started_at=datetime.now(timezone.utc),
                finished_at=datetime.now(timezone.utc),
            )
            self._last_result = result
            self._history.append(result.to_dict())
            return result
        finally:
            self._running = False

    async def _loop(self) -> None:
        """Background loop that runs reflection on the configured interval."""
        # Wait a bit on startup to let the system settle
        await asyncio.sleep(60)

        while True:
            try:
                interval_seconds = self._interval_hours * 3600
                self._next_run = datetime.now(timezone.utc) + timedelta(
                    seconds=interval_seconds
                )
                logger.debug(f"Next reflection at {self._next_run.isoformat()}")

                await asyncio.sleep(interval_seconds)

                if self._enabled:
                    await self._execute_cycle()
                else:
                    logger.debug("Reflection scheduler disabled — skipping cycle")

            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Reflection scheduler error — will retry next cycle")
                await asyncio.sleep(300)  # Back off 5 min on error

    def get_status(self) -> dict:
        """Return current scheduler status for the API/dashboard."""
        cfg = self._build_reflection_config()
        return {
            "running": self._running,
            "scheduler": {
                "enabled": self._enabled,
                "interval_hours": self._interval_hours,
                "next_run": self._next_run.isoformat() if self._next_run else None,
            },
            "config": {
                "gc_threshold": cfg.gc_retrievability_threshold,
                "gc_min_age_days": cfg.gc_min_age_days,
                "gc_exempt_kinds": cfg.gc_exempt_kinds,
                "consolidation_threshold": cfg.consolidation_similarity,
                "min_cluster_size": cfg.min_cluster_size,
                "max_consolidations": cfg.max_consolidations_per_run,
                "conflict_similarity": cfg.conflict_similarity,
                "conflict_age_gap_days": cfg.conflict_age_gap_days,
            },
            "last_run": self._last_result.to_dict() if self._last_result else {},
            "history": list(reversed(self._history[-20:])),
        }
