"""Memory reflection — periodic compaction, consolidation, and conflict resolution.

Three phases run on each reflection cycle:

1. **Garbage Collection** — Mark memories with very low retrievability as
   obsolete.  They still exist in the DB for audit but stop appearing in
   search results.

2. **Consolidation** — Find clusters of semantically similar memories
   within each project and merge them into a single summary memory.
   The originals are marked obsolete with a pointer to the new summary.

3. **Conflict Resolution** — For facts/decisions, find pairs where a newer
   memory contradicts an older one (high similarity, same kind) and
   auto-obsolete the older entry.

All functions are async and operate through the MemoryManager / PostgresStore.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional

from epimneme.activity import EventType, get_activity_bus
from epimneme.core.models import Memory, MemoryKind
from epimneme.decay import calculate_retrievability

if TYPE_CHECKING:
    from epimneme.manager import MemoryManager

logger = logging.getLogger("engram.reflection")


@dataclass
class ReflectionConfig:
    """Tuning knobs for the reflection cycle."""

    # Garbage collection
    gc_retrievability_threshold: float = 0.05
    gc_min_age_days: float = 7.0   # Don't GC memories younger than this
    gc_exempt_kinds: list[str] = field(
        default_factory=lambda: ["decision", "procedure"]
    )

    # Consolidation
    consolidation_similarity: float = 0.88
    min_cluster_size: int = 3
    max_consolidations_per_run: int = 10

    # Conflict resolution
    conflict_similarity: float = 0.85
    conflict_age_gap_days: float = 7.0  # Newer must be ≥ this much newer


@dataclass
class ReflectionResult:
    """Outcome of a single reflection cycle."""

    gc_obsoleted: int = 0
    consolidated: int = 0
    conflicts_resolved: int = 0
    log: str = ""
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    duration_seconds: float = 0.0
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "gc_obsoleted": self.gc_obsoleted,
            "consolidated": self.consolidated,
            "conflicts_resolved": self.conflicts_resolved,
            "log": self.log,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "duration_seconds": self.duration_seconds,
            "error": self.error,
        }


class ReflectionEngine:
    """Runs the three-phase reflection cycle against a MemoryManager."""

    def __init__(self, manager: "MemoryManager", config: Optional[ReflectionConfig] = None):
        self.manager = manager
        self.config = config or ReflectionConfig()
        self._log_lines: list[str] = []

    def _log(self, msg: str) -> None:
        logger.info(msg)
        self._log_lines.append(msg)

    async def run(self) -> ReflectionResult:
        """Execute a full reflection cycle."""
        result = ReflectionResult(started_at=datetime.now(timezone.utc))
        self._log_lines = []
        t0 = time.monotonic()

        try:
            self._log("=== Reflection cycle started ===")

            bus = get_activity_bus()
            await bus.emit(EventType.REFLECT, "Reflection cycle started")

            # Phase 1: Garbage collection
            gc_count = await self._phase_gc()
            result.gc_obsoleted = gc_count
            self._log(f"Phase 1 (GC): {gc_count} memories marked obsolete")
            if gc_count:
                await bus.emit(EventType.REFLECT, f"GC: {gc_count} faded memories obsoleted")

            # Phase 2: Consolidation
            consol_count = await self._phase_consolidate()
            result.consolidated = consol_count
            self._log(f"Phase 2 (Consolidation): {consol_count} memories consolidated")
            if consol_count:
                await bus.emit(EventType.REFLECT, f"Consolidated {consol_count} memory cluster(s)")

            # Phase 3: Conflict resolution
            conflict_count = await self._phase_resolve_conflicts()
            result.conflicts_resolved = conflict_count
            self._log(f"Phase 3 (Conflicts): {conflict_count} stale conflicts resolved")
            if conflict_count:
                await bus.emit(EventType.REFLECT, f"Resolved {conflict_count} stale conflict(s)")

            self._log("=== Reflection cycle complete ===")
            await bus.emit(
                EventType.REFLECT,
                f"Reflection complete: {gc_count} GC'd, {consol_count} consolidated, {conflict_count} conflicts",
                detail=f"Duration: {time.monotonic() - t0:.1f}s",
            )

        except Exception as e:
            result.error = str(e)
            self._log(f"ERROR: {e}")
            logger.exception("Reflection cycle failed")

        result.duration_seconds = time.monotonic() - t0
        result.finished_at = datetime.now(timezone.utc)
        result.log = "\n".join(self._log_lines)
        return result

    # ── Phase 1: Garbage Collection ──────────────────────────────────────

    async def _phase_gc(self) -> int:
        """Mark faded, unused memories as obsolete."""
        store = self.manager.store
        config = self.config
        now = datetime.now(timezone.utc)

        # Get all active memories with decay data
        memories = await store.get_memories_for_gc(
            min_age_days=config.gc_min_age_days,
        )

        # Load persistent project IDs so we can skip their memories
        persistent_project_ids = await store.get_persistent_project_ids()

        obsoleted = 0
        for mem in memories:
            # Skip pinned memories, persistent-project memories, and exempt kinds
            if mem.pinned:
                continue
            if mem.project_id and mem.project_id in persistent_project_ids:
                continue
            if mem.kind.value in config.gc_exempt_kinds:
                continue

            retrievability = calculate_retrievability(
                mem.storage_strength,
                mem.last_accessed or mem.created_at,
                base_stability=self.manager.config.decay_base_stability,
                now=now,
            )

            if retrievability < config.gc_retrievability_threshold:
                await store.mark_obsolete(mem.id)
                obsoleted += 1
                self._log(
                    f"  GC: {mem.id[:8]}… [{mem.kind.value}] "
                    f"retrievability={retrievability:.3f} "
                    f"access_count={mem.access_count} → obsolete"
                )

        return obsoleted

    # ── Phase 2: Consolidation ───────────────────────────────────────────

    async def _phase_consolidate(self) -> int:
        """Find clusters of similar memories and merge them."""
        store = self.manager.store
        config = self.config

        # Get projects to process
        projects = await store.list_projects()
        total_consolidated = 0

        for project in projects:
            if total_consolidated >= config.max_consolidations_per_run:
                break

            clusters = await store.find_similar_clusters(
                project_id=project.id,
                similarity_threshold=config.consolidation_similarity,
                min_cluster_size=config.min_cluster_size,
                limit=config.max_consolidations_per_run - total_consolidated,
            )

            for cluster_memories in clusters:
                if total_consolidated >= config.max_consolidations_per_run:
                    break

                merged = await self._merge_cluster(cluster_memories, project.id)
                if merged:
                    total_consolidated += 1

        # Also process memories with no project
        if total_consolidated < config.max_consolidations_per_run:
            clusters = await store.find_similar_clusters(
                project_id=None,
                similarity_threshold=config.consolidation_similarity,
                min_cluster_size=config.min_cluster_size,
                limit=config.max_consolidations_per_run - total_consolidated,
            )
            for cluster_memories in clusters:
                if total_consolidated >= config.max_consolidations_per_run:
                    break
                merged = await self._merge_cluster(cluster_memories, None)
                if merged:
                    total_consolidated += 1

        return total_consolidated

    async def _merge_cluster(
        self, memories: list[Memory], project_id: Optional[str]
    ) -> bool:
        """Merge a cluster of similar memories into one consolidated entry."""
        if len(memories) < 2:
            return False

        # Sort by: newest first, highest confidence first, most accessed first
        memories.sort(
            key=lambda m: (m.created_at, m.confidence, m.access_count),
            reverse=True,
        )

        # Build consolidated content from the cluster
        # Keep the most recent/strongest as the base
        best = memories[0]
        rest = memories[1:]

        # Combine unique content lines
        contents = []
        seen_content = set()
        for m in memories:
            normalized = m.content.strip().lower()
            if normalized not in seen_content:
                seen_content.add(normalized)
                contents.append(m.content.strip())

        if len(contents) <= 1:
            # All identical content — just obsolete the duplicates
            for m in rest:
                await self.manager.store.mark_obsolete(m.id)
                self._log(f"  Consolidate: obsoleted duplicate {m.id[:8]}… (identical to {best.id[:8]}…)")
            return True

        # Create a consolidated summary
        consolidated_content = "Consolidated from {} memories:\n{}".format(
            len(memories),
            "\n".join(f"- {c}" for c in contents),
        )

        # Collect all tags
        all_tags = set()
        for m in memories:
            all_tags.update(m.tags)
        all_tags.add("consolidated")

        # Use the highest confidence and most common kind
        max_confidence = max(m.confidence for m in memories)
        kind_counts: dict[MemoryKind, int] = {}
        for m in memories:
            kind_counts[m.kind] = kind_counts.get(m.kind, 0) + 1
        dominant_kind = max(kind_counts, key=kind_counts.get)  # type: ignore[arg-type]

        # Resolve project name from project_id so dedup runs in correct scope
        project_name: Optional[str] = None
        if project_id:
            proj = await self.manager.store.get_project_by_id(project_id)
            if proj:
                project_name = proj.name

        # Store the consolidated memory (uses the manager for embedding + dedup)
        from epimneme.core.models import RememberResult

        result = await self.manager.remember(
            content=consolidated_content,
            kind=dominant_kind,
            project_name=project_name,
            subject=best.subject,
            tags=list(all_tags),
            confidence=max_confidence,
        )

        if isinstance(result, str):
            # Dedup blocked it — that's fine, just obsolete the weaker entries
            self._log(f"  Consolidate: dedup blocked merge ({result[:60]}…)")
            for m in rest:
                await self.manager.store.mark_obsolete(m.id)
            return True

        # Mark originals as obsolete
        for m in memories:
            if isinstance(result, RememberResult) and m.id == result.memory.id:
                continue
            await self.manager.store.mark_obsolete(m.id)

        ids_str = ", ".join(m.id[:8] + "…" for m in memories)
        new_id = result.memory.id[:8] if isinstance(result, RememberResult) else "?"
        self._log(
            f"  Consolidate: merged [{ids_str}] → {new_id}… "
            f"[{dominant_kind.value}] ({len(contents)} unique items)"
        )
        return True

    # ── Phase 3: Conflict Resolution ─────────────────────────────────────

    async def _phase_resolve_conflicts(self) -> int:
        """Auto-obsolete older memories contradicted by newer ones."""
        store = self.manager.store
        config = self.config
        resolved = 0

        # Get recent facts and decisions that might contradict older ones
        conflict_kinds = [MemoryKind.FACT, MemoryKind.DECISION]

        for kind in conflict_kinds:
            pairs = await store.find_conflicting_pairs(
                kind=kind.value,
                similarity_threshold=config.conflict_similarity,
                min_age_gap_days=config.conflict_age_gap_days,
                limit=20,
            )

            for newer, older, sim in pairs:
                # Check that the newer one has equal or better signals
                if newer.confidence >= older.confidence:
                    await store.mark_obsolete(older.id)
                    resolved += 1
                    self._log(
                        f"  Conflict: obsoleted {older.id[:8]}… "
                        f"(superseded by newer {newer.id[:8]}… "
                        f"[{kind.value}], sim={sim:.2f})"
                    )

        return resolved
