"""Activity event bus — in-memory ring buffer for the unified logger.

Captures all meaningful engram operations (writes, reads, sessions, entities,
reflection, dedup, etc.) in a lightweight ring buffer that the dashboard polls
via ``GET /api/activity``.

Events are ephemeral — they only live in memory and are lost on restart.
This is intentional: the activity stream is an operational view, not an audit
log.  Persistent history lives in the memories/sessions tables.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional


class EventType(str, Enum):
    """Activity event types shown in the unified logger."""

    WRITE = "write"           # remember() stored a memory
    RECALL = "recall"         # recall() / search
    SESSION = "session"       # session_start / session_end
    ENTITY = "entity"         # entity_track / entity_relate / entity_explore
    REFLECT = "reflect"       # reflection cycle GC / consolidate / conflict resolution
    DEDUP = "dedup"           # duplicate blocked by simhash or semantic dedup
    FORGET = "forget"         # memory marked obsolete
    CONFLICT = "conflict"     # potential conflict surfaced during write


# Color assignments for each type (used by dashboard)
EVENT_COLORS = {
    EventType.WRITE: "#3fb950",      # green
    EventType.RECALL: "#58a6ff",     # blue
    EventType.SESSION: "#bc8cff",    # purple
    EventType.ENTITY: "#d29922",     # yellow
    EventType.REFLECT: "#f0883e",    # orange
    EventType.DEDUP: "#8b949e",      # gray
    EventType.FORGET: "#f85149",     # red
    EventType.CONFLICT: "#d29922",   # yellow/warning
}


@dataclass
class ActivityEvent:
    """A single activity event in the ring buffer."""

    id: int
    ts: str  # ISO 8601
    type: str
    summary: str
    project: Optional[str] = None
    agent: Optional[str] = None
    detail: Optional[str] = None
    memory_id: Optional[str] = None
    _epoch: float = field(default_factory=time.time, repr=False)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("_epoch", None)
        return d


class ActivityBus:
    """Thread-safe in-memory ring buffer for activity events.

    Max ``capacity`` events are kept.  Consumers poll with ``since=<last_id>``
    to get only new events since their last fetch.
    """

    def __init__(self, capacity: int = 2000) -> None:
        self._capacity = capacity
        self._events: list[ActivityEvent] = []
        self._counter = 0
        self._lock = asyncio.Lock()
        # Per-type counters (for dashboard rate display)
        self._type_counts: dict[str, int] = {}
        self._started_at = time.time()

    async def emit(
        self,
        event_type: EventType | str,
        summary: str,
        *,
        project: str | None = None,
        agent: str | None = None,
        detail: str | None = None,
        memory_id: str | None = None,
    ) -> ActivityEvent:
        """Emit an activity event into the ring buffer and text log."""
        etype = event_type.value if isinstance(event_type, EventType) else event_type
        async with self._lock:
            self._counter += 1
            event = ActivityEvent(
                id=self._counter,
                ts=datetime.now(timezone.utc).isoformat(),
                type=etype,
                summary=summary,
                project=project,
                agent=agent,
                detail=detail,
                memory_id=memory_id,
            )
            self._events.append(event)
            # Trim to capacity
            if len(self._events) > self._capacity:
                self._events = self._events[-self._capacity:]
            # Update counter
            self._type_counts[etype] = self._type_counts.get(etype, 0) + 1

        # Write to disk outside the lock (fire-and-forget, never block the bus)
        try:
            from .textlog import log_event_to_disk
            await log_event_to_disk(event)
        except Exception:
            pass  # Disk logging must never break the activity bus

        return event

    async def get_events(
        self,
        since: int = 0,
        types: list[str] | None = None,
        project: str | None = None,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        """Get events newer than ``since`` (event ID), optionally filtered by type/project."""
        async with self._lock:
            events = self._events
            if since:
                events = [e for e in events if e.id > since]
            if types:
                type_set = set(types)
                events = [e for e in events if e.type in type_set]
            if project:
                events = [e for e in events if e.project == project]
            # Return newest first (reverse chronological), capped at limit
            return [e.to_dict() for e in reversed(events[-limit:])]

    async def get_stats(self) -> dict[str, Any]:
        """Get activity stats for the dashboard header."""
        async with self._lock:
            uptime = time.time() - self._started_at
            total = sum(self._type_counts.values())
            rate = total / max(uptime, 1) * 60  # events per minute
            return {
                "total_events": total,
                "events_per_minute": round(rate, 1),
                "by_type": dict(self._type_counts),
                "buffer_size": len(self._events),
                "buffer_capacity": self._capacity,
                "uptime_seconds": round(uptime),
            }


# ── Singleton ────────────────────────────────────────────────────────────────
# Created once at import, shared across the process.

_bus: ActivityBus | None = None


def get_activity_bus() -> ActivityBus:
    """Get the global activity bus (creates lazily)."""
    global _bus
    if _bus is None:
        _bus = ActivityBus()
    return _bus
