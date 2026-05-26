"""Tests for engram.activity — ActivityBus ring buffer and event emission.

Pure async unit tests — no DB, no external dependencies.
"""

from __future__ import annotations

import time

import pytest

from epimneme.activity import (
    EVENT_COLORS,
    ActivityBus,
    ActivityEvent,
    EventType,
    get_activity_bus,
)


# ── EventType enum ───────────────────────────────────────────────────────────


class TestEventType:
    def test_all_types_have_colors(self):
        """Every EventType value should have a color mapped."""
        for et in EventType:
            assert et in EVENT_COLORS, f"Missing color for {et}"

    def test_values_are_strings(self):
        assert EventType.WRITE.value == "write"
        assert EventType.RECALL.value == "recall"
        assert EventType.CONFLICT.value == "conflict"

    def test_total_event_types(self):
        assert len(EventType) == 8


# ── ActivityEvent ─────────────────────────────────────────────────────────────


class TestActivityEvent:
    def test_to_dict_excludes_epoch(self):
        e = ActivityEvent(id=1, ts="2026-01-01T00:00:00", type="write", summary="test")
        d = e.to_dict()
        assert "_epoch" not in d
        assert d["id"] == 1
        assert d["type"] == "write"
        assert d["summary"] == "test"

    def test_optional_fields_default_none(self):
        e = ActivityEvent(id=1, ts="now", type="recall", summary="search")
        assert e.project is None
        assert e.agent is None
        assert e.detail is None
        assert e.memory_id is None


# ── ActivityBus ───────────────────────────────────────────────────────────────


class TestActivityBus:
    @pytest.mark.asyncio
    async def test_emit_returns_event(self):
        bus = ActivityBus(capacity=100)
        event = await bus.emit(EventType.WRITE, "stored a fact")
        assert event.id == 1
        assert event.type == "write"
        assert event.summary == "stored a fact"

    @pytest.mark.asyncio
    async def test_emit_increments_counter(self):
        bus = ActivityBus(capacity=100)
        e1 = await bus.emit(EventType.WRITE, "first")
        e2 = await bus.emit(EventType.RECALL, "second")
        assert e1.id == 1
        assert e2.id == 2

    @pytest.mark.asyncio
    async def test_emit_with_all_fields(self):
        bus = ActivityBus(capacity=100)
        event = await bus.emit(
            EventType.WRITE,
            "stored fact",
            project="myproj",
            agent="copilot",
            detail="some detail",
            memory_id="mem-123",
        )
        assert event.project == "myproj"
        assert event.agent == "copilot"
        assert event.detail == "some detail"
        assert event.memory_id == "mem-123"

    @pytest.mark.asyncio
    async def test_emit_accepts_string_type(self):
        """emit() should accept both EventType enum and raw strings."""
        bus = ActivityBus(capacity=100)
        event = await bus.emit("custom_type", "custom event")
        assert event.type == "custom_type"

    @pytest.mark.asyncio
    async def test_ring_buffer_capacity(self):
        """Events beyond capacity should evict oldest."""
        bus = ActivityBus(capacity=5)
        for i in range(10):
            await bus.emit(EventType.WRITE, f"event {i}")

        events = await bus.get_events()
        assert len(events) == 5
        # Should have events 5-9 (the latest 5), newest first
        assert events[0]["summary"] == "event 9"
        assert events[-1]["summary"] == "event 5"

    @pytest.mark.asyncio
    async def test_get_events_since(self):
        """get_events(since=N) should return only events with id > N."""
        bus = ActivityBus(capacity=100)
        for i in range(5):
            await bus.emit(EventType.WRITE, f"event {i}")

        events = await bus.get_events(since=3)
        assert len(events) == 2
        # newest-first ordering: ids 5, 4
        assert events[0]["id"] == 5
        assert events[1]["id"] == 4

    @pytest.mark.asyncio
    async def test_get_events_type_filter(self):
        """get_events(types=[...]) should filter by event type."""
        bus = ActivityBus(capacity=100)
        await bus.emit(EventType.WRITE, "write 1")
        await bus.emit(EventType.RECALL, "recall 1")
        await bus.emit(EventType.WRITE, "write 2")
        await bus.emit(EventType.SESSION, "session 1")

        events = await bus.get_events(types=["write"])
        assert len(events) == 2
        assert all(e["type"] == "write" for e in events)

    @pytest.mark.asyncio
    async def test_get_events_combined_filters(self):
        """since + types should work together."""
        bus = ActivityBus(capacity=100)
        await bus.emit(EventType.WRITE, "w1")     # id=1
        await bus.emit(EventType.RECALL, "r1")     # id=2
        await bus.emit(EventType.WRITE, "w2")      # id=3
        await bus.emit(EventType.RECALL, "r2")     # id=4

        events = await bus.get_events(since=2, types=["write"])
        assert len(events) == 1
        assert events[0]["summary"] == "w2"

    @pytest.mark.asyncio
    async def test_get_events_limit(self):
        """get_events(limit=N) should cap the result count."""
        bus = ActivityBus(capacity=100)
        for i in range(20):
            await bus.emit(EventType.WRITE, f"e{i}")

        events = await bus.get_events(limit=5)
        assert len(events) == 5
        # Should be the 5 most recent, newest first
        assert events[0]["summary"] == "e19"

    @pytest.mark.asyncio
    async def test_get_events_empty_bus(self):
        bus = ActivityBus(capacity=100)
        events = await bus.get_events()
        assert events == []


# ── ActivityBus stats ─────────────────────────────────────────────────────────


class TestActivityBusStats:
    @pytest.mark.asyncio
    async def test_stats_initial(self):
        bus = ActivityBus(capacity=100)
        stats = await bus.get_stats()
        assert stats["total_events"] == 0
        assert stats["buffer_size"] == 0
        assert stats["buffer_capacity"] == 100
        assert stats["events_per_minute"] == 0.0
        assert stats["by_type"] == {}

    @pytest.mark.asyncio
    async def test_stats_after_events(self):
        bus = ActivityBus(capacity=100)
        await bus.emit(EventType.WRITE, "w1")
        await bus.emit(EventType.WRITE, "w2")
        await bus.emit(EventType.RECALL, "r1")

        stats = await bus.get_stats()
        assert stats["total_events"] == 3
        assert stats["buffer_size"] == 3
        assert stats["by_type"]["write"] == 2
        assert stats["by_type"]["recall"] == 1

    @pytest.mark.asyncio
    async def test_stats_rate_calculation(self):
        """events_per_minute should be positive after emitting events."""
        bus = ActivityBus(capacity=100)
        # Hack the start time to be 60s ago so rate is calculable
        bus._started_at = time.time() - 60
        await bus.emit(EventType.WRITE, "w1")
        await bus.emit(EventType.WRITE, "w2")

        stats = await bus.get_stats()
        assert stats["events_per_minute"] > 0

    @pytest.mark.asyncio
    async def test_stats_type_counts_survive_eviction(self):
        """Type counters should keep total count even after ring buffer eviction."""
        bus = ActivityBus(capacity=3)
        for _ in range(10):
            await bus.emit(EventType.WRITE, "w")

        stats = await bus.get_stats()
        assert stats["total_events"] == 10  # total counter, not buffer size
        assert stats["buffer_size"] == 3    # ring buffer capped
        assert stats["by_type"]["write"] == 10


# ── Singleton ─────────────────────────────────────────────────────────────────


class TestSingleton:
    def test_get_activity_bus_returns_same_instance(self):
        """get_activity_bus() should return the same singleton."""
        bus1 = get_activity_bus()
        bus2 = get_activity_bus()
        assert bus1 is bus2

    def test_singleton_is_activity_bus(self):
        bus = get_activity_bus()
        assert isinstance(bus, ActivityBus)
