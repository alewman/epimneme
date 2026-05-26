"""Tests for offset-based pagination on list/search endpoints."""

from __future__ import annotations

import pytest

from epimneme.core.models import Entity, EntityKind, Memory, MemoryKind, MemoryResult


# ── Entity pagination ────────────────────────────────────────────────────────


class TestEntityPagination:
    @pytest.mark.asyncio
    async def test_entities_returns_pagination_metadata(self, async_client, mock_store):
        mock_store.list_entities.return_value = []
        mock_store.count_entities.return_value = 0

        resp = await async_client.get("/api/entities")
        assert resp.status_code == 200
        body = resp.json()
        assert "total" in body
        assert "limit" in body
        assert "offset" in body
        assert body["total"] == 0
        assert body["limit"] == 200
        assert body["offset"] == 0

    @pytest.mark.asyncio
    async def test_entities_custom_limit_offset(self, async_client, mock_store):
        mock_store.list_entities.return_value = []
        mock_store.count_entities.return_value = 50

        resp = await async_client.get("/api/entities?limit=10&offset=20")
        assert resp.status_code == 200
        body = resp.json()
        assert body["limit"] == 10
        assert body["offset"] == 20
        assert body["total"] == 50

        # Verify the store was called with correct limit/offset
        mock_store.list_entities.assert_awaited_once()
        call_kwargs = mock_store.list_entities.call_args
        assert call_kwargs.kwargs.get("limit") == 10
        assert call_kwargs.kwargs.get("offset") == 20

    @pytest.mark.asyncio
    async def test_entities_with_data(self, async_client, mock_store):
        entities = [
            Entity(name=f"entity-{i}", kind=EntityKind.FILE)
            for i in range(3)
        ]
        mock_store.list_entities.return_value = entities
        mock_store.count_entities.return_value = 25

        resp = await async_client.get("/api/entities?limit=3&offset=0")
        assert resp.status_code == 200
        body = resp.json()
        assert body["count"] == 3
        assert body["total"] == 25
        assert len(body["entities"]) == 3


# ── Search pagination ────────────────────────────────────────────────────────


class TestSearchPagination:
    @pytest.mark.asyncio
    async def test_search_returns_pagination_metadata(self, async_client, mock_store):
        mock_store.search_fulltext.return_value = []
        mock_store.find_similar_by_simhash.return_value = []

        resp = await async_client.get("/api/memories/search?query=test")
        assert resp.status_code == 200
        body = resp.json()
        assert "total" in body
        assert "limit" in body
        assert "offset" in body
        assert "has_more" in body

    @pytest.mark.asyncio
    async def test_search_with_offset(self, async_client, mock_store):
        # Create 5 results
        results = [
            MemoryResult(
                memory=Memory(kind=MemoryKind.FACT, content=f"fact {i}"),
                score=0.9 - i * 0.1,
                source="fulltext",
            )
            for i in range(5)
        ]
        mock_store.search_fulltext.return_value = results
        mock_store.find_similar_by_simhash.return_value = []

        # Request page 2 (offset=2, limit=2)
        resp = await async_client.get("/api/memories/search?query=test&limit=2&offset=2")
        assert resp.status_code == 200
        body = resp.json()
        assert body["count"] == 2
        assert body["offset"] == 2
        assert body["limit"] == 2
        # Should get facts 2 and 3 (0-indexed from the sorted results)
        assert len(body["results"]) == 2


# ── Recent memories (already had offset) ─────────────────────────────────────


class TestRecentPagination:
    @pytest.mark.asyncio
    async def test_recent_returns_pagination_fields(self, async_client, mock_store):
        mock_store.get_recent_memories.return_value = []

        resp = await async_client.get("/api/memories/recent?limit=10&offset=5")
        assert resp.status_code == 200
        body = resp.json()
        assert body["limit"] == 10
        assert body["offset"] == 5
        assert body["count"] == 0
