"""Tests for engram.server — HTTP integration tests via httpx AsyncClient.

Uses the async_client fixture from conftest.py which wires up:
  - A mock MemoryManager (no real DB)
  - Admin auth by default (overrides get_auth / require_admin)
  - httpx AsyncClient talking to the FastAPI app via ASGI transport
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from epimneme.core.models import (
    ContextBundle,
    Entity,
    EntityKind,
    EntityResult,
    Memory,
    MemoryKind,
    MemoryResult,
    Project,
    Relationship,
)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _mem(content="test", kind=MemoryKind.FACT, **kw) -> Memory:
    return Memory(kind=kind, content=content, **kw)


def _proj(name="test-proj") -> Project:
    return Project(name=name, description="test", path="/tmp")


def _ent(name="auth.py", kind=EntityKind.FILE) -> Entity:
    return Entity(name=name, kind=kind)


# ── Health ───────────────────────────────────────────────────────────────────


class TestHealth:
    @pytest.mark.asyncio
    async def test_health_ok(self, async_client, mock_store):
        mock_store.get_memory_count.return_value = 10
        mock_store.get_vector_count.return_value = 8
        mock_store.list_entities.return_value = []
        mock_store.list_projects.return_value = []

        resp = await async_client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert data["memories"] == 10


# ── Dashboard ────────────────────────────────────────────────────────────────


class TestDashboard:
    @pytest.mark.asyncio
    async def test_dashboard_returns_html(self, async_client):
        resp = await async_client.get("/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]


# ── Sessions API ─────────────────────────────────────────────────────────────


class TestSessionsAPI:
    @pytest.mark.asyncio
    async def test_session_start(self, async_client, mock_manager):
        bundle = ContextBundle(
            session_id="sid-123",
            project=_proj("p"),
        )
        mock_manager.session_start = AsyncMock(return_value=bundle)

        resp = await async_client.post("/api/sessions/start", json={
            "project": "p",
            "task": "fix bugs",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "started"
        assert data["session_id"] == "sid-123"

    @pytest.mark.asyncio
    async def test_session_end(self, async_client, mock_manager):
        mock_manager.session_end = AsyncMock(return_value="Session abc ended")

        resp = await async_client.post("/api/sessions/end", json={
            "session_id": "abc",
            "summary": "Done with task",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ended"


# ── Memories API ─────────────────────────────────────────────────────────────


class TestMemoriesAPI:
    @pytest.mark.asyncio
    async def test_create_memory(self, async_client, mock_manager):
        mem = _mem("New fact")
        from epimneme.core.models import RememberResult
        mock_manager.remember = AsyncMock(return_value=RememberResult(memory=mem))

        resp = await async_client.post("/api/memories", json={
            "content": "New fact",
            "kind": "fact",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["kind"] == "fact"
        assert "id" in data

    @pytest.mark.asyncio
    async def test_create_memory_dedup(self, async_client, mock_manager):
        mock_manager.remember = AsyncMock(return_value="Near-duplicate of memory abc")

        resp = await async_client.post("/api/memories", json={
            "content": "Duplicate fact",
            "kind": "fact",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "deduplicated"

    @pytest.mark.asyncio
    async def test_search_memories(self, async_client, mock_manager):
        mem = _mem("Found result")
        mr = MemoryResult(memory=mem, score=0.9, source="semantic")
        mock_manager.recall = AsyncMock(return_value=[mr])

        resp = await async_client.get("/api/memories/search", params={"query": "test"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 1
        assert data["results"][0]["content"] == "Found result"

    @pytest.mark.asyncio
    async def test_search_deep(self, async_client, mock_manager):
        bundle = ContextBundle()
        mock_manager.get_context = AsyncMock(return_value=bundle)

        resp = await async_client.get("/api/memories/search", params={
            "query": "test",
            "deep": "true",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "context" in data

    @pytest.mark.asyncio
    async def test_update_memory(self, async_client, mock_manager):
        updated = _mem("Updated content")
        updated.version = 2
        updated.version_of = "orig-id"
        mock_manager.update_memory = AsyncMock(return_value=updated)

        resp = await async_client.put("/api/memories/some-id", json={
            "content": "Updated content",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["version"] == 2

    @pytest.mark.asyncio
    async def test_update_memory_not_found(self, async_client, mock_manager):
        mock_manager.update_memory = AsyncMock(return_value=None)

        resp = await async_client.put("/api/memories/bad-id", json={
            "content": "Anything",
        })
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_memory_versions(self, async_client, mock_manager):
        v1 = _mem("v1")
        v1.version = 1
        v2 = _mem("v2")
        v2.version = 2
        mock_manager.get_memory_versions = AsyncMock(return_value=[v1, v2])

        resp = await async_client.get("/api/memories/some-id/versions")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["versions"]) == 2

    @pytest.mark.asyncio
    async def test_delete_memory(self, async_client, mock_manager):
        mock_manager.forget = AsyncMock(return_value="Memory abc marked obsolete")

        resp = await async_client.delete("/api/memories/abc")
        assert resp.status_code == 200
        data = resp.json()
        assert "obsolete" in data["message"]


# ── Projects API ─────────────────────────────────────────────────────────────


class TestProjectsAPI:
    @pytest.mark.asyncio
    async def test_list_projects(self, async_client, mock_manager):
        mock_manager.list_projects = AsyncMock(return_value=[_proj("a"), _proj("b")])

        resp = await async_client.get("/api/projects")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["projects"]) == 2

    @pytest.mark.asyncio
    async def test_create_project(self, async_client, mock_manager):
        proj = _proj("new")
        mock_manager.create_project = AsyncMock(return_value=proj)

        resp = await async_client.post("/api/projects", json={"name": "new"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "new"

    @pytest.mark.asyncio
    async def test_project_status(self, async_client, mock_manager):
        proj = _proj("p")
        mock_manager.get_project = AsyncMock(return_value=proj)
        mock_manager.project_status = AsyncMock(return_value={
            "project": proj.model_dump(),
            "memory_count": 5,
            "entity_count": 3,
            "open_issues": 1,
            "last_session": None,
            "recent_decisions": [],
            "entities_by_kind": {"file": 2, "module": 1},
        })

        resp = await async_client.get("/api/projects/p/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["memory_count"] == 5


# ── Entities API ─────────────────────────────────────────────────────────────


class TestEntitiesAPI:
    @pytest.mark.asyncio
    async def test_list_entities(self, async_client, mock_store):
        mock_store.get_project.return_value = None
        mock_store.list_entities.return_value = [
            _ent("a.py", EntityKind.FILE),
            _ent("b.py", EntityKind.FILE),
        ]

        resp = await async_client.get("/api/entities")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 2

    @pytest.mark.asyncio
    async def test_track_entity(self, async_client, mock_manager):
        ent = _ent("new.py")
        mock_manager.track_entity = AsyncMock(return_value=ent)

        resp = await async_client.post("/api/entities", json={
            "name": "new.py",
            "kind": "file",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "new.py"

    @pytest.mark.asyncio
    async def test_relate_entities(self, async_client, mock_manager):
        mock_manager.relate_entities = AsyncMock(return_value="a --[uses]--> b")

        resp = await async_client.post("/api/entities/relate", json={
            "from_entity": "a",
            "relation": "uses",
            "to_entity": "b",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "uses" in data["message"]

    @pytest.mark.asyncio
    async def test_explore_entity(self, async_client, mock_manager):
        ent = _ent("auth")
        rel = Relationship(from_entity="auth", to_entity="db", relation="uses")
        mock_manager.explore_entity = AsyncMock(
            return_value=[EntityResult(entity=ent, relationships=[rel])]
        )

        resp = await async_client.get("/api/entities/auth/explore")
        assert resp.status_code == 200
        data = resp.json()
        assert data["entity"] == "auth"
        assert len(data["connections"]) == 1


# ── Graph API ────────────────────────────────────────────────────────────────


class TestGraphAPI:
    @pytest.mark.asyncio
    async def test_entity_graph(self, async_client, mock_store):
        mock_store.get_graph_data.return_value = {
            "nodes": [{"id": "1", "name": "auth", "kind": "module"}],
            "edges": [],
        }

        resp = await async_client.get("/api/graph/entities")
        assert resp.status_code == 200
        data = resp.json()
        assert "nodes" in data

    @pytest.mark.asyncio
    async def test_similarity_graph(self, async_client, mock_store):
        mock_store.get_similarity_graph.return_value = {
            "nodes": [],
            "edges": [],
        }

        resp = await async_client.get("/api/graph/similarity")
        assert resp.status_code == 200
        data = resp.json()
        assert "nodes" in data


# ── Stats API ────────────────────────────────────────────────────────────────


class TestStatsAPI:
    @pytest.mark.asyncio
    async def test_stats(self, async_client, mock_manager):
        mock_manager.stats = AsyncMock(return_value={
            "total_memories": 100,
            "total_vectors": 80,
            "total_entities": 50,
            "total_projects": 3,
            "embeddings_enabled": True,
        })
        mock_manager.list_projects = AsyncMock(return_value=[_proj("a")])

        resp = await async_client.get("/api/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_memories"] == 100
        assert "version" in data

    @pytest.mark.asyncio
    async def test_detailed_stats(self, async_client, mock_store):
        mock_store.get_detailed_stats.return_value = {
            "total_memories": 100,
            "memories_by_kind": {"fact": 50, "decision": 30},
        }

        resp = await async_client.get("/api/stats/detailed")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_memories"] == 100


# ── Admin API ────────────────────────────────────────────────────────────────


class TestAdminAPI:
    @pytest.mark.asyncio
    async def test_create_key(self, async_client, mock_store):
        mock_store.create_api_key.return_value = "engram_new_key_abc123"

        resp = await async_client.post("/api/admin/keys", json={
            "name": "test-key",
            "role": "agent",
            "projects": ["proj-a"],
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["key"] == "engram_new_key_abc123"
        assert data["name"] == "test-key"

    @pytest.mark.asyncio
    async def test_list_keys(self, async_client, mock_store):
        mock_store.list_api_keys.return_value = [
            {"name": "k1", "role": "agent", "key_hash": "xxx"},
            {"name": "k2", "role": "admin", "key_hash": "yyy"},
        ]

        resp = await async_client.get("/api/admin/keys")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["keys"]) == 2
        # key_hash should be stripped
        for k in data["keys"]:
            assert "key_hash" not in k

    @pytest.mark.asyncio
    async def test_revoke_key(self, async_client, mock_store):
        mock_store.revoke_api_key.return_value = True

        resp = await async_client.delete("/api/admin/keys/test-key")
        assert resp.status_code == 200
        data = resp.json()
        assert "revoked" in data["message"]

    @pytest.mark.asyncio
    async def test_revoke_key_not_found(self, async_client, mock_store):
        mock_store.revoke_api_key.return_value = False

        resp = await async_client.delete("/api/admin/keys/nonexistent")
        assert resp.status_code == 404
