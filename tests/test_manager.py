"""Tests for engram.manager — MemoryManager with mocked PostgresStore.

All methods are async. The store is an AsyncMock so no real DB is needed.
"""

from __future__ import annotations

from datetime import datetime, timezone
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
    RememberResult,
    Session,
)
from epimneme.manager import _group_count


# ── Helper factories ─────────────────────────────────────────────────────────


def _mem(content="test", kind=MemoryKind.FACT, **kw) -> Memory:
    return Memory(kind=kind, content=content, **kw)


def _proj(name="test-proj") -> Project:
    return Project(name=name, description="test", path="/tmp")


def _ent(name="auth.py", kind=EntityKind.FILE) -> Entity:
    return Entity(name=name, kind=kind)


# ── Projects ─────────────────────────────────────────────────────────────────


class TestProjects:
    @pytest.mark.asyncio
    async def test_create_project_new(self, mock_manager, mock_store):
        mock_store.get_project.return_value = None
        proj = _proj("new-proj")
        mock_store.create_project.return_value = None
        mock_store.track_entity.return_value = _ent("new-proj", EntityKind.PROJECT)

        result = await mock_manager.create_project("new-proj", path="/tmp")
        assert result.name == "new-proj"
        mock_store.create_project.assert_awaited_once()
        mock_store.track_entity.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_create_project_existing(self, mock_manager, mock_store):
        """If project already exists, return it without creating."""
        existing = _proj("existing")
        mock_store.get_project.return_value = existing

        result = await mock_manager.create_project("existing")
        assert result.name == "existing"
        mock_store.create_project.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_list_projects(self, mock_manager, mock_store):
        mock_store.list_projects.return_value = [_proj("a"), _proj("b")]
        result = await mock_manager.list_projects()
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_get_project(self, mock_manager, mock_store):
        mock_store.get_project.return_value = _proj("x")
        result = await mock_manager.get_project("x")
        assert result.name == "x"

    @pytest.mark.asyncio
    async def test_project_status_not_found(self, mock_manager, mock_store):
        mock_store.get_project.return_value = None
        result = await mock_manager.project_status("nonexistent")
        assert "error" in result

    @pytest.mark.asyncio
    async def test_project_status_success(self, mock_manager, mock_store):
        proj = _proj("real")
        mock_store.get_project.return_value = proj
        mock_store.get_last_session.return_value = None
        mock_store.get_memory_count.return_value = 42
        mock_store.list_entities.return_value = [_ent("a"), _ent("b")]
        mock_store.get_memories_by_kind.return_value = []

        result = await mock_manager.project_status("real")
        assert result["memory_count"] == 42
        assert result["entity_count"] == 2


# ── Sessions ─────────────────────────────────────────────────────────────────


class TestSessions:
    @pytest.mark.asyncio
    async def test_session_start_no_project(self, mock_manager, mock_store):
        mock_store.create_session.return_value = None
        mock_store.get_previous_session.return_value = None
        mock_store.get_memories_by_kind.return_value = []

        bundle = await mock_manager.session_start()
        assert isinstance(bundle, ContextBundle)
        assert bundle.session_id  # Should have a session ID

    @pytest.mark.asyncio
    async def test_session_start_with_project(self, mock_manager, mock_store):
        proj = _proj("my-proj")
        mock_store.get_project.return_value = proj
        mock_store.create_session.return_value = None
        mock_store.get_previous_session.return_value = None
        mock_store.get_memories_by_kind.return_value = []
        mock_store.list_entities.return_value = []
        mock_store.get_relationships_batch.return_value = {}

        bundle = await mock_manager.session_start(project_name="my-proj")
        assert bundle.project == proj

    @pytest.mark.asyncio
    async def test_session_start_autocreates_project(self, mock_manager, mock_store):
        """If project doesn't exist, session_start should auto-create it."""
        mock_store.get_project.return_value = None
        mock_store.create_project.return_value = None
        mock_store.track_entity.return_value = _ent("new", EntityKind.PROJECT)
        mock_store.create_session.return_value = None
        mock_store.get_previous_session.return_value = None
        mock_store.get_memories_by_kind.return_value = []
        mock_store.list_entities.return_value = []

        bundle = await mock_manager.session_start(project_name="new-proj")
        # create_project should have been called (auto-creation)
        mock_store.create_project.assert_awaited()

    @pytest.mark.asyncio
    async def test_session_end_success(self, mock_manager, mock_store):
        session = Session(project_id="p1", task="fix bug")
        mock_store.get_session_by_id.return_value = session
        mock_store.end_session.return_value = None

        result = await mock_manager.session_end("fake-sid", summary="Done")
        assert "ended" in result

    @pytest.mark.asyncio
    async def test_session_end_not_found(self, mock_manager, mock_store):
        mock_store.get_session_by_id.return_value = None
        result = await mock_manager.session_end("bad-id", summary="Done")
        assert "not found" in result

    @pytest.mark.asyncio
    async def test_session_end_already_ended(self, mock_manager, mock_store):
        session = Session(project_id="p1", task="task")
        session.ended_at = datetime.now(timezone.utc)
        mock_store.get_session_by_id.return_value = session
        result = await mock_manager.session_end(session.id, summary="Done")
        assert "already ended" in result


# ── Remember / Recall / Forget ───────────────────────────────────────────────


class TestMemoryCRUD:
    @pytest.mark.asyncio
    async def test_remember_basic(self, mock_manager, mock_store):
        mock_store.find_similar_by_simhash.return_value = []
        mock_store.store_memory.return_value = None

        result = await mock_manager.remember(
            content="Test fact",
            kind="fact",
        )
        assert isinstance(result, RememberResult)
        assert result.memory.content == "Test fact"
        assert result.memory.kind == MemoryKind.FACT
        mock_store.store_memory.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_remember_with_project(self, mock_manager, mock_store):
        proj = _proj("my-proj")
        mock_store.get_project.return_value = proj
        mock_store.find_similar_by_simhash.return_value = []
        mock_store.store_memory.return_value = None

        result = await mock_manager.remember(
            content="Project fact", kind="fact", project_name="my-proj"
        )
        assert isinstance(result, RememberResult)
        assert result.memory.project_id == proj.id

    @pytest.mark.asyncio
    async def test_remember_dedup_detected(self, mock_manager, mock_store):
        """If a near-duplicate exists, return a dedup message string."""
        existing = _mem("Existing fact")
        mock_store.find_similar_by_simhash.return_value = [existing]

        result = await mock_manager.remember(content="Existing fact copy")
        assert isinstance(result, str)
        assert "Near-duplicate" in result

    @pytest.mark.asyncio
    async def test_remember_with_related_entities(self, mock_manager, mock_store):
        mock_store.find_similar_by_simhash.return_value = []
        mock_store.store_memory.return_value = None
        mock_store.link_memory_to_entities.return_value = None

        result = await mock_manager.remember(
            content="Auth uses JWT", related_to=["auth", "jwt"]
        )
        assert isinstance(result, RememberResult)
        mock_store.link_memory_to_entities.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_recall_empty(self, mock_manager, mock_store):
        mock_store.search_fulltext.return_value = []
        mock_store.search_semantic.return_value = []

        results = await mock_manager.recall("anything")
        assert results == []

    @pytest.mark.asyncio
    async def test_recall_merges_fulltext_and_dedup(self, mock_manager, mock_store):
        mem = _mem("Found item")
        mr = MemoryResult(memory=mem, score=0.8, source="fulltext")

        mock_store.search_fulltext.return_value = [mr]
        mock_store.search_semantic.return_value = []

        results = await mock_manager.recall("Found")
        assert len(results) == 1
        assert results[0].memory.content == "Found item"

    @pytest.mark.asyncio
    async def test_recall_semantic_and_fulltext_merge(self, mock_manager, mock_store):
        """Same memory found by both should get higher RRF score than single-source."""
        mem = _mem("Merged item")
        mr_semantic = MemoryResult(memory=mem, score=0.7, source="semantic")
        mr_fulltext = MemoryResult(memory=mem, score=0.6, source="fulltext")

        mock_store.search_semantic.return_value = [mr_semantic]
        mock_store.search_fulltext.return_value = [mr_fulltext]
        mock_store.update_decay_on_access.return_value = None
        mock_store.log_access.return_value = None

        results = await mock_manager.recall("merged")
        assert len(results) == 1
        # RRF: item in both lists at rank 1 gets 2 × 1/(60+1) ≈ 0.0328
        # Score should be higher than single-source RRF (1/(60+1) ≈ 0.0164)
        single_rrf = 1.0 / (60 + 1)
        assert results[0].score > single_rrf

    @pytest.mark.asyncio
    async def test_forget_not_found(self, mock_manager, mock_store):
        mock_store.get_memory.return_value = None
        result = await mock_manager.forget("nonexistent-id")
        assert "not found" in result

    @pytest.mark.asyncio
    async def test_forget_success(self, mock_manager, mock_store):
        mock_store.get_memory.return_value = _mem("Old fact")
        mock_store.mark_obsolete.return_value = None

        result = await mock_manager.forget("some-id", reason="outdated")
        assert "obsolete" in result
        mock_store.mark_obsolete.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_update_memory(self, mock_manager, mock_store):
        updated = _mem("Updated content")
        updated.version = 2
        mock_store.update_memory.return_value = updated

        result = await mock_manager.update_memory("mem-id", content="Updated content")
        assert result.version == 2

    @pytest.mark.asyncio
    async def test_get_memory_versions(self, mock_manager, mock_store):
        v1 = _mem("v1")
        v1.version = 1
        v2 = _mem("v2")
        v2.version = 2
        mock_store.get_memory_versions.return_value = [v1, v2]

        versions = await mock_manager.get_memory_versions("mem-id")
        assert len(versions) == 2


# ── Entities ─────────────────────────────────────────────────────────────────


class TestEntities:
    @pytest.mark.asyncio
    async def test_track_entity(self, mock_manager, mock_store):
        expected = _ent("server.py")
        mock_store.track_entity.return_value = expected

        result = await mock_manager.track_entity(name="server.py", kind="file")
        assert result.name == "server.py"

    @pytest.mark.asyncio
    async def test_relate_entities(self, mock_manager, mock_store):
        mock_store.relate.return_value = None
        result = await mock_manager.relate_entities("a", "uses", "b")
        assert "a" in result and "b" in result and "uses" in result

    @pytest.mark.asyncio
    async def test_explore_entity(self, mock_manager, mock_store):
        ent = _ent("auth")
        rel = Relationship(from_entity="auth", to_entity="db", relation="uses")
        mock_store.explore.return_value = [EntityResult(entity=ent, relationships=[rel])]

        results = await mock_manager.explore_entity("auth")
        assert len(results) == 1
        assert results[0].entity.name == "auth"


# ── Context ──────────────────────────────────────────────────────────────────


class TestGetContext:
    @pytest.mark.asyncio
    async def test_get_context_empty(self, mock_manager, mock_store):
        mock_store.search_fulltext.return_value = []
        mock_store.search_semantic.return_value = []
        mock_store.get_memories_by_kind.return_value = []
        mock_store.explore.return_value = []
        mock_store.get_memories_for_entity.return_value = []

        bundle = await mock_manager.get_context("test query")
        assert isinstance(bundle, ContextBundle)

    @pytest.mark.asyncio
    async def test_get_context_with_project(self, mock_manager, mock_store):
        proj = _proj("ctx-proj")
        mock_store.get_project.return_value = proj
        mock_store.search_fulltext.return_value = []
        mock_store.search_semantic.return_value = []
        mock_store.get_memories_by_kind.return_value = []
        mock_store.explore.return_value = []
        mock_store.get_memories_for_entity.return_value = []

        bundle = await mock_manager.get_context("query", project_name="ctx-proj")
        assert bundle.project == proj


# ── Stats ────────────────────────────────────────────────────────────────────


class TestStats:
    @pytest.mark.asyncio
    async def test_stats(self, mock_manager, mock_store):
        mock_store.get_memory_count.return_value = 100
        mock_store.get_vector_count.return_value = 80
        mock_store.count_entities.return_value = 2
        mock_store.count_projects.return_value = 1

        stats = await mock_manager.stats()
        assert stats["total_memories"] == 100
        assert stats["total_vectors"] == 80
        assert stats["total_entities"] == 2
        assert stats["total_projects"] == 1
        assert stats["embeddings_enabled"] is False  # mock_config has it off
        assert "dedup" in stats
        assert stats["dedup"]["remember_calls"] == 0
        assert stats["dedup"]["simhash_blocked"] == 0
        assert stats["dedup"]["semantic_blocked"] == 0
        assert stats["dedup"]["conflicts_surfaced"] == 0


# ── _group_count ─────────────────────────────────────────────────────────────


class TestGroupCount:
    def test_empty(self):
        assert _group_count([], lambda x: x) == {}

    def test_simple(self):
        items = [_ent("a", EntityKind.FILE), _ent("b", EntityKind.FILE), _ent("c", EntityKind.MODULE)]
        result = _group_count(items, lambda e: e.kind.value)
        assert result == {"file": 2, "module": 1}


# ── Semantic Dedup + Conflict Surfacing ──────────────────────────────────────


class TestSemanticDedup:
    @pytest.mark.asyncio
    async def test_semantic_dedup_returns_message(self, mock_manager, mock_store, mock_config):
        """When a semantic near-duplicate is found, return a message string."""
        mock_config.semantic_dedup_enabled = True
        mock_config.embeddings_enabled = True

        existing = _mem("The database uses port 5432")
        match = MemoryResult(memory=existing, score=0.95, source="semantic_dedup")

        mock_store.find_similar_by_simhash.return_value = []
        mock_store.find_semantic_duplicates.return_value = [match]

        # Patch _embed to return a fake vector
        mock_manager._embed = AsyncMock(return_value=[0.1] * 384)

        result = await mock_manager.remember(
            content="PostgreSQL listens on port 5432",
            kind="fact",
        )
        assert isinstance(result, str)
        assert "Semantic near-duplicate" in result
        assert "supersedes" in result
        assert mock_manager._counter_semantic_dedup == 1

    @pytest.mark.asyncio
    async def test_semantic_dedup_skipped_when_disabled(self, mock_manager, mock_store, mock_config):
        """Semantic dedup should not fire when disabled in config."""
        mock_config.semantic_dedup_enabled = False
        mock_config.embeddings_enabled = True

        mock_store.find_similar_by_simhash.return_value = []
        mock_store.store_memory.return_value = None

        mock_manager._embed = AsyncMock(return_value=[0.1] * 384)

        result = await mock_manager.remember(content="Some fact")
        assert isinstance(result, RememberResult)
        mock_store.find_semantic_duplicates.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_semantic_dedup_skipped_when_superseding(self, mock_manager, mock_store, mock_config):
        """When supersedes is set, skip semantic dedup (explicit replacement)."""
        mock_config.semantic_dedup_enabled = True
        mock_config.embeddings_enabled = True

        mock_store.find_similar_by_simhash.return_value = []
        mock_store.store_memory.return_value = None

        mock_manager._embed = AsyncMock(return_value=[0.1] * 384)

        result = await mock_manager.remember(
            content="Updated fact",
            supersedes="old-id",
        )
        assert isinstance(result, RememberResult)
        mock_store.find_semantic_duplicates.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_semantic_dedup_skipped_without_embedding(self, mock_manager, mock_store, mock_config):
        """If embedding generation fails/disabled, skip semantic dedup."""
        mock_config.semantic_dedup_enabled = True

        mock_store.find_similar_by_simhash.return_value = []
        mock_store.store_memory.return_value = None

        mock_manager._embed = AsyncMock(return_value=None)

        result = await mock_manager.remember(content="Fact without embedding")
        assert isinstance(result, RememberResult)
        mock_store.find_semantic_duplicates.assert_not_awaited()


class TestConflictSurfacing:
    @pytest.mark.asyncio
    async def test_conflicts_returned_for_facts(self, mock_manager, mock_store, mock_config):
        """Storing a fact should check for potential conflicts."""
        mock_config.embeddings_enabled = True

        conflicting = _mem("We use React for the frontend", kind=MemoryKind.FACT)
        match = MemoryResult(memory=conflicting, score=0.85, source="conflict")

        mock_store.find_similar_by_simhash.return_value = []
        mock_store.find_semantic_duplicates.return_value = []
        mock_store.find_potential_conflicts.return_value = [match]
        mock_store.store_memory.return_value = None

        mock_manager._embed = AsyncMock(return_value=[0.1] * 384)

        result = await mock_manager.remember(
            content="We use Vue for the frontend",
            kind="fact",
        )
        assert isinstance(result, RememberResult)
        assert len(result.potential_conflicts) == 1
        assert result.potential_conflicts[0].memory.content == "We use React for the frontend"
        assert mock_manager._counter_conflicts_surfaced == 1

    @pytest.mark.asyncio
    async def test_conflicts_returned_for_decisions(self, mock_manager, mock_store, mock_config):
        """Storing a decision should also check for potential conflicts."""
        mock_config.embeddings_enabled = True

        conflicting = _mem("Chose PostgreSQL for storage", kind=MemoryKind.DECISION)
        match = MemoryResult(memory=conflicting, score=0.82, source="conflict")

        mock_store.find_similar_by_simhash.return_value = []
        mock_store.find_semantic_duplicates.return_value = []
        mock_store.find_potential_conflicts.return_value = [match]
        mock_store.store_memory.return_value = None

        mock_manager._embed = AsyncMock(return_value=[0.1] * 384)

        result = await mock_manager.remember(
            content="Switching to SQLite for storage",
            kind="decision",
        )
        assert isinstance(result, RememberResult)
        assert len(result.potential_conflicts) == 1

    @pytest.mark.asyncio
    async def test_no_conflicts_for_observations(self, mock_manager, mock_store, mock_config):
        """Non-fact/decision kinds should not trigger conflict surfacing."""
        mock_config.embeddings_enabled = True

        mock_store.find_similar_by_simhash.return_value = []
        mock_store.find_semantic_duplicates.return_value = []
        mock_store.store_memory.return_value = None

        mock_manager._embed = AsyncMock(return_value=[0.1] * 384)

        result = await mock_manager.remember(
            content="The code is well-structured",
            kind="observation",
        )
        assert isinstance(result, RememberResult)
        assert len(result.potential_conflicts) == 0
        mock_store.find_potential_conflicts.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_no_conflicts_without_embedding(self, mock_manager, mock_store, mock_config):
        """No conflict check when embeddings are unavailable."""
        mock_store.find_similar_by_simhash.return_value = []
        mock_store.store_memory.return_value = None

        mock_manager._embed = AsyncMock(return_value=None)

        result = await mock_manager.remember(
            content="Some decision",
            kind="decision",
        )
        assert isinstance(result, RememberResult)
        assert len(result.potential_conflicts) == 0
        mock_store.find_potential_conflicts.assert_not_awaited()
