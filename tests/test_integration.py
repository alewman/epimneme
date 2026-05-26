"""Integration tests against a live PostgreSQL database.

These tests run INSIDE the Docker container where epimneme-db is reachable.
They create a separate 'epimneme_test' database, exercise real SQL (including
pgvector + pg_trgm), and tear down afterwards.

To run:
    docker exec engram python -m pytest tests/test_integration.py -x -v

Skipped automatically when PG is unreachable (e.g. running locally).
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone

import psycopg
import pytest
import pytest_asyncio

from epimneme.core.models import (
    Entity,
    EntityKind,
    Memory,
    MemoryKind,
    Project,
    Relationship,
    Session,
)
from epimneme.stores.postgresql import PostgresStore

# ── Connection helpers ───────────────────────────────────────────────────────

_PG_HOST = os.environ.get("EPIMNEME_PG_HOST", "epimneme-db")
_PG_PORT = os.environ.get("EPIMNEME_PG_PORT", "5432")
_PG_USER = os.environ.get("EPIMNEME_PG_USER", "epimneme")
_PG_PASS = os.environ.get("EPIMNEME_PG_PASSWORD", "epimneme")
_TEST_DB = "epimneme_test"

_ADMIN_DSN = f"postgresql://{_PG_USER}:{_PG_PASS}@{_PG_HOST}:{_PG_PORT}/postgres"
_TEST_DSN = f"postgresql://{_PG_USER}:{_PG_PASS}@{_PG_HOST}:{_PG_PORT}/{_TEST_DB}"


def _pg_available() -> bool:
    """Return True if PostgreSQL is reachable."""
    try:
        conn = psycopg.connect(_ADMIN_DSN, autocommit=True)
        conn.close()
        return True
    except Exception:
        return False


# Skip the entire module if PG is unreachable
pytestmark = [
    pytest.mark.skipif(not _pg_available(), reason="PostgreSQL not reachable"),
    pytest.mark.asyncio,
    pytest.mark.integration,
]


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest_asyncio.fixture(scope="module")
async def store():
    """Create a fresh test database with schema, yield a live PostgresStore,
    then drop the DB on teardown."""
    # Create test database
    conn = psycopg.connect(_ADMIN_DSN, autocommit=True)
    try:
        conn.execute(f"DROP DATABASE IF EXISTS {_TEST_DB}")
        conn.execute(f"CREATE DATABASE {_TEST_DB}")
    finally:
        conn.close()

    # Open store (runs _init_schema → creates all tables + indexes)
    s = PostgresStore(dsn=_TEST_DSN, embedding_dim=384, min_pool=1, max_pool=4)
    await s.open()
    yield s
    await s.close()

    # Drop test database
    conn = psycopg.connect(_ADMIN_DSN, autocommit=True)
    try:
        conn.execute(f"DROP DATABASE IF EXISTS {_TEST_DB}")
    finally:
        conn.close()


@pytest_asyncio.fixture(autouse=True)
async def _clean_tables(store: PostgresStore):
    """Truncate all data tables between tests to isolate state."""
    yield
    async with store.pool.connection() as conn:
        await conn.execute("""
            TRUNCATE memory_entities, memory_access, relationships,
                     memories, sessions, entities, projects, api_keys
            CASCADE
        """)
        await conn.commit()


def _uid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(timezone.utc)


# ── Projects ─────────────────────────────────────────────────────────────────


class TestProjectsCRUD:
    async def test_create_and_get(self, store: PostgresStore):
        proj = Project(name="acme", path="/code/acme", description="Main app")
        created = await store.create_project(proj)
        assert created.name == "acme"
        assert created.id  # auto-generated

        fetched = await store.get_project("acme")
        assert fetched is not None
        assert fetched.id == created.id
        assert fetched.path == "/code/acme"

    async def test_list_projects(self, store: PostgresStore):
        await store.create_project(Project(name="alpha"))
        await store.create_project(Project(name="beta"))
        projects = await store.list_projects()
        names = {p.name for p in projects}
        assert "alpha" in names
        assert "beta" in names

    async def test_count_projects(self, store: PostgresStore):
        assert await store.count_projects() == 0
        await store.create_project(Project(name="one"))
        await store.create_project(Project(name="two"))
        assert await store.count_projects() == 2

    async def test_get_nonexistent(self, store: PostgresStore):
        assert await store.get_project("nope") is None


# ── Sessions ─────────────────────────────────────────────────────────────────


class TestSessionsCRUD:
    async def test_create_and_end(self, store: PostgresStore):
        proj = await store.create_project(Project(name="sess-proj"))
        sid = _uid()
        session = Session(id=sid, project_id=proj.id, task="fix bug")
        await store.create_session(session)

        fetched = await store.get_session_by_id(sid)
        assert fetched is not None
        assert fetched.task == "fix bug"
        assert fetched.ended_at is None

        await store.end_session(sid, summary="Fixed it", handoff="Deploy next")
        ended = await store.get_session_by_id(sid)
        assert ended.summary == "Fixed it"
        assert ended.handoff == "Deploy next"
        assert ended.ended_at is not None

    async def test_last_session(self, store: PostgresStore):
        proj = await store.create_project(Project(name="ls-proj"))
        s1 = Session(id=_uid(), project_id=proj.id, task="first")
        s2 = Session(id=_uid(), project_id=proj.id, task="second")
        await store.create_session(s1)
        await store.create_session(s2)

        last = await store.get_last_session(proj.id)
        assert last is not None
        assert last.task == "second"


# ── Memories ─────────────────────────────────────────────────────────────────


class TestMemoryCRUD:
    async def test_store_and_get(self, store: PostgresStore):
        mem = Memory(kind=MemoryKind.FACT, content="Python uses GIL", subject="python")
        stored = await store.store_memory(mem)
        assert stored.id
        assert stored.kind == MemoryKind.FACT

        fetched = await store.get_memory(stored.id)
        assert fetched is not None
        assert fetched.content == "Python uses GIL"
        assert fetched.subject == "python"

    async def test_store_with_embedding(self, store: PostgresStore):
        mem = Memory(kind=MemoryKind.FACT, content="Vectors work")
        embedding = [0.1] * 384
        stored = await store.store_memory(mem, embedding=embedding)
        assert stored.id

        # Verify vector was stored
        vec_count = await store.get_vector_count()
        assert vec_count >= 1

    async def test_mark_obsolete(self, store: PostgresStore):
        mem = Memory(kind=MemoryKind.FACT, content="Temp fact")
        stored = await store.store_memory(mem)

        await store.mark_obsolete(stored.id)
        fetched = await store.get_memory(stored.id)
        assert fetched.obsolete is True

    async def test_hard_delete(self, store: PostgresStore):
        mem = Memory(kind=MemoryKind.FACT, content="Delete me")
        stored = await store.store_memory(mem)
        assert await store.hard_delete_memory(stored.id) is True
        assert await store.get_memory(stored.id) is None

    async def test_memory_count(self, store: PostgresStore):
        assert await store.get_memory_count() == 0
        await store.store_memory(Memory(kind=MemoryKind.FACT, content="One"))
        await store.store_memory(Memory(kind=MemoryKind.DECISION, content="Two"))
        assert await store.get_memory_count() == 2

    async def test_memory_count_by_project(self, store: PostgresStore):
        p1 = await store.create_project(Project(name="proj-count"))
        await store.store_memory(Memory(kind=MemoryKind.FACT, content="In p1", project_id=p1.id))
        await store.store_memory(Memory(kind=MemoryKind.FACT, content="No proj"))
        assert await store.get_memory_count(p1.id) == 1
        assert await store.get_memory_count() == 2

    async def test_get_memories_by_kind(self, store: PostgresStore):
        await store.store_memory(Memory(kind=MemoryKind.FACT, content="A fact"))
        await store.store_memory(Memory(kind=MemoryKind.DECISION, content="A decision"))
        await store.store_memory(Memory(kind=MemoryKind.FACT, content="Another fact"))

        facts = await store.get_memories_by_kind(MemoryKind.FACT)
        assert len(facts) == 2
        decisions = await store.get_memories_by_kind(MemoryKind.DECISION)
        assert len(decisions) == 1


# ── Memory Updates + Versioning ──────────────────────────────────────────────


class TestMemoryVersioning:
    async def test_update_creates_version(self, store: PostgresStore):
        mem = Memory(kind=MemoryKind.FACT, content="v1 content")
        stored = await store.store_memory(mem)

        updated = await store.update_memory(
            stored.id, content="v2 content", subject="updated"
        )
        assert updated.content == "v2 content"
        assert updated.version == 2

        versions = await store.get_memory_versions(stored.id)
        assert len(versions) == 2  # v1 snapshot + v2 current
        contents = {v.content for v in versions}
        assert "v1 content" in contents
        assert "v2 content" in contents

    async def test_update_with_embedding(self, store: PostgresStore):
        emb1 = [0.1] * 384
        mem = Memory(kind=MemoryKind.FACT, content="has vector")
        stored = await store.store_memory(mem, embedding=emb1)

        emb2 = [0.2] * 384
        updated = await store.update_memory(stored.id, content="new vec", embedding=emb2)
        assert updated.content == "new vec"


# ── Search ───────────────────────────────────────────────────────────────────


class TestSearch:
    async def test_fulltext_search(self, store: PostgresStore):
        await store.store_memory(Memory(kind=MemoryKind.FACT, content="PostgreSQL runs on port 5432"))
        await store.store_memory(Memory(kind=MemoryKind.FACT, content="Redis runs on port 6379"))

        results = await store.search_fulltext("PostgreSQL")
        assert len(results) >= 1
        assert "PostgreSQL" in results[0].memory.content

    async def test_fulltext_no_results(self, store: PostgresStore):
        await store.store_memory(Memory(kind=MemoryKind.FACT, content="Hello world"))
        results = await store.search_fulltext("zzzznonexistent")
        assert len(results) == 0

    async def test_semantic_search(self, store: PostgresStore):
        """Semantic search requires real embeddings — verify it doesn't crash."""
        emb = [0.5] * 384
        await store.store_memory(
            Memory(kind=MemoryKind.FACT, content="The sky is blue"),
            embedding=emb,
        )
        query_emb = [0.5] * 384
        results = await store.search_semantic(query_emb, limit=5)
        assert len(results) >= 1
        assert results[0].memory.content == "The sky is blue"
        assert results[0].score > 0.0

    async def test_semantic_search_respects_project_filter(self, store: PostgresStore):
        p1 = await store.create_project(Project(name="search-proj"))
        emb = [0.3] * 384
        await store.store_memory(
            Memory(kind=MemoryKind.FACT, content="In project", project_id=p1.id),
            embedding=emb,
        )
        await store.store_memory(
            Memory(kind=MemoryKind.FACT, content="No project"),
            embedding=[0.3] * 384,
        )

        results = await store.search_semantic(emb, project_id=p1.id, limit=10)
        assert all(r.memory.project_id == p1.id for r in results)

    async def test_trigram_search(self, store: PostgresStore):
        await store.store_memory(Memory(kind=MemoryKind.FACT, content="authentication middleware handles JWT tokens"))
        results = await store._search_trigram("authentcation middlewre", limit=5)
        # Trigram search is fuzzy — should find the right memory
        assert len(results) >= 1


# ── Entities + Relationships ─────────────────────────────────────────────────


class TestEntitiesAndGraph:
    async def test_track_and_get_entity(self, store: PostgresStore):
        ent = Entity(name="auth.py", kind=EntityKind.FILE, project_id=None)
        created = await store.track_entity(ent)
        assert created.id
        assert created.name == "auth.py"

        fetched = await store.get_entity("auth.py")
        assert fetched is not None
        assert fetched.id == created.id

    async def test_track_entity_upsert(self, store: PostgresStore):
        ent1 = Entity(name="mod.py", kind=EntityKind.FILE, properties={"v": 1})
        created = await store.track_entity(ent1)
        ent2 = Entity(name="mod.py", kind=EntityKind.FILE, properties={"v": 2})
        updated = await store.track_entity(ent2)
        assert updated.id == created.id  # Same entity, updated

    async def test_list_and_count_entities(self, store: PostgresStore):
        await store.track_entity(Entity(name="a.py", kind=EntityKind.FILE))
        await store.track_entity(Entity(name="auth", kind=EntityKind.MODULE))
        assert await store.count_entities() == 2

        entities = await store.list_entities()
        assert len(entities) == 2

        files = await store.list_entities(kind=EntityKind.FILE)
        assert len(files) == 1

    async def test_relate_entities(self, store: PostgresStore):
        a = await store.track_entity(Entity(name="server.py", kind=EntityKind.FILE))
        b = await store.track_entity(Entity(name="auth.py", kind=EntityKind.FILE))

        rel = Relationship(from_entity="server.py", to_entity="auth.py", relation="imports")
        await store.relate(rel)

        rels = await store._get_relationships_for(a.id)
        assert len(rels) == 1
        assert rels[0].relation == "imports"
        assert rels[0].to_entity == "auth.py"

    async def test_get_relationships_batch(self, store: PostgresStore):
        a = await store.track_entity(Entity(name="x.py", kind=EntityKind.FILE))
        b = await store.track_entity(Entity(name="y.py", kind=EntityKind.FILE))
        c = await store.track_entity(Entity(name="z.py", kind=EntityKind.FILE))

        await store.relate(Relationship(from_entity="x.py", to_entity="y.py", relation="imports"))
        await store.relate(Relationship(from_entity="x.py", to_entity="z.py", relation="imports"))
        await store.relate(Relationship(from_entity="y.py", to_entity="z.py", relation="uses"))

        batch = await store.get_relationships_batch([a.id, b.id, c.id])
        assert len(batch[a.id]) == 2  # x imports y and z
        assert len(batch[b.id]) == 1  # y uses z
        assert len(batch[c.id]) == 0  # z has no outgoing

    async def test_explore(self, store: PostgresStore):
        a = await store.track_entity(Entity(name="root", kind=EntityKind.MODULE))
        b = await store.track_entity(Entity(name="child1", kind=EntityKind.FILE))
        c = await store.track_entity(Entity(name="child2", kind=EntityKind.FILE))

        await store.relate(Relationship(from_entity="root", to_entity="child1", relation="contains"))
        await store.relate(Relationship(from_entity="root", to_entity="child2", relation="contains"))

        results = await store.explore("root", depth=1, direction="outgoing")
        names = {r.entity.name for r in results}
        assert "child1" in names
        assert "child2" in names
        assert "root" not in names  # Excludes the pivot

    async def test_link_memory_to_entities(self, store: PostgresStore):
        mem = Memory(kind=MemoryKind.FACT, content="About auth and config")
        stored = await store.store_memory(mem)

        linked = await store.link_memory_to_entities(
            stored.id, ["auth-module", "config-file"]
        )
        assert linked == 2

        # Entities should have been auto-created
        assert await store.get_entity("auth-module") is not None
        assert await store.get_entity("config-file") is not None

        # Memory should be linked
        entities = await store.get_entities_for_memory(stored.id)
        names = {e.name for e in entities}
        assert "auth-module" in names
        assert "config-file" in names


# ── Memory Access + Decay ────────────────────────────────────────────────────


class TestAccessAndDecay:
    async def test_log_access(self, store: PostgresStore):
        mem = Memory(kind=MemoryKind.FACT, content="Track access")
        stored = await store.store_memory(mem)

        await store.log_access(stored.id, context="test")
        # Access log is write-only from store perspective, just verify no crash

    async def test_update_decay_on_access(self, store: PostgresStore):
        mem = Memory(kind=MemoryKind.FACT, content="Decay test")
        stored = await store.store_memory(mem)

        await store.update_decay_on_access(
            stored.id, new_storage=1.0, new_retrieval=0.9, new_count=1
        )
        fetched = await store.get_memory(stored.id)
        assert fetched.access_count == 1
        assert fetched.last_accessed is not None

        await store.update_decay_on_access(
            stored.id, new_storage=1.5, new_retrieval=0.8, new_count=2
        )
        fetched2 = await store.get_memory(stored.id)
        assert fetched2.access_count == 2


# ── SimHash Dedup ────────────────────────────────────────────────────────────


class TestSimHashDedup:
    async def test_find_similar_by_simhash(self, store: PostgresStore):
        m1 = Memory(kind=MemoryKind.FACT, content="Test1", simhash=0b1010101010101010)
        m2 = Memory(kind=MemoryKind.FACT, content="Test2", simhash=0b1010101010101011)  # 1 bit diff
        m3 = Memory(kind=MemoryKind.FACT, content="Test3", simhash=0b0000000000000000)  # very different
        await store.store_memory(m1)
        await store.store_memory(m2)
        await store.store_memory(m3)

        similar = await store.find_similar_by_simhash(0b1010101010101010, threshold=3)
        hashes = {m.simhash for m in similar}
        assert 0b1010101010101011 in hashes
        # m3 should NOT appear (too different)


# ── API Keys ─────────────────────────────────────────────────────────────────


class TestAPIKeys:
    async def test_create_and_validate(self, store: PostgresStore):
        raw_key = await store.create_api_key(name="test-key", role="agent", projects=["proj-a"])
        assert raw_key.startswith("engram_")

        info = await store.validate_api_key(raw_key)
        assert info is not None
        assert info["name"] == "test-key"
        assert info["role"] == "agent"
        assert "proj-a" in info["projects"]

    async def test_validate_invalid_key(self, store: PostgresStore):
        info = await store.validate_api_key("ek_bogus_key_that_does_not_exist")
        assert info is None

    async def test_list_keys(self, store: PostgresStore):
        await store.create_api_key(name="k1", role="admin")
        await store.create_api_key(name="k2", role="agent")
        keys = await store.list_api_keys()
        names = {k["name"] for k in keys}
        assert "k1" in names
        assert "k2" in names

    async def test_revoke_key(self, store: PostgresStore):
        raw = await store.create_api_key(name="revocable", role="agent")
        assert await store.revoke_api_key("revocable") is True

        # Should no longer validate
        assert await store.validate_api_key(raw) is None

    async def test_cycle_key(self, store: PostgresStore):
        old_raw = await store.create_api_key(name="cyclable", role="agent")
        new_raw = await store.cycle_api_key("cyclable")
        assert new_raw is not None
        assert new_raw != old_raw

        # Old key shouldn't work, new key should
        assert await store.validate_api_key(old_raw) is None
        assert await store.validate_api_key(new_raw) is not None


# ── Detailed Stats ───────────────────────────────────────────────────────────


class TestDetailedStats:
    async def test_returns_all_fields(self, store: PostgresStore):
        """Verify the consolidated stats query returns all expected fields."""
        stats = await store.get_detailed_stats()
        expected_keys = {
            "total_memories", "total_vectors", "total_entities",
            "total_projects", "total_sessions", "active_api_keys",
            "obsolete_memories",
        }
        assert expected_keys.issubset(set(stats.keys()))

    async def test_counts_are_accurate(self, store: PostgresStore):
        proj = await store.create_project(Project(name="stats-proj"))
        await store.create_session(Session(id=_uid(), project_id=proj.id, task="t"))
        await store.store_memory(Memory(kind=MemoryKind.FACT, content="m1"))
        m2 = await store.store_memory(Memory(kind=MemoryKind.FACT, content="m2"))
        await store.mark_obsolete(m2.id)
        await store.track_entity(Entity(name="e1", kind=EntityKind.CONCEPT))
        await store.create_api_key(name="sk", role="agent")

        stats = await store.get_detailed_stats()
        assert stats["total_memories"] == 1  # Only non-obsolete
        assert stats["obsolete_memories"] == 1
        assert stats["total_projects"] == 1
        assert stats["total_sessions"] == 1
        assert stats["total_entities"] == 1
        assert stats["active_api_keys"] == 1


# ── Purge + GC ───────────────────────────────────────────────────────────────


class TestPurge:
    async def test_purge_obsolete(self, store: PostgresStore):
        m1 = await store.store_memory(Memory(kind=MemoryKind.FACT, content="Keep"))
        m2 = await store.store_memory(Memory(kind=MemoryKind.FACT, content="Gone"))
        await store.mark_obsolete(m2.id)

        purged = await store.purge_obsolete_memories()
        assert purged == 1
        assert await store.get_memory(m1.id) is not None
        assert await store.get_memory(m2.id) is None
