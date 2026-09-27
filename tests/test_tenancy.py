"""The tenant boundary, against a real database.

These exist because the unit suite cannot see this class of bug: it mocks the
store, and every leak fixed here lived in SQL the mock never ran. Each test
below is a cross-tenant read that used to succeed.

Skipped when PostgreSQL is unreachable, like the rest of the integration tests.
"""

from __future__ import annotations

import os
import uuid

import psycopg
import pytest
import pytest_asyncio

from epimneme.auth import AuthContext
from epimneme.core.models import Entity, EntityKind, Memory, MemoryKind, Project, Session
from epimneme.manager import MemoryManager
from epimneme.migrations.runner import MigrationRunner
from epimneme.stores.postgresql import PostgresStore
from epimneme.tenancy import (
    ALL_OWNERS,
    DEFAULT_OWNER_ID,
    GLOBAL_PROJECT_NAME,
    NO_SUCH_PROJECT,
    global_project_id,
    is_reserved_project_name,
    reset_current_owner,
    set_current_owner,
)

_PG_HOST = os.environ.get("EPIMNEME_PG_HOST", "epimneme-db")
_PG_PORT = os.environ.get("EPIMNEME_PG_PORT", "5432")
_PG_USER = os.environ.get("EPIMNEME_PG_USER", "epimneme")
_PG_PASS = os.environ.get("EPIMNEME_PG_PASSWORD", "epimneme")
_TEST_DB = "epimneme_tenancy_test"

_ADMIN_DSN = f"postgresql://{_PG_USER}:{_PG_PASS}@{_PG_HOST}:{_PG_PORT}/postgres"
_TEST_DSN = f"postgresql://{_PG_USER}:{_PG_PASS}@{_PG_HOST}:{_PG_PORT}/{_TEST_DB}"

OWNER_A = "11111111-1111-1111-1111-111111111111"
OWNER_B = "22222222-2222-2222-2222-222222222222"


def _pg_available() -> bool:
    try:
        psycopg.connect(_ADMIN_DSN, autocommit=True).close()
        return True
    except Exception:
        return False


pytestmark = [
    pytest.mark.skipif(not _pg_available(), reason="PostgreSQL not reachable"),
    pytest.mark.asyncio,
    pytest.mark.integration,
]


@pytest_asyncio.fixture(scope="module")
async def store():
    conn = psycopg.connect(_ADMIN_DSN, autocommit=True)
    try:
        conn.execute(f"DROP DATABASE IF EXISTS {_TEST_DB}")
        conn.execute(f"CREATE DATABASE {_TEST_DB}")
    finally:
        conn.close()

    s = PostgresStore(dsn=_TEST_DSN, embedding_dim=384, min_pool=1, max_pool=4)
    await s.open()
    await MigrationRunner(s.pool).run_pending()
    async with s.pool.connection() as c:
        for oid, name in ((OWNER_A, "tenant-a"), (OWNER_B, "tenant-b")):
            await c.execute(
                "INSERT INTO owners (id, name) VALUES (%s, %s) "
                "ON CONFLICT (id) DO NOTHING",
                (oid, name),
            )
        await c.commit()
    yield s
    await s.close()

    conn = psycopg.connect(_ADMIN_DSN, autocommit=True)
    try:
        conn.execute(f"DROP DATABASE IF EXISTS {_TEST_DB}")
    finally:
        conn.close()


@pytest_asyncio.fixture(autouse=True)
async def _clean(store: PostgresStore):
    token = set_current_owner(DEFAULT_OWNER_ID)
    yield
    reset_current_owner(token)
    async with store.pool.connection() as conn:
        await conn.execute(
            "TRUNCATE memory_entities, memory_access, relationships, "
            "memories, sessions, entities, projects, api_keys CASCADE"
        )
        await conn.commit()


async def _project(store, owner: str, name: str) -> Project:
    """Create ``name`` under ``owner``, whoever is currently being served."""
    return await store.create_project(Project(owner_id=owner, name=name))


async def _memory(store, project_id: str, content: str) -> Memory:
    return await store.store_memory(
        Memory(project_id=project_id, kind=MemoryKind.FACT, content=content)
    )


# ── Names ────────────────────────────────────────────────────────────────────


class TestNameCollision:
    async def test_two_owners_hold_the_same_name(self, store: PostgresStore):
        a = await _project(store, OWNER_A, "peoplesoft")
        b = await _project(store, OWNER_B, "peoplesoft")
        assert a.id != b.id

        assert (await store.get_project("peoplesoft", owner_id=OWNER_A)).id == a.id
        assert (await store.get_project("peoplesoft", owner_id=OWNER_B)).id == b.id

    async def test_name_is_unique_within_an_owner(self, store: PostgresStore):
        first = await _project(store, OWNER_A, "dup")
        second = await store.create_project(Project(owner_id=OWNER_A, name="dup"))
        # Same row back, not a second project shadowing the first.
        assert second.id == first.id
        assert await store.count_projects(owner_id=OWNER_A) == 1

    async def test_lookup_does_not_cross_owners(self, store: PostgresStore):
        await _project(store, OWNER_A, "only-in-a")
        assert await store.get_project("only-in-a", owner_id=OWNER_B) is None

    async def test_listing_does_not_cross_owners(self, store: PostgresStore):
        await _project(store, OWNER_A, "a1")
        await _project(store, OWNER_B, "b1")
        assert {p.name for p in await store.list_projects(owner_id=OWNER_A)} == {"a1"}
        assert {p.name for p in await store.list_projects(owner_id=OWNER_B)} == {"b1"}
        every = {p.name for p in await store.list_projects(owner_id=ALL_OWNERS)}
        assert {"a1", "b1"} <= every


# ── Reads ────────────────────────────────────────────────────────────────────


class TestReadScope:
    async def test_omitting_the_project_stays_inside_the_owner(
        self, store: PostgresStore
    ):
        """The leak this whole change exists for.

        ``project_id=None`` used to append no filter at all, so a caller who
        simply left the project out read every tenant's memories.
        """
        a = await _project(store, OWNER_A, "a-proj")
        b = await _project(store, OWNER_B, "b-proj")
        await _memory(store, a.id, "owner A secret")
        await _memory(store, b.id, "owner B secret")

        token = set_current_owner(OWNER_A)
        try:
            assert await store.get_memory_count() == 1
            rows = await store.get_recent_memories()
            assert [m.content for m in rows] == ["owner A secret"]
        finally:
            reset_current_owner(token)

    async def test_fulltext_search_stays_inside_the_owner(self, store: PostgresStore):
        a = await _project(store, OWNER_A, "a-proj")
        b = await _project(store, OWNER_B, "b-proj")
        await _memory(store, a.id, "the quarterly budget spreadsheet")
        await _memory(store, b.id, "the quarterly budget spreadsheet")

        token = set_current_owner(OWNER_A)
        try:
            hits = await store.search_fulltext("quarterly budget", limit=10)
            assert len(hits) == 1
            assert hits[0].memory.project_id == a.id
        finally:
            reset_current_owner(token)

    async def test_global_memories_are_per_owner(self, store: PostgresStore):
        token = set_current_owner(OWNER_A)
        try:
            gid_a = await store.ensure_global_project()
            assert gid_a == global_project_id(OWNER_A)
            await _memory(store, gid_a, "A unscoped note")
        finally:
            reset_current_owner(token)

        token = set_current_owner(OWNER_B)
        try:
            gid_b = await store.ensure_global_project()
            await _memory(store, gid_b, "B unscoped note")
            rows = await store.get_recent_memories()
            assert [m.content for m in rows] == ["B unscoped note"]
        finally:
            reset_current_owner(token)

    async def test_entities_do_not_cross_owners(self, store: PostgresStore):
        token = set_current_owner(OWNER_A)
        try:
            await store.ensure_global_project()
            await store.track_entity(Entity(name="Acme", kind=EntityKind.CONCEPT))
            assert (await store.get_entity("Acme")) is not None
        finally:
            reset_current_owner(token)

        token = set_current_owner(OWNER_B)
        try:
            await store.ensure_global_project()
            assert (await store.get_entity("Acme")) is None
        finally:
            reset_current_owner(token)

    async def test_sessions_do_not_cross_owners(self, store: PostgresStore):
        a = await _project(store, OWNER_A, "a-proj")
        b = await _project(store, OWNER_B, "b-proj")
        await store.create_session(Session(project_id=a.id, task="A work"))
        await store.create_session(Session(project_id=b.id, task="B work"))

        token = set_current_owner(OWNER_A)
        try:
            assert (await store.get_last_session()).task == "A work"
        finally:
            reset_current_owner(token)


# ── Reserved names ───────────────────────────────────────────────────────────


class TestReservedNames:
    @pytest.mark.parametrize("name", ["*", "__global__", "global-abc", "", "   "])
    def test_rejected(self, name):
        assert is_reserved_project_name(name)

    @pytest.mark.parametrize("name", ["peoplesoft", "rom-farmer", "a*b", "globalish"])
    def test_allowed(self, name):
        assert not is_reserved_project_name(name)

    def test_no_key_can_claim_a_reserved_name(self):
        for role in ("admin", "agent"):
            auth = AuthContext(
                name="k", role=role, projects=["*"], source="api_key"
            )
            assert auth.can_claim_project("*") is False
            assert auth.can_claim_project("__global__") is False
            assert auth.can_claim_project("normal") is True


# ── Keys ─────────────────────────────────────────────────────────────────────


class TestKeyOwnership:
    async def test_validate_returns_the_owner(self, store: PostgresStore):
        raw = await store.create_api_key(
            name=f"k-{uuid.uuid4().hex[:8]}", projects=["p"], owner_id=OWNER_B
        )
        info = await store.validate_api_key(raw)
        assert info["owner_id"] == OWNER_B

    async def test_demotion_drops_the_wildcard(self, store: PostgresStore):
        name = f"k-{uuid.uuid4().hex[:8]}"
        await store.create_api_key(name=name, role="admin")
        assert (await store.list_api_keys(owner_id=ALL_OWNERS))[0]["projects"] == ["*"]

        updated = await store.update_api_key(name=name, role="agent")
        assert updated["projects"] == []

    async def test_rotation_keeps_the_owner(self, store: PostgresStore):
        name = f"k-{uuid.uuid4().hex[:8]}"
        await store.create_api_key(name=name, projects=["p"], owner_id=OWNER_B)
        raw = await store.cycle_api_key(name)
        assert (await store.validate_api_key(raw))["owner_id"] == OWNER_B


# ── Manager-level resolution ─────────────────────────────────────────────────


class TestResolution:
    async def test_unknown_name_matches_nothing_rather_than_everything(
        self, store: PostgresStore
    ):
        """A typo used to widen the search to every project, not narrow it."""
        mgr = MemoryManager.__new__(MemoryManager)
        mgr.store = store
        assert await mgr._resolve_project_id("no-such") == NO_SUCH_PROJECT

    async def test_no_name_resolves_to_the_owners_global_project(
        self, store: PostgresStore
    ):
        mgr = MemoryManager.__new__(MemoryManager)
        mgr.store = store
        token = set_current_owner(OWNER_A)
        try:
            assert await mgr._resolve_project_id(None) == global_project_id(OWNER_A)
        finally:
            reset_current_owner(token)

    async def test_global_project_is_named_consistently(self, store: PostgresStore):
        token = set_current_owner(OWNER_A)
        try:
            gid = await store.ensure_global_project()
            proj = await store.get_project(GLOBAL_PROJECT_NAME, owner_id=OWNER_A)
            assert proj is not None and proj.id == gid
        finally:
            reset_current_owner(token)
