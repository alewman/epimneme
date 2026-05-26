"""Shared fixtures for the engram test suite.

Provides:
  - mock_store: an AsyncMock of PostgresStore for unit tests
  - mock_manager: a MemoryManager wrapping mock_store (embeddings off)
  - async_client: httpx AsyncClient pointed at the FastAPI app (integration tests)
  - admin_auth / agent_auth: pre-built AuthContext helpers
"""

from __future__ import annotations

import os
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio

# Ensure default_config() does not raise on its password guard during tests.
os.environ.setdefault("EPIMNEME_PG_PASSWORD", "test-password")

from epimneme.auth import AuthContext
from epimneme.core.config import EngramConfig
from epimneme.core.models import (
    Entity,
    EntityKind,
    Memory,
    MemoryKind,
    Project,
    Session,
)


# ── Auth helpers ─────────────────────────────────────────────────────────────


@pytest.fixture
def admin_auth() -> AuthContext:
    return AuthContext(
        name="test-admin",
        role="admin",
        projects=["*"],
        source="oauth",
    )


@pytest.fixture
def agent_auth() -> AuthContext:
    return AuthContext(
        name="test-agent",
        role="agent",
        projects=["proj-a"],
        source="api_key",
        api_key_id="key-123",
    )


# ── Mock store ───────────────────────────────────────────────────────────────


@pytest.fixture
def mock_store() -> AsyncMock:
    """A fully-mocked PostgresStore where every method is an AsyncMock."""
    store = AsyncMock()

    # Sensible defaults for commonly-called methods
    store.get_project.return_value = None
    store.list_projects.return_value = []
    store.get_memory_count.return_value = 0
    store.get_vector_count.return_value = 0
    store.list_entities.return_value = []
    store.count_entities.return_value = 0
    store.count_projects.return_value = 0
    store.get_memories_by_kind.return_value = []
    store.get_last_session.return_value = None
    store.get_previous_session.return_value = None
    store.find_similar_by_simhash.return_value = []
    store.find_semantic_duplicates.return_value = []
    store.find_potential_conflicts.return_value = []
    store.search_semantic.return_value = []
    store.search_fulltext.return_value = []
    store.get_memory.return_value = None
    store.get_entities_for_memory.return_value = []
    store.explore.return_value = []
    store.get_memories_for_entity.return_value = []

    # validate_api_key for auth tests
    store.validate_api_key.return_value = None

    return store


# ── Mock manager ─────────────────────────────────────────────────────────────


@pytest.fixture
def mock_config() -> EngramConfig:
    """Minimal config with embeddings off for unit tests."""
    return EngramConfig(
        pg_host="localhost",
        pg_port=5432,
        pg_user="test",
        pg_password="test",
        pg_database="test_engram",
        embeddings_enabled=False,
        dedup_enabled=True,
        dedup_hamming_threshold=3,
    )


@pytest.fixture
def mock_manager(mock_config, mock_store):
    """A MemoryManager backed by the mock store — no real DB needed."""
    from epimneme.manager import MemoryManager

    mgr = MemoryManager(config=mock_config, store=mock_store)
    return mgr


# ── Sample data factories ────────────────────────────────────────────────────


@pytest.fixture
def sample_memory() -> Memory:
    return Memory(
        kind=MemoryKind.FACT,
        content="PostgreSQL runs on port 5432",
        subject="postgresql",
        tags=["db", "config"],
        confidence=1.0,
    )


@pytest.fixture
def sample_project() -> Project:
    return Project(name="test-project", description="Unit test project", path="/tmp/test")


@pytest.fixture
def sample_entity() -> Entity:
    return Entity(name="auth.py", kind=EntityKind.FILE, properties={"language": "python"})


@pytest.fixture
def sample_session() -> Session:
    return Session(project_id="proj-id-123", task="fix auth bug")


# ── httpx async client for integration tests ────────────────────────────────


@pytest_asyncio.fixture
async def async_client(mock_manager, admin_auth):
    """httpx AsyncClient wired to the FastAPI app with mocked manager + auth.

    All endpoints see admin_auth by default.
    The server uses a module-level _manager global (not a DI dependency),
    so we patch it directly. Auth is overridden via FastAPI dependency_overrides.
    """
    from httpx import ASGITransport, AsyncClient

    import epimneme.server as server_module
    from epimneme.server import app
    from epimneme.auth import get_auth, require_admin

    # Patch the module-level _manager global that get_manager() reads
    original_manager = server_module._manager
    server_module._manager = mock_manager

    # Disable rate limiter for integration tests
    import epimneme.ratelimit as rl_mod
    original_rl = rl_mod.RATE_LIMIT_ENABLED
    rl_mod.RATE_LIMIT_ENABLED = False

    # Override auth to always return admin
    async def _mock_auth():
        return admin_auth

    app.dependency_overrides[get_auth] = _mock_auth
    app.dependency_overrides[require_admin] = _mock_auth

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client

    app.dependency_overrides.clear()
    server_module._manager = original_manager
    rl_mod.RATE_LIMIT_ENABLED = original_rl
