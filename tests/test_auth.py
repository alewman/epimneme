"""Tests for engram.auth — unit tests with mocked store.

_resolve_bearer_token is async, so those tests need @pytest.mark.asyncio.
"""

import pytest
from unittest.mock import AsyncMock
from epimneme.auth import AuthContext, _resolve_bearer_token, set_auth_store


# ── AuthContext (pure sync — no DB) ──────────────────────────────────────────

class TestAuthContext:
    def test_admin_can_access_any_project(self):
        auth = AuthContext(name="admin", role="admin", projects=["*"], source="oauth")
        assert auth.can_access_project("any-project") is True
        assert auth.can_access_project(None) is True

    def test_agent_scoped_to_projects(self):
        auth = AuthContext(
            name="agent1", role="agent", projects=["proj-a", "proj-b"], source="api_key"
        )
        assert auth.can_access_project("proj-a") is True
        assert auth.can_access_project("proj-b") is True
        assert auth.can_access_project("proj-c") is False

    def test_agent_global_access(self):
        auth = AuthContext(name="agent1", role="agent", projects=["proj-a"], source="api_key")
        assert auth.can_access_project(None) is True  # global scope accessible

    def test_wildcard_projects(self):
        auth = AuthContext(name="agent1", role="agent", projects=["*"], source="api_key")
        assert auth.can_access_project("any-project") is True

    def test_enforce_raises_on_denied(self):
        from fastapi import HTTPException
        auth = AuthContext(name="agent1", role="agent", projects=["proj-a"], source="api_key")
        with pytest.raises(HTTPException) as exc_info:
            auth.enforce_project_access("proj-b")
        assert exc_info.value.status_code == 403

    def test_enforce_passes_on_allowed(self):
        auth = AuthContext(name="agent1", role="agent", projects=["proj-a"], source="api_key")
        auth.enforce_project_access("proj-a")  # Should not raise

    def test_can_claim_admin(self):
        auth = AuthContext(name="admin", role="admin", projects=["*"], source="oauth")
        assert auth.can_claim_project("new-project") is True

    def test_can_claim_agent(self):
        auth = AuthContext(name="agent1", role="agent", projects=["proj-a"], source="api_key")
        assert auth.can_claim_project("unclaimed-project") is True

    def test_api_key_id_field(self):
        auth = AuthContext(
            name="agent1", role="agent", projects=["proj-a"],
            source="api_key", api_key_id="key-123"
        )
        assert auth.api_key_id == "key-123"


# ── _resolve_bearer_token (async) ────────────────────────────────────────────

class TestResolveBearerToken:
    def setup_method(self):
        self.mock_store = AsyncMock()
        set_auth_store(self.mock_store)

    def teardown_method(self):
        set_auth_store(None)

    @pytest.mark.asyncio
    async def test_valid_token(self):
        self.mock_store.validate_api_key.return_value = {
            "id": "key-1",
            "name": "test-agent",
            "role": "agent",
            "projects": ["proj-a"],
        }
        auth = await _resolve_bearer_token("epimneme_test_token")
        assert auth is not None
        assert auth.name == "test-agent"
        assert auth.role == "agent"
        assert auth.projects == ["proj-a"]
        assert auth.api_key_id == "key-1"
        assert auth.source == "api_key"

    @pytest.mark.asyncio
    async def test_invalid_token(self):
        self.mock_store.validate_api_key.return_value = None
        auth = await _resolve_bearer_token("bad_token")
        assert auth is None

    @pytest.mark.asyncio
    async def test_no_store(self):
        set_auth_store(None)
        auth = await _resolve_bearer_token("any_token")
        assert auth is None
