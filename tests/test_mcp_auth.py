"""MCP transport auth.

Regression guard for a real hole: the FastMCP app is mounted as a
sub-application, so it never ran the FastAPI dependency guarding /api/*.
`GET /sse` returned 200 and issued a session id to anyone, and the only thing
closing that publicly was a Traefik HeadersRegexp rule — a routing rule doing
security's job, which left the endpoint open to anything inside the network.
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from epimneme.mcp_auth import MCPAuthMiddleware


class _App:
    def __init__(self):
        self.called = False

    async def __call__(self, scope, receive, send):
        self.called = True
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"downstream"})


def _collect():
    out = []

    async def send(m):
        out.append(m)

    return out, send


def _scope(path, headers=None):
    return {"type": "http", "path": path, "headers": list((headers or {}).items())}


def _status(sent):
    return next(m for m in sent if m["type"] == "http.response.start")["status"]


@pytest.mark.asyncio
class TestMCPAuth:
    @pytest.mark.parametrize("path", ["/sse", "/messages/", "/messages/?session_id=x"])
    async def test_mcp_paths_require_a_token(self, path):
        app = _App()
        mw = MCPAuthMiddleware(app)
        sent, send = _collect()
        await mw(_scope(path), None, send)
        assert _status(sent) == 401
        assert not app.called, "an unauthenticated MCP request must not reach the transport"

    async def test_401_body_is_json_not_a_redirect(self):
        """Agents were getting an OAuth HTML page instead of a usable error."""
        app = _App()
        mw = MCPAuthMiddleware(app)
        sent, send = _collect()
        await mw(_scope("/sse"), None, send)
        body = next(m for m in sent if m["type"] == "http.response.body")["body"]
        assert json.loads(body)["error"] == "Authentication required"

    async def test_malformed_authorization_is_rejected(self):
        app = _App()
        mw = MCPAuthMiddleware(app)
        for hdr in (b"Basic abc", b"Bearer", b"Bearer    ", b"token abc"):
            sent, send = _collect()
            await mw(_scope("/sse", {b"authorization": hdr}), None, send)
            assert _status(sent) == 401, f"{hdr!r} should not authenticate"
            assert not app.called

    async def test_invalid_token_is_rejected(self):
        app = _App()
        mw = MCPAuthMiddleware(app)
        sent, send = _collect()
        with patch("epimneme.auth._resolve_bearer_token", AsyncMock(return_value=None)):
            await mw(_scope("/sse", {b"authorization": b"Bearer nope"}), None, send)
        assert _status(sent) == 401
        assert not app.called

    async def test_valid_token_reaches_the_transport(self):
        app = _App()
        mw = MCPAuthMiddleware(app)
        sent, send = _collect()
        with patch("epimneme.auth._resolve_bearer_token", AsyncMock(return_value=object())):
            await mw(_scope("/sse", {b"authorization": b"Bearer good"}), None, send)
        assert app.called
        assert _status(sent) == 200

    @pytest.mark.parametrize("path", ["/api/memories/search", "/health", "/", "/messagesfoo"])
    async def test_non_mcp_paths_are_untouched(self, path):
        """/api has its own dependency; /health is public; a path that merely
        starts with the same letters must not be captured."""
        app = _App()
        mw = MCPAuthMiddleware(app)
        sent, send = _collect()
        await mw(_scope(path), None, send)
        assert app.called

    async def test_disabled_flag_bypasses_entirely(self):
        app = _App()
        mw = MCPAuthMiddleware(app, enabled=False)
        sent, send = _collect()
        await mw(_scope("/sse"), None, send)
        assert app.called

    async def test_non_http_scope_passes_through(self):
        app = _App()
        mw = MCPAuthMiddleware(app)
        sent, send = _collect()
        await mw({"type": "lifespan"}, None, send)
        assert app.called
