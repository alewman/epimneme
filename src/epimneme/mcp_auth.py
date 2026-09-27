"""Bearer authentication for the MCP transport paths.

The FastMCP transport is mounted as a sub-application, so it does not run the
FastAPI dependency that guards `/api/*`. Until this existed, `GET /sse`
returned 200 and issued a session id to *anyone* who asked — the only thing
protecting it in production was a Traefik rule matching
`HeadersRegexp(Authorization, Bearer .+)`, i.e. a routing rule doing security's
job. Anything already inside the network (another container, any process on the
host) could open an MCP session and call tools.

This middleware enforces the same Bearer check the REST API uses, in the app,
so the transport is safe regardless of what sits in front of it.
"""

from __future__ import annotations

import json
import logging

logger = logging.getLogger("engram.mcp_auth")

# Paths the MCP transport owns. Everything else is handled by FastAPI routes
# with their own dependencies.
MCP_PREFIXES = ("/sse", "/messages")

_UNAUTHORIZED = json.dumps({
    "error": "Authentication required",
    "detail": "The MCP transport requires a Bearer token. "
              "Send 'Authorization: Bearer <api-key>'.",
}).encode()


class MCPAuthMiddleware:
    """Require a valid Bearer token on the MCP transport paths.

    Pure ASGI, like RateLimitMiddleware: BaseHTTPMiddleware buffers responses
    and breaks the SSE stream this is protecting.
    """

    def __init__(self, app, enabled: bool = True):
        self.app = app
        self.enabled = enabled

    @staticmethod
    def _bearer(headers: dict[bytes, bytes]) -> str | None:
        raw = headers.get(b"authorization")
        if not raw:
            return None
        value = raw.decode("latin-1")
        scheme, _, token = value.partition(" ")
        if scheme.lower() != "bearer" or not token.strip():
            return None
        return token.strip()

    async def _deny(self, send) -> None:
        await send({
            "type": "http.response.start",
            "status": 401,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(_UNAUTHORIZED)).encode("latin-1")),
                (b"www-authenticate", b"Bearer"),
            ],
        })
        await send({"type": "http.response.body", "body": _UNAUTHORIZED})

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http" or not self.enabled:
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        if not any(path == p or path.startswith(p + "/") or path.startswith(p + "?")
                   for p in MCP_PREFIXES):
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers") or [])
        token = self._bearer(headers)
        if not token:
            await self._deny(send)
            return

        from epimneme.auth import _resolve_bearer_token
        try:
            ctx = await _resolve_bearer_token(token)
        except Exception:
            logger.exception("MCP auth check failed")
            ctx = None
        if ctx is None:
            await self._deny(send)
            return

        await self.app(scope, receive, send)
