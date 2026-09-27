"""Authentication middleware for FastAPI — Bearer token + project scoping.

Two auth modes:
  1. API key (Bearer token) — for agents and programmatic access
  2. OAuth passthrough — for browser users already authenticated by Traefik

When Traefik handles OAuth (chain-oauth middleware), it sets X-Forwarded-User.
For API access (chain-api middleware), the app handles auth via Bearer tokens.

MCP tool functions use get_mcp_auth(ctx) to resolve auth from the MCP Context.
"""

from __future__ import annotations

import hmac
import logging
import os
from dataclasses import dataclass
from typing import Optional

DEMO_MODE = os.environ.get("EPIMNEME_DEMO_MODE", "") == "1"

from fastapi import Depends, HTTPException, Request, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

logger = logging.getLogger(__name__)

# HTTPBearer with auto_error=False so we can fall through to OAuth headers
_bearer_scheme = HTTPBearer(auto_error=False)


@dataclass
class AuthContext:
    """Resolved authentication context attached to each request."""

    name: str
    role: str  # "admin" or "agent"
    projects: list[str]  # ["*"] for admin, specific project names for agents
    source: str  # "api_key" or "oauth"
    api_key_id: Optional[str] = None  # DB id of the API key (for project claiming)

    def can_access_project(self, project_name: Optional[str]) -> bool:
        """Check if this auth context can access a given project."""
        if self.role == "admin" or "*" in self.projects:
            return True
        if project_name is None:
            return True  # Global scope accessible to all for reads
        return project_name in self.projects

    def enforce_project_access(self, project_name: Optional[str]) -> None:
        """Raise 403 if project access is denied."""
        if not self.can_access_project(project_name):
            raise HTTPException(
                status_code=403,
                detail=f"API key '{self.name}' does not have access to project {project_name}",
            )

    def can_claim_project(self, project_name: str) -> bool:
        """Check if this key is allowed to claim new project namespaces.

        Admins can always claim. Agents can claim unclaimed namespaces.
        Global scope (None) cannot be claimed.
        """
        if self.role == "admin" or "*" in self.projects:
            return True
        # Agents can claim any project that isn't already taken
        return True


# Store reference — set during app startup
_store = None

# ── Trusted-proxy configuration ──────────────────────────────────────────────
# X-Forwarded-User is set by an authenticating reverse proxy and, when trusted,
# grants admin with projects=["*"]. Anything able to reach the app directly can
# forge it, so trust is OFF by default: a stock deployment cannot be escalated
# by a header alone, whatever sits in front of it.
#
# Enable with EPIMNEME_TRUST_FORWARDED_USER=1, and set EPIMNEME_PROXY_SECRET so
# the header is only honoured when the proxy also sends a matching
# X-Proxy-Secret. Without that secret, trusting the header is only as strong as
# the guarantee that nothing else can reach the port.
TRUST_FORWARDED_USER = os.environ.get("EPIMNEME_TRUST_FORWARDED_USER", "0") == "1"
PROXY_SECRET = os.environ.get("EPIMNEME_PROXY_SECRET", "")


def _forwarded_user(headers) -> Optional[str]:
    """Return the proxy-asserted user, or None if it must not be trusted.

    `headers` is anything with a case-insensitive .get (Starlette Headers).
    """
    if not TRUST_FORWARDED_USER:
        return None
    user = headers.get("x-forwarded-user")
    if not user:
        return None
    if PROXY_SECRET:
        presented = headers.get("x-proxy-secret") or ""
        if not hmac.compare_digest(presented, PROXY_SECRET):
            logger.warning(
                "X-Forwarded-User present but X-Proxy-Secret missing or wrong — "
                "refusing to grant admin. Source may be bypassing the proxy."
            )
            return None
    return user


def set_auth_store(store) -> None:
    """Set the PostgresStore used for API key validation. Called at startup."""
    global _store
    _store = store


def get_store():
    """Get the auth store (for use in MCP auth)."""
    return _store


async def _resolve_bearer_token(token: str) -> Optional[AuthContext]:
    """Validate a Bearer token and return AuthContext or None."""
    if _store is None:
        return None
    key_info = await _store.validate_api_key(token)
    if not key_info:
        return None
    return AuthContext(
        name=key_info["name"],
        role=key_info["role"],
        projects=key_info["projects"],
        source="api_key",
        api_key_id=key_info["id"],
    )


async def get_auth(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Security(_bearer_scheme),
) -> AuthContext:
    """Resolve authentication from Bearer token or Traefik OAuth header.

    Priority:
    1. Bearer token → validate against api_keys table
    2. X-Forwarded-User header → OAuth user (trusted, set by Traefik)
    3. Reject with 401
    """
    # 1. Bearer token
    if credentials and credentials.credentials:
        if _store is None:
            raise HTTPException(status_code=503, detail="Auth store not initialized")

        auth = await _resolve_bearer_token(credentials.credentials)
        if not auth:
            raise HTTPException(status_code=401, detail="Invalid or expired API key")
        return auth

    # 2. OAuth passthrough (Traefik sets X-Forwarded-User)
    forwarded_user = _forwarded_user(request.headers)
    if forwarded_user:
        return AuthContext(
            name=forwarded_user,
            role="admin",  # OAuth users are admins (they passed Traefik OAuth)
            projects=["*"],
            source="oauth",
        )

    # 3. Demo mode — full dashboard access for local/dev use.
    #    This is opt-in (EPIMNEME_DEMO_MODE=1) and intended for cases where
    #    the dashboard is accessed without OAuth (e.g. direct container port).
    #    In production, Traefik OAuth provides admin via X-Forwarded-User above.
    if DEMO_MODE:
        return AuthContext(
            name="demo-guest",
            role="admin",
            projects=["*"],
            source="demo",
        )

    # 4. No auth
    raise HTTPException(
        status_code=401,
        detail="Authentication required. Provide Bearer token or use OAuth.",
        headers={"WWW-Authenticate": "Bearer"},
    )


async def get_mcp_auth(ctx) -> AuthContext:
    """Extract AuthContext from an MCP Context object.

    Reads the Authorization header from the Starlette request attached
    to the MCP message POST. Falls back to X-Forwarded-User for OAuth.
    Raises ValueError if no valid auth is found.
    """
    request = None
    try:
        request = ctx.request_context.request
    except (AttributeError, TypeError):
        pass

    if request is not None:
        # Try Bearer token
        auth_header = request.headers.get("authorization", "")
        if auth_header.lower().startswith("bearer "):
            token = auth_header[7:].strip()
            auth = await _resolve_bearer_token(token)
            if auth:
                return auth

        # Try OAuth passthrough
        forwarded_user = _forwarded_user(request.headers)
        if forwarded_user:
            return AuthContext(
                name=forwarded_user,
                role="admin",
                projects=["*"],
                source="oauth",
            )

    raise ValueError("MCP authentication failed — no valid Bearer token or OAuth header")


async def require_admin(auth: AuthContext = Depends(get_auth)) -> AuthContext:
    """Dependency that requires admin role."""
    if auth.role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    return auth
