"""Lightweight in-memory rate limiter middleware for FastAPI.

Token-bucket algorithm per client IP.  No external dependencies.

Configuration via environment variables:
  EPIMNEME_RATE_LIMIT_RPM      - requests per minute per IP  (default: 120)
  EPIMNEME_RATE_LIMIT_BURST    - burst bucket size            (default: 30)
  EPIMNEME_RATE_LIMIT_ENABLED  - "0" to disable              (default: "1")

Exempt paths: /health, /sse, /messages (SSE/MCP need persistent connections).

Implemented as pure ASGI middleware (not starlette.middleware.base.BaseHTTPMiddleware).
BaseHTTPMiddleware buffers/reconstructs every response -- including ones on an
"exempt" early-return path -- via its call_next()/body_stream() machinery, which
is incompatible with long-lived SSE streams and was the root cause of
intermittent "Unexpected message: http.response.start" ASGI assertion crashes
on /sse. Pure ASGI middleware lets exempt paths pass straight through to the
app with zero response wrapping.
"""

from __future__ import annotations

import os
import time
from collections import defaultdict

from starlette.responses import JSONResponse

# -- Configuration ------------------------------------------------------------

RATE_LIMIT_RPM = int(os.environ.get("EPIMNEME_RATE_LIMIT_RPM", "120"))
RATE_LIMIT_BURST = int(os.environ.get("EPIMNEME_RATE_LIMIT_BURST", "30"))
RATE_LIMIT_ENABLED = os.environ.get("EPIMNEME_RATE_LIMIT_ENABLED", "1") == "1"

# Paths exempt from rate limiting (health checks, SSE streams)
_EXEMPT_PREFIXES = ("/health", "/sse", "/messages")


class _TokenBucket:
    """Simple token-bucket for one client."""

    __slots__ = ("tokens", "last_refill", "capacity", "rate")

    def __init__(self, capacity: int, rate: float) -> None:
        self.capacity = capacity
        self.tokens = float(capacity)
        self.rate = rate  # tokens per second
        self.last_refill = time.monotonic()

    def consume(self) -> bool:
        """Try to consume one token.  Returns True if allowed."""
        now = time.monotonic()
        elapsed = now - self.last_refill
        self.last_refill = now
        self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
        if self.tokens >= 1.0:
            self.tokens -= 1.0
            return True
        return False


class RateLimitMiddleware:
    """Per-IP rate limiter using token-bucket algorithm (pure ASGI).

    Attach to a FastAPI app:
        app.add_middleware(RateLimitMiddleware)
    """

    def __init__(self, app, rpm: int = RATE_LIMIT_RPM, burst: int = RATE_LIMIT_BURST):
        self.app = app
        self._rpm = rpm
        self._burst = burst
        self._rate = rpm / 60.0  # tokens per second
        self._buckets: dict[str, _TokenBucket] = defaultdict(
            lambda: _TokenBucket(capacity=burst, rate=self._rate)
        )
        self._last_cleanup = time.monotonic()

    def _client_ip(self, scope, headers: dict[bytes, bytes]) -> str:
        """Extract client IP, respecting X-Forwarded-For from Traefik."""
        forwarded = headers.get(b"x-forwarded-for")
        if forwarded:
            return forwarded.decode("latin-1").split(",")[0].strip()
        client = scope.get("client")
        return client[0] if client else "unknown"

    def _cleanup_stale_buckets(self) -> None:
        """Periodically drop buckets that haven't been used in 10 minutes."""
        now = time.monotonic()
        if now - self._last_cleanup < 300:  # check every 5 min
            return
        self._last_cleanup = now
        stale_threshold = now - 600  # 10 min inactive
        stale_keys = [
            k for k, b in self._buckets.items() if b.last_refill < stale_threshold
        ]
        for k in stale_keys:
            del self._buckets[k]

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http" or not RATE_LIMIT_ENABLED:
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        if any(path.startswith(p) for p in _EXEMPT_PREFIXES):
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers") or [])
        ip = self._client_ip(scope, headers)
        bucket = self._buckets[ip]

        self._cleanup_stale_buckets()

        if not bucket.consume():
            response = JSONResponse(
                status_code=429,
                content={
                    "error": "Rate limit exceeded",
                    "detail": f"Maximum {self._rpm} requests per minute. Try again shortly.",
                },
                headers={
                    "Retry-After": str(int(60 / max(self._rate, 0.01))),
                    "X-RateLimit-Limit": str(self._rpm),
                },
            )
            await response(scope, receive, send)
            return

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                message.setdefault("headers", []).append(
                    (b"x-ratelimit-limit", str(self._rpm).encode("latin-1"))
                )
            await send(message)

        await self.app(scope, receive, send_wrapper)
