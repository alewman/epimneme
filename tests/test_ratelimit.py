"""Tests for engram.ratelimit — token-bucket rate limiting middleware."""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest


from epimneme.ratelimit import RateLimitMiddleware, _TokenBucket


# ── TokenBucket ───────────────────────────────────────────────────────────────


class TestTokenBucket:
    def test_initial_capacity(self):
        b = _TokenBucket(capacity=10, rate=2.0)
        assert b.tokens == 10.0
        assert b.capacity == 10
        assert b.rate == 2.0

    def test_consume_decrements(self):
        b = _TokenBucket(capacity=5, rate=1.0)
        assert b.consume() is True
        assert b.tokens < 5.0

    def test_consume_until_empty(self):
        b = _TokenBucket(capacity=3, rate=0.0)  # no refill
        assert b.consume() is True
        assert b.consume() is True
        assert b.consume() is True
        assert b.consume() is False  # exhausted

    def test_refill_over_time(self):
        b = _TokenBucket(capacity=5, rate=100.0)  # fast refill
        # Drain
        for _ in range(5):
            b.consume()
        # Force time forward
        b.last_refill = time.monotonic() - 1.0  # simulate 1s passed
        assert b.consume() is True  # should have refilled

    def test_capacity_cap(self):
        """Tokens should never exceed capacity."""
        b = _TokenBucket(capacity=5, rate=1000.0)
        b.last_refill = time.monotonic() - 100  # long time ago
        b.consume()
        assert b.tokens <= 5.0


# ── RateLimitMiddleware ───────────────────────────────────────────────────────


class TestRateLimitMiddleware:
    # _client_ip takes the raw ASGI (scope, headers) since the pure-ASGI
    # rewrite — no Starlette Request. `headers` is the raw bytes->bytes mapping
    # and `scope["client"]` is a (host, port) tuple or absent.
    def test_client_ip_from_forwarded_for(self):
        """Should extract the first hop from X-Forwarded-For (Traefik)."""
        mw = RateLimitMiddleware(app=MagicMock(), rpm=60, burst=10)
        headers = {b"x-forwarded-for": b"1.2.3.4, 10.0.0.1"}
        assert mw._client_ip({"client": None}, headers) == "1.2.3.4"

    def test_forwarded_for_wins_over_peer(self):
        """Behind Traefik the socket peer is the proxy, so the header wins."""
        mw = RateLimitMiddleware(app=MagicMock(), rpm=60, burst=10)
        headers = {b"x-forwarded-for": b"1.2.3.4"}
        assert mw._client_ip({"client": ("10.0.0.9", 51234)}, headers) == "1.2.3.4"

    def test_client_ip_from_scope_peer(self):
        """With no header, fall back to the ASGI scope's client tuple."""
        mw = RateLimitMiddleware(app=MagicMock(), rpm=60, burst=10)
        assert mw._client_ip({"client": ("192.168.1.1", 51234)}, {}) == "192.168.1.1"

    def test_client_ip_unknown(self):
        """No header and no peer → 'unknown' (one shared bucket)."""
        mw = RateLimitMiddleware(app=MagicMock(), rpm=60, burst=10)
        assert mw._client_ip({"client": None}, {}) == "unknown"
        assert mw._client_ip({}, {}) == "unknown"

    def test_cleanup_removes_stale(self):
        """Stale buckets should be removed during cleanup."""
        mw = RateLimitMiddleware(app=MagicMock(), rpm=60, burst=10)
        # Add a bucket and make it stale
        bucket = mw._buckets["1.2.3.4"]
        bucket.last_refill = time.monotonic() - 700  # 11+ min ago
        mw._last_cleanup = time.monotonic() - 400  # force cleanup to run
        mw._cleanup_stale_buckets()
        assert "1.2.3.4" not in mw._buckets


# ── ASGI call path ────────────────────────────────────────────────────────────
# The pure-ASGI rewrite exists because BaseHTTPMiddleware buffers and rebuilds
# every response through call_next(), which is incompatible with long-lived SSE
# streams and caused intermittent "Unexpected message: http.response.start"
# crashes. None of that was covered: __call__ was never invoked by a test.


class _RecordingApp:
    """Minimal downstream ASGI app that records the send it was handed."""

    def __init__(self, messages=None):
        self.called = False
        self.seen_send = None
        self._messages = messages or [
            {"type": "http.response.start", "status": 200, "headers": []},
            {"type": "http.response.body", "body": b"ok"},
        ]

    async def __call__(self, scope, receive, send):
        self.called = True
        self.seen_send = send
        for m in self._messages:
            await send(dict(m))


def _collect():
    out = []

    async def send(message):
        out.append(message)

    return out, send


@pytest.mark.asyncio
class TestASGICallPath:
    async def test_non_http_scope_passes_through_untouched(self):
        app = _RecordingApp()
        mw = RateLimitMiddleware(app=app, rpm=60, burst=10)
        sent, send = _collect()
        await mw({"type": "lifespan"}, None, send)
        assert app.called
        assert app.seen_send is send, "non-http scopes must not be wrapped at all"

    @pytest.mark.parametrize("path", ["/health", "/sse", "/messages/abc"])
    async def test_exempt_paths_pass_the_original_send(self, path):
        """The exempt early return is the path BaseHTTPMiddleware broke — SSE
        lives here, so the downstream app must get the raw send."""
        app = _RecordingApp()
        mw = RateLimitMiddleware(app=app, rpm=60, burst=10)
        sent, send = _collect()
        await mw({"type": "http", "path": path, "headers": []}, None, send)
        assert app.seen_send is send

    async def test_allowed_request_streams_every_body_chunk_in_order(self):
        """The wrapper may add a header to response.start; it must not buffer,
        reorder or swallow body messages."""
        chunks = [{"type": "http.response.start", "status": 200, "headers": []}] + [
            {"type": "http.response.body", "body": f"chunk{i}".encode(), "more_body": i < 2}
            for i in range(3)
        ]
        app = _RecordingApp(chunks)
        mw = RateLimitMiddleware(app=app, rpm=60, burst=10)
        sent, send = _collect()
        await mw({"type": "http", "path": "/api/x", "headers": []}, None, send)
        bodies = [m["body"] for m in sent if m["type"] == "http.response.body"]
        assert bodies == [b"chunk0", b"chunk1", b"chunk2"]
        start = next(m for m in sent if m["type"] == "http.response.start")
        assert (b"x-ratelimit-limit", b"60") in start["headers"]

    async def test_over_limit_returns_429_and_skips_the_app(self):
        app = _RecordingApp()
        mw = RateLimitMiddleware(app=app, rpm=60, burst=1)
        scope = {"type": "http", "path": "/api/x", "headers": []}
        sent, send = _collect()
        await mw(scope, None, send)          # consumes the only token
        app.called = False
        sent2, send2 = _collect()

        async def receive():
            return {"type": "http.request", "body": b"", "more_body": False}

        await mw(scope, receive, send2)
        start = next(m for m in sent2 if m["type"] == "http.response.start")
        assert start["status"] == 429
        assert not app.called, "a throttled request must not reach the app"

    async def test_forwarded_for_buckets_separately(self):
        """Two IPs behind one proxy must not share a bucket."""
        app = _RecordingApp()
        mw = RateLimitMiddleware(app=app, rpm=60, burst=1)
        for ip in (b"1.1.1.1", b"2.2.2.2"):
            sent, send = _collect()
            await mw({"type": "http", "path": "/api/x",
                      "headers": [(b"x-forwarded-for", ip)]}, None, send)
            start = next(m for m in sent if m["type"] == "http.response.start")
            assert start["status"] == 200, f"{ip!r} was throttled by another IP's bucket"
