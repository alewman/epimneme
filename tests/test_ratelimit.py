"""Tests for engram.ratelimit — token-bucket rate limiting middleware."""

from __future__ import annotations

import time
from unittest.mock import MagicMock


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
    def test_client_ip_from_forwarded_for(self):
        """Should extract client IP from X-Forwarded-For header."""
        mw = RateLimitMiddleware(app=MagicMock(), rpm=60, burst=10)
        request = MagicMock()
        request.headers = {"x-forwarded-for": "1.2.3.4, 10.0.0.1"}
        request.client = None
        assert mw._client_ip(request) == "1.2.3.4"

    def test_client_ip_from_client(self):
        """Should fall back to request.client.host."""
        mw = RateLimitMiddleware(app=MagicMock(), rpm=60, burst=10)
        request = MagicMock()
        request.headers = {}
        request.client.host = "192.168.1.1"
        assert mw._client_ip(request) == "192.168.1.1"

    def test_client_ip_unknown(self):
        """No headers and no client → 'unknown'."""
        mw = RateLimitMiddleware(app=MagicMock(), rpm=60, burst=10)
        request = MagicMock()
        request.headers = {}
        request.client = None
        assert mw._client_ip(request) == "unknown"

    def test_cleanup_removes_stale(self):
        """Stale buckets should be removed during cleanup."""
        mw = RateLimitMiddleware(app=MagicMock(), rpm=60, burst=10)
        # Add a bucket and make it stale
        bucket = mw._buckets["1.2.3.4"]
        bucket.last_refill = time.monotonic() - 700  # 11+ min ago
        mw._last_cleanup = time.monotonic() - 400  # force cleanup to run
        mw._cleanup_stale_buckets()
        assert "1.2.3.4" not in mw._buckets
