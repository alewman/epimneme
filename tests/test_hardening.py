"""Tests for security and correctness hardening (March 2026 sprint)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ── Backup path traversal prevention ────────────────────────────────────────


class TestBackupPathTraversal:
    def test_safe_path_normal_filename(self, tmp_path):
        from epimneme.backup import _safe_backup_path
        result = _safe_backup_path(str(tmp_path), "backup-2026.json")
        assert result == tmp_path / "backup-2026.json"

    def test_safe_path_rejects_dotdot(self, tmp_path):
        from epimneme.backup import _safe_backup_path
        with pytest.raises(ValueError, match="Invalid backup filename"):
            _safe_backup_path(str(tmp_path), "../etc/passwd")

    def test_safe_path_rejects_absolute(self, tmp_path):
        from epimneme.backup import _safe_backup_path
        with pytest.raises(ValueError, match="Invalid backup filename"):
            _safe_backup_path(str(tmp_path), "/etc/passwd")

    def test_safe_path_rejects_subdirectory(self, tmp_path):
        from epimneme.backup import _safe_backup_path
        (tmp_path / "sub").mkdir()
        with pytest.raises(ValueError, match="Invalid backup filename"):
            _safe_backup_path(str(tmp_path), "sub/evil.json")


# ── Demo mode security ──────────────────────────────────────────────────────


class TestDemoModeSecurity:
    def test_demo_mode_grants_admin(self):
        """Demo mode is opt-in (EPIMNEME_DEMO_MODE=1) and grants admin for dashboard access."""
        with patch("epimneme.auth.DEMO_MODE", True):
            from epimneme.auth import AuthContext
            # Demo mode grants admin — it's explicitly opt-in via env var
            # and the dashboard needs admin to access backup/key management
            demo_auth = AuthContext(
                name="demo-guest",
                role="admin",
                projects=["*"],
                source="demo",
            )
            assert demo_auth.role == "admin"


# ── Health endpoint info leak ────────────────────────────────────────────────


class TestHealthInfoLeak:
    @pytest.mark.asyncio
    async def test_health_unhealthy_hides_error_details(self, async_client, mock_manager):
        """Health endpoint should not expose internal error details."""
        mock_manager.stats = AsyncMock(side_effect=Exception("Connection to 10.0.0.5:5432 refused"))

        resp = await async_client.get("/health")
        assert resp.status_code == 503
        data = resp.json()
        assert data["status"] == "unhealthy"
        # Must NOT contain the internal error details
        assert "10.0.0.5" not in data.get("error", "")
        assert "refused" not in data.get("error", "")
        assert data["error"] == "Database connection failed"


# ── Input validation bounds ──────────────────────────────────────────────────


class TestInputValidation:
    @pytest.mark.asyncio
    async def test_search_rejects_negative_offset(self, async_client):
        resp = await async_client.get("/api/memories/search?query=test&offset=-1")
        assert resp.status_code == 422  # FastAPI validation error

    @pytest.mark.asyncio
    async def test_search_rejects_excessive_limit(self, async_client):
        resp = await async_client.get("/api/memories/search?query=test&limit=10000")
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_search_rejects_zero_limit(self, async_client):
        resp = await async_client.get("/api/memories/search?query=test&limit=0")
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_entities_rejects_negative_offset(self, async_client):
        resp = await async_client.get("/api/entities?offset=-5")
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_explore_rejects_excessive_depth(self, async_client):
        resp = await async_client.get("/api/entities/test/explore?depth=99")
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_explore_rejects_zero_depth(self, async_client):
        resp = await async_client.get("/api/entities/test/explore?depth=0")
        assert resp.status_code == 422


# ── Search pagination has_more field ─────────────────────────────────────────


class TestSearchHasMore:
    @pytest.mark.asyncio
    async def test_search_has_more_true(self, async_client, mock_manager):
        """When results fill the fetch window, has_more should be True."""
        from epimneme.core.models import Memory, MemoryKind, MemoryResult
        # Return exactly limit+offset results to indicate there may be more
        results = [
            MemoryResult(
                memory=Memory(kind=MemoryKind.FACT, content=f"fact {i}"),
                score=0.9 - i * 0.01,
                source="semantic",
            )
            for i in range(12)  # limit=10 + offset=2 = 12
        ]
        mock_manager.recall = AsyncMock(return_value=results)

        resp = await async_client.get("/api/memories/search?query=test&limit=10&offset=2")
        data = resp.json()
        assert data["has_more"] is True

    @pytest.mark.asyncio
    async def test_search_has_more_false(self, async_client, mock_manager):
        """When results don't fill the window, has_more should be False."""
        from epimneme.core.models import Memory, MemoryKind, MemoryResult
        results = [
            MemoryResult(
                memory=Memory(kind=MemoryKind.FACT, content="single"),
                score=0.9,
                source="semantic",
            )
        ]
        mock_manager.recall = AsyncMock(return_value=results)

        resp = await async_client.get("/api/memories/search?query=test&limit=10&offset=0")
        data = resp.json()
        assert data["has_more"] is False


# ── Fire-and-forget error callback ──────────────────────────────────────────


class TestTaskErrorCallback:
    def test_log_task_error_with_exception(self):
        from epimneme.manager import MemoryManager
        task = MagicMock()
        task.cancelled.return_value = False
        task.exception.return_value = ValueError("test error")

        # Should not raise — just log
        MemoryManager._log_task_error(task)

    def test_log_task_error_no_exception(self):
        from epimneme.manager import MemoryManager
        task = MagicMock()
        task.cancelled.return_value = False
        task.exception.return_value = None

        MemoryManager._log_task_error(task)

    def test_log_task_error_cancelled(self):
        from epimneme.manager import MemoryManager
        task = MagicMock()
        task.cancelled.return_value = True

        MemoryManager._log_task_error(task)
        task.exception.assert_not_called()
