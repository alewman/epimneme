"""Tests for engram.migrations.runner — async unit tests with mocked pool."""

import pytest
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock
from epimneme.migrations.runner import MigrationRunner


class TestMigrationRunner:
    def _mock_pool(self):
        """Create a mock pool that works with ``async with pool.connection() as conn:``."""
        conn = AsyncMock()

        @asynccontextmanager
        async def _connection():
            yield conn

        pool = MagicMock()
        pool.connection = _connection

        return pool, conn

    @pytest.mark.asyncio
    async def test_ensure_table_creates_schema_migrations(self):
        pool, conn = self._mock_pool()
        runner = MigrationRunner(pool)
        await runner._ensure_table()
        # Should have called execute with CREATE TABLE
        calls = [str(c) for c in conn.execute.call_args_list]
        assert any("schema_migrations" in c for c in calls)
        conn.commit.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_applied_versions_empty(self):
        pool, conn = self._mock_pool()
        cursor = AsyncMock()
        cursor.fetchall.return_value = []
        conn.execute.return_value = cursor

        runner = MigrationRunner(pool)
        versions = await runner._applied_versions()
        assert versions == set()

    @pytest.mark.asyncio
    async def test_applied_versions_populated(self):
        pool, conn = self._mock_pool()
        cursor = AsyncMock()
        cursor.fetchall.return_value = [{"version": 1}, {"version": 2}]
        conn.execute.return_value = cursor

        runner = MigrationRunner(pool)
        versions = await runner._applied_versions()
        assert 1 in versions
        assert 2 in versions

    @pytest.mark.asyncio
    async def test_run_pending_no_migrations(self):
        pool, conn = self._mock_pool()
        cursor = AsyncMock()
        cursor.fetchall.return_value = []
        conn.execute.return_value = cursor

        runner = MigrationRunner(pool)
        runner._discover_migrations = MagicMock(return_value=[])
        count = await runner.run_pending()
        assert count == 0

    @pytest.mark.asyncio
    async def test_run_pending_applies_new_migration(self):
        pool, conn = self._mock_pool()

        cursor = AsyncMock()
        cursor.fetchall.return_value = []
        conn.execute.return_value = cursor

        # Create a fake migration module
        fake_module = MagicMock()
        fake_module.up = AsyncMock()

        runner = MigrationRunner(pool)
        runner._discover_migrations = MagicMock(
            return_value=[(1, "001_initial", fake_module)]
        )

        count = await runner.run_pending()
        assert count == 1
        fake_module.up.assert_awaited_once()
