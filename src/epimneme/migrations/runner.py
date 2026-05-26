"""Lightweight forward-only migration runner (async).

Usage:
    from epimneme.migrations.runner import MigrationRunner
    runner = MigrationRunner(pool)
    await runner.run_pending()

Migrations are Python modules in this package named NNN_description.py,
each exposing an `async def up(conn)` function.
"""

from __future__ import annotations

import importlib
import logging
import pkgutil
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from psycopg_pool import AsyncConnectionPool

logger = logging.getLogger(__name__)


class MigrationRunner:
    """Run forward-only numbered migrations (async)."""

    def __init__(self, pool: "AsyncConnectionPool") -> None:
        self.pool = pool

    async def _ensure_table(self) -> None:
        """Create the schema_migrations tracking table if needed."""
        async with self.pool.connection() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version     INTEGER PRIMARY KEY,
                    name        TEXT NOT NULL,
                    applied_at  TIMESTAMPTZ DEFAULT NOW()
                )
            """)
            await conn.commit()

    async def _applied_versions(self) -> set[int]:
        """Get the set of already-applied migration versions."""
        async with self.pool.connection() as conn:
            cur = await conn.execute(
                "SELECT version FROM schema_migrations"
            )
            rows = await cur.fetchall()
        return {r["version"] for r in rows}

    def _discover_migrations(self) -> list[tuple[int, str, object]]:
        """Discover migration modules in this package, sorted by version."""
        import epimneme.migrations as pkg

        migrations: list[tuple[int, str, object]] = []
        for importer, name, ispkg in pkgutil.iter_modules(pkg.__path__):
            if name.startswith("_") or ispkg:
                continue
            parts = name.split("_", 1)
            if not parts[0].isdigit():
                continue
            version = int(parts[0])
            module = importlib.import_module(f"engram.migrations.{name}")
            if not hasattr(module, "up"):
                logger.warning(f"Migration {name} has no up() function — skipping")
                continue
            migrations.append((version, name, module))

        migrations.sort(key=lambda m: m[0])
        return migrations

    async def run_pending(self) -> int:
        """Run all pending migrations. Returns count applied."""
        await self._ensure_table()
        applied = await self._applied_versions()
        migrations = self._discover_migrations()
        count = 0

        for version, name, module in migrations:
            if version in applied:
                continue
            logger.info(f"Running migration {version}: {name}")
            try:
                async with self.pool.connection() as conn:
                    await module.up(conn)
                    await conn.execute(
                        "INSERT INTO schema_migrations (version, name) VALUES (%s, %s)",
                        (version, name),
                    )
                    await conn.commit()
                count += 1
                logger.info(f"Migration {version} applied successfully")
            except Exception:
                logger.exception(f"Migration {version} ({name}) FAILED")
                raise

        if count == 0:
            logger.info("No pending migrations")
        else:
            logger.info(f"Applied {count} migration(s)")
        return count
