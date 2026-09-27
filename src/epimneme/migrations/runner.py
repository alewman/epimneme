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

# Distinct from the schema-init lock in PostgresStore._init_schema (847329001),
# so the two cannot block each other.
_MIGRATION_LOCK_KEY = 847329002


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
            module = importlib.import_module(f"epimneme.migrations.{name}")
            if not hasattr(module, "up"):
                logger.warning(f"Migration {name} has no up() function — skipping")
                continue
            migrations.append((version, name, module))

        migrations.sort(key=lambda m: m[0])
        return migrations

    async def run_pending(self) -> int:
        """Run all pending migrations. Returns count applied.

        Safe to call from every worker at once. The server runs with several
        uvicorn workers, so all of them reach this within the same second; on
        2026-09-27 four of them raced on one ``ALTER TABLE`` and deadlocked,
        and three crash-looped while the fourth finished. Each migration
        therefore runs under a transaction-scoped advisory lock, and re-checks
        whether it is still pending *after* taking it — a worker that queued
        behind another would otherwise re-apply what that one just committed.
        """
        await self._ensure_table()
        migrations = self._discover_migrations()
        count = 0

        for version, name, module in migrations:
            try:
                async with self.pool.connection() as conn:
                    # Released on commit/rollback — no manual unlock needed.
                    await conn.execute("SELECT pg_advisory_xact_lock(%s)",
                                       (_MIGRATION_LOCK_KEY,))
                    cur = await conn.execute(
                        "SELECT 1 FROM schema_migrations WHERE version = %s",
                        (version,),
                    )
                    if await cur.fetchone():
                        continue

                    logger.info(f"Running migration {version}: {name}")
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
