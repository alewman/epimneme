"""Add decay, versioning, and deduplication columns to memories table.

For existing databases upgrading to v0.4.0.  New installs get these
columns from _init_schema so this migration is a no-op for them.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Columns to add with their definitions (idempotent via IF NOT EXISTS pattern)
_NEW_COLUMNS = [
    ("version", "INTEGER DEFAULT 1"),
    ("version_of", "TEXT"),
    ("simhash", "BIGINT"),
    ("storage_strength", "REAL DEFAULT 0.0"),
    ("retrieval_strength", "REAL DEFAULT 1.0"),
    ("access_count", "INTEGER DEFAULT 0"),
    ("last_accessed", "TIMESTAMPTZ"),
]


async def up(conn) -> None:
    """Add new columns to the memories table if they don't exist."""
    for col_name, col_def in _NEW_COLUMNS:
        # PostgreSQL doesn't have ADD COLUMN IF NOT EXISTS before v9.6,
        # but we're on 16+ so this is safe
        await conn.execute(f"""
            DO $$
            BEGIN
                ALTER TABLE memories ADD COLUMN {col_name} {col_def};
            EXCEPTION
                WHEN duplicate_column THEN NULL;
            END $$;
        """)
        logger.debug(f"Ensured column memories.{col_name}")

    # Index on simhash for dedup lookups
    await conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_memories_simhash
        ON memories (simhash)
        WHERE simhash IS NOT NULL
    """)

    # Index on version_of for version chain lookups
    await conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_memories_version_of
        ON memories (version_of)
        WHERE version_of IS NOT NULL
    """)

    logger.info("Added decay/versioning/dedup columns to memories table")
