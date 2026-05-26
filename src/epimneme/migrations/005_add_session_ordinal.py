"""Migration 005 — add session_ordinal to sessions table.

Each session within a project gets a monotonically increasing integer
(1, 2, 3, …) assigned at insert time via a per-project sequence counter.
This lets the retrieval layer apply a cheap recency boost without
relying on wall-clock timestamps, which collapse to near-zero during
high-speed benchmark ingest.

The ordinal is NULL for sessions created before this migration; they
are backfilled using started_at ordering within each project.
"""

MIGRATION_VERSION = 5
MIGRATION_NAME = "add_session_ordinal"


async def up(conn) -> None:
    """Add session_ordinal column and backfill existing rows."""

    # 1. Add the column (idempotent)
    await conn.execute("""
        DO $$ BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_name = 'sessions' AND column_name = 'session_ordinal'
            ) THEN
                ALTER TABLE sessions ADD COLUMN session_ordinal INTEGER;
            END IF;
        END $$
    """)

    # 2. Backfill existing rows using ROW_NUMBER over started_at per project.
    #    Sessions with no project get a global ordinal.
    await conn.execute("""
        UPDATE sessions s
        SET session_ordinal = ranked.rn
        FROM (
            SELECT id,
                   ROW_NUMBER() OVER (
                       PARTITION BY COALESCE(project_id, '::global::')
                       ORDER BY started_at ASC
                   ) AS rn
            FROM sessions
        ) ranked
        WHERE s.id = ranked.id
          AND s.session_ordinal IS NULL
    """)

    # 3. Index for fast MAX(session_ordinal) lookups at insert time
    await conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_sessions_ordinal
            ON sessions (project_id, session_ordinal)
    """)

    await conn.commit()
