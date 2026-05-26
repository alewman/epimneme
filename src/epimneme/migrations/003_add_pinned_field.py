"""Add pinned field to memories table.

Pinned memories always appear in session context and are exempt from
garbage collection decay.
"""


async def up(conn) -> None:
    await conn.execute("""
        DO $$ BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_name = 'memories' AND column_name = 'pinned'
            ) THEN
                ALTER TABLE memories ADD COLUMN pinned BOOLEAN DEFAULT FALSE;
            END IF;
        END $$
    """)
    # Partial index for fast pinned-memory lookups
    await conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_memories_pinned
        ON memories(project_id) WHERE pinned = TRUE AND NOT obsolete
    """)
