"""Migration 004 — add persistent_memories flag to projects table.

When enabled, all memories in the project skip decay and garbage collection
without needing to individually pin each memory.
"""

MIGRATION_VERSION = 4
MIGRATION_NAME = "add_project_persistent_memories"


async def migrate(conn) -> None:
    """Add persistent_memories column to projects table."""
    await conn.execute("""
        DO $$ BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_name = 'projects' AND column_name = 'persistent_memories'
            ) THEN
                ALTER TABLE projects ADD COLUMN persistent_memories BOOLEAN DEFAULT FALSE;
            END IF;
        END $$
    """)
    await conn.commit()
