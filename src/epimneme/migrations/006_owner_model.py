"""Give every project an owner, so project names are unique per tenant.

Before this, ``projects.name`` was globally unique: the first key to claim a
name owned it installation-wide, and any other key granted that name read the
same rows. This introduces an ``owners`` table, hangs projects and API keys off
it, and swaps the global name index for a per-owner one.

It also gives the rows that carried ``project_id IS NULL`` a real home — the
owner's ``__global__`` project. That is the point of the migration as much as
the rename: once every memory has a project and every project has an owner, a
scope filter is a single ``project_id IN (owner's projects)`` with no NULL
branch that can fail open.

Everything that exists today belongs to DEFAULT_OWNER_ID. Nothing an existing
key can reach changes.
"""

from __future__ import annotations

import logging

from epimneme.tenancy import (
    DEFAULT_OWNER_ID,
    DEFAULT_OWNER_NAME,
    GLOBAL_PROJECT_NAME,
    LEGACY_GLOBAL_PROJECT_ID,
    RESERVED_PROJECT_NAMES,
    global_project_id,
)

logger = logging.getLogger(__name__)


async def _has_column(conn, table: str, column: str) -> bool:
    cur = await conn.execute(
        "SELECT 1 FROM information_schema.columns "
        "WHERE table_name = %s AND column_name = %s",
        (table, column),
    )
    return bool(await cur.fetchone())


async def up(conn) -> None:
    # ── owners ──
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS owners (
            id          TEXT PRIMARY KEY,
            name        TEXT NOT NULL UNIQUE,
            created_at  TIMESTAMPTZ DEFAULT NOW()
        )
    """)
    await conn.execute(
        "INSERT INTO owners (id, name) VALUES (%s, %s) ON CONFLICT (id) DO NOTHING",
        (DEFAULT_OWNER_ID, DEFAULT_OWNER_NAME),
    )

    # ── projects.owner_id ──
    if not await _has_column(conn, "projects", "owner_id"):
        await conn.execute("ALTER TABLE projects ADD COLUMN owner_id TEXT")
    await conn.execute(
        "UPDATE projects SET owner_id = %s WHERE owner_id IS NULL",
        (DEFAULT_OWNER_ID,),
    )

    # Free the reserved names before the unique index goes on, so nothing can
    # collide with the __global__ project created below. A project literally
    # named "*" existed in production — an agent's wildcard *grant* leaked into
    # a project *claim*, which is the confusion RESERVED_PROJECT_NAMES ends.
    #
    # Empty ones are dropped. Ones holding real rows are renamed, never
    # deleted: the name is the bug, the data under it is not.
    for reserved in sorted(RESERVED_PROJECT_NAMES):
        cur = await conn.execute(
            """SELECT p.id,
                      (SELECT COUNT(*) FROM memories m WHERE m.project_id = p.id)
                    + (SELECT COUNT(*) FROM sessions s WHERE s.project_id = p.id)
                    + (SELECT COUNT(*) FROM entities e WHERE e.project_id = p.id)
                      AS rows_held
               FROM projects p WHERE p.name = %s""",
            (reserved,),
        )
        for row in await cur.fetchall():
            if row["rows_held"] == 0:
                await conn.execute("DELETE FROM projects WHERE id = %s", (row["id"],))
                logger.info("Removed empty project named %r (reserved name)", reserved)
                continue
            renamed = f"reclaimed-{row['id'][:8]}"
            await conn.execute(
                """UPDATE projects
                   SET name = %s,
                       description = COALESCE(description || ' | ', '')
                                     || %s,
                       updated_at = NOW()
                   WHERE id = %s""",
                (renamed, f"was named {reserved!r}, a reserved name", row["id"]),
            )
            logger.warning(
                "Project %s was named %r (reserved) and holds %d row(s) — "
                "renamed to %r rather than deleted",
                row["id"], reserved, row["rows_held"], renamed,
            )

    await conn.execute("ALTER TABLE projects ALTER COLUMN owner_id SET NOT NULL")
    await conn.execute("""
        DO $$ BEGIN
            ALTER TABLE projects ADD CONSTRAINT projects_owner_fk
                FOREIGN KEY (owner_id) REFERENCES owners(id);
        EXCEPTION WHEN duplicate_object THEN NULL;
        END $$
    """)
    # Swap global name uniqueness for per-owner uniqueness.
    await conn.execute("ALTER TABLE projects DROP CONSTRAINT IF EXISTS projects_name_key")
    await conn.execute("DROP INDEX IF EXISTS projects_name_key")
    await conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS projects_owner_name_key "
        "ON projects(owner_id, name)"
    )
    await conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_projects_owner ON projects(owner_id)"
    )

    # ── api_keys.owner_id ──
    if not await _has_column(conn, "api_keys", "owner_id"):
        await conn.execute("ALTER TABLE api_keys ADD COLUMN owner_id TEXT")
    await conn.execute(
        "UPDATE api_keys SET owner_id = %s WHERE owner_id IS NULL",
        (DEFAULT_OWNER_ID,),
    )
    await conn.execute("ALTER TABLE api_keys ALTER COLUMN owner_id SET NOT NULL")
    await conn.execute("""
        DO $$ BEGIN
            ALTER TABLE api_keys ADD CONSTRAINT api_keys_owner_fk
                FOREIGN KEY (owner_id) REFERENCES owners(id);
        EXCEPTION WHEN duplicate_object THEN NULL;
        END $$
    """)

    # ── the default owner's global project ──
    gid = global_project_id(DEFAULT_OWNER_ID)
    await conn.execute(
        """INSERT INTO projects (id, owner_id, name, description)
           VALUES (%s, %s, %s, %s)
           ON CONFLICT (owner_id, name) DO NOTHING""",
        (gid, DEFAULT_OWNER_ID, GLOBAL_PROJECT_NAME,
         "Memories not scoped to any project"),
    )

    # ── adopt the orphans ──
    for table in ("memories", "sessions"):
        cur = await conn.execute(
            f"UPDATE {table} SET project_id = %s WHERE project_id IS NULL",
            (gid,),
        )
        logger.info("Moved %d %s row(s) into %s", cur.rowcount, table, GLOBAL_PROJECT_NAME)

    # entities used a literal '__global__' string rather than NULL.
    cur = await conn.execute(
        "UPDATE entities SET project_id = %s "
        "WHERE project_id IS NULL OR project_id = %s",
        (gid, LEGACY_GLOBAL_PROJECT_ID),
    )
    logger.info("Moved %d entity row(s) into %s", cur.rowcount, GLOBAL_PROJECT_NAME)

    logger.info("Owner model applied; everything pre-existing belongs to owner %s",
                DEFAULT_OWNER_ID)
