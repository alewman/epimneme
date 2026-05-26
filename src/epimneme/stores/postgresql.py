"""PostgreSQL unified store — async, with pgvector, decay, dedup, versioning.

Replaces the synchronous store with a fully async implementation using
psycopg3's AsyncConnectionPool.  Adds memory decay fields, SimHash
deduplication, and memory versioning.

Requires PostgreSQL extensions:
    CREATE EXTENSION IF NOT EXISTS vector;
    CREATE EXTENSION IF NOT EXISTS pg_trgm;
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional

import psycopg
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from epimneme.core.models import (
    Entity,
    EntityKind,
    EntityResult,
    Memory,
    MemoryKind,
    MemoryResult,
    Project,
    Relationship,
    Session,
)

logger = logging.getLogger(__name__)


class PostgresStore:
    """Async PostgreSQL store with pgvector, decay, dedup, and versioning."""

    def __init__(
        self,
        dsn: str,
        embedding_dim: int = 384,
        min_pool: int = 2,
        max_pool: int = 10,
        pool_timeout: float = 30.0,
        hnsw_ef_search: int = 100,
    ) -> None:
        self.dsn = dsn
        self.embedding_dim = embedding_dim
        self.hnsw_ef_search = hnsw_ef_search
        self.pool = AsyncConnectionPool(
            dsn,
            min_size=min_pool,
            max_size=max_pool,
            timeout=pool_timeout,
            kwargs={"row_factory": dict_row, "autocommit": False},
            open=False,
        )

    async def open(self) -> None:
        """Open the connection pool and initialize schema."""
        await self.pool.open(wait=True)
        await self._init_schema()
        logger.info("PostgreSQL store opened (async)")

    async def close(self) -> None:
        await self.pool.close()

    # ── Schema ───────────────────────────────────────────────────────────

    async def _init_schema(self) -> None:
        """Create all tables, extensions, indexes if they don't exist."""
        async with self.pool.connection() as conn:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            await conn.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")

            # ── Projects ──
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS projects (
                    id          TEXT PRIMARY KEY,
                    name        TEXT NOT NULL UNIQUE,
                    path        TEXT,
                    description TEXT,
                    created_at  TIMESTAMPTZ DEFAULT NOW(),
                    updated_at  TIMESTAMPTZ DEFAULT NOW()
                )
            """)

            # ── Sessions ──
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id          TEXT PRIMARY KEY,
                    project_id  TEXT REFERENCES projects(id) ON DELETE SET NULL,
                    task        TEXT,
                    started_at  TIMESTAMPTZ DEFAULT NOW(),
                    ended_at    TIMESTAMPTZ,
                    summary     TEXT,
                    handoff     TEXT
                )
            """)

            # ── Memories (pgvector + decay + versioning + dedup) ──
            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS memories (
                    id                  TEXT PRIMARY KEY,
                    project_id          TEXT REFERENCES projects(id) ON DELETE SET NULL,
                    session_id          TEXT REFERENCES sessions(id) ON DELETE SET NULL,
                    kind                TEXT NOT NULL,
                    content             TEXT NOT NULL,
                    subject             TEXT,
                    confidence          REAL DEFAULT 1.0,
                    supersedes          TEXT,
                    obsolete            BOOLEAN DEFAULT FALSE,
                    pinned              BOOLEAN DEFAULT FALSE,
                    tags                JSONB DEFAULT '[]'::jsonb,
                    embedding           vector({self.embedding_dim}),
                    content_tsv         TSVECTOR,
                    -- Versioning
                    version             INTEGER DEFAULT 1,
                    version_of          TEXT,
                    -- Deduplication
                    simhash             BIGINT,
                    -- Decay / retrievability
                    storage_strength    REAL DEFAULT 0.0,
                    retrieval_strength  REAL DEFAULT 1.0,
                    access_count        INTEGER DEFAULT 0,
                    last_accessed       TIMESTAMPTZ,
                    -- Timestamps
                    created_at          TIMESTAMPTZ DEFAULT NOW(),
                    updated_at          TIMESTAMPTZ DEFAULT NOW()
                )
            """)

            # ── Add columns for existing DBs (idempotent) ──
            _IDENT_RE = re.compile(r"^[a-z_][a-z0-9_]*$")
            _TYPE_RE = re.compile(r"^[A-Z][A-Z0-9_ .()]*$", re.IGNORECASE)
            for col_name, col_type in [
                ("content_tsv", "TSVECTOR"),
                ("version", "INTEGER DEFAULT 1"),
                ("version_of", "TEXT"),
                ("simhash", "BIGINT"),
                ("storage_strength", "REAL DEFAULT 0.0"),
                ("retrieval_strength", "REAL DEFAULT 1.0"),
                ("access_count", "INTEGER DEFAULT 0"),
                ("last_accessed", "TIMESTAMPTZ"),
                ("pinned", "BOOLEAN DEFAULT FALSE"),
            ]:
                if not _IDENT_RE.match(col_name) or not _TYPE_RE.match(col_type):
                    raise ValueError(f"Invalid column definition: {col_name} {col_type}")
                cur = await conn.execute(
                    "SELECT 1 FROM information_schema.columns "
                    "WHERE table_name = %s AND column_name = %s",
                    ("memories", col_name),
                )
                if not (await cur.fetchone()):
                    # col_name/col_type are validated against strict regexes above
                    await conn.execute(
                        f"ALTER TABLE memories ADD COLUMN {col_name} {col_type}"
                    )

            # ── Add columns for existing projects table (idempotent) ──
            for col_name, col_type in [
                ("persistent_memories", "BOOLEAN DEFAULT FALSE"),
            ]:
                if not _IDENT_RE.match(col_name) or not _TYPE_RE.match(col_type):
                    raise ValueError(f"Invalid column definition: {col_name} {col_type}")
                cur = await conn.execute(
                    "SELECT 1 FROM information_schema.columns "
                    "WHERE table_name = %s AND column_name = %s",
                    ("projects", col_name),
                )
                if not (await cur.fetchone()):
                    await conn.execute(
                        f"ALTER TABLE projects ADD COLUMN {col_name} {col_type}"
                    )

            # ── tsvector trigger ──
            await conn.execute("""
                CREATE OR REPLACE FUNCTION memories_tsv_trigger() RETURNS trigger AS $$
                BEGIN
                    NEW.content_tsv :=
                        setweight(to_tsvector('english', COALESCE(NEW.subject, '')), 'A') ||
                        to_tsvector('english', NEW.content);
                    RETURN NEW;
                END
                $$ LANGUAGE plpgsql
            """)
            await conn.execute("""
                DO $$ BEGIN
                    IF NOT EXISTS (
                        SELECT 1 FROM pg_trigger WHERE tgname = 'trig_memories_tsv'
                    ) THEN
                        CREATE TRIGGER trig_memories_tsv
                        BEFORE INSERT OR UPDATE OF content, subject ON memories
                        FOR EACH ROW EXECUTE FUNCTION memories_tsv_trigger();
                    END IF;
                END $$
            """)

            # Backfill tsvector for existing rows
            await conn.execute("""
                UPDATE memories
                SET content_tsv =
                    setweight(to_tsvector('english', COALESCE(subject, '')), 'A') ||
                    to_tsvector('english', content)
                WHERE content_tsv IS NULL
            """)

            # ── Memory access log ──
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS memory_access (
                    id          SERIAL PRIMARY KEY,
                    memory_id   TEXT REFERENCES memories(id) ON DELETE CASCADE,
                    accessed_at TIMESTAMPTZ DEFAULT NOW(),
                    context     TEXT
                )
            """)

            # ── Entities (graph nodes) ──
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS entities (
                    id          TEXT PRIMARY KEY,
                    name        TEXT NOT NULL,
                    kind        TEXT NOT NULL,
                    project_id  TEXT,
                    properties  JSONB DEFAULT '{}'::jsonb,
                    created_at  TIMESTAMPTZ DEFAULT NOW(),
                    UNIQUE(name, project_id)
                )
            """)

            # ── Relationships (graph edges) ──
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS relationships (
                    id          SERIAL PRIMARY KEY,
                    from_entity TEXT NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
                    to_entity   TEXT NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
                    relation    TEXT NOT NULL,
                    properties  JSONB DEFAULT '{}'::jsonb,
                    UNIQUE(from_entity, to_entity, relation)
                )
            """)

            # ── Memory-Entity links ──
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS memory_entities (
                    memory_id   TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
                    entity_id   TEXT NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
                    relation    TEXT NOT NULL DEFAULT 'about',
                    created_at  TIMESTAMPTZ DEFAULT NOW(),
                    PRIMARY KEY (memory_id, entity_id)
                )
            """)

            # ── API Keys ──
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS api_keys (
                    id          TEXT PRIMARY KEY,
                    key_hash    TEXT NOT NULL UNIQUE,
                    key_prefix  TEXT NOT NULL,
                    name        TEXT NOT NULL UNIQUE,
                    role        TEXT NOT NULL DEFAULT 'agent',
                    projects    TEXT[] NOT NULL DEFAULT '{}',
                    created_at  TIMESTAMPTZ DEFAULT NOW(),
                    expires_at  TIMESTAMPTZ,
                    revoked_at  TIMESTAMPTZ,
                    last_used   TIMESTAMPTZ
                )
            """)

            # ── Schema migrations tracking ──
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version     INTEGER PRIMARY KEY,
                    name        TEXT NOT NULL,
                    applied_at  TIMESTAMPTZ DEFAULT NOW()
                )
            """)

            # ── Indexes ──
            idx = [
                "CREATE INDEX IF NOT EXISTS idx_sessions_project ON sessions(project_id)",
                "CREATE INDEX IF NOT EXISTS idx_memories_project ON memories(project_id)",
                "CREATE INDEX IF NOT EXISTS idx_memories_kind ON memories(kind)",
                "CREATE INDEX IF NOT EXISTS idx_memories_obsolete ON memories(obsolete) WHERE NOT obsolete",
                "CREATE INDEX IF NOT EXISTS idx_memories_content_trgm ON memories USING gin (content gin_trgm_ops)",
                "CREATE INDEX IF NOT EXISTS idx_memories_content_tsv ON memories USING gin (content_tsv)",
                "CREATE INDEX IF NOT EXISTS idx_memories_simhash ON memories(simhash) WHERE simhash IS NOT NULL",
                "CREATE INDEX IF NOT EXISTS idx_memories_version_of ON memories(version_of) WHERE version_of IS NOT NULL",
                "DROP INDEX IF EXISTS idx_memories_embedding",
                (
                    "CREATE INDEX IF NOT EXISTS idx_memories_embedding_hnsw "
                    "ON memories USING hnsw (embedding vector_cosine_ops) "
                    "WITH (m = 32, ef_construction = 200)"
                ),
                "CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name)",
                "CREATE INDEX IF NOT EXISTS idx_entities_project ON entities(project_id)",
                "CREATE INDEX IF NOT EXISTS idx_relationships_from ON relationships(from_entity)",
                "CREATE INDEX IF NOT EXISTS idx_relationships_to ON relationships(to_entity)",
                "CREATE INDEX IF NOT EXISTS idx_memory_entities_mem ON memory_entities(memory_id)",
                "CREATE INDEX IF NOT EXISTS idx_memory_entities_ent ON memory_entities(entity_id)",
                "CREATE INDEX IF NOT EXISTS idx_memory_access_memory ON memory_access(memory_id)",
                "CREATE INDEX IF NOT EXISTS idx_sessions_started ON sessions(started_at DESC)",
                "CREATE INDEX IF NOT EXISTS idx_api_keys_hash ON api_keys(key_hash)",
            ]
            for ddl in idx:
                await conn.execute(ddl)

            await conn.commit()
            logger.info("PostgreSQL schema initialized")

    # ── Projects ─────────────────────────────────────────────────────────

    async def create_project(self, project: Project) -> Project:
        async with self.pool.connection() as conn:
            await conn.execute(
                """INSERT INTO projects (id, name, path, description, persistent_memories, created_at, updated_at)
                   VALUES (%s, %s, %s, %s, %s, %s, %s)
                   ON CONFLICT (name) DO NOTHING""",
                (project.id, project.name, project.path, project.description,
                 project.persistent_memories, project.created_at, project.updated_at),
            )
            await conn.commit()
        return project

    async def get_project(self, name: str) -> Optional[Project]:
        async with self.pool.connection() as conn:
            cur = await conn.execute(
                "SELECT * FROM projects WHERE name = %s", (name,)
            )
            row = await cur.fetchone()
        return self._row_to_project(row) if row else None

    async def get_project_by_id(self, project_id: str) -> Optional[Project]:
        async with self.pool.connection() as conn:
            cur = await conn.execute(
                "SELECT * FROM projects WHERE id = %s", (project_id,)
            )
            row = await cur.fetchone()
        return self._row_to_project(row) if row else None

    async def list_projects(self) -> list[Project]:
        async with self.pool.connection() as conn:
            cur = await conn.execute(
                "SELECT * FROM projects ORDER BY updated_at DESC"
            )
            rows = await cur.fetchall()
        return [self._row_to_project(r) for r in rows]

    async def count_projects(self) -> int:
        """Return total number of projects."""
        async with self.pool.connection() as conn:
            cur = await conn.execute("SELECT COUNT(*) AS cnt FROM projects")
            row = await cur.fetchone()
        return row["cnt"] if row else 0

    async def set_project_persistent(self, project_name: str, enabled: bool) -> bool:
        """Enable or disable persistent memories for a project."""
        async with self.pool.connection() as conn:
            cur = await conn.execute(
                """UPDATE projects SET persistent_memories = %s, updated_at = NOW()
                   WHERE name = %s""",
                (enabled, project_name),
            )
            await conn.commit()
            return cur.rowcount > 0

    async def get_persistent_project_ids(self) -> set[str]:
        """Return the set of project IDs that have persistent_memories enabled."""
        async with self.pool.connection() as conn:
            cur = await conn.execute(
                "SELECT id FROM projects WHERE persistent_memories = TRUE"
            )
            rows = await cur.fetchall()
        return {r["id"] for r in rows}

    # ── Sessions ─────────────────────────────────────────────────────────

    async def create_session(self, session: Session) -> Session:
        async with self.pool.connection() as conn:
            # Assign a per-project monotonic ordinal so the retrieval layer
            # can apply session-recency boosts without relying on timestamps.
            cur = await conn.execute(
                """SELECT COALESCE(MAX(session_ordinal), 0) + 1
                   FROM sessions
                   WHERE project_id IS NOT DISTINCT FROM %s""",
                (session.project_id,),
            )
            row = await cur.fetchone()
            ordinal = list(row.values())[0] if row else 1
            session.session_ordinal = ordinal

            await conn.execute(
                """INSERT INTO sessions (id, project_id, task, started_at, ended_at, summary, handoff, session_ordinal)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
                (session.id, session.project_id, session.task,
                 session.started_at, session.ended_at, session.summary, session.handoff,
                 session.session_ordinal),
            )
            await conn.commit()
        return session

    async def get_session_by_id(self, session_id: str) -> Optional[Session]:
        async with self.pool.connection() as conn:
            cur = await conn.execute(
                "SELECT * FROM sessions WHERE id = %s", (session_id,)
            )
            row = await cur.fetchone()
        return self._row_to_session(row) if row else None

    async def get_session_ordinals(self, session_ids: list[str]) -> dict[str, int]:
        """Return a mapping of session_id → session_ordinal for the given IDs.

        Sessions with no ordinal (pre-migration rows that were never backfilled)
        are omitted from the result rather than defaulting to 0.
        """
        if not session_ids:
            return {}
        placeholders = ", ".join(["%s"] * len(session_ids))
        async with self.pool.connection() as conn:
            cur = await conn.execute(
                f"""SELECT id, session_ordinal FROM sessions
                    WHERE id IN ({placeholders})
                      AND session_ordinal IS NOT NULL""",
                session_ids,
            )
            rows = await cur.fetchall()
        return {r["id"]: r["session_ordinal"] for r in rows}

    async def end_session(self, session_id: str, summary: str, handoff: Optional[str] = None) -> None:
        now = datetime.now(timezone.utc)
        async with self.pool.connection() as conn:
            await conn.execute(
                "UPDATE sessions SET ended_at = %s, summary = %s, handoff = %s WHERE id = %s",
                (now, summary, handoff, session_id),
            )
            await conn.commit()

    async def get_last_session(self, project_id: Optional[str] = None) -> Optional[Session]:
        async with self.pool.connection() as conn:
            if project_id:
                cur = await conn.execute(
                    "SELECT * FROM sessions WHERE project_id = %s ORDER BY started_at DESC LIMIT 1",
                    (project_id,),
                )
            else:
                cur = await conn.execute(
                    "SELECT * FROM sessions ORDER BY started_at DESC LIMIT 1"
                )
            row = await cur.fetchone()
        return self._row_to_session(row) if row else None

    async def get_previous_session(self, project_id: Optional[str], exclude_id: str) -> Optional[Session]:
        async with self.pool.connection() as conn:
            if project_id:
                cur = await conn.execute(
                    """SELECT * FROM sessions
                       WHERE project_id = %s AND id != %s
                       ORDER BY started_at DESC LIMIT 1""",
                    (project_id, exclude_id),
                )
            else:
                cur = await conn.execute(
                    "SELECT * FROM sessions WHERE id != %s ORDER BY started_at DESC LIMIT 1",
                    (exclude_id,),
                )
            row = await cur.fetchone()
        return self._row_to_session(row) if row else None

    async def list_open_sessions(self, older_than_hours: Optional[float] = None) -> list[Session]:
        """List sessions that were started but never ended."""
        conditions = ["ended_at IS NULL"]
        params: list = []
        if older_than_hours is not None:
            conditions.append("started_at < NOW() - (%s || ' hours')::interval")
            params.append(str(older_than_hours))
        where = " AND ".join(conditions)
        async with self.pool.connection() as conn:
            cur = await conn.execute(
                f"SELECT * FROM sessions WHERE {where} ORDER BY started_at DESC",
                params,
            )
            rows = await cur.fetchall()
        return [self._row_to_session(r) for r in rows]

    async def force_close_session(self, session_id: str, summary: Optional[str] = None) -> bool:
        """Force-close an open session. Returns True if found and closed."""
        now = datetime.now(timezone.utc)
        summary = summary or "Session force-closed by admin"
        async with self.pool.connection() as conn:
            result = await conn.execute(
                "UPDATE sessions SET ended_at = %s, summary = %s WHERE id = %s AND ended_at IS NULL",
                (now, summary, session_id),
            )
            await conn.commit()
            return result.rowcount > 0

    # ── Memories ─────────────────────────────────────────────────────────

    async def store_memory(self, memory: Memory, embedding: Optional[list[float]] = None) -> Memory:
        """Store a memory with optional embedding vector."""
        async with self.pool.connection() as conn:
            if memory.supersedes:
                await conn.execute(
                    "UPDATE memories SET obsolete = TRUE, updated_at = NOW() WHERE id = %s",
                    (memory.supersedes,),
                )
            await conn.execute(
                """INSERT INTO memories
                   (id, project_id, session_id, kind, content, subject,
                    confidence, supersedes, obsolete, tags, embedding,
                    version, version_of, simhash,
                    storage_strength, retrieval_strength, access_count, last_accessed,
                    pinned,
                    created_at, updated_at)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s,
                           %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                (memory.id, memory.project_id, memory.session_id,
                 memory.kind.value, memory.content, memory.subject,
                 memory.confidence, memory.supersedes, memory.obsolete,
                 json.dumps(memory.tags), embedding,
                 memory.version, memory.version_of, memory.simhash,
                 memory.storage_strength, memory.retrieval_strength,
                 memory.access_count, memory.last_accessed,
                 memory.pinned,
                 memory.created_at, memory.updated_at),
            )
            await conn.commit()
        return memory

    async def get_memory(self, memory_id: str) -> Optional[Memory]:
        async with self.pool.connection() as conn:
            cur = await conn.execute(
                "SELECT * FROM memories WHERE id = %s", (memory_id,)
            )
            row = await cur.fetchone()
        return self._row_to_memory(row) if row else None

    async def pin_memory(self, memory_id: str) -> bool:
        """Pin a memory so it is never garbage-collected."""
        async with self.pool.connection() as conn:
            result = await conn.execute(
                "UPDATE memories SET pinned = TRUE, updated_at = NOW() WHERE id = %s",
                (memory_id,),
            )
            await conn.commit()
            return result.rowcount > 0

    async def unpin_memory(self, memory_id: str) -> bool:
        """Remove the pin from a memory."""
        async with self.pool.connection() as conn:
            result = await conn.execute(
                "UPDATE memories SET pinned = FALSE, updated_at = NOW() WHERE id = %s",
                (memory_id,),
            )
            await conn.commit()
            return result.rowcount > 0

    async def get_pinned_memories(self, project_id: Optional[str] = None) -> list[Memory]:
        """Return all pinned, non-obsolete memories, optionally scoped to a project."""
        async with self.pool.connection() as conn:
            if project_id:
                cur = await conn.execute(
                    "SELECT * FROM memories WHERE pinned = TRUE AND obsolete = FALSE "
                    "AND project_id = %s ORDER BY created_at",
                    (project_id,),
                )
            else:
                cur = await conn.execute(
                    "SELECT * FROM memories WHERE pinned = TRUE AND obsolete = FALSE "
                    "ORDER BY created_at",
                )
            rows = await cur.fetchall()
        return [self._row_to_memory(r) for r in rows]

    async def boost_tsvector_terms(self, memory_id: str, terms: str) -> None:
        """Append terms at tsvector weight 'A' (highest priority).

        Used to boost preference-associated nouns so full-text search
        surfaces preference memories for vague queries.
        """
        async with self.pool.connection() as conn:
            await conn.execute(
                """
                UPDATE memories
                SET content_tsv = content_tsv ||
                    setweight(to_tsvector('english', %s), 'A')
                WHERE id = %s
                """,
                (terms, memory_id),
            )

    async def update_memory(
        self,
        memory_id: str,
        content: str,
        subject: Optional[str] = None,
        tags: Optional[list[str]] = None,
        confidence: Optional[float] = None,
        embedding: Optional[list[float]] = None,
        simhash: Optional[int] = None,
    ) -> Optional[Memory]:
        """Create a new version of a memory (old version is preserved).

        Returns the new versioned memory, or None if original not found.
        """
        original = await self.get_memory(memory_id)
        if not original:
            return None

        root_id = original.version_of or original.id
        new_version = original.version + 1

        new_memory = Memory(
            project_id=original.project_id,
            session_id=original.session_id,
            kind=original.kind,
            content=content,
            subject=subject if subject is not None else original.subject,
            confidence=confidence if confidence is not None else original.confidence,
            tags=tags if tags is not None else original.tags,
            version=new_version,
            version_of=root_id,
            simhash=simhash,
            storage_strength=original.storage_strength,
            retrieval_strength=original.retrieval_strength,
            access_count=original.access_count,
            last_accessed=original.last_accessed,
        )

        # Atomic: mark old as superseded, insert new, copy entity links
        async with self.pool.connection() as conn:
            await conn.execute(
                "UPDATE memories SET obsolete = TRUE, updated_at = NOW() WHERE id = %s",
                (memory_id,),
            )
            await conn.execute(
                """INSERT INTO memories
                   (id, project_id, session_id, kind, content, subject,
                    confidence, supersedes, obsolete, tags, embedding,
                    version, version_of, simhash,
                    storage_strength, retrieval_strength, access_count, last_accessed,
                    created_at, updated_at)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s,
                           %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                (new_memory.id, new_memory.project_id, new_memory.session_id,
                 new_memory.kind.value, new_memory.content, new_memory.subject,
                 new_memory.confidence, new_memory.supersedes, new_memory.obsolete,
                 json.dumps(new_memory.tags), embedding,
                 new_memory.version, new_memory.version_of, new_memory.simhash,
                 new_memory.storage_strength, new_memory.retrieval_strength,
                 new_memory.access_count, new_memory.last_accessed,
                 new_memory.created_at, new_memory.updated_at),
            )
            await conn.execute(
                """INSERT INTO memory_entities (memory_id, entity_id, relation)
                   SELECT %s, entity_id, relation FROM memory_entities
                   WHERE memory_id = %s
                   ON CONFLICT DO NOTHING""",
                (new_memory.id, memory_id),
            )
            await conn.commit()

        return new_memory

    async def get_memory_versions(self, memory_id: str) -> list[Memory]:
        """Get the full version chain for a memory."""
        original = await self.get_memory(memory_id)
        if not original:
            return []

        root_id = original.version_of or original.id

        async with self.pool.connection() as conn:
            cur = await conn.execute(
                """SELECT * FROM memories
                   WHERE id = %s OR version_of = %s
                   ORDER BY version ASC""",
                (root_id, root_id),
            )
            rows = await cur.fetchall()
        return [self._row_to_memory(r) for r in rows]

    async def mark_obsolete(self, memory_id: str) -> None:
        async with self.pool.connection() as conn:
            await conn.execute(
                "UPDATE memories SET obsolete = TRUE, updated_at = NOW() WHERE id = %s",
                (memory_id,),
            )
            await conn.commit()

    async def hard_delete_memory(self, memory_id: str) -> bool:
        """Permanently delete a memory and its entity links from the database."""
        async with self.pool.connection() as conn:
            await conn.execute(
                "DELETE FROM memory_entities WHERE memory_id = %s", (memory_id,)
            )
            result = await conn.execute(
                "DELETE FROM memories WHERE id = %s", (memory_id,)
            )
            await conn.commit()
            return result.rowcount > 0

    async def purge_obsolete_memories(self) -> int:
        """Permanently delete all obsolete memories. Returns count deleted."""
        async with self.pool.connection() as conn:
            await conn.execute(
                "DELETE FROM memory_entities WHERE memory_id IN (SELECT id FROM memories WHERE obsolete = TRUE)"
            )
            result = await conn.execute(
                "DELETE FROM memories WHERE obsolete = TRUE"
            )
            await conn.commit()
            return result.rowcount

    async def search_semantic(
        self,
        embedding: list[float],
        project_id: Optional[str] = None,
        kind: Optional[str] = None,
        tags: Optional[list[str]] = None,
        limit: int = 20,
    ) -> list[MemoryResult]:
        """Vector similarity search using pgvector."""
        embedding_str = str(embedding)
        conditions = ["NOT obsolete", "embedding IS NOT NULL"]
        params: list = []

        if project_id:
            conditions.append("project_id = %s")
            params.append(project_id)
        if kind:
            conditions.append("kind = %s")
            params.append(kind)
        if tags:
            conditions.append("tags @> %s::jsonb")
            params.append(json.dumps(tags))

        where = " AND ".join(conditions)

        async with self.pool.connection() as conn:
            await conn.execute(f"SET LOCAL hnsw.ef_search = {self.hnsw_ef_search}")
            cur = await conn.execute(
                f"""SELECT *, 1 - (embedding <=> %s::vector) AS score
                    FROM memories
                    WHERE {where}
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s""",
                (embedding_str, *params, embedding_str, limit),
            )
            rows = await cur.fetchall()

        results = []
        for row in rows:
            mem = self._row_to_memory(row)
            score = max(0.0, float(row.get("score", 0.0)))
            results.append(MemoryResult(memory=mem, score=score, source="semantic"))
        return results

    async def find_semantic_duplicates(
        self,
        embedding: list[float],
        project_id: Optional[str] = None,
        threshold: float = 0.92,
        limit: int = 3,
    ) -> list[MemoryResult]:
        """Find non-obsolete memories with cosine similarity >= threshold.

        Used for semantic deduplication — catches same-meaning, different-wording
        duplicates that SimHash misses.
        """
        embedding_str = str(embedding)
        conditions = ["NOT obsolete", "embedding IS NOT NULL"]
        params: list = []

        if project_id:
            conditions.append("project_id = %s")
            params.append(project_id)

        where = " AND ".join(conditions)

        async with self.pool.connection() as conn:
            await conn.execute(f"SET LOCAL hnsw.ef_search = {self.hnsw_ef_search}")
            cur = await conn.execute(
                f"""SELECT *, 1 - (embedding <=> %s::vector) AS score
                    FROM memories
                    WHERE {where}
                      AND 1 - (embedding <=> %s::vector) >= %s
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s""",
                (embedding_str, *params, embedding_str, threshold, embedding_str, limit),
            )
            rows = await cur.fetchall()

        results = []
        for row in rows:
            mem = self._row_to_memory(row)
            score = float(row.get("score", 0.0))
            results.append(MemoryResult(memory=mem, score=score, source="semantic_dedup"))
        return results

    async def find_potential_conflicts(
        self,
        embedding: list[float],
        kind: str,
        project_id: Optional[str] = None,
        threshold: float = 0.80,
        limit: int = 3,
    ) -> list[MemoryResult]:
        """Find existing memories of the same kind that may conflict.

        Used for conflict surfacing — when storing a fact or decision,
        finds semantically similar memories of the same kind so the agent
        can decide whether to supersede.
        """
        embedding_str = str(embedding)
        conditions = [
            "NOT obsolete",
            "embedding IS NOT NULL",
            "kind = %s",
        ]
        params: list = [kind]

        if project_id:
            conditions.append("project_id = %s")
            params.append(project_id)

        where = " AND ".join(conditions)

        async with self.pool.connection() as conn:
            await conn.execute(f"SET LOCAL hnsw.ef_search = {self.hnsw_ef_search}")
            cur = await conn.execute(
                f"""SELECT *, 1 - (embedding <=> %s::vector) AS score
                    FROM memories
                    WHERE {where}
                      AND 1 - (embedding <=> %s::vector) >= %s
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s""",
                (embedding_str, *params, embedding_str, threshold, embedding_str, limit),
            )
            rows = await cur.fetchall()

        results = []
        for row in rows:
            mem = self._row_to_memory(row)
            score = float(row.get("score", 0.0))
            results.append(MemoryResult(memory=mem, score=score, source="conflict"))
        return results

    async def search_fulltext(
        self,
        query: str,
        project_id: Optional[str] = None,
        kind: Optional[MemoryKind] = None,
        tags: Optional[list[str]] = None,
        limit: int = 20,
    ) -> list[MemoryResult]:
        """Full-text search using tsvector/tsquery with trigram fallback."""
        if not query.strip():
            return []

        conditions = ["NOT obsolete", "content_tsv @@ websearch_to_tsquery('english', %s)"]
        params: list = [query]

        if project_id:
            conditions.append("project_id = %s")
            params.append(project_id)
        if kind:
            conditions.append("kind = %s")
            params.append(kind.value)
        if tags:
            conditions.append("tags @> %s::jsonb")
            params.append(json.dumps(tags))

        where = " AND ".join(conditions)
        params.append(query)  # for ts_rank
        params.append(limit)

        sql = f"""
            SELECT *, ts_rank(content_tsv, websearch_to_tsquery('english', %s)) AS score
            FROM memories
            WHERE {where}
            ORDER BY score DESC, confidence DESC, created_at DESC
            LIMIT %s
        """

        async with self.pool.connection() as conn:
            cur = await conn.execute(sql, params)
            rows = await cur.fetchall()

        results = []
        for row in rows:
            mem = self._row_to_memory(row)
            score = float(row.get("score", 0.0))
            results.append(MemoryResult(memory=mem, score=score, source="fulltext"))

        if not results:
            results = await self._search_trigram(query, project_id, kind, tags, limit)

        return results

    async def _search_trigram(
        self,
        query: str,
        project_id: Optional[str] = None,
        kind: Optional[MemoryKind] = None,
        tags: Optional[list[str]] = None,
        limit: int = 20,
    ) -> list[MemoryResult]:
        """Fallback: trigram similarity search for fuzzy matching."""
        conditions = ["NOT obsolete", "similarity(content, %s) > 0.1"]
        params: list = [query]

        if project_id:
            conditions.append("project_id = %s")
            params.append(project_id)
        if kind:
            conditions.append("kind = %s")
            params.append(kind.value)
        if tags:
            conditions.append("tags @> %s::jsonb")
            params.append(json.dumps(tags))

        where = " AND ".join(conditions)

        sql = f"""
            SELECT *, similarity(content, %s) AS score
            FROM memories
            WHERE {where}
            ORDER BY score DESC, confidence DESC, created_at DESC
            LIMIT %s
        """
        all_params = (query, *params, limit)

        async with self.pool.connection() as conn:
            cur = await conn.execute(sql, all_params)
            rows = await cur.fetchall()

        results = []
        for row in rows:
            mem = self._row_to_memory(row)
            score = float(row.get("score", 0.0))
            results.append(MemoryResult(memory=mem, score=score, source="trigram"))
        return results

    async def get_memories_by_kind(
        self,
        kind: MemoryKind,
        project_id: Optional[str] = None,
        limit: int = 50,
    ) -> list[Memory]:
        conditions = ["NOT obsolete", "kind = %s"]
        params: list = [kind.value]
        if project_id:
            conditions.append("project_id = %s")
            params.append(project_id)
        where = " AND ".join(conditions)
        params.append(limit)
        async with self.pool.connection() as conn:
            cur = await conn.execute(
                f"SELECT * FROM memories WHERE {where} ORDER BY created_at DESC LIMIT %s",
                params,
            )
            rows = await cur.fetchall()
        return [self._row_to_memory(r) for r in rows]

    async def get_memory_count(self, project_id: Optional[str] = None) -> int:
        async with self.pool.connection() as conn:
            if project_id:
                cur = await conn.execute(
                    "SELECT COUNT(*) AS cnt FROM memories WHERE NOT obsolete AND project_id = %s",
                    (project_id,),
                )
            else:
                cur = await conn.execute(
                    "SELECT COUNT(*) AS cnt FROM memories WHERE NOT obsolete"
                )
            row = await cur.fetchone()
        return row["cnt"] if row else 0

    async def get_vector_count(self) -> int:
        async with self.pool.connection() as conn:
            cur = await conn.execute(
                "SELECT COUNT(*) AS cnt FROM memories WHERE embedding IS NOT NULL AND NOT obsolete"
            )
            row = await cur.fetchone()
        return row["cnt"] if row else 0

    async def log_access(self, memory_id: str, context: str = "") -> None:
        async with self.pool.connection() as conn:
            await conn.execute(
                "INSERT INTO memory_access (memory_id, context) VALUES (%s, %s)",
                (memory_id, context),
            )
            await conn.commit()

    async def update_decay_on_access(
        self,
        memory_id: str,
        new_storage: float,
        new_retrieval: float,
        new_count: int,
    ) -> None:
        """Update decay fields after a memory is accessed."""
        now = datetime.now(timezone.utc)
        async with self.pool.connection() as conn:
            await conn.execute(
                """UPDATE memories
                   SET storage_strength = %s,
                       retrieval_strength = %s,
                       access_count = %s,
                       last_accessed = %s,
                       updated_at = NOW()
                   WHERE id = %s""",
                (new_storage, new_retrieval, new_count, now, memory_id),
            )
            await conn.commit()

    # ── Deduplication ────────────────────────────────────────────────────

    async def find_similar_by_simhash(
        self,
        simhash: int,
        project_id: Optional[str] = None,
        threshold: int = 3,
    ) -> list[Memory]:
        """Find memories with SimHash within Hamming distance threshold.

        Uses bit_count (PG14+) for efficient Hamming distance calculation.
        Falls back to Python-side check if bit_count is unavailable.
        """
        async with self.pool.connection() as conn:
            try:
                # PostgreSQL 14+ has bit_count
                conditions = [
                    "NOT obsolete",
                    "simhash IS NOT NULL",
                    "bit_count((simhash # %s)::bit(64)) <= %s",
                ]
                params: list = [simhash, threshold]
                if project_id:
                    conditions.append("project_id = %s")
                    params.append(project_id)
                where = " AND ".join(conditions)
                cur = await conn.execute(
                    f"SELECT * FROM memories WHERE {where} LIMIT 10",
                    params,
                )
                rows = await cur.fetchall()
            except psycopg.errors.UndefinedFunction:
                await conn.rollback()
                # Fallback: fetch recent memories and filter in Python
                conditions = ["NOT obsolete", "simhash IS NOT NULL"]
                params = []
                if project_id:
                    conditions.append("project_id = %s")
                    params.append(project_id)
                where = " AND ".join(conditions)
                cur = await conn.execute(
                    f"SELECT * FROM memories WHERE {where} ORDER BY created_at DESC LIMIT 1000",
                    params,
                )
                rows = await cur.fetchall()
                rows = [
                    r for r in rows
                    if bin(r["simhash"] ^ simhash).count("1") <= threshold
                ]

        return [self._row_to_memory(r) for r in rows]

    # ── Memory-Entity Links ──────────────────────────────────────────────

    async def link_memory_to_entities(
        self,
        memory_id: str,
        entity_names: list[str],
        project_id: Optional[str] = None,
        relation: str = "about",
    ) -> int:
        if not entity_names:
            return 0

        # Batch-resolve existing entities
        proj_key = project_id or "__global__"
        async with self.pool.connection() as conn:
            if project_id:
                cur = await conn.execute(
                    """SELECT * FROM entities
                       WHERE name = ANY(%s)
                         AND (project_id = %s OR project_id = '__global__')""",
                    (entity_names, project_id),
                )
            else:
                cur = await conn.execute(
                    "SELECT * FROM entities WHERE name = ANY(%s)",
                    (entity_names,),
                )
            existing_rows = await cur.fetchall()

        existing_by_name = {r["name"]: self._row_to_entity(r) for r in existing_rows}

        # Create missing entities
        for name in entity_names:
            if name not in existing_by_name:
                ent = await self.track_entity(
                    Entity(name=name, kind=EntityKind.CONCEPT, project_id=project_id)
                )
                existing_by_name[name] = ent

        # Batch-insert links in one connection
        async with self.pool.connection() as conn:
            for name in entity_names:
                ent = existing_by_name[name]
                await conn.execute(
                    """INSERT INTO memory_entities (memory_id, entity_id, relation)
                       VALUES (%s, %s, %s)
                       ON CONFLICT (memory_id, entity_id) DO NOTHING""",
                    (memory_id, ent.id, relation),
                )
            await conn.commit()
        return len(entity_names)

    async def get_entities_for_memory(self, memory_id: str) -> list[Entity]:
        async with self.pool.connection() as conn:
            cur = await conn.execute(
                """SELECT e.* FROM entities e
                   JOIN memory_entities me ON me.entity_id = e.id
                   WHERE me.memory_id = %s""",
                (memory_id,),
            )
            rows = await cur.fetchall()
        return [self._row_to_entity(r) for r in rows]

    async def get_entities_for_memories_batch(
        self, memory_ids: list[str]
    ) -> dict[str, list[Entity]]:
        """Batch-fetch entities linked to multiple memories."""
        if not memory_ids:
            return {}
        result: dict[str, list[Entity]] = {mid: [] for mid in memory_ids}
        async with self.pool.connection() as conn:
            cur = await conn.execute(
                """SELECT me.memory_id, e.* FROM entities e
                   JOIN memory_entities me ON me.entity_id = e.id
                   WHERE me.memory_id = ANY(%s)""",
                (memory_ids,),
            )
            rows = await cur.fetchall()
        for row in rows:
            mid = row["memory_id"]
            if mid in result:
                result[mid].append(self._row_to_entity(row))
        return result

    async def get_memories_for_entity(
        self,
        entity_name: str,
        project_id: Optional[str] = None,
        limit: int = 20,
    ) -> list[str]:
        entity = await self.get_entity(entity_name, project_id)
        if not entity:
            return []
        async with self.pool.connection() as conn:
            cur = await conn.execute(
                """SELECT me.memory_id FROM memory_entities me
                   JOIN memories m ON m.id = me.memory_id
                   WHERE me.entity_id = %s AND NOT m.obsolete
                   ORDER BY m.created_at DESC
                   LIMIT %s""",
                (entity.id, limit),
            )
            rows = await cur.fetchall()
        return [r["memory_id"] for r in rows]

    # ── Entities (Graph) ─────────────────────────────────────────────────

    async def track_entity(self, entity: Entity) -> Entity:
        async with self.pool.connection() as conn:
            cur = await conn.execute(
                "SELECT * FROM entities WHERE name = %s AND COALESCE(project_id, '__global__') = %s",
                (entity.name, entity.project_id or "__global__"),
            )
            existing = await cur.fetchone()

            if existing:
                raw_props = existing["properties"]
                old_props = raw_props if isinstance(raw_props, dict) else json.loads(raw_props or "{}")
                merged_props = {**old_props, **entity.properties}
                await conn.execute(
                    "UPDATE entities SET properties = %s::jsonb, kind = %s WHERE id = %s",
                    (json.dumps(merged_props), entity.kind.value, existing["id"]),
                )
                await conn.commit()
                entity.id = existing["id"]
                entity.properties = merged_props
                return entity

            await conn.execute(
                """INSERT INTO entities (id, name, kind, project_id, properties, created_at)
                   VALUES (%s, %s, %s, %s, %s::jsonb, %s)""",
                (entity.id, entity.name, entity.kind.value,
                 entity.project_id or "__global__",
                 json.dumps(entity.properties), entity.created_at),
            )
            await conn.commit()
        return entity

    async def get_entity(self, name: str, project_id: Optional[str] = None) -> Optional[Entity]:
        async with self.pool.connection() as conn:
            if project_id:
                cur = await conn.execute(
                    """SELECT * FROM entities
                       WHERE name = %s AND (project_id = %s OR project_id = '__global__')""",
                    (name, project_id),
                )
            else:
                cur = await conn.execute(
                    "SELECT * FROM entities WHERE name = %s", (name,)
                )
            row = await cur.fetchone()
        return self._row_to_entity(row) if row else None

    async def list_entities(
        self,
        project_id: Optional[str] = None,
        kind: Optional[EntityKind] = None,
        limit: int = 200,
        offset: int = 0,
    ) -> list[Entity]:
        conditions: list[str] = []
        params: list = []
        if project_id:
            conditions.append("project_id = %s")
            params.append(project_id)
        if kind:
            conditions.append("kind = %s")
            params.append(kind.value)
        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        params.extend([limit, offset])
        async with self.pool.connection() as conn:
            cur = await conn.execute(
                f"SELECT * FROM entities {where} ORDER BY created_at DESC LIMIT %s OFFSET %s",
                params,
            )
            rows = await cur.fetchall()
        return [self._row_to_entity(r) for r in rows]

    async def count_entities(
        self,
        project_id: Optional[str] = None,
        kind: Optional[EntityKind] = None,
    ) -> int:
        """Return total count of entities matching filters."""
        conditions: list[str] = []
        params: list = []
        if project_id:
            conditions.append("project_id = %s")
            params.append(project_id)
        if kind:
            conditions.append("kind = %s")
            params.append(kind.value)
        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        async with self.pool.connection() as conn:
            cur = await conn.execute(
                f"SELECT COUNT(*) AS cnt FROM entities {where}", params
            )
            row = await cur.fetchone()
        return row["cnt"] if row else 0

    async def delete_entity(self, entity_id: str) -> bool:
        """Delete an entity and its relationships from the graph.

        Also removes memory_entities links. Returns True if entity existed.
        """
        async with self.pool.connection() as conn:
            # Remove memory-entity links
            await conn.execute(
                "DELETE FROM memory_entities WHERE entity_id = %s", (entity_id,)
            )
            # Remove relationships (both directions)
            await conn.execute(
                "DELETE FROM relationships WHERE from_entity = %s OR to_entity = %s",
                (entity_id, entity_id),
            )
            # Remove entity
            result = await conn.execute(
                "DELETE FROM entities WHERE id = %s", (entity_id,)
            )
            await conn.commit()
            return result.rowcount > 0

    # ── Relationships (Graph) ────────────────────────────────────────────

    async def relate(self, rel: Relationship) -> None:
        from_ent = await self.get_entity(rel.from_entity)
        to_ent = await self.get_entity(rel.to_entity)

        if not from_ent:
            from_ent = await self.track_entity(
                Entity(name=rel.from_entity, kind=EntityKind.CONCEPT)
            )
        if not to_ent:
            to_ent = await self.track_entity(
                Entity(name=rel.to_entity, kind=EntityKind.CONCEPT)
            )

        async with self.pool.connection() as conn:
            await conn.execute(
                """INSERT INTO relationships (from_entity, to_entity, relation, properties)
                   VALUES (%s, %s, %s, %s::jsonb)
                   ON CONFLICT (from_entity, to_entity, relation) DO NOTHING""",
                (from_ent.id, to_ent.id, rel.relation, json.dumps(rel.properties)),
            )
            await conn.commit()

    async def explore(
        self,
        entity_name: str,
        depth: int = 2,
        direction: str = "both",
        project_id: Optional[str] = None,
    ) -> list[EntityResult]:
        depth = min(max(depth, 1), 5)
        entity = await self.get_entity(entity_name, project_id)
        if not entity:
            return []

        if direction == "outgoing":
            edge_join = "r.from_entity = g.entity_id"
            next_entity = "r.to_entity"
        elif direction == "incoming":
            edge_join = "r.to_entity = g.entity_id"
            next_entity = "r.from_entity"
        else:
            edge_join = "(r.from_entity = g.entity_id OR r.to_entity = g.entity_id)"
            next_entity = "CASE WHEN r.from_entity = g.entity_id THEN r.to_entity ELSE r.from_entity END"

        sql = f"""
            WITH RECURSIVE graph AS (
                SELECT %s::text AS entity_id, 0 AS depth
                UNION
                SELECT {next_entity}, g.depth + 1
                FROM graph g
                JOIN relationships r ON {edge_join}
                WHERE g.depth < %s
            )
            SELECT DISTINCT e.*
            FROM graph g
            JOIN entities e ON e.id = g.entity_id
            WHERE e.id != %s
        """

        async with self.pool.connection() as conn:
            cur = await conn.execute(sql, (entity.id, depth, entity.id))
            rows = await cur.fetchall()

        if not rows:
            return []

        entity_ids = [row["id"] for row in rows]
        rels_by_entity = await self.get_relationships_batch(entity_ids)

        results: list[EntityResult] = []
        for row in rows:
            ent = self._row_to_entity(row)
            results.append(EntityResult(entity=ent, relationships=rels_by_entity.get(ent.id, [])))

        return results

    async def _get_relationships_for(self, entity_id: str) -> list[Relationship]:
        result = await self.get_relationships_batch([entity_id])
        return result.get(entity_id, [])

    async def get_relationships_batch(
        self, entity_ids: list[str]
    ) -> dict[str, list[Relationship]]:
        """Batch-fetch outgoing relationships for multiple entities."""
        if not entity_ids:
            return {}
        result: dict[str, list[Relationship]] = {eid: [] for eid in entity_ids}
        async with self.pool.connection() as conn:
            cur = await conn.execute(
                """SELECT r.from_entity, r.relation, e.name AS to_name, r.properties
                   FROM relationships r
                   JOIN entities e ON e.id = r.to_entity
                   WHERE r.from_entity = ANY(%s)""",
                (entity_ids,),
            )
            rows = await cur.fetchall()
        for row in rows:
            props = row["properties"] if isinstance(row["properties"], dict) else {}
            rel = Relationship(
                from_entity=row["from_entity"],
                to_entity=row["to_name"],
                relation=row["relation"],
                properties=props,
            )
            if row["from_entity"] in result:
                result[row["from_entity"]].append(rel)
        return result

    # ── API Key Management ───────────────────────────────────────────────

    @staticmethod
    def hash_key(raw_key: str) -> str:
        return hashlib.sha256(raw_key.encode()).hexdigest()

    async def create_api_key(
        self,
        name: str,
        role: str = "agent",
        projects: Optional[list[str]] = None,
        expires_in_days: Optional[int] = None,
    ) -> str:
        """Create a new API key. Returns the raw key (only shown once)."""
        import uuid
        raw_key = f"engram_{secrets.token_urlsafe(32)}"
        key_hash = self.hash_key(raw_key)
        key_prefix = raw_key[:12]
        key_id = str(uuid.uuid4())

        project_list = ["*"] if role == "admin" else (projects or [])

        expires_at = None
        if expires_in_days:
            expires_at = datetime.now(timezone.utc) + timedelta(days=expires_in_days)

        async with self.pool.connection() as conn:
            await conn.execute(
                """INSERT INTO api_keys (id, key_hash, key_prefix, name, role, projects, expires_at)
                   VALUES (%s, %s, %s, %s, %s, %s, %s)""",
                (key_id, key_hash, key_prefix, name, role, project_list, expires_at),
            )
            await conn.commit()

        logger.info(f"Created API key: name={name}, role={role}, prefix={key_prefix}")
        return raw_key

    async def validate_api_key(self, raw_key: str) -> Optional[dict]:
        key_hash = self.hash_key(raw_key)
        async with self.pool.connection() as conn:
            cur = await conn.execute(
                """UPDATE api_keys SET last_used = NOW()
                   WHERE key_hash = %s
                     AND revoked_at IS NULL
                     AND (expires_at IS NULL OR expires_at > NOW())
                   RETURNING id, name, role, projects""",
                (key_hash,),
            )
            row = await cur.fetchone()
            await conn.commit()

        if not row:
            return None

        return {
            "id": row["id"],
            "name": row["name"],
            "role": row["role"],
            "projects": row["projects"],
        }

    async def list_api_keys(self) -> list[dict]:
        async with self.pool.connection() as conn:
            cur = await conn.execute(
                """SELECT id, key_prefix, name, role, projects,
                          created_at, expires_at, revoked_at, last_used
                   FROM api_keys ORDER BY created_at DESC"""
            )
            rows = await cur.fetchall()
        return [dict(r) for r in rows]

    async def revoke_api_key(self, name: str) -> bool:
        async with self.pool.connection() as conn:
            result = await conn.execute(
                "UPDATE api_keys SET revoked_at = NOW() WHERE name = %s AND revoked_at IS NULL",
                (name,),
            )
            await conn.commit()
            return result.rowcount > 0

    async def cycle_api_key(self, name: str) -> Optional[str]:
        """Rotate an API key: revoke the old, create a new one with same name/role/projects.

        Returns the new raw key, or None if the key wasn't found.
        """
        # Read current key metadata
        async with self.pool.connection() as conn:
            cur = await conn.execute(
                "SELECT role, projects FROM api_keys WHERE name = %s AND revoked_at IS NULL",
                (name,),
            )
            row = await cur.fetchone()
        if not row:
            return None

        role, projects = row["role"], row["projects"]
        # Hard-delete the old key so the UNIQUE(name) constraint allows re-creation.
        # (revoke_api_key only sets revoked_at, leaving the row with the same name.)
        async with self.pool.connection() as conn:
            await conn.execute("DELETE FROM api_keys WHERE name = %s", (name,))
            await conn.commit()
        return await self.create_api_key(name=name, role=role, projects=projects)

    async def add_project_to_api_key(self, key_id: str, project_name: str) -> None:
        async with self.pool.connection() as conn:
            await conn.execute(
                "UPDATE api_keys SET projects = array_append(projects, %s) "
                "WHERE id = %s AND NOT (%s = ANY(projects))",
                (project_name, key_id, project_name),
            )
            await conn.commit()

    async def update_api_key(
        self,
        name: str,
        role: Optional[str] = None,
        projects: Optional[list[str]] = None,
    ) -> Optional[dict]:
        """Update an API key's role and/or project scope."""
        async with self.pool.connection() as conn:
            cur = await conn.execute(
                "SELECT id, name, role, projects FROM api_keys WHERE name = %s AND revoked_at IS NULL",
                (name,),
            )
            row = await cur.fetchone()
            if not row:
                return None
            new_role = role if role is not None else row["role"]
            new_projects = projects if projects is not None else row["projects"]
            # Admin keys always get wildcard
            if new_role == "admin":
                new_projects = ["*"]
            await conn.execute(
                "UPDATE api_keys SET role = %s, projects = %s WHERE id = %s",
                (new_role, new_projects, row["id"]),
            )
            await conn.commit()
        logger.info(f"Updated API key: name={name}, role={new_role}, projects={new_projects}")
        return {"name": name, "role": new_role, "projects": new_projects}

    # ── Backup ───────────────────────────────────────────────────────────

    async def get_recent_memories(
        self,
        project_id: Optional[str] = None,
        kind: Optional[str] = None,
        since: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Memory]:
        """Get recent memories newest-first, with optional filters.

        Args:
            project_id: filter by project
            kind: filter by memory kind
            since: ISO timestamp — only return memories created after this
            limit: max rows
            offset: pagination offset
        """
        conditions = ["NOT obsolete"]
        params: list = []
        if project_id:
            conditions.append("project_id = %s")
            params.append(project_id)
        if kind:
            conditions.append("kind = %s")
            params.append(kind)
        if since:
            conditions.append("created_at > %s")
            params.append(since)
        where = " AND ".join(conditions)
        params.extend([limit, offset])
        async with self.pool.connection() as conn:
            cur = await conn.execute(
                f"SELECT * FROM memories WHERE {where} ORDER BY created_at DESC LIMIT %s OFFSET %s",
                params,
            )
            rows = await cur.fetchall()
        return [self._row_to_memory(r) for r in rows]

    async def get_all_memories(
        self,
        project_id: Optional[str] = None,
        include_obsolete: bool = False,
        limit: int = 10000,
    ) -> list[Memory]:
        """Get all memories, for backup/export."""
        conditions = []
        params: list = []
        if not include_obsolete:
            conditions.append("NOT obsolete")
        if project_id:
            conditions.append("project_id = %s")
            params.append(project_id)
        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        params.append(limit)
        async with self.pool.connection() as conn:
            cur = await conn.execute(
                f"SELECT * FROM memories {where} ORDER BY created_at ASC LIMIT %s",
                params,
            )
            rows = await cur.fetchall()
        return [self._row_to_memory(r) for r in rows]

    async def get_graph_data(
        self,
        project_name: Optional[str] = None,
        limit_nodes: int = 300,
    ) -> dict:
        """Return entity graph data (nodes + edges) for visualization.

        Two modes:
          - Entity graph: entities as nodes, relationships as edges
          - Similarity clusters: if requested, compute cosine-sim between memory embeddings
        """
        async with self.pool.connection() as conn:
            # ── Nodes: entities ──
            if project_name:
                proj = await self.get_project(project_name)
                if not proj:
                    return {"nodes": [], "edges": [], "stats": {}}
                cur = await conn.execute(
                    """SELECT id, name, kind, project_id, properties, created_at
                       FROM entities WHERE project_id = %s
                       ORDER BY created_at DESC LIMIT %s""",
                    (proj.id, limit_nodes),
                )
            else:
                cur = await conn.execute(
                    """SELECT id, name, kind, project_id, properties, created_at
                       FROM entities
                       ORDER BY created_at DESC LIMIT %s""",
                    (limit_nodes,),
                )
            entity_rows = await cur.fetchall()
            id_set = {r["id"] for r in entity_rows}

            # Build node list
            nodes = []
            for r in entity_rows:
                nodes.append({
                    "id": r["id"],
                    "name": r["name"],
                    "kind": r["kind"],
                    "project_id": r["project_id"],
                    "created_at": r["created_at"].isoformat() if r.get("created_at") else None,
                })

            # ── Edges: relationships between visible entities ──
            id_list: list = []
            if id_set:
                id_list = list(id_set)
                cur = await conn.execute(
                    """SELECT r.from_entity, r.to_entity, r.relation,
                              e1.name AS from_name, e2.name AS to_name
                       FROM relationships r
                       JOIN entities e1 ON r.from_entity = e1.id
                       JOIN entities e2 ON r.to_entity = e2.id
                       WHERE r.from_entity = ANY(%s) AND r.to_entity = ANY(%s)""",
                    (id_list, id_list),
                )
                edge_rows = await cur.fetchall()
            else:
                edge_rows = []

            edges = [
                {
                    "source": r["from_entity"],
                    "target": r["to_entity"],
                    "relation": r["relation"],
                    "source_name": r["from_name"],
                    "target_name": r["to_name"],
                }
                for r in edge_rows
            ]

            # ── Weights: count relationships per entity ──
            degree_map: dict[str, int] = {}
            for e in edges:
                degree_map[e["source"]] = degree_map.get(e["source"], 0) + 1
                degree_map[e["target"]] = degree_map.get(e["target"], 0) + 1
            for n in nodes:
                n["weight"] = degree_map.get(n["id"], 0)

            # ── Heat data: aggregate access stats from linked memories ──
            if id_list:
                cur = await conn.execute(
                    """SELECT mel.entity_id,
                              MAX(m.last_accessed) AS last_accessed,
                              SUM(COALESCE(m.access_count, 0)) AS access_count
                       FROM memory_entities mel
                       JOIN memories m ON mel.memory_id = m.id
                       WHERE mel.entity_id = ANY(%s)
                       GROUP BY mel.entity_id""",
                    (id_list,),
                )
                heat_rows = await cur.fetchall()
                heat_map = {
                    r["entity_id"]: {
                        "last_accessed": r["last_accessed"].isoformat()
                        if r["last_accessed"]
                        else None,
                        "access_count": int(r["access_count"] or 0),
                    }
                    for r in heat_rows
                }
            else:
                heat_map = {}
            for n in nodes:
                h = heat_map.get(n["id"], {})
                n["last_accessed"] = h.get("last_accessed")
                n["access_count"] = h.get("access_count", 0)

            # ── Stats ──
            kind_counts: dict[str, int] = {}
            for n in nodes:
                kind_counts[n["kind"]] = kind_counts.get(n["kind"], 0) + 1
            relation_counts: dict[str, int] = {}
            for e in edges:
                relation_counts[e["relation"]] = relation_counts.get(e["relation"], 0) + 1

            return {
                "nodes": nodes,
                "edges": edges,
                "stats": {
                    "total_nodes": len(nodes),
                    "total_edges": len(edges),
                    "kinds": kind_counts,
                    "relations": relation_counts,
                },
            }

    async def get_similarity_graph(
        self,
        project_name: Optional[str] = None,
        limit: int = 100,
        threshold: float = 0.75,
    ) -> dict:
        """Build a similarity graph from memory embeddings using cosine distance."""
        async with self.pool.connection() as conn:
            conditions = ["NOT obsolete", "embedding IS NOT NULL"]
            params: list = []
            if project_name:
                proj = await self.get_project(project_name)
                if not proj:
                    return {"nodes": [], "edges": [], "stats": {}}
                conditions.append("project_id = %s")
                params.append(proj.id)
            where = " AND ".join(conditions)
            params.append(limit)

            cur = await conn.execute(
                f"""SELECT id, kind, subject, content,
                       last_accessed, access_count, created_at
                    FROM memories
                    WHERE {where}
                    ORDER BY created_at DESC LIMIT %s""",
                params,
            )
            rows = await cur.fetchall()

            if len(rows) < 2:
                return {"nodes": [], "edges": [], "stats": {}}

            # Build nodes
            nodes = []
            for r in rows:
                nodes.append({
                    "id": r["id"],
                    "name": r["subject"] or r["id"][:12],
                    "kind": r["kind"],
                    "last_accessed": r["last_accessed"].isoformat() if r.get("last_accessed") else None,
                    "access_count": r["access_count"] or 0,
                    "created_at": r["created_at"].isoformat() if r.get("created_at") else None,
                    "content_preview": (r["content"] or "")[:80],
                })

            # Compute pairwise cosine similarity using pgvector
            # Use a self-join on the memory IDs — cosine distance stays in PostgreSQL
            id_list = [r["id"] for r in rows]
            cur = await conn.execute(
                """SELECT a.id AS id_a, b.id AS id_b,
                          (1 - (a.embedding <=> b.embedding))::double precision AS similarity
                   FROM memories a, memories b
                   WHERE a.id = ANY(%s) AND b.id = ANY(%s)
                     AND a.id < b.id
                     AND (1 - (a.embedding <=> b.embedding))::double precision > %s
                   ORDER BY similarity DESC
                   LIMIT 1000""",
                (id_list, id_list, threshold),
            )
            sim_rows = await cur.fetchall()

            edges = [
                {
                    "source": r["id_a"],
                    "target": r["id_b"],
                    "relation": "similar",
                    "similarity": round(float(r["similarity"]), 3),
                }
                for r in sim_rows
            ]

            return {
                "nodes": nodes,
                "edges": edges,
                "stats": {
                    "total_nodes": len(nodes),
                    "total_edges": len(edges),
                    "threshold": threshold,
                },
            }

    async def get_detailed_stats(self) -> dict:
        """Detailed statistics for the dashboard (single-query counts)."""
        async with self.pool.connection() as conn:
            # Consolidate 7 separate COUNTs into one query
            cur = await conn.execute("""
                SELECT
                    (SELECT COUNT(*) FROM memories WHERE NOT obsolete) AS total_memories,
                    (SELECT COUNT(*) FROM memories WHERE NOT obsolete AND embedding IS NOT NULL) AS total_vectors,
                    (SELECT COUNT(*) FROM entities) AS total_entities,
                    (SELECT COUNT(*) FROM projects) AS total_projects,
                    (SELECT COUNT(*) FROM sessions) AS total_sessions,
                    (SELECT COUNT(*) FROM api_keys WHERE revoked_at IS NULL) AS active_api_keys,
                    (SELECT COUNT(*) FROM memories WHERE obsolete) AS obsolete_memories
            """)
            counts = await cur.fetchone()
            stats = {
                "total_memories": counts["total_memories"],
                "total_vectors": counts["total_vectors"],
                "total_entities": counts["total_entities"],
                "total_projects": counts["total_projects"],
                "total_sessions": counts["total_sessions"],
                "active_api_keys": counts["active_api_keys"],
                "obsolete_memories": counts["obsolete_memories"],
            }

            # Memories by kind
            cur = await conn.execute(
                """SELECT kind, COUNT(*) AS cnt FROM memories
                   WHERE NOT obsolete GROUP BY kind ORDER BY cnt DESC"""
            )
            stats["memories_by_kind"] = {r["kind"]: r["cnt"] for r in await cur.fetchall()}

            # Memories by project
            cur = await conn.execute(
                """SELECT p.name, COUNT(m.id) AS cnt
                   FROM memories m
                   JOIN projects p ON p.id = m.project_id
                   WHERE NOT m.obsolete
                   GROUP BY p.name ORDER BY cnt DESC"""
            )
            stats["by_project"] = {r["name"]: r["cnt"] for r in await cur.fetchall()}

            # Recent activity (last 7 days)
            cur = await conn.execute(
                """SELECT DATE(created_at) AS day, COUNT(*) AS cnt
                   FROM memories
                   WHERE created_at > NOW() - INTERVAL '7 days' AND NOT obsolete
                   GROUP BY DATE(created_at) ORDER BY day"""
            )
            stats["recent_activity"] = [
                {"day": str(r["day"]), "count": r["cnt"]}
                for r in await cur.fetchall()
            ]

            # Recent sessions (last 50)
            cur = await conn.execute(
                """SELECT s.*, p.name AS project_name
                   FROM sessions s
                   LEFT JOIN projects p ON p.id = s.project_id
                   ORDER BY s.started_at DESC LIMIT 50"""
            )
            stats["recent_sessions"] = [
                {
                    "id": r["id"],
                    "task": r["task"],
                    "project_id": r["project_id"],
                    "project_name": r.get("project_name"),
                    "started_at": r["started_at"].isoformat() if r["started_at"] else None,
                    "ended_at": r["ended_at"].isoformat() if r.get("ended_at") else None,
                    "summary": r.get("summary"),
                    "handoff": r.get("handoff"),
                }
                for r in await cur.fetchall()
            ]

        return stats

    # ── Re-embed support ─────────────────────────────────────────────────

    async def alter_embedding_dimension(self, new_dim: int) -> None:
        """Drop HNSW index, null out embeddings, alter column to new dimension.

        Call this before re-embedding when the model dimension changes.
        """
        async with self.pool.connection() as conn:
            await conn.execute("DROP INDEX IF EXISTS idx_memories_embedding_hnsw")
            await conn.execute("UPDATE memories SET embedding = NULL")
            await conn.execute(
                f"ALTER TABLE memories ALTER COLUMN embedding TYPE vector({new_dim})"
            )
            await conn.commit()
        logger.info(f"Embedding column altered to vector({new_dim})")

    async def update_embedding(self, memory_id: str, embedding: list[float]) -> None:
        """Update the embedding vector for a single memory."""
        async with self.pool.connection() as conn:
            await conn.execute(
                "UPDATE memories SET embedding = %s WHERE id = %s",
                (str(embedding), memory_id),
            )
            await conn.commit()

    async def rebuild_hnsw_index(self) -> None:
        """Rebuild the HNSW index on the embedding column."""
        async with self.pool.connection() as conn:
            await conn.execute("DROP INDEX IF EXISTS idx_memories_embedding_hnsw")
            await conn.execute(
                "CREATE INDEX idx_memories_embedding_hnsw "
                "ON memories USING hnsw (embedding vector_cosine_ops) "
                "WITH (m = 32, ef_construction = 200)"
            )
            await conn.commit()
        logger.info("HNSW index rebuilt")

    async def get_memories_needing_embedding(self, limit: int = 10000) -> list[Memory]:
        """Get non-obsolete memories that have no embedding (for re-embed)."""
        async with self.pool.connection() as conn:
            cur = await conn.execute(
                "SELECT * FROM memories WHERE NOT obsolete AND embedding IS NULL "
                "ORDER BY created_at ASC LIMIT %s",
                (limit,),
            )
            rows = await cur.fetchall()
        return [self._row_to_memory(r) for r in rows]

    # ── Reflection helpers ───────────────────────────────────────────────

    async def get_memories_for_gc(
        self,
        min_age_days: float = 7.0,
        limit: int = 5000,
    ) -> list[Memory]:
        """Get active memories old enough to be candidates for GC."""
        async with self.pool.connection() as conn:
            cur = await conn.execute(
                """SELECT * FROM memories
                   WHERE NOT obsolete
                     AND created_at < NOW() - (%s || ' days')::interval
                   ORDER BY access_count ASC, created_at ASC
                   LIMIT %s""",
                (str(min_age_days), limit),
            )
            rows = await cur.fetchall()
        return [self._row_to_memory(r) for r in rows]

    async def find_similar_clusters(
        self,
        project_id: Optional[str],
        similarity_threshold: float = 0.88,
        min_cluster_size: int = 3,
        limit: int = 10,
    ) -> list[list[Memory]]:
        """Find clusters of semantically similar memories within a project.

        Uses pgvector cosine similarity to find groups of memories that are
        very similar to each other.  Returns up to `limit` clusters, each
        containing at least `min_cluster_size` memories.

        Algorithm:
            1. Fetch up to 2000 candidate memories ordered by created_at ASC.
            2. For each candidate ("pivot"), use the HNSW index to find its
               nearest neighbours above `similarity_threshold`.
            3. If a pivot yields >= `min_cluster_size - 1` unused neighbours,
               form a cluster and mark all member IDs as used.
            4. Early termination: if 50 consecutive pivots fail to form a
               cluster, stop — the remaining candidates (ordered by age) are
               unlikely to cluster either, since newer memories tend to be
               more redundant.  This keeps the cost linear in practice even
               with 2000 candidates.

        Optimisation: uses a single DB connection for all KNN probes instead
        of opening a new connection per pivot.
        """
        proj_filter = "AND project_id = %s" if project_id else "AND project_id IS NULL"
        proj_filter_q = "AND m.project_id = %s" if project_id else "AND m.project_id IS NULL"
        proj_param = [project_id] if project_id else []

        async with self.pool.connection() as conn:
            # Get candidate memories with embeddings
            cur = await conn.execute(
                f"""SELECT id, content, kind, subject, confidence, tags,
                       storage_strength, retrieval_strength, access_count,
                       last_accessed, created_at, updated_at, project_id,
                       session_id, supersedes, obsolete, version, version_of,
                       simhash
                   FROM memories
                   WHERE NOT obsolete AND embedding IS NOT NULL
                     {proj_filter}
                   ORDER BY created_at ASC
                   LIMIT 2000""",
                proj_param,
            )
            candidates = await cur.fetchall()

            if len(candidates) < min_cluster_size:
                return []

            used_ids: set[str] = set()
            clusters: list[list[Memory]] = []
            misses = 0  # consecutive pivots that didn't form a cluster

            for pivot_row in candidates:
                if len(clusters) >= limit:
                    break
                # Early termination: if 50 consecutive pivots fail, remaining
                # candidates are unlikely to cluster either.
                if misses > 50:
                    break
                pivot_id = pivot_row["id"]
                if pivot_id in used_ids:
                    continue

                # Find neighbours using the same connection (no pool round-trip)
                cur = await conn.execute(
                    f"""SELECT m.*, 1 - (m.embedding <=> p.embedding) AS sim
                       FROM memories m, memories p
                       WHERE p.id = %s
                         AND m.id != p.id
                         AND NOT m.obsolete
                         AND m.embedding IS NOT NULL
                         {proj_filter_q}
                         AND 1 - (m.embedding <=> p.embedding) >= %s
                       ORDER BY sim DESC
                       LIMIT 20""",
                    [pivot_id] + proj_param + [similarity_threshold],
                )
                neighbour_rows = await cur.fetchall()

                # Filter out already-used IDs
                neighbours = [
                    r for r in neighbour_rows if r["id"] not in used_ids
                ]

                if len(neighbours) < min_cluster_size - 1:
                    misses += 1
                    continue

                misses = 0
                cluster_rows = [pivot_row] + neighbours[: min_cluster_size + 5]
                cluster = [self._row_to_memory(r) for r in cluster_rows]
                for m in cluster:
                    used_ids.add(m.id)
                clusters.append(cluster)

        return clusters

    async def find_conflicting_pairs(
        self,
        kind: str,
        similarity_threshold: float = 0.85,
        min_age_gap_days: float = 7.0,
        limit: int = 20,
    ) -> list[tuple[Memory, Memory]]:
        """Find pairs of semantically similar memories where one is significantly newer.

        Strategy: instead of an O(n²) self-join across all memories of the same
        kind, we iterate over recent "newer" candidates and use pgvector's
        HNSW index via ORDER BY <=> to find their nearest older neighbour.
        This is O(n·log n) at worst and leverages the vector index.

        Returns list of (newer_memory, older_memory, similarity) tuples.
        """
        # Step 1: Get recent candidate memories (the "newer" half of each pair)
        # Use a single connection for both the candidate fetch and KNN probes.
        async with self.pool.connection() as conn:
            cur = await conn.execute(
                """SELECT id, embedding, created_at
                   FROM memories
                   WHERE kind = %s
                     AND NOT obsolete
                     AND embedding IS NOT NULL
                   ORDER BY created_at DESC
                   LIMIT %s""",
                (kind, limit * 5),  # oversample to find enough pairs
            )
            newer_candidates = await cur.fetchall()

        if not newer_candidates:
            return []

        # Step 2: For each newer candidate, find its closest older neighbour
        # using pgvector index (fast KNN, not a full cross-join)
        pair_ids: list[tuple[str, str, float]] = []  # (newer_id, older_id, sim)

        async with self.pool.connection() as conn:
            for nc in newer_candidates:
                if len(pair_ids) >= limit:
                    break
                cur = await conn.execute(
                    """SELECT m.id,
                            1 - (m.embedding <=> %s::vector) AS sim
                       FROM memories m
                       WHERE m.kind = %s
                         AND m.id != %s
                         AND NOT m.obsolete
                         AND m.embedding IS NOT NULL
                         AND m.created_at < %s
                         AND %s::timestamptz - m.created_at > (%s || ' days')::interval
                       ORDER BY m.embedding <=> %s::vector
                       LIMIT 1""",
                    (
                        str(nc["embedding"]),
                        kind,
                        nc["id"],
                        nc["created_at"],
                        nc["created_at"],
                        str(min_age_gap_days),
                        str(nc["embedding"]),
                    ),
                )
                row = await cur.fetchone()
                if row and row["sim"] >= similarity_threshold:
                    pair_ids.append((nc["id"], row["id"], row["sim"]))

        if not pair_ids:
            return []

        # Step 3: Fetch full memory objects for each pair
        all_ids = list({pid for pair in pair_ids for pid in (pair[0], pair[1])})
        async with self.pool.connection() as conn:
            placeholders = ",".join(["%s"] * len(all_ids))
            cur = await conn.execute(
                f"SELECT * FROM memories WHERE id IN ({placeholders})",
                all_ids,
            )
            rows = await cur.fetchall()

        mem_map = {r["id"]: self._row_to_memory(r) for r in rows}

        result: list[tuple[Memory, Memory, float]] = []
        for newer_id, older_id, sim in sorted(pair_ids, key=lambda x: -x[2]):
            newer = mem_map.get(newer_id)
            older = mem_map.get(older_id)
            if newer and older:
                result.append((newer, older, sim))

        return result[:limit]

    async def set_memory_project(self, memory_id: str, project_id: str) -> None:
        """Set the project_id on a memory (used by consolidation)."""
        async with self.pool.connection() as conn:
            await conn.execute(
                "UPDATE memories SET project_id = %s, updated_at = NOW() WHERE id = %s",
                (project_id, memory_id),
            )
            await conn.commit()

    # ── Row Mapping ──────────────────────────────────────────────────────

    @staticmethod
    def _row_to_project(row: dict) -> Project:
        return Project(
            id=row["id"],
            name=row["name"],
            path=row.get("path"),
            description=row.get("description"),
            persistent_memories=row.get("persistent_memories", False),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    @staticmethod
    def _row_to_session(row: dict) -> Session:
        return Session(
            id=row["id"],
            project_id=row.get("project_id"),
            task=row.get("task"),
            started_at=row["started_at"],
            ended_at=row.get("ended_at"),
            summary=row.get("summary"),
            handoff=row.get("handoff"),
            session_ordinal=row.get("session_ordinal"),
        )

    @staticmethod
    def _row_to_memory(row: dict) -> Memory:
        tags = row.get("tags", [])
        if isinstance(tags, str):
            try:
                tags = json.loads(tags)
            except (json.JSONDecodeError, TypeError):
                tags = []

        return Memory(
            id=row["id"],
            project_id=row.get("project_id"),
            session_id=row.get("session_id"),
            kind=MemoryKind(row["kind"]),
            content=row["content"],
            subject=row.get("subject"),
            confidence=row.get("confidence", 1.0),
            supersedes=row.get("supersedes"),
            obsolete=row.get("obsolete", False),
            tags=tags,
            version=row.get("version", 1),
            version_of=row.get("version_of"),
            simhash=row.get("simhash"),
            storage_strength=row.get("storage_strength", 0.0),
            retrieval_strength=row.get("retrieval_strength", 1.0),
            access_count=row.get("access_count", 0),
            last_accessed=row.get("last_accessed"),
            pinned=row.get("pinned", False),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    @staticmethod
    def _row_to_entity(row: dict) -> Entity:
        props = row.get("properties", {})
        if isinstance(props, str):
            try:
                props = json.loads(props)
            except (json.JSONDecodeError, TypeError):
                props = {}

        project_id = row.get("project_id")
        if project_id == "__global__":
            project_id = None

        return Entity(
            id=row["id"],
            name=row["name"],
            kind=EntityKind(row["kind"]),
            project_id=project_id,
            properties=props,
            created_at=row.get("created_at"),
        )
