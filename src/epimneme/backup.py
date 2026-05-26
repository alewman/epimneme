"""Backup & restore — JSON archive of the full engram knowledge base.

Archive format (current = version 2):
  {
    "format_version": 2,
    "epimneme_version": "0.5.0",
    "created_at": "...",
    "metadata": { ... },
    "tables": {
      "projects": [ ... ],
      "sessions": [ ... ],
      "memories": [ ... ],
      "entities": [ ... ],
      "relationships": [ ... ],
      "memory_entities": [ ... ],
      "memory_access": [ ... ]
    }
  }

Embeddings are stored as JSON arrays (list[float]).  On restore they are
written back via ``::vector`` cast.

API keys and schema_migrations are NOT included — keys are security-sensitive
and migrations are auto-managed.

Backward compatibility
─────────────────────
Older backup files are automatically upgraded to the current format_version
before restore.  Each format version bump has a registered upgrade function
that transforms row schemas (adds new columns with defaults, renames columns,
adds missing tables, etc.).  The chain is applied sequentially:

  v1 → v2 → ... → CURRENT_FORMAT_VERSION

To add a new format version:
  1. Bump CURRENT_FORMAT_VERSION
  2. Write ``_upgrade_vN_to_vM(archive)`` that mutates and returns the archive
  3. Register it in _UPGRADE_CHAIN
  4. Update TABLE_COLUMNS to include any new columns
"""

from __future__ import annotations

import copy
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# ── Format version ───────────────────────────────────────────────────────────
# Bump this when TABLE_COLUMNS or TABLE_ORDER change.  Then add an upgrade
# function in _UPGRADE_CHAIN so old backups can be migrated forward.
CURRENT_FORMAT_VERSION = 2

# Tables in FK-safe insertion order
TABLE_ORDER = [
    "projects",
    "sessions",
    "memories",
    "entities",
    "relationships",
    "memory_entities",
    "memory_access",
]

# Columns to export per table (explicit to avoid leaking internal cols)
TABLE_COLUMNS: dict[str, list[str]] = {
    "projects": [
        "id", "name", "path", "description", "created_at", "updated_at",
    ],
    "sessions": [
        "id", "project_id", "task", "started_at", "ended_at", "summary", "handoff",
    ],
    "memories": [
        "id", "project_id", "session_id", "kind", "content", "subject",
        "confidence", "supersedes", "obsolete", "tags", "embedding",
        "created_at", "updated_at",
        # versioning
        "version", "version_of",
        # dedup
        "simhash",
        # decay
        "storage_strength", "retrieval_strength", "access_count", "last_accessed",
        # pinning
        "pinned",
    ],
    "entities": [
        "id", "name", "kind", "project_id", "properties", "created_at",
    ],
    "relationships": [
        "id", "from_entity", "to_entity", "relation", "properties",
    ],
    "memory_entities": [
        "memory_id", "entity_id", "relation", "created_at",
    ],
    "memory_access": [
        "id", "memory_id", "accessed_at", "context",
    ],
}


# ── Format upgrade chain ─────────────────────────────────────────────────────


def _upgrade_v1_to_v2(archive: dict) -> dict:
    """Upgrade a format_version=1 archive to version 2.

    Version 1 → 2 changes (engram ≤0.3.x → 0.4.x):
      - memories: added version, version_of, simhash, storage_strength,
        retrieval_strength, access_count, last_accessed
      - Added memory_entities table (may be absent in v1)
      - Added memory_access table (may be absent in v1)
    """
    tables = archive.get("tables", {})

    # ── memories: add new columns with sensible defaults ──
    _V2_MEMORY_DEFAULTS = {
        "version": 1,
        "version_of": None,
        "simhash": None,
        "storage_strength": 0.0,
        "retrieval_strength": 1.0,
        "access_count": 0,
        "last_accessed": None,
    }
    for row in tables.get("memories", []):
        for col, default in _V2_MEMORY_DEFAULTS.items():
            row.setdefault(col, default)

    # ── ensure new tables exist ──
    tables.setdefault("memory_entities", [])
    tables.setdefault("memory_access", [])

    archive["format_version"] = 2
    archive.setdefault("metadata", {})
    archive["metadata"].setdefault("upgrades_applied", [])
    archive["metadata"]["upgrades_applied"].append(
        {"from": 1, "to": 2, "note": "Added decay/versioning/dedup columns to memories"}
    )
    logger.info("Backup upgrade: v1 → v2 (added decay/versioning/dedup columns)")
    return archive


# Registry: source_version → upgrade function.  Each function takes and returns
# the archive dict, bumping format_version from N to N+1.
_UPGRADE_CHAIN: dict[int, Callable[[dict], dict]] = {
    1: _upgrade_v1_to_v2,
    # Future: 2: _upgrade_v2_to_v3, etc.
}


def upgrade_archive(archive: dict) -> dict:
    """Upgrade a backup archive to CURRENT_FORMAT_VERSION.

    Applies chained upgrades: v1→v2→...→current.  Returns the archive
    (mutated in-place for efficiency).  Raises ValueError if the archive
    version is newer than we support (can't downgrade) or has no upgrade
    path.

    A deep copy is made before mutation so the caller's original is not
    modified.
    """
    archive = copy.deepcopy(archive)
    fv = archive.get("format_version", 1)

    if fv == CURRENT_FORMAT_VERSION:
        return archive

    if fv > CURRENT_FORMAT_VERSION:
        raise ValueError(
            f"Backup format_version {fv} is newer than this engram supports "
            f"(max {CURRENT_FORMAT_VERSION}).  Upgrade engram first."
        )

    if fv < 1:
        raise ValueError(f"Invalid format_version: {fv}")

    while fv < CURRENT_FORMAT_VERSION:
        fn = _UPGRADE_CHAIN.get(fv)
        if fn is None:
            raise ValueError(
                f"No upgrade path from format_version {fv} to "
                f"{CURRENT_FORMAT_VERSION}.  Missing migration for v{fv}→v{fv+1}."
            )
        archive = fn(archive)
        fv = archive["format_version"]

    # Recalculate metadata row counts after upgrades
    tables = archive.get("tables", {})
    total = sum(len(tables.get(t, [])) for t in TABLE_ORDER)
    archive["metadata"]["total_rows"] = total
    archive["metadata"]["tables"] = {
        t: len(tables.get(t, [])) for t in TABLE_ORDER
    }

    return archive


# ── Serialisation helpers ────────────────────────────────────────────────────


def _serialise_row(row: dict) -> dict:
    """Convert a DB row dict to JSON-safe values."""
    out: dict[str, Any] = {}
    for k, v in row.items():
        if isinstance(v, datetime):
            out[k] = v.isoformat()
        elif isinstance(v, memoryview):
            out[k] = bytes(v).hex()
        elif hasattr(v, "tolist"):  # numpy array from pgvector
            out[k] = v.tolist()
        elif isinstance(v, (list, dict)):
            out[k] = v
        else:
            out[k] = v
    return out


async def export_backup(pool, epimneme_version: str = "0.5.0") -> dict:
    """Export the full database as a JSON-serialisable dict.

    Args:
        pool: psycopg_pool.AsyncConnectionPool
        epimneme_version: current engram version for metadata

    Returns:
        dict ready to be written via json.dump()
    """
    tables: dict[str, list[dict]] = {}
    row_count = 0

    async with pool.connection() as conn:
        for table in TABLE_ORDER:
            cols = TABLE_COLUMNS[table]
            col_list = ", ".join(cols)
            cur = await conn.execute(f"SELECT {col_list} FROM {table}")
            rows = await cur.fetchall()
            tables[table] = [_serialise_row(r) for r in rows]
            row_count += len(tables[table])
            logger.info(f"Backup: exported {len(tables[table])} rows from {table}")

    archive = {
        "format_version": CURRENT_FORMAT_VERSION,
        "epimneme_version": epimneme_version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "metadata": {
            "total_rows": row_count,
            "tables": {t: len(tables[t]) for t in TABLE_ORDER},
        },
        "tables": tables,
    }
    return archive


async def save_backup(
    pool,
    backup_dir: str | Path,
    epimneme_version: str = "0.5.0",
    label: Optional[str] = None,
) -> dict:
    """Export database and write to a timestamped JSON file.

    Returns metadata dict with filename, path, size, row counts.
    """
    backup_dir = Path(backup_dir)
    backup_dir.mkdir(parents=True, exist_ok=True)

    archive = await export_backup(pool, epimneme_version)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_label = ""
    if label:
        safe_label = "_" + "".join(c if c.isalnum() or c in "-_" else "_" for c in label)
    filename = f"epimneme_backup_{ts}{safe_label}.json"
    filepath = backup_dir / filename

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(archive, f, ensure_ascii=False, indent=None, separators=(",", ":"))

    size_bytes = filepath.stat().st_size
    logger.info(
        f"Backup saved: {filename} ({size_bytes:,} bytes, "
        f"{archive['metadata']['total_rows']} rows)"
    )
    return {
        "filename": filename,
        "path": str(filepath),
        "size_bytes": size_bytes,
        "created_at": archive["created_at"],
        "metadata": archive["metadata"],
    }


def list_backups(backup_dir: str | Path) -> list[dict]:
    """List available backup files in the backup directory.

    Returns list of dicts sorted newest-first.
    """
    backup_dir = Path(backup_dir)
    if not backup_dir.exists():
        return []

    results = []
    for p in sorted(backup_dir.glob("epimneme_backup_*.json"), reverse=True):
        stat = p.stat()
        # Try to read metadata without loading full file
        meta = _peek_metadata(p)
        results.append({
            "filename": p.name,
            "size_bytes": stat.st_size,
            "modified_at": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
            "format_version": meta.get("format_version"),
            "epimneme_version": meta.get("epimneme_version"),
            "created_at": meta.get("created_at"),
            "metadata": meta.get("metadata", {}),
        })
    return results


def _peek_metadata(filepath: Path) -> dict:
    """Read only metadata fields from a backup file without loading all tables.

    Uses a streaming approach: reads the file character-by-character via
    raw_decode to parse only the top-level scalar keys and the 'metadata'
    object, stopping before the large 'tables' blob.
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            # Read up to 8KB — enough for format_version, epimneme_version,
            # created_at, and metadata{} which precede the large tables{}.
            # Backup files put tables last, so this avoids loading multi-MB data.
            head = f.read(8192)

        result: dict[str, Any] = {}
        # Try to extract each known top-level key from the partial JSON
        import re

        for key in ("format_version", "epimneme_version", "created_at"):
            m = re.search(rf'"{key}"\s*:\s*(".*?"|[\d.]+|null|true|false)', head)
            if m:
                val = m.group(1)
                # Strip quotes from string values
                if val.startswith('"'):
                    result[key] = val[1:-1]
                elif val == "null":
                    result[key] = None
                else:
                    # Try int first, then float
                    try:
                        result[key] = int(val)
                    except ValueError:
                        result[key] = float(val)

        # Extract the metadata object — it's small (table counts, etc.)
        meta_match = re.search(r'"metadata"\s*:\s*(\{)', head)
        if meta_match:
            # Find matching closing brace, handling nesting
            start = meta_match.start(1)
            depth = 0
            for i, ch in enumerate(head[start:], start):
                if ch == '{':
                    depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0:
                        try:
                            result["metadata"] = json.loads(head[start:i + 1])
                        except json.JSONDecodeError:
                            result["metadata"] = {}
                        break
            else:
                result["metadata"] = {}
        else:
            result["metadata"] = {}

        return result
    except Exception:
        return {}


def _safe_backup_path(backup_dir: str | Path, filename: str) -> Path:
    """Resolve a backup filename, rejecting path traversal attempts."""
    base = Path(backup_dir).resolve()
    filepath = (base / filename).resolve()
    if not filepath.parent == base:
        raise ValueError(f"Invalid backup filename: {filename}")
    return filepath


def load_backup_file(backup_dir: str | Path, filename: str) -> dict:
    """Load, validate, and auto-upgrade a backup archive from disk.

    Old format versions are upgraded to CURRENT_FORMAT_VERSION via the
    upgrade chain.  Raises FileNotFoundError or ValueError on problems.

    Returns the archive at CURRENT_FORMAT_VERSION.
    """
    filepath = _safe_backup_path(backup_dir, filename)
    if not filepath.exists():
        raise FileNotFoundError(f"Backup file not found: {filename}")

    with open(filepath, "r", encoding="utf-8") as f:
        archive = json.load(f)

    fv = archive.get("format_version")
    if fv is None or not isinstance(fv, int) or fv < 1:
        raise ValueError(f"Unsupported backup format_version: {fv}")

    if "tables" not in archive:
        raise ValueError("Invalid backup: missing 'tables' key")

    # Auto-upgrade old formats to current
    if fv < CURRENT_FORMAT_VERSION:
        logger.info(f"Upgrading backup {filename} from format v{fv} → v{CURRENT_FORMAT_VERSION}")
        archive = upgrade_archive(archive)
    elif fv > CURRENT_FORMAT_VERSION:
        raise ValueError(
            f"Backup format_version {fv} is newer than this engram supports "
            f"(max {CURRENT_FORMAT_VERSION}).  Upgrade engram first."
        )

    return archive


def delete_backup(backup_dir: str | Path, filename: str) -> bool:
    """Delete a backup file. Returns True if deleted, False if not found."""
    filepath = _safe_backup_path(backup_dir, filename)
    if not filepath.exists():
        return False
    filepath.unlink()
    logger.info(f"Deleted backup: {filename}")
    return True


def rotate_backups(
    backup_dir: str | Path,
    keep_last: int = 10,
    keep_days: Optional[int] = None,
) -> list[str]:
    """Delete old backups according to a retention policy.

    Keeps:
      - The most recent ``keep_last`` backups (regardless of age), AND
      - Any backup newer than ``keep_days`` days (if set).

    Returns a list of deleted filenames.
    """
    backups = list_backups(backup_dir)  # sorted newest-first
    if not backups:
        return []

    now = datetime.now(timezone.utc)
    deleted: list[str] = []

    for i, bk in enumerate(backups):
        # Always keep the first keep_last entries
        if i < keep_last:
            continue

        # If keep_days is set, also keep anything within the window
        if keep_days is not None and bk.get("modified_at"):
            try:
                mod_at = datetime.fromisoformat(bk["modified_at"])
                age_days = (now - mod_at).total_seconds() / 86400
                if age_days < keep_days:
                    continue
            except (ValueError, TypeError):
                pass

        # This backup is beyond both retention windows — delete it
        fn = bk["filename"]
        filepath = _safe_backup_path(backup_dir, fn)
        if filepath.exists():
            filepath.unlink()
            deleted.append(fn)
            logger.info(f"Rotated old backup: {fn}")

    if deleted:
        logger.info(f"Backup rotation: deleted {len(deleted)} old backup(s)")
    return deleted


# ── Restore ──────────────────────────────────────────────────────────────────

# SQL for each table — parameterised INSERT with ON CONFLICT handling
# so restore is idempotent (re-running won't fail on existing rows).

_RESTORE_SQL: dict[str, str] = {
    "projects": """
        INSERT INTO projects (id, name, path, description, created_at, updated_at)
        VALUES (%(id)s, %(name)s, %(path)s, %(description)s,
                %(created_at)s, %(updated_at)s)
        ON CONFLICT (id) DO UPDATE SET
            name=EXCLUDED.name, path=EXCLUDED.path, description=EXCLUDED.description,
            updated_at=EXCLUDED.updated_at
    """,
    "sessions": """
        INSERT INTO sessions (id, project_id, task, started_at, ended_at, summary, handoff)
        VALUES (%(id)s, %(project_id)s, %(task)s, %(started_at)s,
                %(ended_at)s, %(summary)s, %(handoff)s)
        ON CONFLICT (id) DO UPDATE SET
            summary=EXCLUDED.summary, handoff=EXCLUDED.handoff, ended_at=EXCLUDED.ended_at
    """,
    "memories": """
        INSERT INTO memories (
            id, project_id, session_id, kind, content, subject,
            confidence, supersedes, obsolete, tags, embedding,
            created_at, updated_at,
            version, version_of, simhash,
            storage_strength, retrieval_strength, access_count, last_accessed,
            pinned
        ) VALUES (
            %(id)s, %(project_id)s, %(session_id)s, %(kind)s, %(content)s, %(subject)s,
            %(confidence)s, %(supersedes)s, %(obsolete)s, %(tags)s::jsonb,
            %(embedding)s::vector,
            %(created_at)s, %(updated_at)s,
            %(version)s, %(version_of)s, %(simhash)s,
            %(storage_strength)s, %(retrieval_strength)s, %(access_count)s, %(last_accessed)s,
            %(pinned)s
        ) ON CONFLICT (id) DO UPDATE SET
            content=EXCLUDED.content, subject=EXCLUDED.subject,
            confidence=EXCLUDED.confidence, obsolete=EXCLUDED.obsolete,
            tags=EXCLUDED.tags, embedding=EXCLUDED.embedding,
            updated_at=EXCLUDED.updated_at,
            version=EXCLUDED.version, version_of=EXCLUDED.version_of,
            simhash=EXCLUDED.simhash,
            storage_strength=EXCLUDED.storage_strength,
            retrieval_strength=EXCLUDED.retrieval_strength,
            access_count=EXCLUDED.access_count,
            last_accessed=EXCLUDED.last_accessed,
            pinned=EXCLUDED.pinned
    """,
    "entities": """
        INSERT INTO entities (id, name, kind, project_id, properties, created_at)
        VALUES (%(id)s, %(name)s, %(kind)s, %(project_id)s, %(properties)s::jsonb,
                %(created_at)s)
        ON CONFLICT (id) DO UPDATE SET
            name=EXCLUDED.name, kind=EXCLUDED.kind, properties=EXCLUDED.properties
    """,
    "relationships": """
        INSERT INTO relationships (id, from_entity, to_entity, relation, properties)
        VALUES (%(id)s, %(from_entity)s, %(to_entity)s, %(relation)s, %(properties)s::jsonb)
        ON CONFLICT (id) DO NOTHING
    """,
    "memory_entities": """
        INSERT INTO memory_entities (memory_id, entity_id, relation, created_at)
        VALUES (%(memory_id)s, %(entity_id)s, %(relation)s, %(created_at)s)
        ON CONFLICT (memory_id, entity_id) DO NOTHING
    """,
    "memory_access": """
        INSERT INTO memory_access (id, memory_id, accessed_at, context)
        VALUES (%(id)s, %(memory_id)s, %(accessed_at)s, %(context)s)
        ON CONFLICT (id) DO NOTHING
    """,
}


def _prepare_row_for_restore(table: str, row: dict) -> dict:
    """Normalise a row dict for parameterised INSERT.

    Handles:
    - Embedding lists → JSON string for ::vector cast
    - JSONB fields → JSON string
    - Missing columns → None
    """
    out: dict[str, Any] = {}
    expected_cols = TABLE_COLUMNS[table]

    for col in expected_cols:
        val = row.get(col)

        # Embedding: list[float] → stringified for ::vector cast
        if col == "embedding" and isinstance(val, list):
            val = json.dumps(val)
        elif col == "embedding" and val is None:
            pass  # keep None

        # JSONB columns: ensure they're JSON strings for ::jsonb casts
        if col in ("tags", "properties"):
            if isinstance(val, (list, dict)):
                val = json.dumps(val)
            elif val is None:
                val = "[]" if col == "tags" else "{}"

        out[col] = val

    return out


async def restore_backup(
    pool,
    archive: dict,
    mode: str = "merge",
) -> dict:
    """Restore a backup archive into the database.

    The archive is auto-upgraded to CURRENT_FORMAT_VERSION before
    restoring, so old backups are always compatible.

    Args:
        pool: psycopg_pool.AsyncConnectionPool
        archive: parsed backup dict (from load_backup_file or direct)
        mode: "merge" (ON CONFLICT UPDATE/IGNORE) or "clean" (wipe + insert)

    Returns:
        dict with per-table restore counts and any errors.
    """
    # Ensure archive is at current format before inserting
    original_fv = archive.get("format_version", 1)
    if original_fv != CURRENT_FORMAT_VERSION:
        archive = upgrade_archive(archive)

    tables = archive.get("tables", {})
    fv = archive.get("format_version", CURRENT_FORMAT_VERSION)
    results: dict[str, int] = {}
    errors: list[str] = []
    upgraded_from = original_fv if original_fv != fv else None

    async with pool.connection() as conn:
        if mode == "clean":
            # Delete in reverse FK order
            for table in reversed(TABLE_ORDER):
                if table in tables:
                    await conn.execute(f"DELETE FROM {table}")
                    logger.info(f"Restore (clean): cleared {table}")
            await conn.commit()

        for table in TABLE_ORDER:
            rows = tables.get(table, [])
            if not rows:
                results[table] = 0
                continue

            sql = _RESTORE_SQL.get(table)
            if not sql:
                errors.append(f"No restore SQL for table: {table}")
                continue

            count = 0
            for row in rows:
                try:
                    prepared = _prepare_row_for_restore(table, row)
                    await conn.execute("SAVEPOINT row_sp")
                    await conn.execute(sql, prepared)
                    await conn.execute("RELEASE SAVEPOINT row_sp")
                    count += 1
                except Exception as e:
                    err_msg = f"{table} row {row.get('id', '?')}: {e}"
                    logger.warning(f"Restore error: {err_msg}")
                    errors.append(err_msg)
                    # Roll back only this row, keep other rows intact
                    await conn.execute("ROLLBACK TO SAVEPOINT row_sp")

            await conn.commit()
            results[table] = count
            logger.info(f"Restore: {count}/{len(rows)} rows into {table}")

    # Rebuild tsvector for full-text search
    try:
        async with pool.connection() as conn:
            await conn.execute("""
                UPDATE memories SET content_tsv =
                    setweight(to_tsvector('english', coalesce(subject,'')), 'A') ||
                    setweight(to_tsvector('english', content), 'B')
                WHERE content_tsv IS NULL
            """)
            await conn.commit()
            logger.info("Restore: rebuilt tsvector index")
    except Exception as e:
        errors.append(f"tsvector rebuild: {e}")

    result = {
        "mode": mode,
        "format_version": fv,
        "rows_restored": results,
        "total_restored": sum(results.values()),
        "errors": errors,
    }
    if upgraded_from is not None:
        result["upgraded_from_format"] = upgraded_from
    return result
