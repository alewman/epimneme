"""Bulk import — ingest files, chat exports, and project directories into Engram.

Supported formats:
  - Project files: .py, .js, .ts, .md, .txt, .yaml, .json, .sh, .go, .rs,
    .rb, .java, .html, .css, .sql, .toml, .cfg, .ini, .xml, .csv, .c, .cpp,
    .h, .hpp, .swift (25 types)
  - Claude.ai JSON exports: [{"role": "user", "content": "..."}]
  - ChatGPT conversations.json: nested mapping tree
  - Claude Code JSONL: type:"human"/"assistant" entries
  - Plain text / markdown chat transcripts (> markers)

Chunking:
  - Project files: paragraph-boundary chunks (800 chars, 100 overlap)
  - Chat transcripts: exchange-pair chunking (user turn + AI response)
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# File types we can ingest
CODE_EXTENSIONS = frozenset({
    ".py", ".js", ".ts", ".jsx", ".tsx", ".go", ".rs", ".rb", ".java",
    ".c", ".cpp", ".h", ".hpp", ".swift", ".sh", ".bash", ".zsh",
})
DOC_EXTENSIONS = frozenset({
    ".md", ".txt", ".rst", ".adoc",
})
CONFIG_EXTENSIONS = frozenset({
    ".yaml", ".yml", ".json", ".toml", ".cfg", ".ini", ".xml", ".env",
})
DATA_EXTENSIONS = frozenset({
    ".csv", ".sql", ".html", ".css",
})
ALL_EXTENSIONS = CODE_EXTENSIONS | DOC_EXTENSIONS | CONFIG_EXTENSIONS | DATA_EXTENSIONS

# Directories to skip during project scanning
SKIP_DIRS = frozenset({
    ".git", "node_modules", "__pycache__", ".venv", "venv", "env",
    "build", "dist", ".tox", ".mypy_cache", ".pytest_cache",
    ".ruff_cache", "coverage", "htmlcov", ".eggs", ".cache",
    ".idea", ".vscode", ".DS_Store",
})

# Chunking parameters (overridable via env vars for different embedding model windows)
CHUNK_SIZE = int(os.environ.get("EPIMNEME_CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.environ.get("EPIMNEME_CHUNK_OVERLAP", "100"))
MIN_CHUNK_SIZE = 50


@dataclass
class ImportChunk:
    """A single chunk ready for storage as a memory."""

    content: str
    kind: str = "fact"  # fact, decision, procedure, etc.
    subject: str = ""
    tags: list[str] = field(default_factory=list)
    source_file: str = ""
    chunk_index: int = 0


@dataclass
class ImportResult:
    """Summary of a bulk import operation."""

    files_processed: int = 0
    files_skipped: int = 0
    chunks_created: int = 0
    chunks_deduplicated: int = 0
    errors: list[str] = field(default_factory=list)


def _chunk_text(text: str, source: str = "", tags: list[str] | None = None) -> list[ImportChunk]:
    """Split text into paragraph-boundary chunks with overlap."""
    if not text.strip():
        return []

    chunks: list[ImportChunk] = []
    paragraphs = text.split("\n\n")
    current = ""
    chunk_idx = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if len(current) + len(para) + 2 > CHUNK_SIZE and current:
            if len(current) >= MIN_CHUNK_SIZE:
                chunks.append(ImportChunk(
                    content=current.strip(),
                    subject=os.path.basename(source) if source else "",
                    tags=list(tags or []),
                    source_file=source,
                    chunk_index=chunk_idx,
                ))
                chunk_idx += 1
            # Keep overlap from end of current chunk
            overlap_text = current[-CHUNK_OVERLAP:] if len(current) > CHUNK_OVERLAP else ""
            current = overlap_text + "\n\n" + para if overlap_text else para
        else:
            current = current + "\n\n" + para if current else para

    # Final chunk
    if current.strip() and len(current.strip()) >= MIN_CHUNK_SIZE:
        chunks.append(ImportChunk(
            content=current.strip(),
            subject=os.path.basename(source) if source else "",
            tags=list(tags or []),
            source_file=source,
            chunk_index=chunk_idx,
        ))

    return chunks


def _detect_room(filepath: str) -> str:
    """Detect a topic/room from file path (for tagging)."""
    parts = Path(filepath).parts
    lower_parts = [p.lower() for p in parts]

    mappings = {
        "frontend": ["frontend", "ui", "components", "views", "pages", "client"],
        "backend": ["backend", "server", "api", "routes", "handlers"],
        "database": ["database", "db", "migrations", "models", "schema"],
        "tests": ["tests", "test", "__tests__", "spec", "specs"],
        "docs": ["docs", "documentation", "wiki", "guides"],
        "config": ["config", "configuration", "settings", ".github"],
        "scripts": ["scripts", "tools", "bin", "utils"],
        "infra": ["docker", "deploy", "infrastructure", "terraform", "k8s"],
    }

    for room, keywords in mappings.items():
        if any(kw in lower_parts for kw in keywords):
            return room

    # Fallback: use file extension to guess
    ext = Path(filepath).suffix.lower()
    if ext in CODE_EXTENSIONS:
        return "code"
    if ext in DOC_EXTENSIONS:
        return "docs"
    if ext in CONFIG_EXTENSIONS:
        return "config"
    return "general"


# ── Project file scanning ────────────────────────────────────────────────────


def scan_project_files(
    directory: str,
    *,
    limit: int = 0,
) -> list[tuple[str, str]]:
    """Scan a directory for importable files.

    Returns list of (filepath, content) tuples.
    """
    results: list[tuple[str, str]] = []
    root = Path(directory)

    if not root.is_dir():
        return results

    for dirpath, dirnames, filenames in os.walk(root):
        # Prune skip directories in-place
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]

        for fname in sorted(filenames):
            if limit and len(results) >= limit:
                return results

            fpath = Path(dirpath) / fname
            if fpath.suffix.lower() not in ALL_EXTENSIONS:
                continue

            try:
                content = fpath.read_text(encoding="utf-8", errors="replace")
                if content.strip():
                    results.append((str(fpath), content))
            except (OSError, UnicodeDecodeError) as e:
                logger.warning(f"Skipping {fpath}: {e}")

    return results


def import_project_files(
    directory: str,
    *,
    wing: str = "",
    limit: int = 0,
) -> tuple[list[ImportChunk], ImportResult]:
    """Scan and chunk a project directory for import.

    Args:
        directory: Path to project root.
        wing: Project/wing name (used as tag).
        limit: Max files to process (0 = all).

    Returns:
        (chunks, result) tuple.
    """
    result = ImportResult()
    all_chunks: list[ImportChunk] = []

    files = scan_project_files(directory, limit=limit)

    for filepath, content in files:
        room = _detect_room(filepath)
        tags = [f"source:{filepath}"]
        if wing:
            tags.append(f"wing:{wing}")
        if room:
            tags.append(f"room:{room}")

        chunks = _chunk_text(content, source=filepath, tags=tags)
        if chunks:
            all_chunks.extend(chunks)
            result.files_processed += 1
        else:
            result.files_skipped += 1

    result.chunks_created = len(all_chunks)
    return all_chunks, result


# ── Chat format normalization ────────────────────────────────────────────────


def _normalize_plain_text(text: str) -> list[tuple[str, str]]:
    """Parse plain text with > markers into (role, content) pairs."""
    exchanges: list[tuple[str, str]] = []
    current_role = "user"
    current_content: list[str] = []

    for line in text.split("\n"):
        if line.startswith("> ") or line.startswith(">"):
            # New user turn
            if current_content:
                exchanges.append((current_role, "\n".join(current_content).strip()))
                current_content = []
            current_role = "user"
            current_content.append(line.lstrip("> ").strip())
        elif current_content or line.strip():
            if current_role == "user" and not line.startswith("> ") and current_content:
                # Switch to assistant after user turn
                exchanges.append(("user", "\n".join(current_content).strip()))
                current_content = []
                current_role = "assistant"
            current_content.append(line)

    if current_content:
        exchanges.append((current_role, "\n".join(current_content).strip()))

    return exchanges


def _normalize_claude_json(data: list[dict]) -> list[tuple[str, str]]:
    """Parse Claude.ai JSON format: [{"role": "user", "content": "..."}]."""
    exchanges: list[tuple[str, str]] = []
    for msg in data:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if isinstance(content, list):
            # Claude sometimes uses content blocks
            content = "\n".join(
                block.get("text", str(block)) if isinstance(block, dict) else str(block)
                for block in content
            )
        if content.strip():
            exchanges.append((role, content.strip()))
    return exchanges


def _normalize_chatgpt_json(data: list[dict]) -> list[tuple[str, str]]:
    """Parse ChatGPT conversations.json format."""
    exchanges: list[tuple[str, str]] = []

    for convo in data:
        mapping = convo.get("mapping", {})
        # Build parent→children tree, walk in order
        nodes = {}
        children_of: dict[str | None, list[str]] = {}
        for node_id, node in mapping.items():
            msg = node.get("message")
            parent = node.get("parent")
            if msg and msg.get("content", {}).get("parts"):
                role = msg.get("author", {}).get("role", "user")
                parts = msg["content"]["parts"]
                content = "\n".join(str(p) for p in parts if isinstance(p, str))
                if content.strip() and role in ("user", "assistant"):
                    nodes[node_id] = (role, content.strip())
            # Track children for all nodes (even non-message ones) to
            # follow the tree structure correctly.
            children_of.setdefault(parent, []).append(node_id)

        # DFS from roots (nodes whose parent isn't in mapping)
        ordered: list[tuple[str, str]] = []
        visited: set[str] = set()

        def _walk(nid: str) -> None:
            if nid in visited:
                return
            visited.add(nid)
            if nid in nodes:
                ordered.append(nodes[nid])
            for child_id in children_of.get(nid, []):
                _walk(child_id)

        # Find roots and walk
        for nid in mapping:
            parent = mapping[nid].get("parent")
            if parent not in mapping:
                _walk(nid)
        # Add remaining unvisited (disconnected nodes)
        for nid in nodes:
            if nid not in visited:
                ordered.append(nodes[nid])

        exchanges.extend(ordered)

    return exchanges


def _normalize_claude_code_jsonl(text: str) -> list[tuple[str, str]]:
    """Parse Claude Code JSONL format: one JSON object per line."""
    exchanges: list[tuple[str, str]] = []
    for line in text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            msg_type = obj.get("type", "")
            content = obj.get("content", "")
            if isinstance(content, list):
                content = "\n".join(
                    block.get("text", str(block)) if isinstance(block, dict) else str(block)
                    for block in content
                )
            role = "user" if msg_type == "human" else "assistant"
            if content.strip():
                exchanges.append((role, content.strip()))
        except json.JSONDecodeError:
            continue
    return exchanges


def normalize_chat(text: str) -> list[tuple[str, str]]:
    """Auto-detect chat format and normalize to (role, content) pairs.

    Returns list of (role, content) tuples where role is 'user' or 'assistant'.
    """
    stripped = text.strip()

    # Try JSON array (Claude.ai or ChatGPT)
    if stripped.startswith("["):
        try:
            data = json.loads(stripped)
            if isinstance(data, list) and data:
                first = data[0]
                if isinstance(first, dict):
                    if "mapping" in first:
                        return _normalize_chatgpt_json(data)
                    if "role" in first:
                        return _normalize_claude_json(data)
        except json.JSONDecodeError:
            pass

    # Try JSONL (Claude Code)
    if stripped.startswith("{"):
        lines = stripped.split("\n")
        try:
            json.loads(lines[0])
            return _normalize_claude_code_jsonl(stripped)
        except json.JSONDecodeError:
            pass

    # Fallback: plain text with > markers
    return _normalize_plain_text(stripped)


def _exchange_pair_chunks(
    exchanges: list[tuple[str, str]],
    source: str = "",
    tags: list[str] | None = None,
) -> list[ImportChunk]:
    """Chunk chat exchanges into user+assistant pairs."""
    chunks: list[ImportChunk] = []
    chunk_idx = 0
    i = 0

    while i < len(exchanges):
        role, content = exchanges[i]

        if role == "user":
            pair = f"User: {content}"
            # Look for following assistant response
            if i + 1 < len(exchanges) and exchanges[i + 1][0] == "assistant":
                pair += f"\n\nAssistant: {exchanges[i + 1][1]}"
                i += 1
        else:
            pair = f"Assistant: {content}"

        if len(pair) >= MIN_CHUNK_SIZE:
            chunks.append(ImportChunk(
                content=pair,
                kind="fact",
                subject=os.path.basename(source) if source else "conversation",
                tags=list(tags or []),
                source_file=source,
                chunk_index=chunk_idx,
            ))
            chunk_idx += 1

        i += 1

    return chunks


def import_chat_file(
    filepath: str,
    *,
    wing: str = "",
) -> tuple[list[ImportChunk], ImportResult]:
    """Import a single chat export file.

    Auto-detects format (Claude JSON, ChatGPT, Claude Code JSONL, plain text).

    Returns:
        (chunks, result) tuple.
    """
    result = ImportResult()
    fpath = Path(filepath)

    try:
        content = fpath.read_text(encoding="utf-8", errors="replace")
    except (OSError, UnicodeDecodeError) as e:
        result.errors.append(f"Cannot read {filepath}: {e}")
        result.files_skipped += 1
        return [], result

    exchanges = normalize_chat(content)
    if not exchanges:
        result.files_skipped += 1
        return [], result

    tags = [f"source:{filepath}", "import:chat"]
    if wing:
        tags.append(f"wing:{wing}")

    chunks = _exchange_pair_chunks(exchanges, source=filepath, tags=tags)
    result.files_processed = 1
    result.chunks_created = len(chunks)

    return chunks, result


def import_chat_directory(
    directory: str,
    *,
    wing: str = "",
    limit: int = 0,
) -> tuple[list[ImportChunk], ImportResult]:
    """Import all chat files from a directory.

    Looks for .json, .jsonl, .txt, .md files.
    """
    result = ImportResult()
    all_chunks: list[ImportChunk] = []
    root = Path(directory)

    chat_extensions = {".json", ".jsonl", ".txt", ".md"}

    for fpath in sorted(root.iterdir()):
        if limit and result.files_processed >= limit:
            break
        if not fpath.is_file() or fpath.suffix.lower() not in chat_extensions:
            continue

        chunks, file_result = import_chat_file(str(fpath), wing=wing)
        all_chunks.extend(chunks)
        result.files_processed += file_result.files_processed
        result.files_skipped += file_result.files_skipped
        result.chunks_created += file_result.chunks_created
        result.errors.extend(file_result.errors)

    return all_chunks, result
