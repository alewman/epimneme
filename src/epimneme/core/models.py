"""Pydantic models for Engram memory system."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _new_id() -> str:
    return str(uuid.uuid4())


# ── Enums ────────────────────────────────────────────────────────────────────


class MemoryKind(str, Enum):
    """Types of memories an agent can store."""

    FACT = "fact"
    DECISION = "decision"
    PROCEDURE = "procedure"
    PATTERN = "pattern"
    PREFERENCE = "preference"
    OBSERVATION = "observation"
    ISSUE = "issue"


class EntityKind(str, Enum):
    """Types of entities tracked in the knowledge graph."""

    PROJECT = "project"
    FILE = "file"
    MODULE = "module"
    CONCEPT = "concept"
    TOOL = "tool"
    PERSON = "person"
    LIBRARY = "library"
    CONFIG = "config"
    COMMAND = "command"


# ── Core models ──────────────────────────────────────────────────────────────


class Project(BaseModel):
    """A registered project / workspace."""

    id: str = Field(default_factory=_new_id)
    name: str
    path: Optional[str] = None
    description: Optional[str] = None
    persistent_memories: bool = False  # When True, memories skip decay and GC
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)


class Session(BaseModel):
    """A single agent working session."""

    id: str = Field(default_factory=_new_id)
    project_id: Optional[str] = None
    task: Optional[str] = None
    started_at: datetime = Field(default_factory=_utcnow)
    ended_at: Optional[datetime] = None
    summary: Optional[str] = None
    handoff: Optional[str] = None
    # Monotonically increasing integer per project, assigned at session_start.
    # Used for session-recency boosting in retrieval without relying on
    # wall-clock time (which collapses to ~0 during benchmark ingest).
    session_ordinal: Optional[int] = None


class Memory(BaseModel):
    """A single unit of long-term memory."""

    id: str = Field(default_factory=_new_id)
    project_id: Optional[str] = None
    session_id: Optional[str] = None
    kind: MemoryKind
    content: str
    subject: Optional[str] = None
    confidence: float = 1.0
    supersedes: Optional[str] = None
    obsolete: bool = False
    pinned: bool = False  # Pinned memories always appear in session context and skip GC
    tags: list[str] = Field(default_factory=list)
    # Versioning
    version: int = 1
    version_of: Optional[str] = None  # ID of the original memory
    # Deduplication
    simhash: Optional[int] = None
    # Decay / retrievability
    storage_strength: float = 0.0
    retrieval_strength: float = 1.0
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    # Derived at recall time: ordinal of the session that created this memory.
    # NULL for memories without a session or created before migration 005.
    session_ordinal: Optional[int] = None
    # Timestamps
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)


class Entity(BaseModel):
    """A tracked entity in the knowledge graph."""

    id: str = Field(default_factory=_new_id)
    name: str
    kind: EntityKind
    project_id: Optional[str] = None
    properties: dict = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=_utcnow)


class Relationship(BaseModel):
    """A directed edge between two entities."""

    from_entity: str
    to_entity: str
    relation: str
    properties: dict = Field(default_factory=dict)


# ── Search / response models ────────────────────────────────────────────────


class MemoryResult(BaseModel):
    """A memory with relevance score from search."""

    memory: Memory
    score: float = 0.0
    source: str = ""


class RememberResult(BaseModel):
    """Result of a remember() call — stored memory + any detected conflicts."""

    memory: Memory
    potential_conflicts: list["MemoryResult"] = Field(default_factory=list)


class EntityResult(BaseModel):
    """An entity with its relationships from graph traversal."""

    entity: Entity
    relationships: list[Relationship] = Field(default_factory=list)


class ContextBundle(BaseModel):
    """Rich context returned at session start or on contextual query."""

    session_id: Optional[str] = None
    project: Optional[Project] = None
    last_session: Optional[Session] = None
    relevant_memories: list[MemoryResult] = Field(default_factory=list)
    related_entities: list[EntityResult] = Field(default_factory=list)
    recent_decisions: list[Memory] = Field(default_factory=list)
    known_issues: list[Memory] = Field(default_factory=list)
    procedures: list[Memory] = Field(default_factory=list)
    preferences: list[Memory] = Field(default_factory=list)
    pinned_memories: list[Memory] = Field(default_factory=list)

    def to_prompt(self, max_tokens: int = 8000, tier: str = "full") -> str:
        """Format as tiered context block suitable for injecting into a prompt.

        Tiers (each includes all lower tiers):
            L0 — Identity: project header only (~50-100 tokens)
            L1 — Essential: + handoff, pinned, key decisions (~500-800 tokens)
            L2 — Working: + procedures, preferences, issues (~1500-2500 tokens)
            L3/full — Deep: + semantic matches, graph entities (up to max_tokens)

        Args:
            max_tokens: Approximate token budget (~4 chars/token).
            tier: One of "L0", "L1", "L2", "L3", "full".
                  "full" is equivalent to "L3".
        """
        tier = tier.upper() if tier != "full" else "L3"
        max_chars = max_tokens * 4
        parts: list[str] = []
        seen_ids: set[str] = set()
        used = 0

        def _budget_left() -> int:
            return max_chars - used

        def _add(text: str) -> bool:
            """Append text if within budget. Returns False when over budget."""
            nonlocal used
            if used + len(text) + 1 > max_chars:
                remaining = max_chars - used - 1
                if remaining > 20:
                    parts.append(text[:remaining] + "…")
                    used = max_chars
                return False
            parts.append(text)
            used += len(text) + 1
            return True

        def _add_memory(m: Memory, prefix: str = "- ", max_content: int = 1500) -> bool:
            """Add a memory line, deduplicating by ID."""
            if m.id in seen_ids:
                return True
            seen_ids.add(m.id)
            content = m.content
            if len(content) > max_content:
                content = content[: max_content - 3] + "…"
            return _add(f"{prefix}{content}")

        # ── L0: Identity (always included) ──────────────────────────────
        if self.project:
            _add(f"## Project: {self.project.name}")
            if self.project.description:
                _add(self.project.description)
            if self.project.path:
                _add(f"Path: {self.project.path}")

        if tier == "L0":
            return "\n".join(parts)

        # ── L1: Essential (handoff + pinned + decisions) ────────────────
        if self.last_session:
            _add("\n## Previous Session")
            if self.last_session.summary:
                _add(self.last_session.summary)
            if self.last_session.handoff:
                _add(f"\n### Handoff Notes\n{self.last_session.handoff}")

        if self.pinned_memories and _budget_left() > 50:
            _add("\n## Pinned Memories")
            for m in self.pinned_memories:
                if not _add_memory(m, prefix="- 📌 "):
                    break

        if self.recent_decisions and _budget_left() > 50:
            _add("\n## Key Decisions")
            for m in self.recent_decisions:
                if not _add_memory(m):
                    break

        if tier == "L1":
            return "\n".join(parts)

        # ── L2: Working context (issues + procedures + preferences) ─────
        if self.known_issues and _budget_left() > 50:
            _add("\n## Known Issues")
            for m in self.known_issues:
                if not _add_memory(m):
                    break

        if self.procedures and _budget_left() > 50:
            _add("\n## Procedures")
            for m in self.procedures:
                if m.id in seen_ids:
                    continue
                seen_ids.add(m.id)
                label = m.subject or "General"
                content = m.content
                if len(content) > 1500:
                    content = content[:1497] + "…"
                if not _add(f"### {label}\n{content}"):
                    break

        if self.preferences and _budget_left() > 50:
            _add("\n## User Preferences")
            for m in self.preferences:
                if not _add_memory(m):
                    break

        if tier == "L2":
            return "\n".join(parts)

        # ── L3: Deep context (semantic matches + graph) ─────────────────
        if self.relevant_memories and _budget_left() > 50:
            _add("\n## Relevant Context")
            for mr in self.relevant_memories:
                m = mr.memory
                if m.id in seen_ids:
                    continue
                seen_ids.add(m.id)
                content = m.content
                if len(content) > 1500:
                    content = content[:1497] + "…"
                if not _add(f"- [{m.kind.value}] {content}"):
                    break

        if self.related_entities and _budget_left() > 50:
            _add("\n## Key Entities")
            for er in self.related_entities:
                e = er.entity
                rels = ", ".join(
                    f"{r.relation} → {r.to_entity}" for r in er.relationships
                )
                line = f"- **{e.name}** ({e.kind.value})"
                if rels:
                    line += f": {rels}"
                if not _add(line):
                    break

        return "\n".join(parts)


# ── Request models (for JSON POST bodies) ────────────────────────────────────


class StoreMemoryRequest(BaseModel):
    """POST /api/memories"""
    content: str
    kind: str = "fact"
    project: Optional[str] = None
    subject: Optional[str] = None
    tags: list[str] = Field(default_factory=list)
    confidence: float = 1.0
    supersedes: Optional[str] = None
    related_to: list[str] = Field(default_factory=list)
    session_id: Optional[str] = None


class UpdateMemoryRequest(BaseModel):
    """PUT /api/memories/{id}"""
    content: str
    subject: Optional[str] = None
    tags: Optional[list[str]] = None
    confidence: Optional[float] = None


class SearchRequest(BaseModel):
    """POST /api/memories/search"""
    query: str
    project: Optional[str] = None
    kind: Optional[str] = None
    limit: int = 10
    deep: bool = False


class SessionStartRequest(BaseModel):
    """POST /api/sessions/start"""
    project: Optional[str] = None
    task: Optional[str] = None
    tier: Optional[str] = None  # L0, L1, L2, L3, or full (default: full)


class SessionEndRequest(BaseModel):
    """POST /api/sessions/end"""
    session_id: str
    summary: str
    handoff: Optional[str] = None


class CreateProjectRequest(BaseModel):
    """POST /api/projects"""
    name: str
    description: Optional[str] = None
    path: Optional[str] = None
    persistent_memories: bool = False


class TrackEntityRequest(BaseModel):
    """POST /api/entities"""
    name: str
    kind: str
    project: Optional[str] = None
    properties: dict = Field(default_factory=dict)


class RelateEntitiesRequest(BaseModel):
    """POST /api/entities/relate"""
    from_entity: str
    relation: str
    to_entity: str


class CreateKeyRequest(BaseModel):
    """POST /api/admin/keys"""
    name: str
    role: str = "agent"
    projects: list[str] = Field(default_factory=list)
    expires_in_days: Optional[int] = None


class UpdateKeyRequest(BaseModel):
    """PUT /api/admin/keys/{name}"""
    role: Optional[str] = None
    projects: Optional[list[str]] = None


class ForgetRequest(BaseModel):
    """DELETE /api/memories/{id}"""
    reason: str = ""


# ── Bulk operation request models ────────────────────────────────────────────


class BulkMemoryItem(BaseModel):
    """A single item in a batch remember request."""
    content: str
    kind: str = "fact"
    subject: Optional[str] = None
    tags: list[str] = Field(default_factory=list)
    confidence: float = 1.0
    related_to: list[str] = Field(default_factory=list)


class BulkRememberRequest(BaseModel):
    """POST /api/bulk/memories"""
    project: Optional[str] = None
    session_id: Optional[str] = None
    memories: list[BulkMemoryItem] = Field(..., min_length=1, max_length=100)


class BulkEntityItem(BaseModel):
    """A single item in a batch entity_track request."""
    name: str
    kind: str
    properties: dict = Field(default_factory=dict)


class BulkEntityRequest(BaseModel):
    """POST /api/bulk/entities"""
    project: Optional[str] = None
    entities: list[BulkEntityItem] = Field(..., min_length=1, max_length=200)


class BulkRelateItem(BaseModel):
    """A single item in a batch entity_relate request."""
    from_entity: str
    relation: str
    to_entity: str


class BulkRelateRequest(BaseModel):
    """POST /api/bulk/relationships"""
    relationships: list[BulkRelateItem] = Field(..., min_length=1, max_length=200)


class BulkImportRequest(BaseModel):
    """POST /api/bulk/import"""
    directory: str = Field(..., description="Absolute path to directory to import")
    mode: str = Field("project", description="Import mode: 'project' or 'chat'")
    project: Optional[str] = Field(None, description="Project to store memories under")
    limit: int = Field(500, ge=1, le=5000, description="Max chunks to import")
