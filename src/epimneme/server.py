"""Engram server — FastAPI REST API + MCP/SSE dual transport + web dashboard.

Three ways to access:
  1. REST API at /api/* — standard HTTP JSON endpoints
  2. MCP over SSE at /sse — for MCP-compatible clients (VS Code, Cursor, etc.)
  3. Web dashboard at / — for humans (behind Traefik OAuth)
"""

from __future__ import annotations

import json
import logging
import os
from contextlib import asynccontextmanager
from typing import Any, Optional

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.server import Context
from mcp.server.transport_security import TransportSecuritySettings

from epimneme.activity import get_activity_bus, EventType
from epimneme.auth import (
    AuthContext,
    get_auth,
    get_mcp_auth,
    require_admin,
    set_auth_store,
)
from epimneme.skills import INSTRUCTIONS, register_skills

# ── SSE keepalive — patch before MCP imports use EventSourceResponse ─────────
# Reverse proxies (Traefik, Cloudflare) drop idle SSE connections.
# Inject a 15-second ping so the stream stays alive between tool calls.
import sse_starlette.sse as _sse_mod

_OriginalESR = _sse_mod.EventSourceResponse


class _PatchedEventSourceResponse(_OriginalESR):  # type: ignore[misc]
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("ping", 15)
        super().__init__(*args, **kwargs)


_sse_mod.EventSourceResponse = _PatchedEventSourceResponse  # type: ignore[misc]
from epimneme.core.config import default_config
from epimneme.core.models import (
    CreateKeyRequest,
    CreateProjectRequest,
    BulkImportRequest,
    BulkRelateRequest,
    BulkRememberRequest,
    BulkEntityRequest,
    RelateEntitiesRequest,
    SessionEndRequest,
    SessionStartRequest,
    StoreMemoryRequest,
    TrackEntityRequest,
    UpdateKeyRequest,
    UpdateMemoryRequest,
)
from epimneme.dashboard import DASHBOARD_HTML
from epimneme.bulk_import import (
    import_project_files,
    import_chat_directory,
)
from epimneme.manager import MemoryManager
from epimneme.migrations.runner import MigrationRunner

# ── Logging ──────────────────────────────────────────────────────────────────

_log_level = getattr(
    logging, os.environ.get("LOG_LEVEL", "INFO").upper(), logging.INFO
)
_log_format = os.environ.get("LOG_FORMAT", "text").lower()

if _log_format == "json":
    from epimneme.logging import JSONFormatter

    _handler = logging.StreamHandler()
    _handler.setFormatter(JSONFormatter())
    logging.root.handlers = [_handler]
    logging.root.setLevel(_log_level)
else:
    logging.basicConfig(
        level=_log_level,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
logger = logging.getLogger("engram.server")

from epimneme import __version__ as VERSION

# ── Globals (initialised in lifespan) ────────────────────────────────────────

_manager: Optional[MemoryManager] = None
_reflection_scheduler: Optional["ReflectionScheduler"] = None


def get_manager() -> MemoryManager:
    assert _manager is not None, "Manager not initialized"
    return _manager


def get_scheduler() -> "ReflectionScheduler":
    assert _reflection_scheduler is not None, "Scheduler not initialized"
    return _reflection_scheduler


def _validate_import_path(directory: str) -> str | None:
    """Validate that *directory* is under an allowed base directory.

    Returns None if valid, or an error message string if rejected.
    """
    mgr = get_manager()
    resolved = os.path.realpath(directory)
    for allowed in mgr.config.import_allowed_dirs:
        allowed_real = os.path.realpath(allowed)
        if resolved == allowed_real or resolved.startswith(allowed_real + os.sep):
            return None
    return (
        f"Directory {directory!r} is outside allowed import paths "
        f"({mgr.config.import_allowed_dirs})"
    )


async def _mcp_enforce_project(ctx: Context, project: Optional[str]) -> AuthContext:
    """Authenticate MCP request and enforce project access.

    If the agent's key lacks access but the project doesn't exist yet,
    the agent "claims" it.
    """
    auth = await get_mcp_auth(ctx)
    await _try_claim_project(auth, project)
    return auth


async def _try_claim_project(auth: AuthContext, project: Optional[str]) -> None:
    """If *auth* can't access *project* yet, auto-claim if it doesn't exist."""
    if not project or auth.can_access_project(project):
        return

    mgr = get_manager()
    existing = await mgr.get_project(project)
    if existing:
        raise ValueError(
            f"Project '{project}' is already claimed by another key. "
            f"Use a different project name, or ask an admin to grant '{auth.name}' access."
        )
    if auth.can_claim_project(project) and auth.api_key_id:
        await mgr.store.add_project_to_api_key(auth.api_key_id, project)
        auth.projects.append(project)
        logger.info(f"Key '{auth.name}' claimed project '{project}'")
    else:
        raise ValueError(
            f"API key '{auth.name}' cannot claim project '{project}'"
        )


# ── Lifespan ─────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _manager, _reflection_scheduler
    config = default_config()
    _manager = await MemoryManager.create(config)
    set_auth_store(_manager.store)

    # Run pending database migrations
    runner = MigrationRunner(_manager.store.pool)
    await runner.run_pending()

    # Start the reflection scheduler
    from epimneme.scheduler import ReflectionScheduler
    _reflection_scheduler = ReflectionScheduler(_manager)
    _reflection_scheduler.start()

    logger.info(
        f"Engram v{VERSION} started — "
        f"embeddings={'on' if _manager._embeddings_enabled else 'off'}, "
        f"reflection={'on' if config.reflection_enabled else 'off'} "
        f"(every {config.reflection_interval_hours}h)"
    )
    yield
    await _reflection_scheduler.stop()
    await _manager.close()
    logger.info("Engram server stopped")


# ── FastAPI App ──────────────────────────────────────────────────────────────

_config = default_config()

app = FastAPI(
    title="Engram",
    description="Persistent memory service for AI coding agents",
    version=VERSION,
    lifespan=lifespan,
)

# If no explicit origins are configured, fall back to '*' WITHOUT credentials.
# Browsers reject the '*' + credentials combination, so we mirror spec behaviour.
_cors_origins = _config.cors_origins
_cors_allow_credentials = bool(_cors_origins)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins or ["*"],
    allow_credentials=_cors_allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate limiting (per-IP token bucket) — must be added AFTER CORS so CORS
# headers are present even on 429 responses.
from epimneme.ratelimit import RATE_LIMIT_ENABLED, RateLimitMiddleware

if RATE_LIMIT_ENABLED:
    app.add_middleware(RateLimitMiddleware)


# ── Dashboard ────────────────────────────────────────────────────────────────


@app.get("/", include_in_schema=False)
async def dashboard():
    """Serve the web dashboard (browser access via Traefik OAuth)."""
    return HTMLResponse(content=DASHBOARD_HTML)


# ── Health Check (no auth) ───────────────────────────────────────────────────


@app.get("/health")
async def health():
    mgr = get_manager()
    try:
        stats = await mgr.stats()
        return {
            "status": "healthy",
            "version": VERSION,
            "memories": stats["total_memories"],
            "vectors": stats["total_vectors"],
            "entities": stats["total_entities"],
            "embeddings": stats["embeddings_enabled"],
        }
    except Exception as e:
        logger.error("Health check failed: %s", e, exc_info=True)
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "error": "Database connection failed"},
        )


# ── REST API: Sessions ──────────────────────────────────────────────────────


@app.post("/api/sessions/start")
async def api_session_start(
    req: SessionStartRequest,
    auth: AuthContext = Depends(get_auth),
):
    try:
        await _try_claim_project(auth, req.project)
    except ValueError as exc:
        return JSONResponse(status_code=403, content={"error": str(exc)})
    mgr = get_manager()
    bundle = await mgr.session_start(project_name=req.project, task=req.task, tier=req.tier)
    prompt = bundle.to_prompt(tier=req.tier) if req.tier else bundle.to_prompt()
    return {
        "status": "started",
        "session_id": bundle.session_id,
        "context": prompt if prompt.strip() else "No previous context found.",
        "project": bundle.project.model_dump() if bundle.project else None,
        "last_session": bundle.last_session.model_dump() if bundle.last_session else None,
        "memory_count": len(bundle.relevant_memories),
        "entity_count": len(bundle.related_entities),
    }


@app.post("/api/sessions/end")
async def api_session_end(
    req: SessionEndRequest,
    auth: AuthContext = Depends(get_auth),
):
    mgr = get_manager()
    result = await mgr.session_end(
        session_id=req.session_id, summary=req.summary, handoff=req.handoff
    )
    return {"status": "ended", "message": result}


# ── REST API: Memories ───────────────────────────────────────────────────────


@app.post("/api/memories")
async def api_remember(
    req: StoreMemoryRequest,
    auth: AuthContext = Depends(get_auth),
):
    try:
        await _try_claim_project(auth, req.project)
    except ValueError as exc:
        return JSONResponse(status_code=403, content={"error": str(exc)})
    mgr = get_manager()
    result = await mgr.remember(
        content=req.content,
        kind=req.kind,
        project_name=req.project,
        subject=req.subject,
        tags=req.tags,
        confidence=req.confidence,
        supersedes=req.supersedes,
        related_to=req.related_to if req.related_to else None,
        session_id=req.session_id,
    )
    if isinstance(result, str):
        return {"status": "deduplicated", "message": result}
    response = {
        "id": result.memory.id,
        "kind": result.memory.kind.value,
        "content": result.memory.content[:200],
    }
    if result.potential_conflicts:
        response["potential_conflicts"] = [
            {
                "id": c.memory.id,
                "kind": c.memory.kind.value,
                "content": c.memory.content[:200],
                "similarity": round(c.score, 3),
            }
            for c in result.potential_conflicts
        ]
    return response


@app.put("/api/memories/{memory_id}")
async def api_update_memory(
    memory_id: str,
    req: UpdateMemoryRequest,
    auth: AuthContext = Depends(get_auth),
):
    mgr = get_manager()
    new_version = await mgr.update_memory(
        memory_id=memory_id,
        content=req.content,
        subject=req.subject,
        tags=req.tags,
        confidence=req.confidence,
    )
    if not new_version:
        return JSONResponse(status_code=404, content={"error": "Memory not found"})
    return {
        "id": new_version.id,
        "version": new_version.version,
        "version_of": new_version.version_of,
    }


@app.get("/api/memories/{memory_id}/versions")
async def api_memory_versions(
    memory_id: str,
    auth: AuthContext = Depends(get_auth),
):
    mgr = get_manager()
    versions = await mgr.get_memory_versions(memory_id)
    return {
        "memory_id": memory_id,
        "versions": [
            {
                "id": m.id,
                "version": m.version,
                "kind": m.kind.value if hasattr(m.kind, 'value') else m.kind,
                "content": m.content,
                "subject": m.subject,
                "tags": m.tags,
                "confidence": m.confidence,
                "created_at": m.created_at.isoformat(),
            }
            for m in versions
        ],
    }


@app.get("/api/memories/search")
async def api_recall(
    query: str,
    project: Optional[str] = None,
    kind: Optional[str] = None,
    tags: Optional[str] = None,
    limit: int = Query(default=10, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    deep: bool = False,
    reference_date: Optional[str] = None,
    auth: AuthContext = Depends(get_auth),
):
    auth.enforce_project_access(project)
    mgr = get_manager()

    # Parse comma-separated tags
    tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else None

    if deep:
        bundle = await mgr.get_context(query, project_name=project)
        return {"context": bundle.to_prompt() or f"No context found for: {query}"}

    # Fetch limit+offset so we can slice for offset-based pagination
    fetch_count = limit + offset
    results = await mgr.recall(query, project_name=project, kind=kind, tags=tag_list, limit=fetch_count, reference_date=reference_date)
    page = results[offset : offset + limit]
    # total is a lower bound — true total requires a separate count query
    has_more = len(results) == fetch_count
    return {
        "count": len(page),
        "total": len(results),
        "has_more": has_more,
        "limit": limit,
        "offset": offset,
        "results": [
            {
                "id": r.memory.id,
                "kind": r.memory.kind.value,
                "content": r.memory.content,
                "subject": r.memory.subject,
                "score": round(r.score, 3),
                "source": r.source,
                "project_id": r.memory.project_id,
                "version": r.memory.version,
                "version_of": r.memory.version_of,
                "tags": r.memory.tags,
                "pinned": r.memory.pinned,
                "confidence": r.memory.confidence,
                "session_ordinal": r.memory.session_ordinal,
            }
            for r in page
        ],
    }


@app.delete("/api/memories/{memory_id}")
async def api_forget(
    memory_id: str,
    reason: str = "",
    hard: bool = False,
    auth: AuthContext = Depends(get_auth),
):
    mgr = get_manager()
    if hard:
        result = await mgr.hard_forget(memory_id, reason=reason)
    else:
        result = await mgr.forget(memory_id, reason=reason)
    return {"message": result}


@app.post("/api/memories/{memory_id}/pin")
async def api_pin_memory(
    memory_id: str,
    auth: AuthContext = Depends(get_auth),
):
    """Pin a memory so it is always included in context and never garbage-collected."""
    mgr = get_manager()
    ok = await mgr.store.pin_memory(memory_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Memory not found")
    return {"message": f"Memory {memory_id[:8]}… pinned"}


@app.post("/api/memories/{memory_id}/unpin")
async def api_unpin_memory(
    memory_id: str,
    auth: AuthContext = Depends(get_auth),
):
    """Remove the pin from a memory."""
    mgr = get_manager()
    ok = await mgr.store.unpin_memory(memory_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Memory not found")
    return {"message": f"Memory {memory_id[:8]}… unpinned"}


# ── REST API: Projects ───────────────────────────────────────────────────────


@app.get("/api/projects")
async def api_list_projects(auth: AuthContext = Depends(get_auth)):
    mgr = get_manager()
    projects = await mgr.list_projects()
    visible = [p for p in projects if auth.can_access_project(p.name)]
    return {"projects": [p.model_dump() for p in visible]}


@app.post("/api/projects")
async def api_create_project(
    req: CreateProjectRequest,
    auth: AuthContext = Depends(get_auth),
):
    auth.enforce_project_access(req.name)
    mgr = get_manager()
    project = await mgr.create_project(
        name=req.name, path=req.path, description=req.description,
        persistent_memories=req.persistent_memories,
    )
    return project.model_dump()


@app.get("/api/projects/{project_name}/status")
async def api_project_status(
    project_name: str,
    auth: AuthContext = Depends(get_auth),
):
    auth.enforce_project_access(project_name)
    mgr = get_manager()
    existing = await mgr.get_project(project_name)
    if not existing:
        await mgr.create_project(name=project_name)
    status = await mgr.project_status(project_name)
    if "error" in status:
        return JSONResponse(status_code=404, content=status)
    return status


@app.post("/api/projects/{project_name}/persistent")
async def api_set_project_persistent(
    project_name: str,
    auth: AuthContext = Depends(get_auth),
):
    """Enable persistent memories for a project (memories skip decay/GC)."""
    auth.enforce_project_access(project_name)
    mgr = get_manager()
    existing = await mgr.get_project(project_name)
    if not existing:
        return JSONResponse(status_code=404, content={"error": f"Project '{project_name}' not found"})
    ok = await mgr.set_project_persistent(project_name, True)
    if ok:
        bus = get_activity_bus()
        await bus.emit(EventType.SESSION, f"Persistent memories enabled for {project_name}", project=project_name)
    return {"project": project_name, "persistent_memories": True}


@app.delete("/api/projects/{project_name}/persistent")
async def api_unset_project_persistent(
    project_name: str,
    auth: AuthContext = Depends(get_auth),
):
    """Disable persistent memories for a project (memories resume normal decay)."""
    auth.enforce_project_access(project_name)
    mgr = get_manager()
    existing = await mgr.get_project(project_name)
    if not existing:
        return JSONResponse(status_code=404, content={"error": f"Project '{project_name}' not found"})
    ok = await mgr.set_project_persistent(project_name, False)
    if ok:
        bus = get_activity_bus()
        await bus.emit(EventType.SESSION, f"Persistent memories disabled for {project_name}", project=project_name)
    return {"project": project_name, "persistent_memories": False}


# ── REST API: Entities ───────────────────────────────────────────────────────


@app.get("/api/entities")
async def api_list_entities(
    project: Optional[str] = None,
    kind: Optional[str] = None,
    limit: int = Query(default=200, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    auth: AuthContext = Depends(get_auth),
):
    """List entities, optionally filtered by project and/or kind.

    Supports pagination via limit/offset.  Returns total count for UI paging.
    """
    auth.enforce_project_access(project)
    mgr = get_manager()
    project_id = None
    if project:
        proj = await mgr.get_project(project)
        if proj:
            project_id = proj.id

    from epimneme.core.models import EntityKind

    kind_enum = EntityKind(kind) if kind else None
    entities = await mgr.store.list_entities(
        project_id=project_id, kind=kind_enum, limit=limit, offset=offset,
    )
    total = await mgr.store.count_entities(
        project_id=project_id, kind=kind_enum,
    )
    results = []
    for e in entities:
        results.append({
            "id": e.id,
            "name": e.name,
            "kind": e.kind.value,
            "project_id": e.project_id,
            "properties": e.properties,
            "created_at": e.created_at.isoformat() if e.created_at else None,
        })
    return {
        "count": len(results),
        "total": total,
        "limit": limit,
        "offset": offset,
        "entities": results,
    }


@app.post("/api/entities")
async def api_track_entity(
    req: TrackEntityRequest,
    auth: AuthContext = Depends(get_auth),
):
    auth.enforce_project_access(req.project)
    mgr = get_manager()
    entity = await mgr.track_entity(
        name=req.name,
        kind=req.kind,
        project_name=req.project,
        properties=req.properties,
    )
    return {"id": entity.id, "name": entity.name, "kind": entity.kind.value}


@app.post("/api/entities/relate")
async def api_relate_entities(
    req: RelateEntitiesRequest,
    auth: AuthContext = Depends(get_auth),
):
    mgr = get_manager()
    result = await mgr.relate_entities(
        from_entity=req.from_entity,
        relation=req.relation,
        to_entity=req.to_entity,
    )
    return {"message": result}


@app.get("/api/entities/{entity_name}/explore")
async def api_explore_entity(
    entity_name: str,
    depth: int = Query(default=2, ge=1, le=5),
    direction: str = "both",
    project: Optional[str] = None,
    auth: AuthContext = Depends(get_auth),
):
    auth.enforce_project_access(project)
    mgr = get_manager()
    results = await mgr.explore_entity(
        entity_name=entity_name,
        depth=depth,
        direction=direction,
        project_name=project,
    )
    return {
        "entity": entity_name,
        "connections": [
            {
                "entity": {"name": er.entity.name, "kind": er.entity.kind.value},
                "relationships": [
                    {"relation": r.relation, "to": r.to_entity}
                    for r in er.relationships
                ],
            }
            for er in results
        ],
    }


@app.delete("/api/entities/{entity_id}")
async def api_delete_entity(
    entity_id: str,
    auth: AuthContext = Depends(get_auth),
):
    """Delete an entity and its relationships from the knowledge graph."""
    mgr = get_manager()
    deleted = await mgr.store.delete_entity(entity_id)
    if deleted:
        return {"message": f"Entity '{entity_id}' deleted"}
    return JSONResponse(status_code=404, content={"error": "Entity not found"})


# ── REST API: Graph ──────────────────────────────────────────────────────────


@app.get("/api/graph/entities")
async def api_entity_graph(
    project: Optional[str] = None,
    limit: int = 300,
    auth: AuthContext = Depends(get_auth),
):
    """Entity relationship graph — nodes are entities, edges are relationships."""
    auth.enforce_project_access(project)
    mgr = get_manager()
    return await mgr.store.get_graph_data(project_name=project, limit_nodes=limit)


@app.get("/api/graph/similarity")
async def api_similarity_graph(
    project: Optional[str] = None,
    limit: int = 100,
    threshold: float = 0.75,
    auth: AuthContext = Depends(get_auth),
):
    """Memory similarity graph — nodes are memories, edges are cosine sim > threshold."""
    auth.enforce_project_access(project)
    mgr = get_manager()
    return await mgr.store.get_similarity_graph(
        project_name=project, limit=limit, threshold=threshold
    )


# ── REST API: Bulk Operations ────────────────────────────────────────────────


@app.post("/api/bulk/memories")
async def api_bulk_remember(
    req: BulkRememberRequest,
    auth: AuthContext = Depends(get_auth),
):
    """Store multiple memories in one request. Max 100 per call.

    Dedup and conflict detection run for each item. Items that are
    deduplicated are reported but don't block the rest.
    """
    try:
        await _try_claim_project(auth, req.project)
    except ValueError as exc:
        return JSONResponse(status_code=403, content={"error": str(exc)})

    mgr = get_manager()

    # Pre-compute all embeddings in a single batched model.encode() call
    # (much faster than N individual encode calls inside remember())
    texts = [item.content for item in req.memories]
    embeddings = await mgr._embed_batch(texts)

    results = []
    for item, precomputed in zip(req.memories, embeddings):
        result = await mgr.remember(
            content=item.content,
            kind=item.kind,
            project_name=req.project,
            subject=item.subject,
            tags=item.tags,
            confidence=item.confidence,
            related_to=item.related_to if item.related_to else None,
            session_id=req.session_id,
            _precomputed_embedding=precomputed,
        )
        if isinstance(result, str):
            results.append({"status": "deduplicated", "message": result})
        else:
            entry = {"status": "stored", "id": result.memory.id, "kind": result.memory.kind.value}
            if result.potential_conflicts:
                entry["conflicts"] = len(result.potential_conflicts)
            results.append(entry)

    stored = sum(1 for r in results if r["status"] == "stored")
    deduped = sum(1 for r in results if r["status"] == "deduplicated")
    return {
        "total": len(results),
        "stored": stored,
        "deduplicated": deduped,
        "results": results,
    }


@app.post("/api/bulk/entities")
async def api_bulk_entity_track(
    req: BulkEntityRequest,
    auth: AuthContext = Depends(get_auth),
):
    """Track multiple entities in one request. Max 200 per call."""
    auth.enforce_project_access(req.project)
    mgr = get_manager()
    results = []
    for item in req.entities:
        entity = await mgr.track_entity(
            name=item.name,
            kind=item.kind,
            project_name=req.project,
            properties=item.properties,
        )
        results.append({"id": entity.id, "name": entity.name, "kind": entity.kind.value})
    return {"total": len(results), "entities": results}


@app.post("/api/bulk/relationships")
async def api_bulk_relate(
    req: BulkRelateRequest,
    auth: AuthContext = Depends(get_auth),
):
    """Create multiple entity relationships in one request. Max 200 per call."""
    mgr = get_manager()
    results = []
    for item in req.relationships:
        msg = await mgr.relate_entities(
            from_entity=item.from_entity,
            relation=item.relation,
            to_entity=item.to_entity,
        )
        results.append(msg)
    return {"total": len(results), "relationships": results}


@app.post("/api/bulk/import")
async def api_bulk_import(
    req: BulkImportRequest,
    auth: AuthContext = Depends(get_auth),
):
    """Import project files or chat exports as memories in bulk.

    Scans a directory, chunks content, deduplicates, and stores as memories.
    Modes:
        project — import source code, configs, docs (25 file types)
        chat    — import conversation exports (Claude JSON, ChatGPT JSON,
                  Claude Code JSONL, plain text)
    """
    import os

    directory = os.path.abspath(req.directory)
    if not os.path.isdir(directory):
        return JSONResponse(status_code=400, content={"error": f"Not a directory: {directory}"})

    path_err = _validate_import_path(directory)
    if path_err:
        return JSONResponse(status_code=403, content={"error": path_err})

    try:
        await _try_claim_project(auth, req.project)
    except ValueError as exc:
        return JSONResponse(status_code=403, content={"error": str(exc)})

    mgr = get_manager()

    if req.mode == "chat":
        chunks, result = import_chat_directory(directory, wing=req.project or "default", limit=req.limit)
    else:
        chunks, result = import_project_files(directory, wing=req.project or "default", limit=req.limit)

    # Store each chunk as a memory
    stored = 0
    store_errors: list[str] = []
    for chunk in chunks:
        try:
            await mgr.remember(
                content=chunk.content,
                kind=chunk.kind,
                project_name=req.project,
                subject=chunk.subject,
                tags=chunk.tags,
            )
            stored += 1
        except Exception as e:
            store_errors.append(f"{chunk.source_file}: {e}")

    return {
        "status": "completed",
        "directory": directory,
        "mode": req.mode,
        "files_processed": result.files_processed,
        "files_skipped": result.files_skipped,
        "chunks_found": result.chunks_created,
        "stored": stored,
        "errors": (result.errors + store_errors)[:20],
    }


# ── REST API: Stats ──────────────────────────────────────────────────────────


@app.get("/api/memories/recent")
async def api_recent_memories(
    request: Request,
    auth: AuthContext = Depends(get_auth),
):
    """Get recent memories newest-first with full content.  For the Logger tab.

    Query params:
        limit:   max rows (default 100, max 500)
        offset:  pagination offset (default 0)
        project: filter by project name
        kind:    filter by memory kind
        since:   ISO timestamp — only memories created after this
    """
    mgr = get_manager()
    params = request.query_params
    limit = min(int(params.get("limit", "100")), 500)
    offset = int(params.get("offset", "0"))
    project = params.get("project") or None
    kind = params.get("kind") or None
    since = params.get("since") or None

    # Resolve project name → id
    project_id = None
    if project:
        p = await mgr.store.get_project(project)
        if p:
            project_id = p.id
        else:
            return {"memories": [], "total": 0}

    memories = await mgr.store.get_recent_memories(
        project_id=project_id,
        kind=kind,
        since=since,
        limit=limit,
        offset=offset,
    )

    return {
        "memories": [
            {
                "id": m.id,
                "kind": m.kind.value,
                "content": m.content,
                "subject": m.subject,
                "project_id": m.project_id,
                "session_id": m.session_id,
                "confidence": m.confidence,
                "tags": m.tags,
                "version": m.version,
                "version_of": m.version_of,
                "created_at": m.created_at.isoformat() if m.created_at else None,
            }
            for m in memories
            if auth.can_access_project(m.project_id or "")
        ],
        "count": len(memories),
        "limit": limit,
        "offset": offset,
    }


@app.get("/api/stats")
async def api_stats(auth: AuthContext = Depends(get_auth)):
    mgr = get_manager()
    stats = await mgr.stats()
    projects = await mgr.list_projects()
    visible = [p.name for p in projects if auth.can_access_project(p.name)]
    stats["projects"] = visible
    stats["version"] = VERSION
    return stats


@app.get("/api/stats/detailed")
async def api_detailed_stats(auth: AuthContext = Depends(require_admin)):
    """Detailed stats for the dashboard — admin only."""
    mgr = get_manager()
    return await mgr.store.get_detailed_stats()


# ── REST API: Admin ──────────────────────────────────────────────────────────


@app.post("/api/admin/keys")
async def api_create_key(
    req: CreateKeyRequest,
    auth: AuthContext = Depends(require_admin),
):
    mgr = get_manager()
    raw_key = await mgr.store.create_api_key(
        name=req.name,
        role=req.role,
        projects=req.projects,
        expires_in_days=req.expires_in_days,
    )
    return {
        "key": raw_key,
        "name": req.name,
        "role": req.role,
        "projects": req.projects,
    }


@app.get("/api/admin/keys")
async def api_list_keys(auth: AuthContext = Depends(require_admin)):
    mgr = get_manager()
    keys = await mgr.store.list_api_keys()
    for k in keys:
        k.pop("key_hash", None)
    return {"keys": keys}


@app.put("/api/admin/keys/{name}")
async def api_update_key(
    name: str,
    req: UpdateKeyRequest,
    auth: AuthContext = Depends(require_admin),
):
    mgr = get_manager()
    result = await mgr.store.update_api_key(
        name=name, role=req.role, projects=req.projects
    )
    if not result:
        return JSONResponse(
            status_code=404, content={"error": "Key not found or revoked"}
        )
    return result


@app.delete("/api/admin/keys/{name}")
async def api_revoke_key(
    name: str,
    auth: AuthContext = Depends(require_admin),
):
    mgr = get_manager()
    if await mgr.store.revoke_api_key(name):
        return {"message": f"Key '{name}' revoked"}
    return JSONResponse(
        status_code=404, content={"error": "Key not found or already revoked"}
    )


@app.post("/api/admin/keys/{name}/cycle")
async def api_cycle_key(
    name: str,
    auth: AuthContext = Depends(require_admin),
):
    """Rotate an API key: revoke the old one and issue a new key with the same name/role/projects."""
    mgr = get_manager()
    new_raw_key = await mgr.store.cycle_api_key(name)
    if not new_raw_key:
        return JSONResponse(
            status_code=404, content={"error": "Key not found or already revoked"}
        )
    return {"key": new_raw_key, "name": name, "message": "Key cycled — old key is now revoked"}


@app.post("/api/admin/purge-obsolete")
async def api_purge_obsolete(
    auth: AuthContext = Depends(require_admin),
):
    """Permanently delete all memories marked obsolete. Irreversible."""
    mgr = get_manager()
    count = await mgr.store.purge_obsolete_memories()
    logger.info(f"Purged {count} obsolete memories")
    return {"purged": count, "message": f"Permanently deleted {count} obsolete memories"}


# ── REST API: Session Management ─────────────────────────────────────────────


@app.get("/api/admin/sessions/open")
async def api_list_open_sessions(
    older_than_hours: Optional[float] = None,
    auth: AuthContext = Depends(require_admin),
):
    """List sessions that were started but never ended (potential leaks)."""
    mgr = get_manager()
    sessions = await mgr.store.list_open_sessions(older_than_hours=older_than_hours)
    return {
        "count": len(sessions),
        "sessions": [
            {
                "id": s.id,
                "project_id": s.project_id,
                "task": s.task,
                "started_at": s.started_at.isoformat() if s.started_at else None,
            }
            for s in sessions
        ],
    }


@app.post("/api/admin/sessions/{session_id}/close")
async def api_force_close_session(
    session_id: str,
    request: Request,
    auth: AuthContext = Depends(require_admin),
):
    """Force-close an open session that was never ended."""
    body = {}
    try:
        body = await request.json()
    except Exception:
        pass
    summary = body.get("summary", "Session force-closed by admin")

    mgr = get_manager()
    closed = await mgr.store.force_close_session(session_id, summary=summary)
    if closed:
        return {"message": f"Session {session_id[:8]}… force-closed"}
    return JSONResponse(
        status_code=404, content={"error": "Session not found or already ended"}
    )


# ── REST API: Backup & Restore ───────────────────────────────────────────────


@app.post("/api/admin/backup")
async def api_create_backup(
    request: Request,
    auth: AuthContext = Depends(require_admin),
):
    """Create a new backup archive of the full database."""
    from epimneme.backup import rotate_backups, save_backup

    mgr = get_manager()
    body = {}
    try:
        body = await request.json()
    except Exception:
        pass
    label = body.get("label", "")

    result = await save_backup(
        pool=mgr.store.pool,
        backup_dir=mgr.config.backup_dir,
        epimneme_version=VERSION,
        label=label,
    )

    # Auto-rotate old backups
    deleted = rotate_backups(
        mgr.config.backup_dir,
        keep_last=mgr.config.backup_keep_last,
        keep_days=mgr.config.backup_keep_days,
    )
    if deleted:
        result["rotated"] = deleted

    return result


@app.get("/api/admin/backups")
async def api_list_backups(auth: AuthContext = Depends(require_admin)):
    """List available backup files."""
    from epimneme.backup import list_backups

    mgr = get_manager()
    backups = list_backups(mgr.config.backup_dir)
    return {"backups": backups}


@app.post("/api/admin/restore/{filename}")
async def api_restore_backup(
    filename: str,
    request: Request,
    auth: AuthContext = Depends(require_admin),
):
    """Restore from a backup file. Body: {"mode": "merge"} or {"mode": "clean"}."""
    from epimneme.backup import load_backup_file, restore_backup

    mgr = get_manager()
    body = {}
    try:
        body = await request.json()
    except Exception:
        pass
    mode = body.get("mode", "merge")
    if mode not in ("merge", "clean"):
        return JSONResponse(status_code=400, content={"error": "mode must be 'merge' or 'clean'"})

    try:
        archive = load_backup_file(mgr.config.backup_dir, filename)
    except FileNotFoundError:
        return JSONResponse(status_code=404, content={"error": f"Backup not found: {filename}"})
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

    result = await restore_backup(
        pool=mgr.store.pool,
        archive=archive,
        mode=mode,
    )
    return result


@app.delete("/api/admin/backups/{filename}")
async def api_delete_backup(
    filename: str,
    auth: AuthContext = Depends(require_admin),
):
    """Delete a backup file."""
    from epimneme.backup import delete_backup

    mgr = get_manager()
    if delete_backup(mgr.config.backup_dir, filename):
        return {"message": f"Backup '{filename}' deleted"}
    return JSONResponse(status_code=404, content={"error": "Backup not found"})


@app.get("/api/admin/backups/{filename}/download")
async def api_download_backup(
    filename: str,
    auth: AuthContext = Depends(require_admin),
):
    """Download a backup file."""
    from fastapi.responses import FileResponse

    mgr = get_manager()
    from epimneme.backup import _safe_backup_path
    try:
        filepath = _safe_backup_path(mgr.config.backup_dir, filename)
    except ValueError:
        return JSONResponse(status_code=400, content={"error": "Invalid filename"})
    if not filepath.exists():
        return JSONResponse(status_code=404, content={"error": "Backup not found"})
    return FileResponse(
        path=str(filepath),
        filename=filename,
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ── Reflection API ───────────────────────────────────────────────────────────


@app.get("/api/admin/reflection/status")
async def api_reflection_status(auth: AuthContext = Depends(require_admin)):
    """Get the current reflection scheduler status & history."""
    scheduler = get_scheduler()
    return scheduler.get_status()


@app.post("/api/admin/reflection/run")
async def api_reflection_run(auth: AuthContext = Depends(require_admin)):
    """Trigger an immediate reflection cycle."""
    import asyncio

    scheduler = get_scheduler()

    # Run in background so the HTTP response returns immediately
    async def _run_and_log():
        result = await scheduler.run_now()
        if result.error:
            logger.warning(f"Manual reflection failed: {result.error}")

    asyncio.create_task(_run_and_log())
    return {"message": "Reflection cycle started", "running": True}


# ── Activity Stream API ──────────────────────────────────────────────────────


@app.get("/api/activity")
async def api_activity(
    request: Request,
    auth: AuthContext = Depends(get_auth),
):
    """Poll the activity event stream for the unified logger.

    Query params:
        since:   event ID — only return events newer than this (default 0)
        types:   comma-separated event types to include (default all)
        project: filter by project name (default all)
        limit:   max events (default 200, max 500)
    """
    params = request.query_params
    since = int(params.get("since", "0"))
    types_str = params.get("types", "")
    types = [t.strip() for t in types_str.split(",") if t.strip()] or None
    project = params.get("project") or None
    limit = min(int(params.get("limit", "200")), 500)

    bus = get_activity_bus()
    events = await bus.get_events(since=since, types=types, project=project, limit=limit)
    stats = await bus.get_stats()

    return {
        "events": events,
        "stats": stats,
        "note": "Events are ephemeral (in-memory ring buffer) and lost on restart.",
    }


# ── MCP Server (SSE transport) ──────────────────────────────────────────────

_extra_hosts = [
    h.strip()
    for h in os.environ.get("EPIMNEME_ALLOWED_HOSTS", "").split(",")
    if h.strip()
]
if _extra_hosts:
    _transport_security = TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=["127.0.0.1:*", "localhost:*", "[::1]:*"]
        + [f"{h}:*" for h in _extra_hosts]
        + _extra_hosts,
        allowed_origins=[
            "http://127.0.0.1:*",
            "http://localhost:*",
            "http://[::1]:*",
        ]
        + [f"https://{h}" for h in _extra_hosts],
    )
else:
    _transport_security = None  # SDK defaults

mcp = FastMCP(
    "epimneme",
    instructions=INSTRUCTIONS,
    transport_security=_transport_security,
)
register_skills(mcp)


# ── MCP Tools ────────────────────────────────────────────────────────────────


@mcp.tool()
async def session_start(
    ctx: Context,
    project: Optional[str] = None,
    task: Optional[str] = None,
    tier: Optional[str] = None,
) -> str:
    """Start a working session — call this FIRST in every conversation.

    Returns previous session context, decisions, known issues, and handoff
    notes so you can pick up where the last agent left off.

    IMPORTANT: The response includes a session_id. You MUST save it and pass
    it to session_end when you are done, and optionally to remember calls.

    Args:
        project: Project name to scope the session to.
        task: Brief description of what you'll work on — improves context relevance.
        tier: Context loading tier — controls how much context is returned.
              L0 = minimal (project header only, ~170 tokens)
              L1 = handoff + pinned + decisions
              L2 = + issues, procedures, preferences
              L3/full = everything including semantic matches and graph (default)
    """
    await _mcp_enforce_project(ctx, project)
    mgr = get_manager()
    bundle = await mgr.session_start(project_name=project, task=task, tier=tier)
    prompt = bundle.to_prompt(tier=tier) if tier else bundle.to_prompt()
    header = f"session_id: {bundle.session_id}\n\n"
    if not prompt.strip():
        return (
            header
            + "New session started. No previous context found — this may be a fresh project."
        )
    return header + f"Session started. Here's what I know:\n\n{prompt}"


@mcp.tool()
async def session_end(
    ctx: Context,
    session_id: str,
    summary: str,
    handoff: Optional[str] = None,
) -> str:
    """End your session — call this LAST before you finish.

    Your summary and handoff notes will be shown to the next agent at the
    start of their session. Write them as if briefing a colleague.

    Args:
        session_id: The session_id from session_start (required)
        summary: What you accomplished — be specific about changes made
        handoff: Instructions for the next agent — next steps, blockers, etc.
    """
    await get_mcp_auth(ctx)
    mgr = get_manager()
    return await mgr.session_end(
        session_id=session_id, summary=summary, handoff=handoff
    )


@mcp.tool()
async def remember(
    ctx: Context,
    content: str,
    kind: str = "fact",
    project: Optional[str] = None,
    subject: Optional[str] = None,
    tags: Optional[str] = None,
    confidence: float = 1.0,
    supersedes: Optional[str] = None,
    related_to: Optional[str] = None,
    forget_id: Optional[str] = None,
    session_id: Optional[str] = None,
) -> str:
    """Store a memory that persists across sessions and agents.

    Use this to record anything a future agent should know. Write content as
    a self-contained statement — another agent will read it without your context.

    Kind guide:
      fact       — objective truth ("PostgreSQL runs on port 5432")
      decision   — a choice that was made and why
      procedure  — how to do something
      pattern    — recurring observation
      preference — user or team preference
      observation — something noticed
      issue      — a known problem

    Args:
        content: The memory — be specific, self-contained, and concise
        kind: One of: fact, decision, procedure, pattern, preference, observation, issue
        project: Project name to scope to (omit for cross-project knowledge)
        subject: Topic identifier — a file path, module name, or concept
        tags: Comma-separated tags (e.g. "auth,security,login")
        confidence: 0.0-1.0 certainty (default 1.0, lower if speculative)
        supersedes: ID of a previous memory this corrects or replaces
        related_to: Comma-separated entity names to link in the knowledge graph
        forget_id: Pass a memory ID here to mark it obsolete (content becomes the reason)
        session_id: Session ID from session_start (links memory to this session)
    """
    await _mcp_enforce_project(ctx, project)
    mgr = get_manager()

    if forget_id:
        return await mgr.forget(forget_id, reason=content)

    tag_list = [t.strip() for t in tags.split(",")] if tags else []
    entity_list = [e.strip() for e in related_to.split(",")] if related_to else []

    result = await mgr.remember(
        content=content,
        kind=kind,
        project_name=project,
        subject=subject,
        tags=tag_list,
        confidence=confidence,
        supersedes=supersedes,
        related_to=entity_list if entity_list else None,
        session_id=session_id,
    )
    if isinstance(result, str):
        return result  # dedup message
    msg = f"Stored [{result.memory.kind.value}] {result.memory.id[:8]}…: {content[:100]}"
    if result.potential_conflicts:
        msg += "\n\n⚠️ Potential conflicts with existing memories:"
        for c in result.potential_conflicts:
            msg += (
                f"\n- ({c.score:.2f}) {c.memory.kind.value}: "
                f"{c.memory.content[:120]}\n"
                f"  id: {c.memory.id}  "
                f"(use supersedes='{c.memory.id}' to replace)"
            )
    return msg


@mcp.tool()
async def recall(
    ctx: Context,
    query: str,
    project: Optional[str] = None,
    kind: Optional[str] = None,
    tags: Optional[str] = None,
    limit: int = 10,
    deep: bool = False,
) -> str:
    """Search for relevant memories using semantic similarity and keywords.

    Use this when you need to find what's been recorded about a topic.
    Results are ranked by relevance. Use deep=True for rich context including
    knowledge graph traversal.

    Args:
        query: What you're looking for — natural language works best
        project: Scope search to a specific project
        kind: Filter by memory kind (e.g. "decision", "issue")
        tags: Comma-separated tags to filter by (e.g. "auth,security") — only returns memories with ALL specified tags
        limit: Maximum results (default 10)
        deep: If True, returns rich context bundle with graph traversal
    """
    await _mcp_enforce_project(ctx, project)
    mgr = get_manager()

    tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else None

    if deep:
        bundle = await mgr.get_context(query, project_name=project)
        return bundle.to_prompt() or f"No context found for: {query}"

    results = await mgr.recall(query, project_name=project, kind=kind, tags=tag_list, limit=limit)
    if not results:
        return f"No memories found for: {query}"

    lines = [f"Found {len(results)} memories:\n"]
    for r in results:
        m = r.memory
        score = f"{r.score:.2f}"
        project_tag = f" @{m.project_id[:8]}" if m.project_id else " @global"
        subject_tag = f" [{m.subject}]" if m.subject else ""
        lines.append(
            f"- ({score}) **{m.kind.value}**{subject_tag}{project_tag}: {m.content}"
        )
        lines.append(f"  id: {m.id}")
    return "\n".join(lines)


@mcp.tool()
async def project_status(
    ctx: Context,
    project: str,
    description: Optional[str] = None,
    path: Optional[str] = None,
) -> str:
    """Get a project's status, stats, and recent activity. Auto-registers new projects.

    Call this to understand a project's current state: memory count, entity
    count, open issues, last session summary, and recent decisions.
    If the project doesn't exist yet, it will be created.

    Args:
        project: Project name (e.g. "epimneme", "dns-redirect")
        description: One-line description (only needed for new projects)
        path: Filesystem path to the project root (only for new projects)
    """
    await _mcp_enforce_project(ctx, project)
    mgr = get_manager()

    existing = await mgr.get_project(project)
    if not existing:
        await mgr.create_project(name=project, path=path, description=description)

    status = await mgr.project_status(project)
    if "error" in status:
        return status["error"]

    stats = await mgr.stats()
    lines = [f"## {project}\n"]
    lines.append(f"- Memories: {status['memory_count']}")
    lines.append(f"- Entities: {status['entity_count']}")
    lines.append(f"- Open issues: {status['open_issues']}")

    if status.get("entities_by_kind"):
        lines.append(
            f"- Entity breakdown: {json.dumps(status['entities_by_kind'])}"
        )

    if status.get("last_session"):
        s = status["last_session"]
        lines.append("\n### Last Session")
        if s.get("task"):
            lines.append(f"Task: {s['task']}")
        if s.get("summary"):
            lines.append(f"Summary: {s['summary'][:300]}")

    if status.get("recent_decisions"):
        lines.append("\n### Recent Decisions")
        for d in status["recent_decisions"]:
            lines.append(f"- {d['content'][:200]}")

    all_projects = await mgr.list_projects()
    lines.append("\n### Engram Stats")
    lines.append(f"- Total memories: {stats['total_memories']}")
    lines.append(f"- Total entities: {stats['total_entities']}")
    lines.append(
        f"- Projects: {', '.join(p.name for p in all_projects)}"
    )
    lines.append(
        f"- Embeddings: {'enabled' if stats['embeddings_enabled'] else 'disabled (FTS only)'}"
    )

    return "\n".join(lines)


@mcp.tool()
async def entity_track(
    ctx: Context,
    name: str,
    kind: str,
    project: Optional[str] = None,
    properties: Optional[dict[str, Any]] = None,
) -> str:
    """Track an entity in the knowledge graph. Creates or updates.

    Use this to register important things you encounter — files, modules,
    tools, people, concepts. Once tracked, entities can be linked to memories
    and to each other.

    Entity kinds: project, file, module, concept, tool, person, library, config, command

    Args:
        name: Entity name (e.g. "postgresql.py", "auth-module", "React")
        kind: One of: project, file, module, concept, tool, person, library, config, command
        project: Project to scope to (entities without a project are global)
        properties: Key-value metadata (e.g. {"language": "python", "version": "3.12"})
    """
    await _mcp_enforce_project(ctx, project)
    mgr = get_manager()
    entity = await mgr.track_entity(
        name=name,
        kind=kind,
        project_name=project,
        properties=properties or {},
    )
    return f"Tracking entity: {entity.name} ({entity.kind.value})"


@mcp.tool()
async def entity_relate(
    ctx: Context,
    from_entity: str,
    relation: str,
    to_entity: str,
) -> str:
    """Create a directed relationship between two entities.

    Builds the knowledge graph edges. Both entities are auto-created if
    they don't exist yet (as kind=concept).

    Common relations: depends_on, part_of, uses, implements, created_by,
    replaces, tests, configures, deploys, calls, extends, contains

    Args:
        from_entity: Source entity name (the subject)
        relation: Relationship type (verb-like, e.g. "depends_on")
        to_entity: Target entity name (the object)
    """
    await get_mcp_auth(ctx)
    mgr = get_manager()
    return await mgr.relate_entities(
        from_entity=from_entity, relation=relation, to_entity=to_entity
    )


@mcp.tool()
async def entity_explore(
    ctx: Context,
    entity: str,
    depth: int = 2,
    direction: str = "both",
    project: Optional[str] = None,
) -> str:
    """Explore the knowledge graph from an entity — see what's connected.

    Use this to understand the relationships around a concept, file, or module.
    Returns connected entities and their relationships up to N hops away.

    Args:
        entity: Starting entity name (e.g. "auth-module")
        depth: How many hops to traverse (1-5, default 2)
        direction: "outgoing" (what it uses), "incoming" (what uses it), or "both"
        project: Scope traversal to a project
    """
    await _mcp_enforce_project(ctx, project)
    mgr = get_manager()
    results = await mgr.explore_entity(
        entity_name=entity,
        depth=depth,
        direction=direction,
        project_name=project,
    )

    if not results:
        return f"No connections found for entity: {entity}"

    lines = [f"Graph from '{entity}' (depth={depth}, {direction}):\n"]
    for er in results:
        e = er.entity
        lines.append(f"- **{e.name}** ({e.kind.value})")
        for r in er.relationships:
            lines.append(f"    --[{r.relation}]--> {r.to_entity}")
    return "\n".join(lines)


@mcp.tool()
async def project_set_persistent(
    ctx: Context,
    project: str,
    enabled: bool = True,
) -> str:
    """Enable or disable persistent memories for a project.

    When enabled, ALL memories in the project skip decay and garbage
    collection — they never fade.  When disabled, memories resume
    normal decay behavior.  Default for all projects is non-persistent
    (memories fade naturally over time).

    Agents can always delete individual memories regardless of this setting.

    Args:
        project: Project name
        enabled: True to make memories persistent, False to allow normal decay
    """
    await _mcp_enforce_project(ctx, project)
    mgr = get_manager()
    existing = await mgr.get_project(project)
    if not existing:
        return f"Error: project '{project}' not found"
    ok = await mgr.set_project_persistent(project, enabled)
    if ok:
        bus = get_activity_bus()
        state = "enabled" if enabled else "disabled"
        await bus.emit(EventType.SESSION, f"Persistent memories {state} for {project}", project=project)
        return f"Persistent memories {'enabled' if enabled else 'disabled'} for project '{project}'"
    return f"No changes made (project '{project}' may already have that setting)"


@mcp.tool()
async def bulk_import(
    ctx: Context,
    directory: str,
    mode: str = "project",
    project: Optional[str] = None,
    limit: int = 500,
) -> str:
    """Import files from a directory into engram as memories.

    Scans a directory, chunks the content, and stores each chunk as a
    memory. Supports two modes:

    - **project**: Import source code, configs, docs, and other project
      files (25 file types including .py, .ts, .md, .yaml, etc.)
    - **chat**: Import conversation exports — auto-detects Claude JSON,
      ChatGPT JSON, Claude Code JSONL, and plain text formats.

    Args:
        directory: Absolute path to the directory to import.
        mode: Import mode — "project" for code/docs, "chat" for conversations.
        project: Project to store the memories under.
        limit: Maximum number of files to process (default 500, max 5000).
    """
    import os

    await _mcp_enforce_project(ctx, project)
    directory = os.path.abspath(directory)
    if not os.path.isdir(directory):
        return f"Error: not a directory: {directory}"

    path_err = _validate_import_path(directory)
    if path_err:
        return f"Error: {path_err}"

    mgr = get_manager()
    limit = max(1, min(limit, 5000))

    if mode == "chat":
        chunks, result = import_chat_directory(directory, wing=project or "default", limit=limit)
    else:
        chunks, result = import_project_files(directory, wing=project or "default", limit=limit)

    stored = 0
    store_errors: list[str] = []
    for chunk in chunks:
        try:
            await mgr.remember(
                content=chunk.content,
                kind=chunk.kind,
                project_name=project,
                subject=chunk.subject,
                tags=chunk.tags,
            )
            stored += 1
        except Exception as e:
            store_errors.append(str(e))

    lines = [
        f"Bulk import completed ({mode} mode):",
        f"  Directory: {directory}",
        f"  Files processed: {result.files_processed}",
        f"  Files skipped: {result.files_skipped}",
        f"  Chunks created: {result.chunks_created}",
        f"  Memories stored: {stored}",
    ]
    if result.errors or store_errors:
        all_errs = (result.errors + store_errors)[:10]
        lines.append(f"  Errors ({len(result.errors) + len(store_errors)} total):")
        for err in all_errs:
            lines.append(f"    - {err}")
    return "\n".join(lines)


# ── Mount MCP on / ──────────────────────────────────────────────────────────
# The FastMCP SSE transport is mounted as a sub-application.
# Clients connect to /sse for the SSE stream and /messages for posting.
# Main app routes (/, /api/*, /health) take precedence.

mcp_app = mcp.sse_app()
app.mount("/", mcp_app)


# ── Entry point ──────────────────────────────────────────────────────────────


def main():
    import uvicorn

    uvicorn.run(
        "engram.server:app",
        host="0.0.0.0",
        port=8000,
        log_level="info",
    )


if __name__ == "__main__":
    main()
