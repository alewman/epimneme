"""Async memory manager — coordinates PostgreSQL store + embeddings + decay + dedup.

All public methods are async.  Embeddings run in a thread pool to avoid
blocking the event loop.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

from epimneme.activity import EventType, get_activity_bus
from epimneme.core.config import EngramConfig, default_config
from epimneme.core.models import (
    ContextBundle,
    Entity,
    EntityKind,
    EntityResult,
    Memory,
    MemoryKind,
    MemoryResult,
    Project,
    Relationship,
    RememberResult,
    Session,
)
from epimneme.decay import calculate_retrievability, decay_score_boost, update_on_access
from epimneme.dateutil import extract_date_terms
from epimneme.dedup import compute_simhash, entities_diverge
from epimneme.fusion import (
    adaptive_keyword_weight,
    apply_preference_signal_boost,
    apply_recency_boost,
    apply_temporal_boost,
    apply_turn_pair_boost,
    bm25_rank,
    date_proximity_rank,
    entity_overlap_rank,
    extract_context_entities,
    extract_prf_terms,
    extract_preference_terms,
    extract_proper_nouns,
    gap_aware_tiebreak,
    has_recency_intent,
    is_counting_query,
    is_vague_query,
    mmr_rerank,
    parse_target_date,
    rrf_fuse,
    session_recency_rank,
    temporal_hard_filter,
    turn_pair_rank,
)
from epimneme.rerank import keyword_rerank
from epimneme.stores.postgresql import PostgresStore

logger = logging.getLogger(__name__)


class MemoryManager:
    """Async unified API backed by PostgreSQL.

    Create via the async classmethod::

        mgr = await MemoryManager.create(config)
    """

    def __init__(self, config: EngramConfig, store: PostgresStore) -> None:
        self.config = config
        self.store = store
        self._embedder = None
        self._embeddings_enabled = config.embeddings_enabled
        # Limit concurrent model.encode() calls — prevents timeout with many workers
        self._embed_semaphore: asyncio.Semaphore = asyncio.Semaphore(2)

        # Dedup / conflict counters (reset on restart, lightweight)
        self._counter_remember_calls = 0
        self._counter_simhash_dedup = 0
        self._counter_semantic_dedup = 0
        self._counter_conflicts_surfaced = 0

    @classmethod
    async def create(cls, config: Optional[EngramConfig] = None) -> "MemoryManager":
        """Factory: create manager with an open store."""
        config = config or default_config()
        store = PostgresStore(dsn=config.pg_dsn, embedding_dim=config.embedding_dim, pool_timeout=config.pg_pool_timeout, hnsw_ef_search=config.hnsw_ef_search)
        await store.open()
        mgr = cls(config, store)
        logger.info(
            f"MemoryManager ready — embeddings={'on' if mgr._embeddings_enabled else 'off'}"
        )
        return mgr

    async def close(self) -> None:
        await self.store.close()

    @staticmethod
    def _log_task_error(task: asyncio.Task) -> None:
        """Callback for fire-and-forget tasks to log exceptions."""
        if task.cancelled():
            return
        exc = task.exception()
        if exc:
            logger.error("Background task failed: %s", exc, exc_info=exc)

    # ── Embedding (runs in thread pool) ──────────────────────────────────

    @property
    def embedder(self):
        """Lazy-load sentence-transformers model on first use."""
        if self._embedder is None and self._embeddings_enabled:
            try:
                from sentence_transformers import SentenceTransformer
                self._embedder = SentenceTransformer(self.config.embedding_model)
                logger.info(f"Loaded embedding model: {self.config.embedding_model}")
            except Exception as e:
                logger.warning(f"Failed to load embeddings: {e}")
                self._embeddings_enabled = False
        return self._embedder

    @property
    def _maxsim(self):
        """Lazy-load MaxSim reranker (reuses the same bi-encoder model)."""
        if not hasattr(self, "_maxsim_reranker"):
            self._maxsim_reranker = None
        if self._maxsim_reranker is None and self.embedder is not None:
            try:
                from epimneme.maxsim import MaxSimReranker
                self._maxsim_reranker = MaxSimReranker(
                    self.embedder,
                    cache_size=self.config.maxsim_cache_size,
                )
                logger.info("MaxSim reranker initialised (cache=%d)", self.config.maxsim_cache_size)
            except Exception as e:
                logger.warning("MaxSim reranker failed to init: %s", e)
        return self._maxsim_reranker

    def _embed_sync(self, text: str) -> Optional[list[float]]:
        """Synchronous embedding — called inside thread pool."""
        if not self._embeddings_enabled or self.embedder is None:
            return None
        try:
            vec = self.embedder.encode(text, normalize_embeddings=True)
            return vec.tolist()
        except Exception as e:
            logger.warning(f"Embedding failed: {e}")
            return None

    def _embed_query_sync(self, text: str) -> Optional[list[float]]:
        """Synchronous query embedding — prepends query_prefix if configured."""
        prefix = self.config.embedding_query_prefix
        return self._embed_sync(f"{prefix}{text}" if prefix else text)

    async def _embed(self, text: str) -> Optional[list[float]]:
        """Generate embedding in a worker thread (non-blocking)."""
        if not self._embeddings_enabled:
            return None
        return await asyncio.to_thread(self._embed_sync, text)

    async def _embed_query(self, text: str) -> Optional[list[float]]:
        """Generate query embedding with optional instruction prefix (non-blocking)."""
        if not self._embeddings_enabled:
            return None
        return await asyncio.to_thread(self._embed_query_sync, text)

    def _embed_batch_sync(self, texts: list[str]) -> list[Optional[list[float]]]:
        """Encode a batch of texts in a single model.encode() call (much faster than N individual calls)."""
        if not self._embeddings_enabled or self.embedder is None:
            return [None] * len(texts)
        try:
            import numpy as np
            vecs = self.embedder.encode(texts, normalize_embeddings=True, batch_size=64, show_progress_bar=False)
            return [v.tolist() for v in vecs]
        except Exception:
            logger.exception("Batch embedding failed")
            return [None] * len(texts)

    async def _embed_batch(self, texts: list[str]) -> list[Optional[list[float]]]:
        """Non-blocking batch embedding — encodes all texts in one GPU/CPU kernel call."""
        if not self._embeddings_enabled:
            return [None] * len(texts)
        async with self._embed_semaphore:
            return await asyncio.to_thread(self._embed_batch_sync, texts)

    # ── Projects ─────────────────────────────────────────────────────────

    async def create_project(
        self,
        name: str,
        path: Optional[str] = None,
        description: Optional[str] = None,
        persistent_memories: bool = False,
    ) -> Project:
        existing = await self.store.get_project(name)
        if existing:
            return existing

        project = Project(name=name, path=path, description=description,
                          persistent_memories=persistent_memories)
        await self.store.create_project(project)

        await self.store.track_entity(
            Entity(
                id=project.id,
                name=name,
                kind=EntityKind.PROJECT,
                properties={"path": path or "", "description": description or ""},
            )
        )
        return project

    async def list_projects(self) -> list[Project]:
        return await self.store.list_projects()

    async def get_project(self, name: str) -> Optional[Project]:
        return await self.store.get_project(name)

    async def set_project_persistent(self, project_name: str, enabled: bool) -> bool:
        """Enable or disable persistent memories for a project.

        When enabled, all memories in the project skip decay and GC.
        """
        return await self.store.set_project_persistent(project_name, enabled)

    async def project_status(self, project_name: str) -> dict:
        project = await self.store.get_project(project_name)
        if not project:
            return {"error": f"Project '{project_name}' not found"}

        pid = project.id
        # Parallel independent reads (was 5 sequential round-trips)
        last_session, memory_count, entities, issues, decisions = await asyncio.gather(
            self.store.get_last_session(pid),
            self.store.get_memory_count(pid),
            self.store.list_entities(pid),
            self.store.get_memories_by_kind(MemoryKind.ISSUE, pid, limit=10),
            self.store.get_memories_by_kind(MemoryKind.DECISION, pid, limit=10),
        )

        return {
            "project": project.model_dump(),
            "memory_count": memory_count,
            "entity_count": len(entities),
            "open_issues": len(issues),
            "last_session": last_session.model_dump() if last_session else None,
            "recent_decisions": [d.model_dump() for d in decisions[:5]],
            "entities_by_kind": _group_count(entities, lambda e: e.kind.value),
        }

    # ── Sessions ─────────────────────────────────────────────────────────

    async def session_start(
        self,
        project_name: Optional[str] = None,
        task: Optional[str] = None,
        tier: Optional[str] = None,
    ) -> ContextBundle:
        project = None
        project_id = None
        tier_upper = (tier or "full").upper()
        if tier_upper == "FULL":
            tier_upper = "L3"

        if project_name:
            project = await self.store.get_project(project_name)
            if not project:
                project = await self.create_project(project_name)
                logger.info(f"Auto-created project '{project_name}' on session start")
            project_id = project.id

        session = Session(project_id=project_id, task=task)
        await self.store.create_session(session)

        last_session = await self.store.get_previous_session(project_id, exclude_id=session.id)

        # L0 = project header only — skip all DB-heavy queries
        decisions: list[Memory] = []
        issues: list[Memory] = []
        procedures: list[Memory] = []
        preferences: list[Memory] = []
        relevant_memories: list[MemoryResult] = []
        related_entities: list[EntityResult] = []
        pinned_memories: list[Memory] = []

        # L1+ : decisions + pinned
        if tier_upper >= "L1":
            decisions, pinned_memories = await asyncio.gather(
                self.store.get_memories_by_kind(MemoryKind.DECISION, project_id, limit=20),
                self.store.get_pinned_memories(project_id),
            )

        # L2+ : procedures, preferences, issues
        if tier_upper >= "L2":
            issues, procedures, preferences = await asyncio.gather(
                self.store.get_memories_by_kind(MemoryKind.ISSUE, project_id, limit=10),
                self.store.get_memories_by_kind(MemoryKind.PROCEDURE, project_id, limit=10),
                self.store.get_memories_by_kind(MemoryKind.PREFERENCE, project_id, limit=10),
            )

        # L3/full : semantic recall + entity graph
        if tier_upper >= "L3":
            if task:
                relevant_memories = await self.recall(task, project_name=project_name, limit=15)
            if project_id:
                entities = await self.store.list_entities(project_id)
                top_entities = entities[:20]
                if top_entities:
                    rels_map = await self.store.get_relationships_batch(
                        [e.id for e in top_entities]
                    )
                    for ent in top_entities:
                        related_entities.append(
                            EntityResult(entity=ent, relationships=rels_map.get(ent.id, []))
                        )

        bundle = ContextBundle(
            session_id=session.id,
            project=project,
            last_session=last_session,
            relevant_memories=relevant_memories,
            related_entities=related_entities,
            recent_decisions=decisions,
            known_issues=issues,
            procedures=procedures,
            preferences=preferences,
            pinned_memories=pinned_memories,
        )

        logger.info(
            f"Session started: {session.id[:8]}… | "
            f"{len(relevant_memories)} memories | "
            f"{len(related_entities)} entities"
        )

        bus = get_activity_bus()
        await bus.emit(
            EventType.SESSION,
            f"Session started for {project_name or 'global'}: {task or 'no task'}",
            project=project_name,
            detail=f"{len(relevant_memories)} memories, {len(related_entities)} entities loaded",
        )

        return bundle

    async def session_end(
        self,
        session_id: str,
        summary: str,
        handoff: Optional[str] = None,
    ) -> str:
        session = await self.store.get_session_by_id(session_id)
        if not session:
            return f"Session {session_id[:8]}… not found"
        if session.ended_at:
            return f"Session {session_id[:8]}… already ended"

        await self.store.end_session(session_id, summary=summary, handoff=handoff)

        bus = get_activity_bus()
        proj_name = None
        if session.project_id:
            proj = await self.store.get_project_by_id(session.project_id)
            proj_name = proj.name if proj else None
        await bus.emit(
            EventType.SESSION,
            f"Session ended: {summary[:80]}",
            project=proj_name,
            detail=handoff[:200] if handoff else None,
        )

        return f"Session {session_id[:8]}… ended"

    # ── Memory CRUD ──────────────────────────────────────────────────────

    async def remember(
        self,
        content: str,
        kind: str | MemoryKind = MemoryKind.FACT,
        project_name: Optional[str] = None,
        subject: Optional[str] = None,
        tags: Optional[list[str]] = None,
        confidence: float = 1.0,
        supersedes: Optional[str] = None,
        related_to: Optional[list[str]] = None,
        session_id: Optional[str] = None,
        _precomputed_embedding: Optional[list[float]] = None,
    ) -> RememberResult | str:
        """Store a new memory with dedup check, embedding, and graph links.

        Returns RememberResult on success (memory + any potential conflicts),
        or a string message if deduplicated.
        """
        if isinstance(kind, str):
            kind = MemoryKind(kind)

        self._counter_remember_calls += 1

        project_id = None
        if project_name:
            project = await self.store.get_project(project_name)
            if project:
                project_id = project.id

        # ── Pass 1: SimHash deduplication (fast, O(1) hash compare) ──
        simhash_val = None
        if self.config.dedup_enabled:
            simhash_val = compute_simhash(content)
            dupes = await self.store.find_similar_by_simhash(
                simhash_val,
                project_id=project_id,
                threshold=self.config.dedup_hamming_threshold,
            )
            if dupes:
                dupe = dupes[0]
                # Entity isolation: don't merge if key entities differ
                if entities_diverge(content, dupe.content):
                    logger.info(
                        f"SimHash match {dupe.id[:8]}… but entities diverge — storing"
                    )
                else:
                    self._counter_simhash_dedup += 1
                    logger.info(
                        f"Near-duplicate detected (simhash): new content matches {dupe.id[:8]}…"
                    )
                    bus = get_activity_bus()
                    await bus.emit(
                        EventType.DEDUP,
                        f"SimHash blocked duplicate of {dupe.id[:8]}…",
                        project=project_name,
                        detail=content[:120],
                        memory_id=dupe.id,
                    )
                    return (
                        f"Near-duplicate of memory {dupe.id[:8]}… — "
                        f"existing: {dupe.content[:100]}"
                    )

        # ── Generate embedding (needed for semantic dedup + storage) ──
        embedding = _precomputed_embedding if _precomputed_embedding is not None else await self._embed(content)

        # ── Pass 2: Semantic deduplication (catches same-meaning, different-wording) ──
        if (
            self.config.semantic_dedup_enabled
            and embedding is not None
            and not supersedes  # skip if explicitly replacing a memory
        ):
            sem_dupes = await self.store.find_semantic_duplicates(
                embedding,
                project_id=project_id,
                threshold=self.config.semantic_dedup_threshold,
            )
            if sem_dupes:
                best = sem_dupes[0]
                # Entity isolation: don't merge if key entities differ
                if entities_diverge(content, best.memory.content):
                    logger.info(
                        f"Semantic match {best.memory.id[:8]}… ({best.score:.2f}) "
                        f"but entities diverge — storing"
                    )
                else:
                    sim = f"{best.score:.2f}"
                    self._counter_semantic_dedup += 1
                    logger.info(
                        f"Semantic near-duplicate detected ({sim}): "
                        f"matches {best.memory.id[:8]}…"
                    )
                    bus = get_activity_bus()
                    await bus.emit(
                        EventType.DEDUP,
                        f"Semantic dedup blocked ({sim} sim) of {best.memory.id[:8]}…",
                        project=project_name,
                        detail=content[:120],
                        memory_id=best.memory.id,
                    )
                    return (
                        f"Semantic near-duplicate of memory {best.memory.id[:8]}… "
                        f"({sim} similarity). "
                        f"Existing: {best.memory.content[:120]}\n"
                        f"Use supersedes='{best.memory.id}' to explicitly replace it."
                    )
            else:
                logger.debug(
                    "Semantic dedup: no near-duplicates found "
                    f"(threshold={self.config.semantic_dedup_threshold})"
                )

        # ── Conflict surfacing for facts/decisions ──
        potential_conflicts: list[MemoryResult] = []
        if (
            embedding is not None
            and kind in (MemoryKind.FACT, MemoryKind.DECISION)
            and not supersedes
        ):
            potential_conflicts = await self.store.find_potential_conflicts(
                embedding,
                kind=kind.value,
                project_id=project_id,
                threshold=0.80,
                limit=3,
            )
            if potential_conflicts:
                self._counter_conflicts_surfaced += 1
                scores = ", ".join(f"{c.score:.2f}" for c in potential_conflicts)
                logger.debug(
                    f"Conflict surfacing: {len(potential_conflicts)} potential "
                    f"conflict(s) for [{kind.value}] (scores: {scores})"
                )

        # ── Preference noun-weighting ──────────────────────────────────
        # Extract key nouns from preference-bearing sentences.  After the
        # memory is saved the store boosts these terms at weight 'A' in
        # the tsvector so full-text search surfaces preference-relevant
        # memories for vague queries like "any tips on my camera?".
        pref_terms = extract_preference_terms(content)

        memory = Memory(
            project_id=project_id,
            session_id=session_id,
            kind=kind,
            content=content,
            subject=subject,
            confidence=confidence,
            supersedes=supersedes,
            tags=tags or [],
            simhash=simhash_val,
        )

        await self.store.store_memory(memory, embedding=embedding)

        # Boost preference terms in tsvector at weight 'A' (post-INSERT)
        if pref_terms:
            await self.store.boost_tsvector_terms(
                memory.id, " ".join(pref_terms[:20])
            )

        # Inject date normalizations into tsvector at weight 'B' (post-INSERT).
        # Expands "May 1st, 2022" → ISO, US, UK, spelled-out forms so that
        # FTS queries using any date format can find this memory.
        date_terms = extract_date_terms(content)
        if date_terms:
            await self.store.boost_tsvector_terms(
                memory.id, " ".join(date_terms[:40])
            )

        if related_to:
            await self.store.link_memory_to_entities(
                memory_id=memory.id,
                entity_names=related_to,
                project_id=project_id,
                relation="about",
            )

        logger.info(f"Stored [{kind.value}]: {content[:80]}…")

        bus = get_activity_bus()
        await bus.emit(
            EventType.WRITE,
            f"Stored [{kind.value}] {memory.id[:8]}…: {content[:80]}",
            project=project_name,
            detail=content[:300],
            memory_id=memory.id,
        )
        if potential_conflicts:
            await bus.emit(
                EventType.CONFLICT,
                f"{len(potential_conflicts)} potential conflict(s) for [{kind.value}]",
                project=project_name,
                detail="\n".join(f"({c.score:.2f}) {c.memory.content[:80]}" for c in potential_conflicts),
                memory_id=memory.id,
            )

        return RememberResult(memory=memory, potential_conflicts=potential_conflicts)

    async def update_memory(
        self,
        memory_id: str,
        content: str,
        subject: Optional[str] = None,
        tags: Optional[list[str]] = None,
        confidence: Optional[float] = None,
    ) -> Optional[Memory]:
        """Update a memory, creating a new version (history preserved)."""
        simhash_val = compute_simhash(content) if self.config.dedup_enabled else None
        embedding = await self._embed(content)

        return await self.store.update_memory(
            memory_id,
            content=content,
            subject=subject,
            tags=tags,
            confidence=confidence,
            embedding=embedding,
            simhash=simhash_val,
        )

    async def get_memory_versions(self, memory_id: str) -> list[Memory]:
        """Get the full version chain for a memory."""
        return await self.store.get_memory_versions(memory_id)

    async def recall(
        self,
        query: str,
        project_name: Optional[str] = None,
        kind: Optional[str] = None,
        tags: Optional[list[str]] = None,
        limit: int = 20,
        reference_date: Optional[str] = None,
    ) -> list[MemoryResult]:
        """Search memories with multi-signal RRF hybrid fusion.

        Pipeline:
          1. Over-fetch from semantic + full-text search (3× limit).
          2. Build additional ranked lists from the candidate pool:
             BM25, entity-overlap, date-proximity (temporal queries),
             session-recency (recency-intent queries), turn-pair completeness.
          3. Fuse all ranked lists via RRF with per-signal adaptive weights.
          4. Apply proper-noun boost, decay scoring, keyword reranking.
          5. Populate session ordinals; apply recency + vague-query boosts.
          6. MaxSim rerank top-N (if enabled).
          7. PRF: second FTS pass with expanded query (if enabled + vague).
          8. Existing soft boosts: preference signal, temporal Gaussian, turn-pair.
          9. MMR session diversification (if enabled + counting query).
         10. Gap-aware deterministic tiebreaker (if enabled).
         11. Temporal hard-filter (if enabled + day-precision date resolved).
         12. Sort, truncate, fire decay updates.
        """
        project_id = None
        if project_name:
            project = await self.store.get_project(project_name)
            if project:
                project_id = project.id

        kind_enum = MemoryKind(kind) if kind else None

        # Over-fetch: cast a wider net, fuse later
        fetch_limit = max(limit * self.config.rrf_overfetch_multiplier, 30)

        # ── 1. Primary retrieval ────────────────────────────────────────
        embedding = await self._embed_query(query)
        semantic_results: list[MemoryResult] = []
        if embedding:
            semantic_results = await self.store.search_semantic(
                embedding,
                project_id=project_id,
                kind=kind,
                tags=tags,
                limit=fetch_limit,
            )

        fulltext_results = await self.store.search_fulltext(
            query,
            project_id=project_id,
            kind=kind_enum,
            tags=tags,
            limit=fetch_limit,
        )

        # ── 2. Build candidate pool for derived ranked lists ────────────
        # Union of semantic + FTS (pre-fusion) — all downstream rankers
        # operate on this same pool so there is no additional DB cost.
        candidate_map: dict[str, MemoryResult] = {}
        for mr in [*semantic_results, *fulltext_results]:
            if mr.memory.id not in candidate_map:
                candidate_map[mr.memory.id] = mr
        candidates = list(candidate_map.values())

        # Resolve reference date once (used by date-proximity and temporal boost)
        from datetime import date as _date
        parsed_ref: _date | None = None
        if reference_date:
            try:
                raw = reference_date[:10].replace("/", "-")
                parsed_ref = _date.fromisoformat(raw)
            except ValueError:
                pass

        # ── 3. Additional ranked-list signals ───────────────────────────
        kw_weight = adaptive_keyword_weight(query, self.config.rrf_keyword_weight)
        rrf_lists: list[list[MemoryResult]] = [semantic_results, fulltext_results]
        rrf_weights: list[float] = [self.config.rrf_vector_weight, kw_weight]

        if self.config.bm25_signal_enabled and candidates:
            rrf_lists.append(bm25_rank(query, candidates))
            rrf_weights.append(self.config.bm25_signal_weight)

        if self.config.entity_signal_enabled and candidates:
            rrf_lists.append(entity_overlap_rank(query, candidates))
            rrf_weights.append(self.config.entity_signal_weight)

        # Date-proximity signal: resolve target date from query
        _target_date: _date | None = None
        if candidates and parsed_ref is not None:
            _target_date = parse_target_date(query, parsed_ref)
        if _target_date is not None:
            rrf_lists.append(date_proximity_rank(candidates, _target_date))
            rrf_weights.append(self.config.date_signal_weight)

        # Session-recency signal: added after ordinals are known; placeholder here —
        # we defer to step 5 where ordinals are populated.

        # Turn-pair signal (cheap, always safe)
        if candidates:
            rrf_lists.append(turn_pair_rank(candidates))
            rrf_weights.append(self.config.turn_pair_signal_weight)

        # ── 4. RRF fusion over all ranked lists ─────────────────────────
        fused = rrf_fuse(*rrf_lists, weights=rrf_weights)

        # ── 4b. Proper-noun boost (mild, name-matching) ─────────────────
        proper_nouns = extract_proper_nouns(query)
        if proper_nouns:
            for mr in fused.values():
                content_lower = mr.memory.content.lower()
                hits = sum(1 for n in proper_nouns if n.lower() in content_lower)
                if hits:
                    mr.score += hits * 0.004

        # ── 4c. Decay scoring ────────────────────────────────────────────
        for mr in fused.values():
            m = mr.memory
            retrievability = calculate_retrievability(
                m.storage_strength,
                m.last_accessed,
                base_stability=self.config.decay_base_stability,
            )
            mr.score *= decay_score_boost(retrievability)

        # ── 4d. Keyword reranking ────────────────────────────────────────
        rerank_input = [
            (mr.memory.id, mr.score, mr.memory.content)
            for mr in fused.values()
        ]
        if rerank_input:
            reranked = keyword_rerank(query, rerank_input)
            for rr in reranked:
                if rr.memory_id in fused:
                    fused[rr.memory_id].score = rr.final_score

        # ── 5. Session ordinals + recency/vague-query boosts ─────────────
        session_ids = list({
            mr.memory.session_id
            for mr in fused.values()
            if mr.memory.session_id
        })
        ordinals: dict[str, int] = {}
        if session_ids:
            ordinals = await self.store.get_session_ordinals(session_ids)
            for mr in fused.values():
                sid = mr.memory.session_id
                if sid and sid in ordinals:
                    mr.memory.session_ordinal = ordinals[sid]

            # Session-recency as additional RRF input (now that ordinals are known)
            if has_recency_intent(query) and candidates:
                recency_list = session_recency_rank(candidates, ordinals)
                recency_fused = rrf_fuse(recency_list, weights=[self.config.recency_signal_weight])
                for mid, mr in recency_fused.items():
                    if mid in fused:
                        fused[mid].score += mr.score

            if has_recency_intent(query):
                apply_recency_boost(fused, ordinals)

            if is_vague_query(query):
                ctx_entities = extract_context_entities(list(fused.values()), ordinals)
                if ctx_entities:
                    for mr in fused.values():
                        content_lower = mr.memory.content.lower()
                        hits = sum(1 for e in ctx_entities if e.lower() in content_lower)
                        if hits:
                            mr.score += hits * 0.012

        # ── 6. MaxSim rerank (token-level late interaction) ──────────────
        if self.config.maxsim_enabled and self._maxsim is not None:
            sorted_for_maxsim = sorted(
                fused.values(), key=lambda r: r.score, reverse=True
            )
            reranked_maxsim = await asyncio.to_thread(
                self._maxsim.rerank,
                query,
                sorted_for_maxsim,
                self.config.maxsim_top_n,
            )
            # Assign ranks as scores so the ordering is preserved through
            # subsequent boosts (which are small relative to rank-gaps).
            n = len(reranked_maxsim)
            for rank, mr in enumerate(reranked_maxsim):
                fused[mr.memory.id].score = (n - rank) / n

        # ── 7. PRF: second FTS pass with expanded query ──────────────────
        if self.config.prf_enabled and is_vague_query(query):
            top_for_prf = sorted(
                fused.values(), key=lambda r: r.score, reverse=True
            )[: self.config.prf_top_k]
            expansion_terms = extract_prf_terms(top_for_prf, n_terms=self.config.prf_n_terms)
            if expansion_terms:
                expanded_query = query + " " + " ".join(expansion_terms)
                prf_results = await self.store.search_fulltext(
                    expanded_query,
                    project_id=project_id,
                    kind=kind_enum,
                    tags=tags,
                    limit=fetch_limit,
                )
                if prf_results:
                    prf_fused = rrf_fuse(prf_results, weights=[self.config.prf_fts_weight])
                    for mid, mr in prf_fused.items():
                        if mid in fused:
                            fused[mid].score += mr.score
                        else:
                            fused[mid] = mr

        # ── 8. Remaining soft boosts ─────────────────────────────────────
        apply_preference_signal_boost(fused, query)
        apply_temporal_boost(fused, query, reference_date=parsed_ref)
        apply_turn_pair_boost(fused)

        # ── 9. Sort for MMR / tiebreaker / filter stages ─────────────────
        results = sorted(
            fused.values(),
            key=lambda r: (r.score, -(r.memory.session_ordinal or 0), r.memory.id),
            reverse=True,
        )

        # ── 10. MMR session diversification (single-fact queries only) ──
        # Counting queries need ALL instances visible to count; MMR's
        # session_cap would hide repeated events and cause misses.
        if self.config.mmr_enabled and not is_counting_query(query):
            results = mmr_rerank(
                results,
                lambda_=self.config.mmr_lambda,
                session_cap=self.config.mmr_session_cap,
                limit=limit,
            )

        # ── 11. Gap-aware deterministic tiebreaker ───────────────────────
        if self.config.tiebreak_enabled and len(results) >= 2:
            results = gap_aware_tiebreak(
                results,
                query,
                ordinals,
                eps=self.config.tiebreak_eps,
                target_date=_target_date,
            )

        # ── 12. Temporal hard-filter (optional, day-precision only) ──────
        if (
            self.config.temporal_hard_filter_enabled
            and _target_date is not None
            and self.config.temporal_hard_filter_sigma <= 7.0  # safety: day-precision only
        ):
            filtered = temporal_hard_filter(
                {r.memory.id: r for r in results},
                _target_date,
                sigma_days=self.config.temporal_hard_filter_sigma,
                min_keep=limit,
            )
            if len(filtered) >= limit:
                results = sorted(
                    filtered.values(),
                    key=lambda r: (r.score, -(r.memory.session_ordinal or 0), r.memory.id),
                    reverse=True,
                )

        # Update decay fields for top results (fire-and-forget)
        for r in results[:5]:
            m = r.memory
            new_storage, new_retrieval, new_count = update_on_access(
                m.storage_strength,
                m.access_count,
                growth_factor=self.config.decay_growth_factor,
            )
            t1 = asyncio.create_task(
                self.store.update_decay_on_access(
                    m.id, new_storage, new_retrieval, new_count
                )
            )
            t2 = asyncio.create_task(
                self.store.log_access(m.id, context=query[:200])
            )
            t1.add_done_callback(self._log_task_error)
            t2.add_done_callback(self._log_task_error)

        final = results[:limit]

        bus = get_activity_bus()
        await bus.emit(
            EventType.RECALL,
            f"Recalled {len(final)} memories for '{query[:60]}'",
            project=project_name,
            detail=f"Top: {final[0].memory.content[:80]}" if final else None,
        )

        return final

    async def forget(self, memory_id: str, reason: str = "") -> str:
        memory = await self.store.get_memory(memory_id)
        if not memory:
            return f"Memory {memory_id} not found"

        await self.store.mark_obsolete(memory_id)
        logger.info(f"Forgot memory {memory_id[:8]}…: {reason}")

        bus = get_activity_bus()
        proj_name = None
        if memory.project_id:
            proj = await self.store.get_project_by_id(memory.project_id)
            proj_name = proj.name if proj else None
        await bus.emit(
            EventType.FORGET,
            f"Forgot {memory_id[:8]}…: {reason[:60] or 'no reason'}",
            project=proj_name,
            detail=memory.content[:200],
            memory_id=memory_id,
        )

        return f"Memory {memory_id[:8]}… marked obsolete"

    async def hard_forget(self, memory_id: str, reason: str = "") -> str:
        """Permanently delete a memory from the database (irreversible)."""
        memory = await self.store.get_memory(memory_id)
        if not memory:
            return f"Memory {memory_id} not found"

        content_preview = memory.content[:200]
        deleted = await self.store.hard_delete_memory(memory_id)
        if not deleted:
            return f"Memory {memory_id} could not be deleted"

        logger.info(f"Hard-deleted memory {memory_id[:8]}…: {reason}")

        bus = get_activity_bus()
        proj_name = None
        if memory.project_id:
            proj = await self.store.get_project_by_id(memory.project_id)
            proj_name = proj.name if proj else None
        await bus.emit(
            EventType.FORGET,
            f"Hard-deleted {memory_id[:8]}…: {reason[:60] or 'no reason'}",
            project=proj_name,
            detail=content_preview,
            memory_id=memory_id,
        )

        return f"Memory {memory_id[:8]}… permanently deleted"

    # ── Context ──────────────────────────────────────────────────────────

    async def get_context(
        self,
        query: str,
        project_name: Optional[str] = None,
    ) -> ContextBundle:
        project = None
        project_id = None
        if project_name:
            project = await self.store.get_project(project_name)
            if project:
                project_id = project.id

        relevant = await self.recall(query, project_name=project_name, limit=15)

        # Batch-fetch entities linked to the top relevant memories
        top_memory_ids = [r.memory.id for r in relevant[:5]]
        entities_by_memory = await self.store.get_entities_for_memories_batch(top_memory_ids)

        entity_names: set[str] = set()
        for r in relevant[:5]:
            if r.memory.subject:
                entity_names.add(r.memory.subject)
            for ent in entities_by_memory.get(r.memory.id, []):
                entity_names.add(ent.name)
        for word in query.split():
            if len(word) > 2:
                entity_names.add(word)

        # Batch explore: gather all explore calls concurrently
        related_entities: list[EntityResult] = []
        seen_entity_ids: set[str] = set()
        if entity_names:
            explore_results = await asyncio.gather(
                *(self.store.explore(name, depth=1, project_id=project_id)
                  for name in entity_names)
            )
            for results in explore_results:
                for er in results:
                    if er.entity.id not in seen_entity_ids:
                        seen_entity_ids.add(er.entity.id)
                        related_entities.append(er)

        # Batch fetch linked memory IDs: gather all calls concurrently
        seen_memory_ids = {r.memory.id for r in relevant}
        augmented: list[MemoryResult] = []
        if entity_names:
            linked_results = await asyncio.gather(
                *(self.store.get_memories_for_entity(name, project_id, limit=5)
                  for name in entity_names)
            )
            all_linked_ids: list[str] = []
            for linked_ids in linked_results:
                for mid in linked_ids:
                    if mid not in seen_memory_ids:
                        seen_memory_ids.add(mid)
                        all_linked_ids.append(mid)
            # Fetch all linked memories in parallel
            if all_linked_ids:
                mems = await asyncio.gather(
                    *(self.store.get_memory(mid) for mid in all_linked_ids)
                )
                for mem in mems:
                    if mem and not mem.obsolete:
                        augmented.append(MemoryResult(
                            memory=mem, score=0.5, source="graph"
                        ))

        all_relevant = relevant + augmented

        decisions, issues, procedures, preferences = await asyncio.gather(
            self.store.get_memories_by_kind(MemoryKind.DECISION, project_id, limit=10),
            self.store.get_memories_by_kind(MemoryKind.ISSUE, project_id, limit=5),
            self.store.get_memories_by_kind(MemoryKind.PROCEDURE, project_id, limit=5),
            self.store.get_memories_by_kind(MemoryKind.PREFERENCE, project_id, limit=5),
        )

        return ContextBundle(
            project=project,
            relevant_memories=all_relevant,
            related_entities=related_entities,
            recent_decisions=decisions,
            known_issues=issues,
            procedures=procedures,
            preferences=preferences,
        )

    # ── Entities ─────────────────────────────────────────────────────────

    async def track_entity(
        self,
        name: str,
        kind: str | EntityKind,
        project_name: Optional[str] = None,
        properties: Optional[dict] = None,
    ) -> Entity:
        if isinstance(kind, str):
            kind = EntityKind(kind)

        project_id = None
        if project_name:
            project = await self.store.get_project(project_name)
            if project:
                project_id = project.id

        entity = Entity(
            name=name,
            kind=kind,
            project_id=project_id,
            properties=properties or {},
        )
        result = await self.store.track_entity(entity)

        bus = get_activity_bus()
        await bus.emit(
            EventType.ENTITY,
            f"Tracked entity: {name} ({kind.value if hasattr(kind, 'value') else kind})",
            project=project_name,
        )

        return result

    async def relate_entities(
        self,
        from_entity: str,
        relation: str,
        to_entity: str,
        properties: Optional[dict] = None,
    ) -> str:
        rel = Relationship(
            from_entity=from_entity,
            to_entity=to_entity,
            relation=relation,
            properties=properties or {},
        )
        await self.store.relate(rel)

        bus = get_activity_bus()
        await bus.emit(
            EventType.ENTITY,
            f"{from_entity} --[{relation}]--> {to_entity}",
        )

        return f"{from_entity} --[{relation}]--> {to_entity}"

    async def explore_entity(
        self,
        entity_name: str,
        depth: int = 2,
        direction: str = "both",
        project_name: Optional[str] = None,
    ) -> list[EntityResult]:
        project_id = None
        if project_name:
            project = await self.store.get_project(project_name)
            if project:
                project_id = project.id

        return await self.store.explore(
            entity_name, depth=depth, direction=direction, project_id=project_id
        )

    # ── Stats ────────────────────────────────────────────────────────────

    async def stats(self) -> dict:
        mem_count, vec_count, ent_count, proj_count = await asyncio.gather(
            self.store.get_memory_count(),
            self.store.get_vector_count(),
            self.store.count_entities(),
            self.store.count_projects(),
        )
        return {
            "total_memories": mem_count,
            "total_vectors": vec_count,
            "total_entities": ent_count,
            "total_projects": proj_count,
            "embeddings_enabled": self._embeddings_enabled,
            "dedup": {
                "remember_calls": self._counter_remember_calls,
                "simhash_blocked": self._counter_simhash_dedup,
                "semantic_blocked": self._counter_semantic_dedup,
                "conflicts_surfaced": self._counter_conflicts_surfaced,
            },
        }


def _group_count(items: list, key_fn) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        k = key_fn(item)
        counts[k] = counts.get(k, 0) + 1
    return counts
