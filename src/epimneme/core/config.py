"""Engram configuration — PostgreSQL + optional embeddings."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class EngramConfig:
    """Configuration for the Engram memory service.

    All settings can be overridden via environment variables prefixed with EPIMNEME_.
    """

    # PostgreSQL connection
    pg_host: str = "epimneme-db"
    pg_port: int = 5432
    pg_user: str = "epimneme"
    pg_password: str = "epimneme"
    pg_database: str = "epimneme"

    # Embedding model
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384
    embeddings_enabled: bool = True
    # Optional instruction prefix prepended to *query* embeddings only (e.g. BGE models)
    embedding_query_prefix: str = ""

    # Memory decay (power-law retrievability)
    decay_base_stability: float = 1.0  # base half-life in days
    decay_growth_factor: float = 0.5

    # Deduplication (SimHash)
    dedup_enabled: bool = True
    dedup_hamming_threshold: int = 3

    # Semantic deduplication (vector similarity)
    semantic_dedup_enabled: bool = True
    semantic_dedup_threshold: float = 0.92

    # Backup
    backup_dir: str = "/backups"
    backup_keep_last: int = 10    # always keep the N most recent backups
    backup_keep_days: int = 30    # also keep backups newer than this many days

    # Reflection (periodic memory compaction)
    reflection_enabled: bool = True
    reflection_interval_hours: float = 24.0
    reflection_gc_threshold: float = 0.05  # retrievability below this → obsolete
    reflection_gc_min_age_days: float = 7.0
    reflection_consolidation_similarity: float = 0.88
    reflection_min_cluster_size: int = 3
    reflection_max_consolidations: int = 10
    reflection_conflict_similarity: float = 0.85
    reflection_conflict_age_gap_days: float = 7.0

    # CORS
    cors_origins: list[str] | None = None  # None = derive from allowed_hosts

    # Allowed hosts (for Traefik DNS-rebinding protection)
    allowed_hosts: list[str] = field(default_factory=list)

    # Hybrid search fusion
    rrf_vector_weight: float = 1.0
    rrf_keyword_weight: float = 0.75
    rrf_overfetch_multiplier: int = 3  # fetch N×limit from each source

    # Chunking parameters (for bulk import)
    chunk_size: int = 800       # max chars per chunk
    chunk_overlap: int = 100    # overlap between consecutive chunks

    # HNSW retrieval quality
    hnsw_ef_search: int = 100   # higher = better recall, slightly slower queries

    # ── Pure-math recall improvement (no new models or API calls) ────────────

    # Phase A: Additional ranked-list signals fed into multi-signal RRF
    # Each signal produces an additional sorted candidate list that RRF
    # combines with the existing semantic + FTS lists.
    bm25_signal_enabled: bool = True       # in-process BM25 over fetched candidates
    bm25_signal_weight: float = 0.5        # RRF weight for the BM25 list
    entity_signal_enabled: bool = True     # proper nouns + numbers overlap ranking
    entity_signal_weight: float = 0.3      # RRF weight for entity-overlap list
    date_signal_weight: float = 0.6        # RRF weight for date-proximity list (temporal)
    recency_signal_weight: float = 0.2     # RRF weight for session-recency list (recency intent)
    turn_pair_signal_weight: float = 0.15  # RRF weight for turn-pair-completeness list

    # Phase B: Token-level MaxSim (ColBERT-style, reuses existing bi-encoder)
    maxsim_enabled: bool = False    # off by default — opt-in per deployment
    maxsim_top_n: int = 20          # re-rank this many candidates
    maxsim_cache_size: int = 2048   # LRU doc-embedding cache entries

    # Phase C: Pseudo-relevance feedback (Rocchio text expansion, second FTS pass)
    prf_enabled: bool = False    # gated to vague/preference queries only
    prf_top_k: int = 5           # top-K initial results used for term extraction
    prf_n_terms: int = 8         # max expansion terms to append to FTS query
    prf_fts_weight: float = 0.3  # RRF weight for the PRF FTS result list

    # Phase D: Gap-aware deterministic tiebreaker
    tiebreak_enabled: bool = True  # fire when top-2 gap <= tiebreak_eps
    tiebreak_eps: float = 0.005    # score-gap threshold to trigger tiebreaker

    # Phase E: MMR session diversification (counting/aggregation queries)
    mmr_enabled: bool = True   # gated to is_counting_query() detection
    mmr_lambda: float = 0.7    # relevance weight (0=pure diversity, 1=pure relevance)
    mmr_session_cap: int = 2   # max chunks from any single session_id in output

    # Phase F: Temporal hard-filter (optional, gated by config — risky, off by default)
    temporal_hard_filter_enabled: bool = False  # pre-filter candidates to date window
    temporal_hard_filter_sigma: float = 3.5     # half-window in days

    # Bulk import — allowed base directories (path traversal guard)
    import_allowed_dirs: list[str] = field(default_factory=lambda: ["/app"])

    # PostgreSQL connection pool timeout (seconds)
    pg_pool_timeout: float = 30.0

    @property
    def pg_dsn(self) -> str:
        """PostgreSQL connection string."""
        return (
            f"host={self.pg_host} port={self.pg_port} "
            f"dbname={self.pg_database} user={self.pg_user} "
            f"password={self.pg_password}"
        )

    @property
    def pg_dsn_async(self) -> str:
        """Async PostgreSQL connection URL for asyncpg."""
        return (
            f"postgresql://{self.pg_user}:{self.pg_password}"
            f"@{self.pg_host}:{self.pg_port}/{self.pg_database}"
        )


def default_config() -> EngramConfig:
    """Create config from environment variables with sensible defaults."""
    # Require an explicit PG password unless running in demo mode.
    # This prevents accidental deployments with the default "epimneme" password.
    pg_password = os.environ.get("EPIMNEME_PG_PASSWORD", "")
    demo_mode = os.environ.get("EPIMNEME_DEMO_MODE", "") == "1"
    if not pg_password:
        if demo_mode:
            pg_password = "epimneme"  # safe: demo mode only
        else:
            raise RuntimeError(
                "EPIMNEME_PG_PASSWORD is not set. Set a strong password in your "
                "environment (or .env), or set EPIMNEME_DEMO_MODE=1 for local "
                "testing with the default password."
            )
    elif pg_password == "epimneme" and not demo_mode:
        raise RuntimeError(
            "EPIMNEME_PG_PASSWORD is set to the default value 'epimneme'. "
            "Choose a strong password, or set EPIMNEME_DEMO_MODE=1 to override "
            "for local testing."
        )

    # Parse allowed hosts
    allowed_hosts_raw = os.environ.get("EPIMNEME_ALLOWED_HOSTS", "")
    allowed_hosts = [h.strip() for h in allowed_hosts_raw.split(",") if h.strip()]

    # Parse CORS origins: explicit list, or derive from allowed hosts
    cors_raw = os.environ.get("EPIMNEME_CORS_ORIGINS", "")
    if cors_raw:
        cors_origins = [o.strip() for o in cors_raw.split(",") if o.strip()]
    elif allowed_hosts:
        cors_origins = [f"https://{h}" for h in allowed_hosts]
    else:
        cors_origins = None

    config = EngramConfig(
        pg_host=os.environ.get("EPIMNEME_PG_HOST", "epimneme-db"),
        pg_port=int(os.environ.get("EPIMNEME_PG_PORT", "5432")),
        pg_user=os.environ.get("EPIMNEME_PG_USER", "epimneme"),
        pg_password=pg_password,
        pg_database=os.environ.get("EPIMNEME_PG_DATABASE", "epimneme"),
        embedding_model=os.environ.get("EPIMNEME_EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
        embedding_dim=int(os.environ.get("EPIMNEME_EMBEDDING_DIM", "384")),
        decay_base_stability=float(os.environ.get("EPIMNEME_DECAY_STABILITY", "1.0")),
        decay_growth_factor=float(os.environ.get("EPIMNEME_DECAY_GROWTH", "0.5")),
        dedup_enabled=os.environ.get("EPIMNEME_DEDUP_ENABLED", "1") == "1",
        dedup_hamming_threshold=int(os.environ.get("EPIMNEME_DEDUP_THRESHOLD", "3")),
        semantic_dedup_enabled=os.environ.get("EPIMNEME_SEMANTIC_DEDUP_ENABLED", "1") == "1",
        semantic_dedup_threshold=float(os.environ.get("EPIMNEME_SEMANTIC_DEDUP_THRESHOLD", "0.92")),
        backup_dir=os.environ.get("EPIMNEME_BACKUP_DIR", "/backups"),
        backup_keep_last=int(os.environ.get("EPIMNEME_BACKUP_KEEP_LAST", "10")),
        backup_keep_days=int(os.environ.get("EPIMNEME_BACKUP_KEEP_DAYS", "30")),
        reflection_enabled=os.environ.get("EPIMNEME_REFLECTION_ENABLED", "1") == "1",
        reflection_interval_hours=float(os.environ.get("EPIMNEME_REFLECTION_INTERVAL_HOURS", "24")),
        reflection_gc_threshold=float(os.environ.get("EPIMNEME_REFLECTION_GC_THRESHOLD", "0.05")),
        reflection_gc_min_age_days=float(os.environ.get("EPIMNEME_REFLECTION_GC_MIN_AGE_DAYS", "7")),
        reflection_consolidation_similarity=float(os.environ.get("EPIMNEME_REFLECTION_CONSOLIDATION_SIM", "0.88")),
        reflection_min_cluster_size=int(os.environ.get("EPIMNEME_REFLECTION_MIN_CLUSTER", "3")),
        reflection_max_consolidations=int(os.environ.get("EPIMNEME_REFLECTION_MAX_CONSOLIDATIONS", "10")),
        reflection_conflict_similarity=float(os.environ.get("EPIMNEME_REFLECTION_CONFLICT_SIM", "0.85")),
        reflection_conflict_age_gap_days=float(os.environ.get("EPIMNEME_REFLECTION_CONFLICT_AGE_GAP", "7")),
        cors_origins=cors_origins,
        allowed_hosts=allowed_hosts,
        import_allowed_dirs=[
            d.strip()
            for d in os.environ.get("EPIMNEME_IMPORT_ALLOWED_DIRS", "/app").split(",")
            if d.strip()
        ],
        pg_pool_timeout=float(os.environ.get("EPIMNEME_PG_POOL_TIMEOUT", "30")),
        rrf_vector_weight=float(os.environ.get("EPIMNEME_RRF_VECTOR_WEIGHT", "1.0")),
        rrf_keyword_weight=float(os.environ.get("EPIMNEME_RRF_KEYWORD_WEIGHT", "0.75")),
        rrf_overfetch_multiplier=int(os.environ.get("EPIMNEME_RRF_OVERFETCH", "3")),
        chunk_size=int(os.environ.get("EPIMNEME_CHUNK_SIZE", "800")),
        chunk_overlap=int(os.environ.get("EPIMNEME_CHUNK_OVERLAP", "100")),
        hnsw_ef_search=int(os.environ.get("EPIMNEME_HNSW_EF_SEARCH", "100")),
        embedding_query_prefix=os.environ.get("EPIMNEME_EMBEDDING_QUERY_PREFIX", ""),
        # Phase A: additional ranked-list signals
        bm25_signal_enabled=os.environ.get("EPIMNEME_BM25_SIGNAL_ENABLED", "1") == "1",
        bm25_signal_weight=float(os.environ.get("EPIMNEME_BM25_SIGNAL_WEIGHT", "0.5")),
        entity_signal_enabled=os.environ.get("EPIMNEME_ENTITY_SIGNAL_ENABLED", "1") == "1",
        entity_signal_weight=float(os.environ.get("EPIMNEME_ENTITY_SIGNAL_WEIGHT", "0.3")),
        date_signal_weight=float(os.environ.get("EPIMNEME_DATE_SIGNAL_WEIGHT", "0.6")),
        recency_signal_weight=float(os.environ.get("EPIMNEME_RECENCY_SIGNAL_WEIGHT", "0.2")),
        turn_pair_signal_weight=float(os.environ.get("EPIMNEME_TURN_PAIR_SIGNAL_WEIGHT", "0.15")),
        # Phase B: MaxSim
        maxsim_enabled=os.environ.get("EPIMNEME_MAXSIM_ENABLED", "0") == "1",
        maxsim_top_n=int(os.environ.get("EPIMNEME_MAXSIM_TOP_N", "20")),
        maxsim_cache_size=int(os.environ.get("EPIMNEME_MAXSIM_CACHE_SIZE", "2048")),
        # Phase C: PRF
        prf_enabled=os.environ.get("EPIMNEME_PRF_ENABLED", "0") == "1",
        prf_top_k=int(os.environ.get("EPIMNEME_PRF_TOP_K", "5")),
        prf_n_terms=int(os.environ.get("EPIMNEME_PRF_N_TERMS", "8")),
        prf_fts_weight=float(os.environ.get("EPIMNEME_PRF_FTS_WEIGHT", "0.3")),
        # Phase D: gap-aware tiebreaker
        tiebreak_enabled=os.environ.get("EPIMNEME_TIEBREAK_ENABLED", "1") == "1",
        tiebreak_eps=float(os.environ.get("EPIMNEME_TIEBREAK_EPS", "0.005")),
        # Phase E: MMR
        mmr_enabled=os.environ.get("EPIMNEME_MMR_ENABLED", "1") == "1",
        mmr_lambda=float(os.environ.get("EPIMNEME_MMR_LAMBDA", "0.7")),
        mmr_session_cap=int(os.environ.get("EPIMNEME_MMR_SESSION_CAP", "2")),
        # Phase F: temporal hard-filter
        temporal_hard_filter_enabled=os.environ.get("EPIMNEME_TEMPORAL_HARD_FILTER", "0") == "1",
        temporal_hard_filter_sigma=float(os.environ.get("EPIMNEME_TEMPORAL_HARD_FILTER_SIGMA", "3.5")),
    )

    # Auto-detect sentence-transformers availability
    try:
        import sentence_transformers  # noqa: F401
        config.embeddings_enabled = True
    except ImportError:
        config.embeddings_enabled = False

    return config
