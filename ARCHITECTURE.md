# Architecture

Engram is a persistent memory service for AI coding agents. It exposes three surfaces:

1. **MCP over SSE** at `/sse` + `/messages` — for MCP-compatible clients (VS Code, Cursor, Claude Desktop, n8n).
2. **REST API** at `/api/*` — for scripts, automation, and non-MCP clients.
3. **Web dashboard** at `/` — for humans, intended to sit behind SSO/OAuth at your reverse proxy.

Everything runs as a single FastAPI process against a single PostgreSQL database with the `pgvector` extension.

---

## Request Lifecycle

```
client ──► (reverse proxy / OAuth) ──► FastAPI ──► AuthMiddleware
                                                      │
                                         ┌────────────┴───────────┐
                                         ▼                        ▼
                                    MemoryManager           FastMCP (SSE)
                                         │                        │
                                         └────────┬───────────────┘
                                                  ▼
                                        PostgreSQL + pgvector
```

### Authentication

Two paths, resolved by `auth.get_auth`:

1. **Bearer token** (`Authorization: Bearer engram_…`) — validated against the `api_keys` table in constant time. Keys are project-scoped (`agent`) or global (`admin`).
2. **OAuth passthrough** — the reverse proxy sets `X-Forwarded-User` after its own auth. Trusted as admin.

`ENGRAM_DEMO_MODE=1` short-circuits both with an unauthenticated admin identity for local development.

MCP requests reuse the same resolver via `get_mcp_auth(ctx)`, which reads the `Authorization` header from the SSE handshake.

### Project scoping

Every tenant-facing operation calls `AuthContext.enforce_project_access(project_name)`. Agents can "claim" a project name that doesn't yet exist — this makes self-service onboarding possible without an admin interventions step.

---

## Storage Layer (`stores/postgresql.py`)

A single `PostgresStore` class wraps a `psycopg` async connection pool and owns all SQL. Tables:

| Table | Purpose |
|---|---|
| `projects` | Tenant namespaces. `persistent_memories` flag exempts all memories from decay/GC. |
| `sessions` | Agent working sessions — task, summary, handoff notes. |
| `memories` | The core table. 20+ columns: kind, content, subject, `embedding vector(384)`, `simhash bigint`, decay state, version chain, `pinned`, `obsolete`, tags (JSONB), `content_tsv` (tsvector). |
| `memory_access` | Access log used by the decay model. |
| `entities` | Graph nodes — file, module, concept, tool, person, library, config, command. |
| `relationships` | Directed graph edges. |
| `memory_entities` | Link table associating memories with the entities they mention. |
| `api_keys` | Hashed keys only. Prefix is visible for identification. |
| `schema_migrations` | Applied migration versions. |

### Indexes

- `memories.embedding` — HNSW, `vector_cosine_ops`, `m=16, ef_construction=64`.
- `memories.content_tsv` — GiST for full-text search.
- `memories.content` — trigram GIN for fuzzy matching.
- Multicolumn covering indexes on `(project_id, kind)`, `(from_entity)`, `(to_entity)`.

### Migrations

`migrations/runner.py` applies numbered Python migration modules in order and records each in `schema_migrations`. Current migrations:

1. Entity tracking bootstrap (pseudo-entity cleanup)
2. Decay fields, versioning, simhash
3. Pinned memories
4. Per-project persistent flag

---

## MemoryManager (`manager.py`)

The unified API consumed by both the REST handlers and MCP tools.

### `remember()` pipeline

1. **SimHash dedup** — O(1) Hamming distance against recent memories in the same project. Configurable threshold (default 3 bits).
2. **Embedding** — generated in a thread-pool (non-blocking async).
3. **Semantic dedup** — cosine distance against existing embeddings above `semantic_dedup_threshold` (default 0.92).
4. **Conflict surfacing** — find fact/decision memories with similarity ≥ 0.80. Returned to the caller so the agent can explicitly supersede or reconcile.
5. **Store** — with version-chain pointers if `supersedes` was supplied.

### `recall()`

Runs **two retrieval passes in parallel**, then fuses:

- **Semantic** — pgvector `<=>` cosine search, HNSW-accelerated.
- **Keyword** — `ts_rank` against `content_tsv` plus trigram similarity.

Ranks are fused via **Reciprocal Rank Fusion** with configurable weights (`rrf_vector_weight`, `rrf_keyword_weight`). Decay-based score boost (range `[0.3, 1.2]`) is applied last.

Top-5 results have their decay state updated on access (`update_decay_on_access`).

### Knowledge graph

`entity_explore()` runs a recursive CTE over `relationships`, configurable depth and direction (`incoming`/`outgoing`/`both`). Returns entities + the memories linked to them.

---

## Decay Model (`decay.py`)

FSRS-inspired power-law retrievability:

- `R = exp(-t / S)` where `S = base_stability * (1 + storage_strength)`.
- `update_on_access` boosts `S` with diminishing returns: `S_new = S + growth_factor / (1 + 0.5 * S)`.
- At query time, `decay_score_boost` returns a ranking multiplier in `[0.3, 1.2]`.

Pinned memories, memories in persistent projects, and memories of kind `decision` or `procedure` are exempt from GC.

---

## Reflection (`reflection.py`, `scheduler.py`)

An async background job runs every `ENGRAM_REFLECTION_INTERVAL_HOURS` (default 24). Three phases:

1. **GC** — mark obsolete any memory with retrievability < `ENGRAM_REFLECTION_GC_THRESHOLD` (default 0.05), older than `ENGRAM_REFLECTION_GC_MIN_AGE_DAYS` (default 7), and not in an exempt class.
2. **Consolidation** — cluster high-similarity memories (threshold 0.88), merge into a single summary memory, mark originals obsolete with a pointer.
3. **Conflict resolution** — find fact/decision pairs with similarity ≥ 0.85 and age gap ≥ 7 days; obsolete the older one.

Limits (`reflection_max_consolidations` = 10) prevent churn. The scheduler exposes `/api/admin/reflection/run` for manual triggering.

---

## Activity & Logging

- `activity.py` — in-memory ring buffer (~2000 events). Event types: `WRITE`, `RECALL`, `SESSION`, `ENTITY`, `REFLECT`, `DEDUP`, `FORGET`, `CONFLICT`.
- `textlog.py` — best-effort persistent log tailed by the dashboard.
- `logging.py` — optional JSON log formatter (`LOG_FORMAT=json`).

---

## Backup (`backup.py`)

JSON archive format v2 including all tables except `api_keys` and `schema_migrations`. Embeddings serialize as JSON arrays. Backward-compatible v1→v2 upgrade. Restore supports `merge` (upsert) or `clean` (truncate first). Auto-rotation keeps `N` recent plus any younger than `X` days.

---

## Rate Limiting (`ratelimit.py`)

Starlette middleware, per-IP token bucket.

- Defaults: burst 30, refill 120/min.
- Exempt paths: `/health`, `/sse`, `/messages`.
- Respects `X-Forwarded-For` when behind a proxy.
- Returns `429` with `Retry-After`.

---

## Configuration

All via environment variables. Defaults live in [`src/engram/core/config.py`](src/engram/core/config.py).

### Required

| Variable | Notes |
|---|---|
| `ENGRAM_PG_PASSWORD` | Must not be literal `engram`. Server refuses to start otherwise unless `ENGRAM_DEMO_MODE=1`. |

### PostgreSQL

`ENGRAM_PG_HOST`, `ENGRAM_PG_PORT`, `ENGRAM_PG_USER`, `ENGRAM_PG_DATABASE`, `ENGRAM_PG_POOL_TIMEOUT`.

### Embeddings & retrieval

`ENGRAM_EMBEDDING_MODEL`, `ENGRAM_EMBEDDING_DIM`, `ENGRAM_RRF_VECTOR_WEIGHT`, `ENGRAM_RRF_KEYWORD_WEIGHT`, `ENGRAM_RRF_OVERFETCH`.

### Decay

`ENGRAM_DECAY_STABILITY`, `ENGRAM_DECAY_GROWTH`.

### Deduplication

`ENGRAM_DEDUP_ENABLED`, `ENGRAM_DEDUP_THRESHOLD`, `ENGRAM_SEMANTIC_DEDUP_ENABLED`, `ENGRAM_SEMANTIC_DEDUP_THRESHOLD`.

### Reflection

`ENGRAM_REFLECTION_ENABLED`, `ENGRAM_REFLECTION_INTERVAL_HOURS`, `ENGRAM_REFLECTION_GC_THRESHOLD`, `ENGRAM_REFLECTION_GC_MIN_AGE_DAYS`, `ENGRAM_REFLECTION_CONSOLIDATION_SIM`, `ENGRAM_REFLECTION_MIN_CLUSTER`, `ENGRAM_REFLECTION_MAX_CONSOLIDATIONS`, `ENGRAM_REFLECTION_CONFLICT_SIM`, `ENGRAM_REFLECTION_CONFLICT_AGE_GAP`.

### Backup

`ENGRAM_BACKUP_DIR`, `ENGRAM_BACKUP_KEEP_LAST`, `ENGRAM_BACKUP_KEEP_DAYS`.

### Network / auth

`ENGRAM_ALLOWED_HOSTS`, `ENGRAM_CORS_ORIGINS`, `ENGRAM_DEMO_MODE`, `ENGRAM_RATE_LIMIT_RPM`, `ENGRAM_RATE_LIMIT_BURST`.

### Logging

`LOG_LEVEL`, `LOG_FORMAT`.

### Bulk import (path traversal guard)

`ENGRAM_IMPORT_ALLOWED_DIRS` — comma-separated list of base directories under which `/api/bulk/import` is allowed to read. Default `/app`.

---

## Known Limitations

- Embeddings are currently synchronous per-memory at write time (batched via a thread-pool executor). For very high-throughput workloads, consider pre-batching client-side.
- HNSW index quality depends on insert order; a periodic `REINDEX` can help after heavy ingestion. See `engram-manage re-embed` for the supported rebuild path.
- The activity ring buffer is per-process; in a multi-replica deployment, consume the text log instead.
