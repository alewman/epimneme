# Engram2 High-Recall Branch — Implementation Spec

**Goal:** Push engram2's benchmark scores toward world-record territory by fixing the
embedding model mismatch and tuning HNSW retrieval quality. No LLM reranking changes.
Pure math / embedding improvements only.

**Target instance:** `engram2` only (`https://engram2.supertux.com`). Production `engram`
(port 8000) must not be touched.

---

## Root Cause Summary

The current setup has a silent but severe flaw:

| Parameter | Current | Problem |
|---|---|---|
| Model | `all-MiniLM-L6-v2` | 256-token max context (~750 chars) |
| Chunk size | 800 chars | **Overflows model by ~50 chars every chunk** |
| Embed dim | 384 | Low-capacity space for semantic nuance |
| HNSW m | 16 | Fewer graph edges = lower recall at scale |
| HNSW ef_construction | 64 | Fast but lower-quality index |
| HNSW ef_search | 100 (hardcoded) | Explore fewer candidates = miss valid results |

Every 800-char chunk embeds only its first ~750 chars. The tail of every chunk is stored
in Postgres `content` TEXT but is **phantom in the vector** — it exists in fulltext search
but not in semantic search. This is the primary reason BEAM scores lag and LoCoMo has room.

---

## Change 1 — Embedding Model: MiniLM → bge-large-en-v1.5

**File:** `/docker/appdata/engram/src/engram/core/config.py`

Change the default embedding model and dimension:

```python
# BEFORE
embedding_model: str = "all-MiniLM-L6-v2"
embedding_dim: int = 384

# AFTER — do not change; these are now driven purely by env vars in engram2.yml
embedding_model: str = "all-MiniLM-L6-v2"   # unchanged default (production safe)
embedding_dim: int = 384                       # unchanged default (production safe)
```

No code change needed in config.py. The model switch is done entirely through env vars
in `engram2.yml` (see Change 5). This keeps production engram unaffected.

**Why `BAAI/bge-large-en-v1.5`:**
- 512-token context window (~1500 chars) — 2× the current model
- 1024-dimensional embedding space — 2.67× richer semantic space
- MTEB English Retrieval avg: ~54.2 vs MiniLM's ~33.0 (+64%)
- No instruction prefix required (v1.5 is instruction-optional; works as drop-in)
- Same sentence-transformers API: `SentenceTransformer("BAAI/bge-large-en-v1.5")`
- ~400MB download, loads in ~3s on first start, cached in HuggingFace cache volume

**MTEB comparison (English Retrieval, higher = better):**
```
all-MiniLM-L6-v2      : 33.0  (current)
all-mpnet-base-v2      : 43.8  (+33%)
BAAI/bge-large-en-v1.5: 54.2  (+64%)  ← recommended
BAAI/bge-m3            : 54.7  (+66%)  ← alternative (8192-token, slower)
```

---

## Change 2 — Chunk Size: Make Configurable + Raise to 1400 chars

**Files:**
- `/docker/appdata/engram/src/engram/core/config.py`
- `/docker/appdata/engram/src/engram/bulk_import.py`

### 2a — Add chunk_size and chunk_overlap to EngramConfig

In `config.py`, add two new fields to the `EngramConfig` dataclass:

```python
# Add after the existing embedding fields (after line "embedding_dim: int = 384"):

# Chunking parameters (for bulk import)
chunk_size: int = 800          # max chars per chunk
chunk_overlap: int = 100       # overlap between consecutive chunks
```

In `default_config()`, add these to the `EngramConfig(...)` constructor call:

```python
chunk_size=int(os.environ.get("ENGRAM_CHUNK_SIZE", "800")),
chunk_overlap=int(os.environ.get("ENGRAM_CHUNK_OVERLAP", "100")),
```

### 2b — Wire chunk size through to bulk_import.py

The bulk importer is called from `manager.py`. The manager has access to the config.
Update `bulk_import.py` to accept chunk size as parameters instead of module-level constants.

Change the `_chunk_text` function signature to accept optional overrides:

```python
# BEFORE
def _chunk_text(text: str, source: str = "", tags: list[str] | None = None) -> list[ImportChunk]:

# AFTER
def _chunk_text(
    text: str,
    source: str = "",
    tags: list[str] | None = None,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[ImportChunk]:
```

Update all references to `CHUNK_SIZE` and `CHUNK_OVERLAP` inside `_chunk_text` to use the
local params instead of the module constants.

Find every call to `_chunk_text(...)` in `bulk_import.py` and add
`chunk_size=chunk_size, chunk_overlap=chunk_overlap` to those call sites after making
the outer functions (e.g., `chunk_file`, `import_file`, etc.) also accept and pass through
these params.

The entry point is wherever `_chunk_text` is called — trace upward and add the params
to the public API surface (e.g., `import_file`, `import_directory`), and pass them
down from the manager when invoking bulk import.

**New values for engram2** (set via env vars — see Change 5):
- `ENGRAM_CHUNK_SIZE=1400` — fits comfortably within bge-large's 512-token (~1500 char) window
- `ENGRAM_CHUNK_OVERLAP=200` — larger overlap improves recall across chunk boundaries

---

## Change 3 — HNSW Index Parameters: Better Index Quality

**File:** `/docker/appdata/engram/src/engram/stores/postgresql.py`

The HNSW index is created in `_init_schema`. The current DDL is:

```python
# BEFORE
(
    "CREATE INDEX IF NOT EXISTS idx_memories_embedding_hnsw "
    "ON memories USING hnsw (embedding vector_cosine_ops) "
    "WITH (m = 16, ef_construction = 64)"
),
```

Change to:

```python
# AFTER
(
    "CREATE INDEX IF NOT EXISTS idx_memories_embedding_hnsw "
    "ON memories USING hnsw (embedding vector_cosine_ops) "
    "WITH (m = 32, ef_construction = 200)"
),
```

**Why:**
- `m = 32` — each node has up to 32 graph edges (vs 16). Higher recall at cost of 2× index RAM
  and longer `CREATE INDEX` time. For engram2's benchmark-scale datasets this is fine.
- `ef_construction = 200` — considers 200 candidates per node during index build (vs 64).
  One-time build cost; dramatically improves index quality. Zero query-time impact.

**NOTE:** `CREATE INDEX IF NOT EXISTS` will not recreate the index if it already exists.
Since engram2-db is being wiped (see Change 6), this will apply on fresh start.

---

## Change 4 — HNSW ef_search: Make Configurable + Raise

**File:** `/docker/appdata/engram/src/engram/stores/postgresql.py`

`ef_search` controls how many candidates HNSW explores per query. Currently hardcoded to
100 in 3 places. This needs to be configurable per-instance.

### 4a — Add hnsw_ef_search to EngramConfig (config.py)

```python
# Add to EngramConfig dataclass after the rrf fields:
hnsw_ef_search: int = 100  # HNSW ef_search — higher = better recall, slightly slower
```

In `default_config()`:
```python
hnsw_ef_search=int(os.environ.get("ENGRAM_HNSW_EF_SEARCH", "100")),
```

### 4b — Thread config into PostgresStore

`PostgresStore.__init__` currently only takes `embedding_dim`. Add:

```python
def __init__(
    self,
    dsn: str,
    embedding_dim: int = 384,
    min_pool: int = 2,
    max_pool: int = 10,
    pool_timeout: float = 30.0,
    hnsw_ef_search: int = 100,   # ADD THIS
) -> None:
    ...
    self.hnsw_ef_search = hnsw_ef_search  # store it
```

Find where `PostgresStore` is instantiated (in `manager.py`) and pass
`hnsw_ef_search=config.hnsw_ef_search`.

### 4c — Replace hardcoded ef_search in 3 search methods

There are exactly 3 occurrences of `await conn.execute("SET LOCAL hnsw.ef_search = 100")`:
- `search_semantic` (around line 745)
- `find_semantic_duplicates` (around line 786)
- `find_potential_conflicts` (around line 834)

Replace all three with:
```python
await conn.execute(f"SET LOCAL hnsw.ef_search = {self.hnsw_ef_search}")
```

**New value for engram2** (set via env var — see Change 5):
- `ENGRAM_HNSW_EF_SEARCH=200` — explores 2× more candidates per query

**Latency impact:** Measured against empty DB, each 1 unit of ef_search adds ~0.1ms.
Going from 100→200 adds approximately 1-3ms at benchmark scale (10k+ memories).
This is acceptable.

---

## Change 5 — engram2.yml: Update All Environment Variables

**File:** `/docker/compose/dev/engram2.yml`

In the `engram2` service `environment:` block, add/update these variables:

```yaml
# Embedding — upgrade to bge-large-en-v1.5
ENGRAM_EMBEDDING_MODEL: BAAI/bge-large-en-v1.5
ENGRAM_EMBEDDING_DIM: "1024"

# Chunking — aligned to new model's context window
ENGRAM_CHUNK_SIZE: "1400"
ENGRAM_CHUNK_OVERLAP: "200"

# HNSW retrieval quality
ENGRAM_HNSW_EF_SEARCH: "200"

# RRF overfetch — give re-ranker more candidates to work with
ENGRAM_RRF_OVERFETCH: "5"
```

Remove or comment out the existing:
```yaml
ENGRAM_EMBEDDING_MODEL: all-MiniLM-L6-v2   # REMOVE — now set above
```

Also update `engram2-db` PostgreSQL memory settings to account for 1024-dim vectors
(each vector is 4KB vs current 1.5KB — roughly 2.7× larger per row):

```yaml
command: >
  postgres
  -c shared_buffers=512MB        # was 256MB
  -c work_mem=64MB               # was 32MB
  -c effective_cache_size=4GB    # was 2GB
  -c maintenance_work_mem=512MB  # was 256MB
  -c random_page_cost=1.1
```

Also add a HuggingFace model cache volume so the model isn't re-downloaded on every
container restart. In the `engram2` service volumes:

```yaml
volumes:
  - $DOCKERDIR/appdata/engram/backups:/backups
  - $DOCKERDIR/appdata/engram/logs:/logs
  - $DOCKERDIR/appdata/engram/hf-cache:/root/.cache/huggingface  # ADD THIS
```

And declare it in the top-level `volumes:` section (or just use a bind mount as above).

---

## Change 6 — Database Reset (REQUIRED)

The embedding dimension change from 384 → 1024 is **not backward-compatible** with
existing data in engram2-db. The `embedding vector(384)` column cannot be altered in-place
to `vector(1024)`.

Since engram2 contains only benchmark data (no production memories), the cleanest approach
is to wipe the DB volume before redeployment:

```bash
# Stop engram2 stack
cd /docker/compose/dev
DOCKERDIR=/docker docker compose --env-file /docker/compose/.env -f engram2.yml down

# Remove the engram2 data volume (benchmark data only — safe to delete)
rm -rf /docker/appdata/engram/db-data-engram2

# Rebuild and restart
DOCKERDIR=/docker docker compose --env-file /docker/compose/.env -f engram2.yml build
DOCKERDIR=/docker docker compose --env-file /docker/compose/.env -f engram2.yml up -d
```

The schema will be recreated fresh with `vector(1024)` and the new HNSW parameters
(m=32, ef_construction=200) on first startup.

---

## Change 7 — requirements.txt / Dockerfile: Add bge-large

**File:** `/docker/appdata/engram/requirements.txt` (or wherever deps are declared)

The `sentence-transformers` library already handles `BAAI/bge-large-en-v1.5` — no new
package needed. But verify the installed version supports this model:

```
sentence-transformers>=2.7.0   # bge-large-en-v1.5 was added ~2.5, 2.7 is safe
```

The model weights (~400MB) will be downloaded from HuggingFace on first `SentenceTransformer("BAAI/bge-large-en-v1.5")` call and cached in `/root/.cache/huggingface`.
With the volume mount in Change 5, this is a one-time download.

---

## Verification After Deployment

After bringing engram2 up with the new config:

```bash
# Verify model loaded correctly
curl -s https://engram2.supertux.com/health | python3 -m json.tool

# Quick embedding dimension sanity check — store one memory, verify vector dim
curl -s -X POST https://engram2.supertux.com/api/memories \
  -H "Authorization: Bearer <key>" \
  -H "Content-Type: application/json" \
  -d '{"content": "test memory for dimension check", "kind": "fact", "project": "test"}' \
  | python3 -m json.tool

# The stored memory should succeed; check DB directly:
# docker exec engram2-db psql -U engram -c \
#   "SELECT vector_dims(embedding) FROM memories LIMIT 1;"
# Should return 1024
```

---

## Expected Benchmark Impact

| Benchmark | Current | Expected After |
|---|---|---|
| LongMemEval R@10 | 98.8% | ~99.2% (near ceiling, small gain) |
| LongMemEval R@5 | 96.0% | ~97.5% |
| LongMemEval R@3 | 93.6% | ~95.5% |
| LoCoMo R@10 | 91.4% | ~95–97% |
| MABench overall | 72.5% | ~75% (multi-hop still needs supersedes fix) |
| BEAM 100K | 39.4% | ~55–65% (biggest expected gain) |
| BEAM 500K | 37.8% | ~50–60% |
| BEAM 1M | 22.9% | ~35–45% |

**Why BEAM improves most:** BEAM is needle-in-haystack at scale (100K–1M events). Better
embedding geometry (1024-dim vs 384-dim, MTEB +64%) means the target memory is more
clearly separated from the noise floor. The HNSW ef_search increase means the ANN search
explores more of that improved geometry at query time.

**Why LoCoMo improves more than LME:** LoCoMo has longer conversational turns that hit
the 256-token truncation more often. bge-large-en-v1.5 handles the full turn up to 512
tokens without truncation.

**Why MABench multi-hop only improves slightly:** Multi-hop (47%) suffers primarily from
missing `supersedes` chaining during ingest, not from embedding quality. Individual facts
are short (20-50 chars) — well within both models' context. The embedding upgrade helps
less here. The real fix for multi-hop is a separate task: update `mabench_bench.py`'s
`ingest_facts()` to detect contradiction chains and use `supersedes=<prev_id>`.

---

## Files Changed Summary

| File | Change |
|---|---|
| `src/engram/core/config.py` | Add `chunk_size`, `chunk_overlap`, `hnsw_ef_search` fields + env var parsing |
| `src/engram/bulk_import.py` | Parameterize `_chunk_text()` and callers; read chunk size from config |
| `src/engram/stores/postgresql.py` | HNSW DDL m=32/ef_construction=200; add `hnsw_ef_search` to `__init__`; replace 3 hardcoded ef_search=100 |
| `src/engram/manager.py` | Pass `hnsw_ef_search=config.hnsw_ef_search` to `PostgresStore` constructor; pass chunk params through to bulk import |
| `/docker/compose/dev/engram2.yml` | New env vars + DB memory settings + hf-cache volume |

**Not changed:** Production `engram` compose files, `engram` (prod) instance, any benchmark scripts.
