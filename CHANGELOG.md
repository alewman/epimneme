# Changelog

All notable changes to this project will be documented here. The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- `src/epimneme/assembly.py` — deterministic, $0/query context assembly for the reader: temporal scaffolding (precomputed date deltas, no reader-side date arithmetic), chronological presentation, supersession pruning (linked + SimHash-detected), adaptive K/char budgeting, session grouping, and parent-document (small-to-big) expansion. Opt-in via `assemble=true` on `GET /api/memories/search` and the MCP `recall` tool.
- `recall_all@k` and `evidence_completeness@k` metrics in the LME benchmark harness, alongside the existing `recall_any@k` — `recall_any` was masking incomplete multi-session evidence (turn-level `EvidenceCompleteness@10` measured at 0.527 on the current LME-S baseline, vs. `recall_any@10` reading 0.982).
- Temporal partition rerank (`fusion.temporal_partition`): for day-precision temporal queries, structurally promotes in-window candidates ahead of out-of-window ones rather than relying on an additive score boost.
- New config knobs: `EPIMNEME_ASSEMBLY_BUDGET_CHARS`, `EPIMNEME_ASSEMBLY_K_SINGLE`, `EPIMNEME_ASSEMBLY_K_COUNTING`, `EPIMNEME_ASSEMBLY_PARENT_EXPANSION`, `EPIMNEME_TEMPORAL_PARTITION_ENABLED`.
- MIT license, `NOTICE`, `CONTRIBUTING.md`, `SECURITY.md`, `ARCHITECTURE.md`.
- Startup guard: the server refuses to start if `EPIMNEME_PG_PASSWORD` is unset or equal to the default `epimneme`, unless `EPIMNEME_DEMO_MODE=1`.
- `.env.example` with documented minimum settings.
- OCI image labels on the Dockerfile.

### Fixed
- `fusion.extract_logical_date` (used by the temporal boost) only matched ISO-hyphenated dates; real LongMemEval haystacks use slash-delimited dates with a weekday/time suffix (`[Date: 2023/05/20 (Sat) 09:05]`), so the temporal boost was silently falling back to `created_at` (ingestion wall-clock time) on every benchmark memory — effectively a no-op on the exact data it's benchmarked against.
- `fusion.mmr_rerank` was O(n·k²) in the requested result limit (measured: 0.8s → 174s going from `limit=10` to `limit=200` on a ~2,500-memory project). Fixed to O(n·k) by maintaining a running max-similarity-to-selected per candidate instead of recomputing it from scratch every iteration. Output is unchanged (verified by differential test against the original algorithm).
- Two leftover hardcoded `engram.*` references from the Engram→Epimneme rename that broke a fresh container build entirely: the Dockerfile's `CMD` (`engram.server:app`) and `migrations/runner.py`'s `importlib.import_module(f"engram.migrations.{name}")`.
- `benchmarks/epimneme_client.py`'s `clear_project()` now enumerates a project's memories directly via `/api/memories/recent` instead of a relevance-scored `search(query="*")` loop — not a correctness bug in practice, but not a *guaranteed* complete enumeration either.
- `pyproject.toml`'s `mcp[cli]` dependency had no upper bound; `mcp` 2.x renamed `FastMCP`, breaking `skills.py`'s import on any fresh install. Pinned `<2`.

### Changed
- `benchmarks/*.py` defaults now point at `http://localhost:8000` instead of a hard-coded LAN address.
- README rewritten for a public audience; architectural deep-dive moved to `ARCHITECTURE.md`.
- CORS middleware now disables credentials when `allow_origins=["*"]` (matches browser spec).
- License switched from Apache 2.0 to MIT.

### Removed
- Internal-only `TECHNICAL_REVIEW.md` (content preserved in `ARCHITECTURE.md`).
- Private domain and hostnames from documentation.

### Investigated (no change)
- Phase 5 (two-stage session-then-chunk retrieval at LME-M scale): NO-GO. Best case tied flat chunk search (0.0pp R@1 gain, well short of the +3pp bar); narrower session cuts regressed R@10 recall_all by up to 8.4pp. See `benchmarks/lme_m_hierarchical_experiment.py` and its commit message for the full results table.

## [0.7.0] — 2026-04

Historical releases tracked via git history. Highlights:

- LongMemEval + LoCoMo benchmark harness, adaptive keyword weight for vague queries.
- Reciprocal-Rank Fusion hybrid search, preference boosting, entity-aware dedup.
- Persistent-project flag, pinning, versioning, SimHash dedup, FSRS-inspired decay.
- Reflection / compaction scheduler (GC, consolidation, conflict resolution).
- Dashboard (activity stream, graph viz, backup/restore), REST project claiming.
- Pagination, structured JSON logging, CI pipeline.
- API key cycling, hard-forget endpoint, 403 hints.
- Backup rotation, integration tests.
- Initial PostgreSQL + pgvector rewrite (replacing DuckDB + LanceDB + Kuzu).
