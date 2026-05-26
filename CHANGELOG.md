# Changelog

All notable changes to this project will be documented here. The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- Apache 2.0 license, `NOTICE`, `CONTRIBUTING.md`, `SECURITY.md`, `ARCHITECTURE.md`.
- Startup guard: the server refuses to start if `ENGRAM_PG_PASSWORD` is unset or equal to the default `engram`, unless `ENGRAM_DEMO_MODE=1`.
- `.env.example` with documented minimum settings.
- OCI image labels on the Dockerfile.

### Changed
- `benchmarks/*.py` defaults now point at `http://localhost:8000` instead of a hard-coded LAN address.
- README rewritten for a public audience; architectural deep-dive moved to `ARCHITECTURE.md`.
- CORS middleware now disables credentials when `allow_origins=["*"]` (matches browser spec).

### Removed
- Internal-only `TECHNICAL_REVIEW.md` (content preserved in `ARCHITECTURE.md`).
- Private domain and hostnames from documentation.

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
