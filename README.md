# Epimneme

> *Pronounced **ep-im-NEE-mee**. From Greek: ἐπί (epi-, "upon") + μνήμη (mnēmē, "memory") — meta-memory, a layer that sits upon memory itself.*

**Persistent memory for AI coding agents.** PostgreSQL + pgvector backend, accessed via MCP or REST. Stop re-explaining your codebase every session.

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](pyproject.toml)
[![CI](https://github.com/alewman/epimneme/actions/workflows/ci.yml/badge.svg)](https://github.com/alewman/epimneme/actions)

---

## Why Epimneme?

Every new chat with your coding agent starts from zero. You re-paste the same design decisions, re-explain the same gotchas, re-answer the same questions. Epimneme gives agents a long-term memory: facts, decisions, procedures, and a knowledge graph — all searchable, versioned, and deduplicated.

Agents connect via **MCP** (VS Code, Cursor, Claude Desktop, etc.) or through a plain **REST API**. Both use Bearer-token auth scoped per-project for multi-tenant safety.

## Benchmarks

On **LongMemEval** (500 questions, 6 question types) with **no LLM reranking** and **no benchmark-specific tuning**:

| Metric | Epimneme (no LLM) |
|--------|------------------|
| Recall @ 1   | 79.0% |
| Recall @ 5   | 96.0% |
| Recall @ 10  | **98.8%** |
| Recall @ 50  | **100%** |
| NDCG @ 10    | 0.887 |
| LLM cost     | **$0 / query** |

See [benchmarks/BENCHMARK_RESULTS.md](benchmarks/BENCHMARK_RESULTS.md) for the full write-up, including LoCoMo numbers and a per-category breakdown against [MemPalace](https://github.com/Chessnl/mempalace).

## Quick Start

```bash
# 1. Clone
git clone https://github.com/alewman/epimneme.git
cd epimneme

# 2. Configure
cp .env.example .env
# Edit .env — at minimum set EPIMNEME_PG_PASSWORD to a strong value

# 3. Bring up Postgres + Epimneme
cp docker-compose.example.yml docker-compose.yml   # or edit in place
docker compose up -d --build

# 4. Wait for health, then create an admin API key
curl http://localhost:8000/health
docker exec epimneme python -m epimneme.manage create-key \
  --name admin --role admin

# Save the key it prints — it will not be shown again.

# 5. Create your first project
docker exec epimneme python -m epimneme.manage create-project \
  --name my-project --description "My awesome project"
```

Epimneme now listens on `http://localhost:8000`. See [Connecting an Agent](#connecting-an-agent) below.

## Connecting an Agent

### VS Code / Cursor / Claude Desktop (MCP over SSE)

```json
{
  "mcpServers": {
    "epimneme": {
      "url": "http://localhost:8000/sse",
      "headers": {
        "Authorization": "Bearer epimneme_YOUR_KEY_HERE"
      }
    }
  }
}
```

Once connected, the agent sees these tools:

| Tool | Purpose |
|------|---------|
| `session_start` | Begin a session, receive previous context (summary, decisions, issues, entities) |
| `session_end` | Close session with summary + handoff notes for the next agent |
| `remember` | Store a memory (fact, decision, procedure, pattern, preference, observation, issue) |
| `recall` | Search memories by semantic similarity + keyword. `deep=true` for graph traversal |
| `project_status` | Get a project overview. Auto-registers new projects |
| `entity_track` | Add a node to the knowledge graph (file, module, concept, tool, person, library, config, command) |
| `entity_relate` | Add an edge (`depends_on`, `part_of`, `uses`, `implements`, …) |
| `entity_explore` | Traverse the graph from an entity (configurable depth + direction) |

### curl

```bash
curl -H "Authorization: Bearer epimneme_YOUR_KEY" \
  "http://localhost:8000/api/memories/search?query=database+config&project=my-project"
```

Full REST reference: [docs/API.md](docs/API.md) *(or run the server and visit `/docs` for the live OpenAPI UI).*

## What Gets Stored

Seven memory kinds. Pick the right one and `recall` works much better later.

| Kind | Use for |
|------|---------|
| `fact` | Discrete knowledge — "CLI entry point is `main.py`" |
| `decision` | Why something was done — "Chose PostgreSQL because pgvector + recursive CTEs cover both search and graph needs" |
| `procedure` | Step-by-step instructions |
| `pattern` | Recurring conventions — "All tests use pytest fixtures from `conftest.py`" |
| `preference` | Working style — "Prefers small PRs with single-purpose commits" |
| `observation` | General notes |
| `issue` | Known bugs, limitations, tech debt |

Plus a knowledge graph of **entities** (files, modules, concepts) and **relationships** (`depends_on`, `uses`, `part_of`, `implements`, …) that `recall` can traverse when `deep=true`.

## Features

- **Hybrid retrieval** — pgvector HNSW semantic search fused with PostgreSQL full-text + trigram keyword search via Reciprocal Rank Fusion.
- **FSRS-inspired decay** — memories fade without access, stabilize with repetition. The retrieval ranker boosts well-used memories.
- **Dual dedup** — SimHash (O(1) Hamming) catches minor rewordings; semantic cosine catches reworded but equivalent facts.
- **Versioning** — `update_memory` creates a new version instead of overwriting. Full history preserved.
- **Conflict surfacing** — when you store a new fact/decision similar to an old one, the response flags the potential conflict so the agent can resolve it.
- **Periodic reflection** — background job garbage-collects low-retrievability memories, consolidates clusters, and resolves detected conflicts. Pinned, persistent-project, decision, and procedure memories are exempt.
- **Multi-tenant** — projects are namespaces; API keys are project-scoped (or global `admin`).
- **Dual auth** — Bearer tokens for agents/programmatic access, OAuth passthrough (`X-Forwarded-User`) for browsers behind a reverse proxy.
- **Rate limited** — per-IP token bucket, honours `X-Forwarded-For`.
- **Activity stream** — in-memory ring buffer + text log for auditing.
- **Dashboard** — self-contained web UI for inspecting memories, entities, activity, and backups.

## Architecture

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  VS Code /   │     │   n8n /      │     │  Browser     │
│  Cursor      │     │   Scripts    │     │  (OAuth)     │
│  (MCP/SSE)   │     │  (REST API)  │     │              │
└──────┬───────┘     └──────┬───────┘     └──────┬───────┘
       │  Bearer            │  Bearer            │  Reverse proxy OAuth
       └────────────┬───────┴────────────────────┘
                    │
          ┌─────────▼─────────┐
          │   FastAPI + MCP   │  ← /api/*, /sse, /messages, /, /health
          │   (port 8000)     │
          └─────────┬─────────┘
                    │
          ┌─────────▼─────────┐
          │   MemoryManager   │  ← sessions, memories, entities, decay, dedup
          └─────────┬─────────┘
                    │
          ┌─────────▼─────────┐
          │  PostgreSQL 16    │
          │  + pgvector       │
          │                   │
          │  vectors +        │
          │  full-text +      │
          │  graph (rCTE)     │
          └───────────────────┘
```

Full details: [ARCHITECTURE.md](ARCHITECTURE.md).

## Configuration

All settings are environment variables (`EPIMNEME_*`). See [.env.example](.env.example) and the **Configuration** section of [ARCHITECTURE.md](ARCHITECTURE.md#configuration) for the full list.

Minimum required:

| Variable | Notes |
|---|---|
| `EPIMNEME_PG_PASSWORD` | Must not be the literal string `epimneme`. The server refuses to start otherwise. |

## Deploying Behind a Reverse Proxy

The bundled `docker-compose.example.yml` has commented-out Traefik labels. Uncomment and adjust for your proxy. Recommended setup:

- Browser requests (`/*`) go through your OAuth/SSO middleware — users authenticate as admin.
- API / SSE requests (`/api/*`, `/sse`, `/messages`, `/health`) skip OAuth and use Bearer tokens instead. **Do not apply gzip compression to `/sse`** — it breaks SSE streaming.

## Development

```bash
# Editable install with dev extras
pip install -e '.[dev]'

# Run tests
make test                # local (uses mocks, no DB needed)
make test-cov            # with coverage report

# Lint
make lint                # ruff check
make lint-fix            # ruff check --fix
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for full contributor guidelines.

## Security

If you discover a security vulnerability, please open a GitHub **Security Advisory** rather than a public issue. See [SECURITY.md](SECURITY.md).

## License

Apache License 2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE).

## Credits

- **Author & maintainer**: [Alewman](https://github.com/alewman)
- **Design & implementation assistance**: Anthropic's **Claude** (via GitHub Copilot Chat and Claude Code). Large portions of the code, tests, migrations, reranking, and documentation were authored in close collaboration with Claude across many sessions — many of which Epimneme itself made possible by persisting the context.
- Built on FastAPI, PostgreSQL + pgvector, sentence-transformers, FlashRank, and the Model Context Protocol.
- Benchmarked against [LongMemEval](https://github.com/xiaowu0162/LongMemEval) and [LoCoMo](https://github.com/snap-research/locomo); compared to [MemPalace](https://github.com/Chessnl/mempalace).
