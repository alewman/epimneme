# Epimneme

> *Pronounced **ep-im-NEE-mee**. From Greek: ἐπί (epi-, "upon") + μνήμη (mnēmē, "memory") — meta-memory, a layer that sits upon memory itself.*

**Persistent memory for AI coding agents.** PostgreSQL + pgvector backend, accessed via MCP or REST. Stop re-explaining your codebase every session.

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](pyproject.toml)
[![CI](https://github.com/alewman/epimneme/actions/workflows/ci.yml/badge.svg)](https://github.com/alewman/epimneme/actions)

---

## Why Epimneme?

Every new chat with your coding agent starts from zero. You re-paste the same design decisions, re-explain the same gotchas, re-answer the same questions. Epimneme gives agents a long-term memory: facts, decisions, procedures, and a knowledge graph — all searchable, versioned, and deduplicated.

Agents connect via **MCP** (VS Code, Cursor, Claude Desktop, etc.) or through a plain **REST API**. Both use Bearer-token auth scoped per-project for multi-tenant safety.

## Benchmarks

On **LongMemEval-S** (500 questions, 6 question types) with **no LLM reranking** and **no benchmark-specific tuning**:

| Metric | Epimneme (no LLM) |
|--------|------------------|
| Recall @ 1   | 86.0% |
| Recall @ 5   | 96.8% |
| Recall @ 10  | **98.2%** |
| Recall @ 50  | **100%** |
| NDCG @ 10    | 0.894 |
| LLM cost     | **$0 / query** |

Retrieval alone (the numbers above) is only half the story — see [`assembly.py`](src/epimneme/assembly.py) for a deterministic, still-$0 context-assembly stage (`assemble=true` on search / the MCP `recall` tool) that precomputes date arithmetic, prunes superseded facts, and groups by session before handing context to a reader model.

**End to end**, with a local reader (`qwen3.8:27b` via Ollama) answering from assembled context, judged over the same 500 questions: **0.784**.

Retrieval is close to saturated — R@10 is 98.2% and turn-level evidence completeness 0.871 — so most remaining error is reader-side, not retrieval-side. Two prompt-level fixes found by reading generated answers rather than aggregate scores were worth +5.2pp end to end, against nothing measurable from an extensive retrieval and embedder programme. If you are building on top of `recall()`, the pitfalls are written up in the `epimneme://recipes/answering-from-recall` MCP resource ([`skills.py`](src/epimneme/skills.py)) — briefly: never place the final answer last under a token cap, and never offer a blanket "say Unknown if not found" for answers that must be computed.

### Latency

Measured on a 277-document project, `limit=10` (the production shape), four uvicorn workers on CPU, warm:

| concurrent clients | median | p95 | slowest | throughput |
|---|---|---|---|---|
| 1 | 79 ms | 94 ms | 105 ms | 12.5 req/s |
| 4 | 82 ms | 94 ms | 97 ms | 45.6 req/s |
| 8 | 111 ms | 135 ms | 151 ms | 60.9 req/s |
| 16 | 149 ms | 206 ms | 233 ms | 78.1 req/s |

No LLM is called on the query path, so there is no token latency and no per-query cost. Embedding the
query is ~3 ms of that; the rest is pgvector search, six-channel RRF fusion and the post-fusion rerank
stack. At `limit=50` (the depth the benchmarks score at, not a typical production call) the median is
~179 ms.

Two caveats. The **first** query after a restart loads the embedding model — about 2.5 s, once per
worker; everything above is warm. And latency grows with corpus size, so re-measure if a single
project reaches the thousands of memories.

See [benchmarks/BENCHMARK_RESULTS.md](benchmarks/BENCHMARK_RESULTS.md) for the full write-up, including LoCoMo numbers and a per-category breakdown against [MemPalace](https://github.com/Chessnl/mempalace).

### MCP transports

Two are mounted, both requiring `Authorization: Bearer <api-key>`:

| endpoint | transport | workers |
|---|---|---|
| **`/mcp`** (or `/mcp/`) | Streamable HTTP, **stateless** | any number |
| `/sse` + `/messages` | SSE (legacy) | **1 only** |

The SSE transport keeps its session map in process, so a `POST /messages` has to
reach the same worker that holds the stream. Measured on four workers, only 2 of
6 posts landed. **Prefer `/mcp`;** `/sse` is kept for existing clients and is the
reason the server currently runs `--workers 1`.

If you put a reverse proxy in front, remember to route `/mcp` to the service — it is easy to add the transport and leave it unreachable from outside.

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
