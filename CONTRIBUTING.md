# Contributing to Engram

Thanks for wanting to make Engram better. This document covers the basics; if anything is unclear, open a discussion or draft PR and we'll iterate.

## Quick setup

```bash
git clone https://github.com/alewman/engram.git
cd engram
pip install -e '.[dev]'
```

The unit test suite runs entirely against mocks — no PostgreSQL needed:

```bash
make test          # fast, mocked
make test-cov      # with coverage
make lint          # ruff
```

Integration tests (marked `@pytest.mark.integration`) need a live PostgreSQL with pgvector. The CI pipeline provides one automatically; locally:

```bash
cp .env.example .env   # edit ENGRAM_PG_PASSWORD
docker compose up -d engram-db
pytest -m integration
```

## Pull requests

- **One logical change per PR.** If you find yourself writing "and also…" in the description, split it.
- **Include or update tests.** For new behaviour, a unit test in `tests/` is usually enough; add an integration test only if the behaviour depends on real SQL/pgvector semantics.
- **Keep the public API stable** unless the change is explicitly a breaking one. Breaking changes need a note in `CHANGELOG.md` under an `## Unreleased` section.
- **Run `make lint test` before pushing.** CI will do the same.
- **Commit messages**: short imperative subject, optional body. Conventional-Commits prefixes (`feat:`, `fix:`, `perf:`, `docs:`, …) are used in history but not required for PRs.

## Coding style

- Python 3.11+. Type hints on everything new. `from __future__ import annotations` at the top of new modules.
- `async` all the way down for anything touching the DB or HTTP.
- Prefer parameterized SQL. Never f-string user-supplied values into queries; composing whitelisted column/table names is fine.
- Log with `logger = logging.getLogger(__name__)`. No `print()` in library code.

## Database changes

Schema changes ship as numbered migrations in `src/engram/migrations/`:

1. Create `NNN_short_name.py` with an `async def apply(pool)` function.
2. Make it idempotent — `IF NOT EXISTS`, `ADD COLUMN IF NOT EXISTS`, etc.
3. Add a test in `tests/test_migrations.py`.

The runner applies pending migrations on server startup and records them in `schema_migrations`.

## Security

If you believe you've found a security vulnerability, **do not open a public issue**. Follow the process in [SECURITY.md](SECURITY.md).

## Code of Conduct

Be kind. Assume good faith. If someone else isn't, flag it to the maintainers rather than escalating publicly.

## License

By contributing, you agree that your contributions will be licensed under the [Apache License 2.0](LICENSE).
