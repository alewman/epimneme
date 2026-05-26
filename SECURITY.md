# Security Policy

## Reporting a vulnerability

Please **do not** open a public GitHub issue for security vulnerabilities.

Instead, open a private [GitHub Security Advisory](https://github.com/alewman/engram/security/advisories/new). Include:

- A description of the issue and its impact
- Steps to reproduce (ideally a minimal PoC)
- Any suggested remediation, if you have one

We will acknowledge receipt within a few days and coordinate a fix and disclosure timeline with you.

## Scope

In scope:

- Authentication and authorization bypass (Bearer token handling, project scoping, OAuth passthrough)
- SQL injection, command injection, path traversal, SSRF
- Secret exposure (logs, error responses, backup archives)
- Denial of service against the default configuration

Out of scope:

- Issues that require already-compromised administrator credentials
- Rate-limit bypass when `ENGRAM_DEMO_MODE=1` (demo mode is explicitly unauthenticated)
- Issues in third-party dependencies that have no impact on Engram's security posture

## Hardening recommendations for operators

- Set a strong, unique `ENGRAM_PG_PASSWORD`. Engram will refuse to start with the default `engram` value.
- Run behind a reverse proxy (Traefik, Caddy, nginx) with TLS terminated at the proxy.
- Put the dashboard (`/`) behind your own OAuth/SSO middleware. The API/SSE routes (`/api/*`, `/sse`, `/messages`, `/health`) should bypass OAuth and rely on Bearer tokens.
- Keep API keys scoped to single projects for agents. Reserve `admin` keys for CLI use.
- Rotate API keys periodically: revoke old keys with `engram-manage revoke-key` and issue new ones with `engram-manage create-key`.
- Back up the database regularly (`scripts/backup.sh`). Backups are produced with `pg_dump` and contain **all** tables — including the hashed `api_keys` table — so treat backup files as sensitive and store them encrypted at rest.
