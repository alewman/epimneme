"""Epimneme web dashboard — self-contained HTML SPA.

Loaded from templates/dashboard.html at import time.
Served at / behind Traefik OAuth.  Uses vanilla JS + fetch() to talk to /api/*.

The template carries two placeholders the route fills per request:
``__CSP_NONCE__`` on its inline <script>/<style> blocks, and ``__D3_SRI__`` on
the pinned d3 tag.
"""

from pathlib import Path

_TEMPLATE_DIR = Path(__file__).parent / "templates"

DASHBOARD_HTML: str = (_TEMPLATE_DIR / "dashboard.html").read_text(encoding="utf-8")

# Subresource integrity for the pinned d3 build. Recompute when bumping:
#   curl -sL https://cdn.jsdelivr.net/npm/d3@<ver>/dist/d3.min.js \
#     | openssl dgst -sha384 -binary | openssl base64 -A
D3_VERSION = "7.9.0"
D3_SRI = "sha384-CjloA8y00+1SDAUkjs099PVfnY2KmDC2BZnws9kh8D/lX1s46w6EPhpXdqMfjK6i"
