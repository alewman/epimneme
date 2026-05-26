"""Engram web dashboard — self-contained HTML SPA.

Loaded from templates/dashboard.html at import time.
Served at / behind Traefik OAuth.  Uses vanilla JS + fetch() to talk to /api/*.
"""

from pathlib import Path

_TEMPLATE_DIR = Path(__file__).parent / "templates"

DASHBOARD_HTML: str = (_TEMPLATE_DIR / "dashboard.html").read_text(encoding="utf-8")
