"""Static guards for the dashboard template.

The dashboard builds its DOM with innerHTML from template strings. An entity
named ``<img onerror=…>`` executed in it (external review, 2026-09-25), and the
fix was a rule rather than a line: nothing from the server is interpolated
raw, and no handler is an inline ``onclick`` with data spliced into it.

These tests keep the rule without a browser. They are deliberately narrow —
they cannot prove the page is safe, only that the two patterns that were
exploited do not come back.
"""

from __future__ import annotations

import re

import pytest

from epimneme.dashboard import DASHBOARD_HTML

# Server-supplied fields a template might interpolate. A `${…}` whose expression
# mentions one of these must go through esc()/keep()/encodeURIComponent().
_SERVER_FIELDS = (
    "name", "content", "subject", "task", "summary", "filename", "message",
    "project_name", "project", "label", "key", "detail", "relation", "to",
    "kind", "role", "owner_id", "version", "tags", "log", "status",
)
_SAFE_WRAPPERS = ("esc(", "escapeHtml(", "keep(", "encodeURIComponent(", "badge(", "ago(")

# A `${ … }` with balanced-enough matching: no nested `${` inside.
_INTERP = re.compile(r"\$\{([^{}]*)\}")


def _script_body() -> str:
    m = re.search(r"<script nonce=\"__CSP_NONCE__\">(.*)</script>", DASHBOARD_HTML, re.S)
    assert m, "dashboard script block not found"
    return m.group(1)


class TestNoInlineHandlers:
    def test_no_on_attributes_in_markup(self):
        # `on*=` inside the HTML/template text is the shape CSP blocks and the
        # shape a quote in a name used to break out of.
        hits = [ln for ln in DASHBOARD_HTML.splitlines()
                if re.search(r"\son[a-z]+=\s*[\"'`]", ln)]
        assert hits == [], "inline event handlers present:\n" + "\n".join(h.strip()[:120] for h in hits)

    def test_no_window_globals_for_handlers(self):
        # Globals existed only so inline onclick strings could reach them.
        assert not re.search(r"^window\.\w+\s*=\s*(async )?function", _script_body(), re.M)

    def test_json_in_attribute_helper_is_gone(self):
        assert "jsonAttr" not in DASHBOARD_HTML


class TestEscaping:
    def test_server_fields_are_escaped(self):
        body = _script_body()
        offenders = []
        for ln_no, line in enumerate(body.splitlines(), 1):
            # Not HTML: text goes through textContent, Error(), confirm().
            if re.search(r"\btoast\(|throw new Error|\.textContent\s*=|\bconfirm\(", line):
                continue
            for m in _INTERP.finditer(line):
                expr = m.group(1)
                if any(expr.lstrip().startswith(w) for w in _SAFE_WRAPPERS):
                    continue
                if not re.search(r"\.(%s)\b" % "|".join(_SERVER_FIELDS), expr):
                    continue
                # Attribute-context helpers that are inherently safe
                if expr.strip() in ("t.icon", "t.color") or "KIND_COLORS" in expr:
                    continue
                # A ternary whose both branches are string literals yields a
                # constant whatever the field holds: `x==='admin' ? 'a' : 'b'`.
                if re.search(r"\?\s*'[^']*'\s*:\s*'[^']*'\s*$", expr):
                    continue
                # `x ? `<a …>${esc(x)}</a>` : '—'` — the outer expression is a
                # ternary whose branches are templates; inspect those instead.
                if "`" in expr:
                    continue
                offenders.append(f"{ln_no}: ${{{expr.strip()}}}")
        assert offenders == [], "unescaped server data interpolated:\n" + "\n".join(offenders)

    def test_escape_helper_covers_quotes(self):
        # Attribute values are double-quoted in the template, but a value that
        # lands in a single-quoted context must not break out either.
        body = _script_body()
        fn = re.search(r"function escapeHtml\(s\) \{(.*?)\n\}", body, re.S).group(1)
        for ent in ("&amp;", "&lt;", "&gt;", "&quot;", "&#39;"):
            assert ent in fn


class TestCSP:
    @pytest.mark.asyncio
    async def test_dashboard_sends_nonce_csp(self, async_client):
        resp = await async_client.get("/")
        assert resp.status_code == 200
        csp = resp.headers["content-security-policy"]
        nonce = re.search(r"'nonce-([A-Za-z0-9_-]+)'", csp).group(1)
        assert "'unsafe-inline'" not in re.search(r"script-src[^;]*", csp).group(0)
        assert f'<script nonce="{nonce}">' in resp.text
        assert "__CSP_NONCE__" not in resp.text
        assert resp.headers["x-frame-options"] == "DENY"
        assert resp.headers["x-content-type-options"] == "nosniff"

    @pytest.mark.asyncio
    async def test_nonce_is_per_request(self, async_client):
        a = (await async_client.get("/")).headers["content-security-policy"]
        b = (await async_client.get("/")).headers["content-security-policy"]
        assert a != b

    @pytest.mark.asyncio
    async def test_d3_is_pinned_with_integrity(self, async_client):
        text = (await async_client.get("/")).text
        tag = re.search(r'<script src="https://cdn.jsdelivr.net/npm/d3@[^"]+"[^>]*>', text).group(0)
        assert "@7." in tag and "@7/" not in tag, "d3 must be pinned to an exact version for SRI"
        assert re.search(r'integrity="sha384-[A-Za-z0-9+/=]{64}"', tag)
        assert 'crossorigin="anonymous"' in tag
        assert "__D3_SRI__" not in text
