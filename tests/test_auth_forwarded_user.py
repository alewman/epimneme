"""Trusted-proxy handling for X-Forwarded-User.

Regression guard for a CRIT found by an external review: the header alone
granted admin with projects=["*"], so anything able to reach the app directly
could escalate. It was briefly reachable from the internet on 2026-09-27 when a
Traefik rule that had been incidentally shielding it was removed.

Trust is now opt-in, and can additionally be bound to a shared proxy secret.
"""
from __future__ import annotations

import importlib

import pytest


class _Headers:
    """Case-insensitive mapping, like Starlette's Headers."""

    def __init__(self, d):
        self._d = {k.lower(): v for k, v in d.items()}

    def get(self, k, default=None):
        return self._d.get(k.lower(), default)


def _auth_with(monkeypatch, **env):
    import epimneme.auth as mod
    for k in ("EPIMNEME_TRUST_FORWARDED_USER", "EPIMNEME_PROXY_SECRET"):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    return importlib.reload(mod)


@pytest.fixture(autouse=True)
def _restore():
    yield
    import epimneme.auth as mod
    importlib.reload(mod)


class TestForwardedUserTrust:
    def test_untrusted_by_default(self, monkeypatch):
        """A stock deployment must not be escalatable by a header."""
        a = _auth_with(monkeypatch)
        assert a.TRUST_FORWARDED_USER is False
        assert a._forwarded_user(_Headers({"X-Forwarded-User": "attacker@evil"})) is None

    def test_trusted_when_explicitly_enabled(self, monkeypatch):
        a = _auth_with(monkeypatch, EPIMNEME_TRUST_FORWARDED_USER="1")
        assert a._forwarded_user(_Headers({"X-Forwarded-User": "real@user"})) == "real@user"

    def test_secret_required_when_configured(self, monkeypatch):
        a = _auth_with(monkeypatch, EPIMNEME_TRUST_FORWARDED_USER="1",
                       EPIMNEME_PROXY_SECRET="s3cret")
        assert a._forwarded_user(_Headers({"X-Forwarded-User": "u"})) is None
        assert a._forwarded_user(
            _Headers({"X-Forwarded-User": "u", "X-Proxy-Secret": "wrong"})) is None
        assert a._forwarded_user(
            _Headers({"X-Forwarded-User": "u", "X-Proxy-Secret": "s3cret"})) == "u"

    def test_no_header_is_none_even_when_trusted(self, monkeypatch):
        a = _auth_with(monkeypatch, EPIMNEME_TRUST_FORWARDED_USER="1")
        assert a._forwarded_user(_Headers({})) is None

    def test_only_exactly_1_enables_trust(self, monkeypatch):
        """A truthy-looking value must not enable it by accident."""
        for v in ("0", "true", "yes", "", "2"):
            a = _auth_with(monkeypatch, EPIMNEME_TRUST_FORWARDED_USER=v)
            assert a.TRUST_FORWARDED_USER is False, v
            assert a._forwarded_user(_Headers({"X-Forwarded-User": "x"})) is None


class TestMissingProjectIsNotAWildcard:
    """A missing `project` used to return True from can_access_project, and the
    store treats project_id=None as "every project" (the filter is only added
    `if project_id`). A key scoped to one project could therefore read every
    tenant by omitting the parameter — reproduced live on 2026-09-27.
    """

    @staticmethod
    def _ctx(role, projects):
        from epimneme.auth import AuthContext
        return AuthContext(name="k", role=role, projects=projects, source="api_key")

    def test_scoped_key_cannot_omit_project(self):
        assert self._ctx("agent", ["proj-a"]).can_access_project(None) is False

    def test_scoped_key_still_reaches_its_own_project(self):
        c = self._ctx("agent", ["proj-a"])
        assert c.can_access_project("proj-a") is True
        assert c.can_access_project("proj-b") is False

    def test_admin_and_wildcard_unaffected(self):
        assert self._ctx("admin", ["*"]).can_access_project(None) is True
        assert self._ctx("agent", ["*"]).can_access_project(None) is True
        assert self._ctx("agent", ["a", "*"]).can_access_project(None) is True

    def test_missing_project_is_400_not_403(self):
        """403 would imply the project exists and is someone else's; the caller
        simply has to name one."""
        import pytest as _p
        from fastapi import HTTPException
        c = self._ctx("agent", ["proj-a"])
        with _p.raises(HTTPException) as e:
            c.enforce_project_access(None)
        assert e.value.status_code == 400
        assert "proj-a" in e.value.detail

    def test_foreign_project_is_still_403(self):
        import pytest as _p
        from fastapi import HTTPException
        with _p.raises(HTTPException) as e:
            self._ctx("agent", ["proj-a"]).enforce_project_access("proj-b")
        assert e.value.status_code == 403
