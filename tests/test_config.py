"""Tests for engram.core.config — pure unit tests for EngramConfig + default_config."""

import os
from unittest.mock import patch


from epimneme.core.config import EngramConfig, default_config


# ── EngramConfig dataclass ───────────────────────────────────────────────────


class TestEngramConfig:
    def test_default_values(self):
        c = EngramConfig()
        assert c.pg_host == "epimneme-db"
        assert c.pg_port == 5432
        assert c.pg_user == "epimneme"
        assert c.pg_password == "epimneme"
        assert c.pg_database == "epimneme"
        assert c.embedding_model == "all-MiniLM-L6-v2"
        assert c.embedding_dim == 384
        assert c.embeddings_enabled is True
        assert c.decay_base_stability == 1.0
        assert c.decay_growth_factor == 0.5
        assert c.dedup_enabled is True
        assert c.dedup_hamming_threshold == 3

    def test_pg_dsn_property(self):
        c = EngramConfig(
            pg_host="myhost", pg_port=5433,
            pg_user="myuser", pg_password="secret", pg_database="mydb"
        )
        dsn = c.pg_dsn
        assert "host=myhost" in dsn
        assert "port=5433" in dsn
        assert "dbname=mydb" in dsn
        assert "user=myuser" in dsn
        assert "password=secret" in dsn

    def test_pg_dsn_async_property(self):
        c = EngramConfig(
            pg_host="myhost", pg_port=5433,
            pg_user="myuser", pg_password="secret", pg_database="mydb"
        )
        url = c.pg_dsn_async
        assert url == "postgresql://myuser:secret@myhost:5433/mydb"

    def test_cors_origins_none_by_default(self):
        c = EngramConfig()
        assert c.cors_origins is None

    def test_allowed_hosts_empty_by_default(self):
        c = EngramConfig()
        assert c.allowed_hosts == []

    def test_custom_overrides(self):
        c = EngramConfig(
            embedding_dim=768,
            decay_base_stability=2.5,
            dedup_hamming_threshold=5,
        )
        assert c.embedding_dim == 768
        assert c.decay_base_stability == 2.5
        assert c.dedup_hamming_threshold == 5


# ── default_config() from env ────────────────────────────────────────────────


class TestDefaultConfig:
    def _clean_env(self):
        """Return a dict of EPIMNEME_ env vars to clear."""
        return {k: "" for k in os.environ if k.startswith("EPIMNEME_")}

    @patch.dict(os.environ, {}, clear=False)
    def test_defaults_without_env(self):
        """When no EPIMNEME_ env vars are set (demo mode), defaults should apply."""
        # Remove any pre-existing EPIMNEME_ vars, but keep demo mode so the
        # password guard doesn't raise.
        with patch.dict(os.environ, {k: v for k, v in os.environ.items()
                                      if not k.startswith("EPIMNEME_")} | {"EPIMNEME_DEMO_MODE": "1"}, clear=True):
            c = default_config()
            assert c.pg_host == "epimneme-db"
            assert c.pg_port == 5432
            assert c.dedup_enabled is True
            assert c.pg_password == "epimneme"  # demo-mode fallback

    def test_password_guard_raises_without_password(self):
        with patch.dict(os.environ, {k: v for k, v in os.environ.items()
                                      if not k.startswith("EPIMNEME_")}, clear=True):
            import pytest
            with pytest.raises(RuntimeError, match="EPIMNEME_PG_PASSWORD is not set"):
                default_config()

    def test_password_guard_rejects_default_password(self):
        env = {k: v for k, v in os.environ.items() if not k.startswith("EPIMNEME_")}
        env["EPIMNEME_PG_PASSWORD"] = "epimneme"
        with patch.dict(os.environ, env, clear=True):
            import pytest
            with pytest.raises(RuntimeError, match="default value"):
                default_config()

    def test_pg_host_from_env(self):
        with patch.dict(os.environ, {"EPIMNEME_PG_HOST": "custom-host"}, clear=False):
            c = default_config()
            assert c.pg_host == "custom-host"

    def test_pg_port_from_env(self):
        with patch.dict(os.environ, {"EPIMNEME_PG_PORT": "15432"}, clear=False):
            c = default_config()
            assert c.pg_port == 15432

    def test_pg_password_from_env(self):
        with patch.dict(os.environ, {"EPIMNEME_PG_PASSWORD": "s3cr3t"}, clear=False):
            c = default_config()
            assert c.pg_password == "s3cr3t"

    def test_dedup_disabled_from_env(self):
        with patch.dict(os.environ, {"EPIMNEME_DEDUP_ENABLED": "0"}, clear=False):
            c = default_config()
            assert c.dedup_enabled is False

    def test_dedup_enabled_from_env(self):
        with patch.dict(os.environ, {"EPIMNEME_DEDUP_ENABLED": "1"}, clear=False):
            c = default_config()
            assert c.dedup_enabled is True

    def test_allowed_hosts_from_env(self):
        with patch.dict(os.environ, {"EPIMNEME_ALLOWED_HOSTS": "host1.com,host2.com"}, clear=False):
            c = default_config()
            assert c.allowed_hosts == ["host1.com", "host2.com"]

    def test_cors_origins_explicit(self):
        with patch.dict(os.environ, {
            "EPIMNEME_CORS_ORIGINS": "https://a.com,https://b.com"
        }, clear=False):
            c = default_config()
            assert c.cors_origins == ["https://a.com", "https://b.com"]

    def test_cors_derived_from_allowed_hosts(self):
        """When CORS_ORIGINS is not set but ALLOWED_HOSTS is, derive CORS from hosts."""
        env = {"EPIMNEME_ALLOWED_HOSTS": "app.example.com"}
        # Clear EPIMNEME_CORS_ORIGINS if present
        env["EPIMNEME_CORS_ORIGINS"] = ""
        with patch.dict(os.environ, env, clear=False):
            c = default_config()
            # cors_raw is empty string → no explicit cors → derive from hosts
            # But empty string still splits to [""] which gets filtered...
            # The code: cors_raw = os.environ.get("EPIMNEME_CORS_ORIGINS", "")
            # if cors_raw: → empty string is falsy → falls through
            assert c.cors_origins == ["https://app.example.com"]

    def test_embedding_dim_from_env(self):
        with patch.dict(os.environ, {"EPIMNEME_EMBEDDING_DIM": "768"}, clear=False):
            c = default_config()
            assert c.embedding_dim == 768

    def test_decay_stability_from_env(self):
        with patch.dict(os.environ, {"EPIMNEME_DECAY_STABILITY": "2.5"}, clear=False):
            c = default_config()
            assert c.decay_base_stability == 2.5

    def test_decay_growth_from_env(self):
        with patch.dict(os.environ, {"EPIMNEME_DECAY_GROWTH": "0.8"}, clear=False):
            c = default_config()
            assert c.decay_growth_factor == 0.8

    def test_backup_dir_default(self):
        with patch.dict(os.environ, {k: v for k, v in os.environ.items()
                                      if not k.startswith("EPIMNEME_")} | {"EPIMNEME_DEMO_MODE": "1"}, clear=True):
            c = default_config()
            assert c.backup_dir == "/backups"

    def test_backup_dir_from_env(self):
        with patch.dict(os.environ, {"EPIMNEME_BACKUP_DIR": "/data/backups"}, clear=False):
            c = default_config()
            assert c.backup_dir == "/data/backups"
