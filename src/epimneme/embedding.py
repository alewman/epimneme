"""Embedding backends.

Embedding used to be hard-wired to an in-process SentenceTransformer. That is
fine for a 22M model and painful for anything larger: on this project's CPU a
308M model runs ~22x slower to ingest and adds ~40ms to every recall, and the
strongest candidate (`google/embeddinggemma-300m`) is gated on HuggingFace, so
the weights cannot even be fetched without a token.

Both problems go away if the embedder lives outside the process. `OllamaBackend`
talks to an Ollama server, which serves its own converted weights (no auth) and
can sit on a GPU host. Measured: 29 texts/s on Apple Silicon against 1.6 on this
CPU for the same model.

The two backends are NOT interchangeable for an existing corpus. A different
model means different vectors and usually a different dimension, so switching
requires altering the pgvector column and re-embedding everything. The dimension
is therefore checked against the configured one at startup, loudly.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from typing import Optional, Protocol

logger = logging.getLogger("engram.embedding")


class EmbeddingBackend(Protocol):
    """Minimal surface the manager needs."""

    name: str

    def encode(self, texts: list[str], *, is_query: bool) -> Optional[list[list[float]]]:
        """Embed texts. Returns None if embedding is unavailable — callers
        degrade to lexical search rather than failing the request."""
        ...

    @property
    def st_model(self):
        """The underlying SentenceTransformer, or None. MaxSim needs token-level
        embeddings, which only the in-process backend can produce."""
        ...


def _l2(vecs: list[list[float]]) -> list[list[float]]:
    out = []
    for v in vecs:
        n = sum(x * x for x in v) ** 0.5
        out.append([x / n for x in v] if n > 1e-12 else v)
    return out


class SentenceTransformerBackend:
    """In-process sentence-transformers. The historical behaviour."""

    name = "sentence-transformers"

    def __init__(self, model_name: str, query_prefix: str = "", doc_prefix: str = ""):
        self._model_name = model_name
        self._query_prefix = query_prefix
        self._doc_prefix = doc_prefix
        self._model = None
        self._failed = False

    @property
    def st_model(self):
        if self._model is None and not self._failed:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self._model_name)
                logger.info("Loaded embedding model: %s", self._model_name)
            except Exception as e:
                logger.warning("Failed to load embeddings: %s", e)
                self._failed = True
        return self._model

    def encode(self, texts: list[str], *, is_query: bool) -> Optional[list[list[float]]]:
        model = self.st_model
        if model is None:
            return None
        prefix = self._query_prefix if is_query else self._doc_prefix
        payload = [prefix + t for t in texts] if prefix else texts
        try:
            vecs = model.encode(payload, normalize_embeddings=True,
                                batch_size=64, show_progress_bar=False)
            return [v.tolist() for v in vecs]
        except Exception:
            logger.exception("Embedding failed")
            return None


class OllamaBackend:
    """Remote embedding via Ollama's /api/embed.

    Degrades to None rather than raising: a recall with no vector still has the
    lexical channels, which is better than a 500.
    """

    name = "ollama"

    def __init__(self, model_name: str, url: str, *, query_prefix: str = "",
                 doc_prefix: str = "", timeout: float = 60.0, batch_size: int = 64,
                 attempts: int = 3, expected_dim: Optional[int] = None):
        self._model = model_name
        self._url = url.rstrip("/") + "/api/embed"
        self._query_prefix = query_prefix
        self._doc_prefix = doc_prefix
        self._timeout = timeout
        self._batch = max(1, batch_size)
        self._attempts = max(1, attempts)
        self._expected_dim = expected_dim

    @property
    def st_model(self):
        return None  # no token-level embeddings over HTTP

    def _post(self, batch: list[str]) -> list[list[float]]:
        body = json.dumps({"model": self._model, "input": batch}).encode()
        last: Optional[Exception] = None
        for i in range(self._attempts):
            try:
                req = urllib.request.Request(
                    self._url, data=body, headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=self._timeout) as r:
                    return json.loads(r.read())["embeddings"]
            except urllib.error.HTTPError as e:
                # 4xx is deterministic — a bad model name or oversized input will
                # fail identically on every retry, so fail fast and say why.
                if e.code < 500:
                    raise
                last = e
            except (urllib.error.URLError, OSError, KeyError, json.JSONDecodeError) as e:
                last = e
            if i + 1 < self._attempts:
                time.sleep(0.5 * (2 ** i))
        raise RuntimeError(f"ollama embed failed after {self._attempts} attempts: {last}")

    def encode(self, texts: list[str], *, is_query: bool) -> Optional[list[list[float]]]:
        if not texts:
            return []
        prefix = self._query_prefix if is_query else self._doc_prefix
        payload = [prefix + t for t in texts] if prefix else texts
        try:
            out: list[list[float]] = []
            for i in range(0, len(payload), self._batch):
                out.extend(self._post(payload[i:i + self._batch]))
            # Enforce the dimension on every response, not just at startup:
            # if the backend was unreachable when the manager started, the
            # startup probe never ran, and writing wrong-width vectors into a
            # fixed-width pgvector column fails the insert anyway. Refusing here
            # keeps a misconfiguration from looking like an embedding outage.
            if self._expected_dim and out and len(out[0]) != self._expected_dim:
                logger.error(
                    "Ollama model %r returned %d dims, expected %d — refusing to "
                    "embed. Fix EPIMNEME_EMBEDDING_DIM, migrate the column and "
                    "re-embed before using this model.",
                    self._model, len(out[0]), self._expected_dim)
                return None
            return _l2(out)
        except Exception as e:
            logger.warning("Ollama embedding failed (%s): %s", self._url, e)
            return None


def build_backend(config) -> EmbeddingBackend:
    """Construct the configured backend and verify its dimension.

    A dimension mismatch is not survivable — the pgvector column is fixed width,
    so every insert would fail. Better to find out at startup than per-request.
    """
    query_prefix = getattr(config, "embedding_query_prefix", "") or ""
    doc_prefix = getattr(config, "embedding_doc_prefix", "") or ""
    backend_name = getattr(config, "embedding_backend", "sentence-transformers")

    if backend_name == "ollama":
        kwargs = dict(
            query_prefix=query_prefix, doc_prefix=doc_prefix,
            timeout=getattr(config, "ollama_timeout", 60.0),
            batch_size=getattr(config, "ollama_batch_size", 64),
        )
        url = getattr(config, "ollama_url", "http://localhost:11434")
        # Probe with the dimension check OFF so a mismatch can be reported as a
        # mismatch. An instance that already enforces the dimension would refuse
        # the probe and be indistinguishable from the backend being down.
        probe = OllamaBackend(config.embedding_model, url, **kwargs).encode(
            ["dimension probe"], is_query=False)
        backend = OllamaBackend(config.embedding_model, url,
                                expected_dim=config.embedding_dim, **kwargs)
        if probe is None:
            logger.error(
                "Ollama backend unreachable at %s — semantic search will be "
                "disabled until it responds; lexical channels still work.",
                getattr(config, "ollama_url", ""))
        elif len(probe[0]) != config.embedding_dim:
            raise ValueError(
                f"embedding dimension mismatch: model {config.embedding_model!r} "
                f"returns {len(probe[0])} dims but EPIMNEME_EMBEDDING_DIM is "
                f"{config.embedding_dim}. The pgvector column is fixed width — "
                f"set the dim, migrate the column and re-embed before switching."
            )
        else:
            logger.info("Ollama embedding backend ready: %s @ %s (%d dims)",
                        config.embedding_model,
                        getattr(config, "ollama_url", ""), config.embedding_dim)
        return backend

    return SentenceTransformerBackend(
        config.embedding_model, query_prefix=query_prefix, doc_prefix=doc_prefix)
