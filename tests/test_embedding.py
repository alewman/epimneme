"""Embedding backend tests.

The failure modes that matter here are operational: a remote backend that is
down, one that returns the wrong dimension, and prefixes applied to the wrong
side. Each of those silently corrupts retrieval if unhandled.
"""
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from epimneme.embedding import (
    OllamaBackend,
    SentenceTransformerBackend,
    build_backend,
)


def _cfg(**kw):
    base = dict(embedding_model="all-minilm", embedding_dim=384,
                embedding_backend="ollama", ollama_url="http://stub:11434",
                embedding_query_prefix="", embedding_doc_prefix="",
                ollama_timeout=5.0, ollama_batch_size=64)
    base.update(kw)
    return SimpleNamespace(**base)


class _Resp:
    def __init__(self, payload):
        self._b = json.dumps(payload).encode()
    def read(self):
        return self._b
    def __enter__(self):
        return self
    def __exit__(self, *a):
        return False


class TestOllamaBackend:
    def test_normalizes_vectors(self):
        """Ollama does not guarantee unit vectors and scoring is a dot product."""
        b = OllamaBackend("m", "http://x")
        with patch("urllib.request.urlopen", return_value=_Resp({"embeddings": [[3.0, 4.0]]})):
            v = b.encode(["hi"], is_query=False)
        assert v is not None
        assert pytest.approx(sum(x * x for x in v[0]), rel=1e-6) == 1.0

    def test_unreachable_degrades_to_none(self):
        """A down embedder must not 500 the request — lexical channels remain."""
        b = OllamaBackend("m", "http://x", attempts=1)
        with patch("urllib.request.urlopen", side_effect=OSError("refused")):
            assert b.encode(["hi"], is_query=False) is None

    def test_4xx_is_not_retried(self):
        """A 400 is deterministic; retrying it wasted an hour of a real run."""
        import urllib.error
        calls = []

        def boom(*a, **k):
            calls.append(1)
            raise urllib.error.HTTPError("u", 400, "Bad Request", {}, None)

        b = OllamaBackend("m", "http://x", attempts=4)
        with patch("urllib.request.urlopen", side_effect=boom):
            assert b.encode(["hi"], is_query=False) is None
        assert len(calls) == 1

    def test_5xx_is_retried(self):
        import urllib.error
        calls = []

        def flaky(*a, **k):
            calls.append(1)
            if len(calls) < 3:
                raise urllib.error.HTTPError("u", 503, "nope", {}, None)
            return _Resp({"embeddings": [[1.0, 0.0]]})

        b = OllamaBackend("m", "http://x", attempts=4)
        with patch("urllib.request.urlopen", side_effect=flaky):
            assert b.encode(["hi"], is_query=False) is not None
        assert len(calls) == 3

    def test_query_and_doc_prefixes_go_to_different_sides(self):
        """Instruction-tuned models use different prefixes per side; swapping
        them, or applying both, measurably degrades retrieval."""
        seen = []

        def capture(req, **k):
            seen.append(json.loads(req.data)["input"])
            return _Resp({"embeddings": [[1.0, 0.0]]})

        b = OllamaBackend("m", "http://x", query_prefix="Q: ", doc_prefix="D: ")
        with patch("urllib.request.urlopen", side_effect=capture):
            b.encode(["text"], is_query=True)
            b.encode(["text"], is_query=False)
        assert seen == [["Q: text"], ["D: text"]]

    def test_batches_large_inputs(self):
        sizes = []

        def capture(req, **k):
            n = len(json.loads(req.data)["input"])
            sizes.append(n)
            return _Resp({"embeddings": [[1.0, 0.0]] * n})

        b = OllamaBackend("m", "http://x", batch_size=4)
        with patch("urllib.request.urlopen", side_effect=capture):
            out = b.encode([f"t{i}" for i in range(10)], is_query=False)
        assert sizes == [4, 4, 2]
        assert len(out) == 10

    def test_no_token_embeddings(self):
        assert OllamaBackend("m", "http://x").st_model is None


class TestBuildBackend:
    def test_dimension_mismatch_raises(self):
        """The pgvector column is fixed width — a mismatch must fail at startup,
        not once per insert."""
        with patch("urllib.request.urlopen", return_value=_Resp({"embeddings": [[0.0] * 768]})):
            with pytest.raises(ValueError, match="dimension mismatch"):
                build_backend(_cfg(embedding_dim=384))

    def test_matching_dimension_builds(self):
        with patch("urllib.request.urlopen", return_value=_Resp({"embeddings": [[0.0] * 384]})):
            b = build_backend(_cfg(embedding_dim=384))
        assert b.name == "ollama"

    def test_unreachable_backend_still_builds(self):
        """Startup must not hard-fail because the embedder is briefly down."""
        with patch("urllib.request.urlopen", side_effect=OSError("refused")):
            b = build_backend(_cfg())
        assert b.name == "ollama"

    def test_default_is_in_process(self):
        b = build_backend(_cfg(embedding_backend="sentence-transformers"))
        assert isinstance(b, SentenceTransformerBackend)


class TestDimensionEnforcement:
    def test_wrong_dimension_refused_at_encode_time(self):
        """If the backend was down at startup the probe never ran, so the
        dimension must also be enforced on every response."""
        b = OllamaBackend("m", "http://x", expected_dim=384)
        with patch("urllib.request.urlopen",
                   return_value=_Resp({"embeddings": [[0.0] * 768]})):
            assert b.encode(["hi"], is_query=False) is None

    def test_right_dimension_passes(self):
        b = OllamaBackend("m", "http://x", expected_dim=384)
        with patch("urllib.request.urlopen",
                   return_value=_Resp({"embeddings": [[1.0] + [0.0] * 383]})):
            assert b.encode(["hi"], is_query=False) is not None
