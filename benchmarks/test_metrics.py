"""Unit tests for benchmarks/metrics.py — recall_all / evidence_completeness.

Run directly: python3 -m pytest benchmarks/test_metrics.py -v
(Not part of the main `pytest` run — testpaths is scoped to tests/ for the
installed package; this covers the standalone benchmark harness.)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from metrics import evaluate_retrieval, evidence_completeness, ndcg  # noqa: E402


def test_recall_any_hits_on_partial_overlap():
    ranked = ["a", "b", "c", "d", "e"]
    gold = {"a", "c", "z"}
    ra, rl, nd = evaluate_retrieval(ranked, gold, 3)
    assert ra == 1.0  # at least one gold id (a, c) in top-3
    assert rl == 0.0  # not all gold ids present (z missing)


def test_recall_all_requires_every_gold_id():
    ranked = ["a", "b", "c"]
    gold = {"a", "c"}
    ra, rl, nd = evaluate_retrieval(ranked, gold, 3)
    assert ra == 1.0
    assert rl == 1.0  # both gold ids present in top-3


def test_evidence_completeness_is_fractional():
    ranked = ["a", "b", "c", "d", "e"]
    gold = {"a", "c", "z"}
    ec = evidence_completeness(ranked, gold, 3)
    assert abs(ec - 2 / 3) < 1e-9  # 2 of 3 gold ids present


def test_evidence_completeness_full_and_zero():
    ranked = ["a", "b", "c"]
    assert evidence_completeness(ranked, {"a", "b", "c"}, 3) == 1.0
    assert evidence_completeness(ranked, {"x", "y"}, 3) == 0.0


def test_evidence_completeness_empty_gold_is_vacuously_complete():
    assert evidence_completeness(["a", "b"], set(), 5) == 1.0


def test_evidence_completeness_matches_recall_any_when_single_gold():
    """With exactly one gold id, evidence_completeness collapses to recall_any."""
    ranked = ["x", "y", "a"]
    gold = {"a"}
    ra, _, _ = evaluate_retrieval(ranked, gold, 3)
    ec = evidence_completeness(ranked, gold, 3)
    assert ra == ec == 1.0

    ra_miss, _, _ = evaluate_retrieval(ranked, gold, 1)
    ec_miss = evidence_completeness(ranked, gold, 1)
    assert ra_miss == ec_miss == 0.0
