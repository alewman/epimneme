"""Retrieval metrics — identical to MemPalace's implementations for like-for-like comparison.

Metrics:
  - DCG / NDCG @ k
  - Recall@k (any / all)
  - F1 score (token-level, normalized)
"""

from __future__ import annotations

import math
import re
import string
from collections import Counter


# ── Ranking metrics ──────────────────────────────────────────────────────


def dcg(relevances: list[float], k: int) -> float:
    """Discounted Cumulative Gain."""
    score = 0.0
    for i, rel in enumerate(relevances[:k]):
        score += rel / math.log2(i + 2)
    return score


def ndcg(ranked_ids: list[str], correct_ids: set[str], k: int) -> float:
    """Normalized DCG over ranked result IDs."""
    relevances = [1.0 if rid in correct_ids else 0.0 for rid in ranked_ids[:k]]
    ideal = sorted(relevances, reverse=True)
    idcg = dcg(ideal, k)
    if idcg == 0:
        return 0.0
    return dcg(relevances, k) / idcg


def evaluate_retrieval(
    ranked_ids: list[str], correct_ids: set[str], k: int
) -> tuple[float, float, float]:
    """Evaluate retrieval at rank k.

    Returns (recall_any, recall_all, ndcg_score).
    """
    top_k_ids = set(ranked_ids[:k])
    recall_any = float(any(cid in top_k_ids for cid in correct_ids))
    recall_all = float(all(cid in top_k_ids for cid in correct_ids))
    ndcg_score = ndcg(ranked_ids, correct_ids, k)
    return recall_any, recall_all, ndcg_score


# ── Text metrics ─────────────────────────────────────────────────────────


def normalize_answer(s: str) -> str:
    """Normalize answer for F1 comparison (matches LoCoMo's evaluation.py)."""
    s = s.replace(",", "")
    s = re.sub(r"\b(a|an|the|and)\b", " ", s)
    s = " ".join(s.split())
    s = "".join(ch for ch in s if ch not in string.punctuation)
    return s.lower().strip()


def f1_score(prediction: str, ground_truth: str) -> float:
    """Token-level F1 with normalization."""
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(ground_truth).split()
    if not pred_tokens or not truth_tokens:
        return float(pred_tokens == truth_tokens)
    common = Counter(pred_tokens) & Counter(truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(truth_tokens)
    return (2 * precision * recall) / (precision + recall)


# ── Helpers ──────────────────────────────────────────────────────────────


def session_id_from_corpus_id(corpus_id: str) -> str:
    """Extract session ID from a corpus ID (handles turn-level IDs like sess_123_turn_4)."""
    if "_turn_" in corpus_id:
        return corpus_id.rsplit("_turn_", 1)[0]
    return corpus_id
