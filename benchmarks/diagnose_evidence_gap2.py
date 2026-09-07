#!/usr/bin/env python3
"""Offline follow-up to diagnose_evidence_gap.py — no live queries.

Three things, all computed from the already-logged v700 results file plus
the original dataset (for reconstructing turn text/answer-session mapping):

  A) Corrected recall_all@10 / evidence_completeness@10 (turn-level), where
     "gold" = only the turn(s) in the answer session whose text actually
     contains the answer string (substring match on normalized text) —
     not every turn in the session. Falls back to "all turns in session"
     only when no turn matches (reported separately, since that's the
     heuristic's blind spot, not a retrieval fact).

  B) Simulated per-session cap: re-derive top-10 from the ALREADY-FETCHED
     top-50 ranked_items by greedily walking rank order but capping how
     many slots any single session can take (cap in {1,2,3}), before
     recomputing recall_all@10 (both corrected-turn-level and true
     session-level). No new retrieval — this only reorders/truncates data
     already in the log.

  C) A 20-item sample of gold turns (corrected definition) that are
     completely absent from the top-50 pool, with their reconstructed
     text length in characters — to see whether length correlates with
     being missed.
"""

from __future__ import annotations

import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from longmemeval_bench import build_corpus
from metrics import session_id_from_corpus_id, normalize_answer


def corrected_gold_turns(entry: dict, corpus: list[str], corpus_ids: list[str]) -> tuple[set[str], bool]:
    """Gold turns = those in the answer session(s) whose text contains the
    normalized answer string. Returns (gold_ids, used_fallback)."""
    answer_sids = set(entry["answer_session_ids"])
    session_turns = [
        (cid, text) for cid, text in zip(corpus_ids, corpus)
        if session_id_from_corpus_id(cid) in answer_sids
    ]
    norm_answer = normalize_answer(str(entry["answer"]))
    if not norm_answer:
        # Abstention / empty-answer questions — no literal answer to match.
        return {cid for cid, _ in session_turns}, True

    matches = {cid for cid, text in session_turns if norm_answer in normalize_answer(text)}
    if matches:
        return matches, False
    # Heuristic missed (paraphrased / synthesized answer, no literal quote) —
    # fall back to the full session so we don't manufacture a false failure,
    # but flag it so the fallback rate itself is visible.
    return {cid for cid, _ in session_turns}, True


def capped_top_k(ranked_ids: list[str], k: int, session_cap: int) -> list[str]:
    """Greedily take ranked_ids in order, skipping any id whose session has
    already contributed session_cap items, until k slots are filled."""
    out: list[str] = []
    counts: dict[str, int] = defaultdict(int)
    # First pass: respect the cap.
    for cid in ranked_ids:
        if len(out) >= k:
            break
        sid = session_id_from_corpus_id(cid)
        if counts[sid] < session_cap:
            out.append(cid)
            counts[sid] += 1
    # Second pass: if the cap left slots empty (too few distinct sessions
    # in the pool), fill remaining slots from whatever's left in rank order.
    if len(out) < k:
        for cid in ranked_ids:
            if len(out) >= k:
                break
            if cid not in out:
                out.append(cid)
    return out


def main() -> None:
    results_file = sys.argv[1] if len(sys.argv) > 1 else "benchmarks/results_engram_lme_v700-baseline_20260904.jsonl"
    data_file = sys.argv[2] if len(sys.argv) > 2 else "benchmarks/data/longmemeval_s_cleaned.json"

    with open(data_file) as f:
        dataset = {e["question_id"]: e for e in json.load(f)}

    rows = [json.loads(l) for l in open(results_file)]

    # ── A) Corrected metric ──────────────────────────────────────────────
    fallback_count = 0
    total_scored = 0
    corrected_fail = 0
    by_type_fail = Counter()
    by_type_total = Counter()
    ec_sum = 0.0

    # Cache per-question data we'll reuse in B and C.
    cache: dict[str, dict] = {}

    for row in rows:
        base_qid = row["question_id"].replace("_abs", "")
        entry = dataset.get(base_qid) or dataset.get(row["question_id"])
        if entry is None:
            continue
        corpus, corpus_ids, _ = build_corpus(entry, granularity="turn-pair")
        gold, used_fallback = corrected_gold_turns(entry, corpus, corpus_ids)
        if not gold:
            continue

        fallback_count += used_fallback
        total_scored += 1
        by_type_total[row["question_type"]] += 1

        ranked = [it["corpus_id"] for it in row["retrieval_results"]["ranked_items"]]
        top10 = set(ranked[:10])
        found = len(gold & top10)
        ec = found / len(gold)
        ec_sum += ec
        if found < len(gold):
            corrected_fail += 1
            by_type_fail[row["question_type"]] += 1

        cache[row["question_id"]] = {
            "entry": entry,
            "corpus": corpus,
            "corpus_ids": corpus_ids,
            "gold": gold,
            "used_fallback": used_fallback,
            "ranked": ranked,
            "answer_sids": set(entry["answer_session_ids"]),
        }

    print("=" * 78)
    print("  A) CORRECTED recall_all@10 / evidence_completeness@10 (turn-level)")
    print("     gold = only turns whose text contains the answer string")
    print("=" * 78)
    print(f"  n={total_scored}, fallback-to-full-session rate: {fallback_count}/{total_scored} "
          f"({fallback_count / total_scored * 100:.1f}%) — these questions' 'corrected' score is really the old definition")
    print(f"  Corrected recall_all@10 (exact):        {(1 - corrected_fail / total_scored) * 100:.1f}%  "
          f"(was 9.4% under the old all-turns definition)")
    print(f"  Corrected evidence_completeness@10 avg: {ec_sum / total_scored * 100:.1f}%  (was 52.7%)")
    print("\n  Corrected failure rate by type:")
    for qt in sorted(by_type_total):
        f, t = by_type_fail[qt], by_type_total[qt]
        print(f"    {qt:28} {f:3}/{t:3}  ({f / t * 100:5.1f}%)")

    # ── B) Simulated per-session cap ─────────────────────────────────────
    print("\n" + "=" * 78)
    print("  B) SIMULATED per-session cap on the EXISTING top-50 pool (no new retrieval)")
    print("=" * 78)

    for cap in (1, 2, 3, 99):
        # Session-level recall_all@10 (true gold = answer_session_ids), and
        # corrected turn-level recall_all@10, under the capped re-selection.
        sess_fail = Counter()
        sess_total = Counter()
        turn_fail = Counter()
        turn_total = Counter()
        for row in rows:
            c = cache.get(row["question_id"])
            if c is None:
                continue
            capped = capped_top_k(c["ranked"], k=10, session_cap=cap)
            capped_sessions = {session_id_from_corpus_id(cid) for cid in capped}

            qt = row["question_type"]
            sess_total[qt] += 1
            if not c["answer_sids"].issubset(capped_sessions):
                sess_fail[qt] += 1

            turn_total[qt] += 1
            if not c["gold"].issubset(set(capped)):
                turn_fail[qt] += 1

        label = "uncapped (cap=99, ~baseline)" if cap == 99 else f"session_cap={cap}"
        print(f"\n  --- {label} ---")
        for qt in ("multi-session", "knowledge-update", "temporal-reasoning"):
            sf, st = sess_fail[qt], sess_total[qt]
            tf, tt = turn_fail[qt], turn_total[qt]
            print(f"    {qt:22} session-level recall_all@10={100 * (1 - sf / st):5.1f}%   "
                  f"corrected turn-level recall_all@10={100 * (1 - tf / tt):5.1f}%")
        overall_sf = sum(sess_fail.values())
        overall_st = sum(sess_total.values())
        print(f"    {'OVERALL':22} session-level recall_all@10={100 * (1 - overall_sf / overall_st):5.1f}%")

    # ── C) Sample of genuinely-absent gold turns, with lengths ───────────
    print("\n" + "=" * 78)
    print("  C) 20-item sample of CORRECTED gold turns absent from top-50, with lengths")
    print("=" * 78)

    absent_examples = []
    for row in rows:
        c = cache.get(row["question_id"])
        if c is None or c["used_fallback"]:
            continue  # skip fallback cases — those aren't a confirmed single answer-turn
        ranked_set = set(c["ranked"])
        missing = c["gold"] - ranked_set
        for mid in missing:
            idx = c["corpus_ids"].index(mid)
            text = c["corpus"][idx]
            absent_examples.append({
                "question_id": row["question_id"],
                "question_type": row["question_type"],
                "corpus_id": mid,
                "char_len": len(text),
                "text_preview": text[:100].replace("\n", " "),
            })

    random.Random(7).shuffle(absent_examples)
    sample = absent_examples[:20]
    all_lens = [e["char_len"] for e in absent_examples]
    print(f"  Total genuinely-absent (non-fallback) gold turns: {len(absent_examples)}")
    if all_lens:
        print(f"  Length stats (chars): min={min(all_lens)} median={sorted(all_lens)[len(all_lens)//2]} "
              f"mean={sum(all_lens)/len(all_lens):.0f} max={max(all_lens)}")
    print()
    for e in sample:
        print(f"  [{e['char_len']:4} chars] {e['question_type']:22} {e['question_id']:16} {e['corpus_id']:28} {e['text_preview']!r}")


if __name__ == "__main__":
    main()
