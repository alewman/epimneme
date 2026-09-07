#!/usr/bin/env python3
"""Diagnose the recall_any@k vs recall_all@k / evidence_completeness@k gap.

For every question with turn-level recall_all@10 == 0 (the "failing" set),
reconstructs the exact gold turn-id set (build_corpus + answer_session_ids,
mirroring longmemeval_bench.py exactly) and cross-references it against the
question's actual top-50 ranked_items from the results log to classify each
missing gold turn as either:

  - "ranked 11-50"   : present in the real candidate pool, just outside K=10
                       (a ranking/prioritization problem — ordering, not fetch)
  - "absent from top-50" : never surfaced at all in the final fused list
                       (a candidate-pool/scoring problem, not a K-budget one)

Usage:
    python benchmarks/diagnose_evidence_gap.py \
        benchmarks/results_engram_lme_v700-baseline_20260904.jsonl \
        benchmarks/data/longmemeval_s_cleaned.json
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from longmemeval_bench import build_corpus
from metrics import session_id_from_corpus_id


def main() -> None:
    results_file, data_file = sys.argv[1], sys.argv[2]

    with open(data_file) as f:
        dataset = {e["question_id"]: e for e in json.load(f)}

    rows = [json.loads(l) for l in open(results_file)]

    total = len(rows)
    failing = []
    passing = []

    rank_bucket_counter = Counter()  # for MISSING gold turns only
    by_type_fail = Counter()
    by_type_total = Counter()
    fully_absent_questions = 0
    partially_ranked_questions = 0
    examples_absent = []
    examples_ranked_low = []

    for row in rows:
        qid = row["question_id"].replace("_abs", "")  # abstention variants share the base entry
        entry = dataset.get(qid) or dataset.get(row["question_id"])
        if entry is None:
            continue

        by_type_total[row["question_type"]] += 1

        _, corpus_ids, _ = build_corpus(entry, granularity="turn-pair")
        answer_sids = set(entry["answer_session_ids"])
        gold_turn_ids = {
            cid for cid in corpus_ids if session_id_from_corpus_id(cid) in answer_sids
        }
        if not gold_turn_ids:
            continue

        ranked = [it["corpus_id"] for it in row["retrieval_results"]["ranked_items"]]
        rank_of = {cid: i + 1 for i, cid in enumerate(ranked)}  # 1-indexed

        top10 = set(ranked[:10])
        recall_all_10 = gold_turn_ids.issubset(top10)

        if recall_all_10:
            passing.append(row)
            continue

        failing.append(row)
        by_type_fail[row["question_type"]] += 1

        missing = gold_turn_ids - top10
        n_absent = 0
        n_ranked_11_50 = 0
        for mid in missing:
            r = rank_of.get(mid)
            if r is None:
                rank_bucket_counter["absent_from_top50"] += 1
                n_absent += 1
            else:
                bucket = "ranked_11-30" if r <= 30 else "ranked_31-50"
                rank_bucket_counter[bucket] += 1
                n_ranked_11_50 += 1

        if n_absent > 0:
            fully_absent_questions += 1
            if len(examples_absent) < 5:
                examples_absent.append((row["question_id"], row["question_type"], len(gold_turn_ids), n_absent, n_ranked_11_50))
        else:
            partially_ranked_questions += 1
            if len(examples_ranked_low) < 5:
                examples_ranked_low.append((row["question_id"], row["question_type"], len(gold_turn_ids), n_ranked_11_50,
                                             [rank_of[m] for m in missing]))

    print("=" * 78)
    print(f"  recall_all@10 (turn-level) — {len(failing)}/{len(failing) + len(passing)} FAILING "
          f"({len(failing) / (len(failing) + len(passing)) * 100:.1f}%)")
    print("=" * 78)

    print("\nFailure rate by question_type:")
    for qt in sorted(by_type_total):
        f, t = by_type_fail[qt], by_type_total[qt]
        print(f"  {qt:28} {f:3}/{t:3}  ({f / t * 100:5.1f}%)")

    print(f"\nOf {len(failing)} failing questions, classified by WHERE the missing gold turn(s) went:")
    print(f"  >=1 gold turn completely ABSENT from top-50 candidate pool: {fully_absent_questions} "
          f"({fully_absent_questions / len(failing) * 100:.1f}%)")
    print(f"  ALL missing gold turns present in top-50, just ranked 11-50: {partially_ranked_questions} "
          f"({partially_ranked_questions / len(failing) * 100:.1f}%)")

    print(f"\nMissing-turn rank distribution (across all failing questions, {sum(rank_bucket_counter.values())} missing turns total):")
    for bucket, n in rank_bucket_counter.most_common():
        print(f"  {bucket:22} {n:4}")

    print("\n--- Example: gold turn ABSENT from top-50 entirely ---")
    for qid, qt, n_gold, n_abs, n_ranked in examples_absent:
        print(f"  {qid:20} type={qt:26} gold_turns={n_gold} absent={n_abs} ranked_but_low={n_ranked}")

    print("\n--- Example: gold turns present but ranked below 10 ---")
    for qid, qt, n_gold, n_ranked, ranks in examples_ranked_low:
        print(f"  {qid:20} type={qt:26} gold_turns={n_gold} missing_ranked_at={sorted(ranks)}")


if __name__ == "__main__":
    main()
