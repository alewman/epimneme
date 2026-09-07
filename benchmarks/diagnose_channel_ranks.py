#!/usr/bin/env python3
"""Per-channel RRF pathology check on a live sample of known misses.

For a sample of questions where a has_answer gold turn missed the top-10
in the v700 run, re-ingests the haystack, re-queries with debug=true, and
records each channel's PRE-FUSION rank for:
  - the missing gold (has_answer) turn
  - the actual #1 final winner (whatever occupies rank 1 post-fusion)

Classifies each miss as:
  - "single-channel casualty": the gold turn ranks well in >=1 channel but
    badly/absent in >=1 other — RRF pathology, fixable via per-channel
    weighting or a rank floor.
  - "uniformly weak": no channel ranks the gold turn well — an
    embedding/semantics problem, not a fusion problem.

Usage:
    python benchmarks/diagnose_channel_ranks.py --sample 20
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from epimneme_client import EngramClient
from longmemeval_bench import build_corpus, ingest_corpus, cleanup_project, load_data
from metrics import session_id_from_corpus_id
from diagnose_evidence_gap3 import build_corpus_with_has_answer

GOOD_RANK_THRESHOLD = 10  # "ranks well" cutoff within a channel's own list


def find_misses(results_file: str, data_file: str) -> list[dict]:
    with open(data_file) as f:
        dataset = {e["question_id"]: e for e in json.load(f)}
    rows = [json.loads(l) for l in open(results_file)]

    misses = []
    for row in rows:
        qid = row["question_id"]
        base_qid = qid.replace("_abs", "")
        entry = dataset.get(qid) or dataset.get(base_qid)
        if entry is None:
            continue
        corpus, corpus_ids, offsets = build_corpus_with_has_answer(entry)
        answer_sids = set(entry["answer_session_ids"])
        gold = {
            cid for cid, off in zip(corpus_ids, offsets)
            if off >= 0 and session_id_from_corpus_id(cid) in answer_sids
        }
        if not gold:
            continue
        ranked = [it["corpus_id"] for it in row["retrieval_results"]["ranked_items"]]
        top10 = set(ranked[:10])
        missing = gold - top10
        if missing:
            misses.append({
                "question_id": qid,
                "question_type": row["question_type"],
                "question": row["question"],
                "missing_turn": sorted(missing)[0],  # one representative turn per question
            })
    return misses


async def diagnose_one(client: EngramClient, entry: dict, miss: dict) -> dict | None:
    project_name = f"_chandiag_{miss['question_id']}"
    corpus, corpus_ids, corpus_timestamps = build_corpus(entry, granularity="turn-pair")
    await client.create_project(project_name)
    await ingest_corpus(client, project_name, corpus, corpus_ids, corpus_timestamps)

    # Reliable subject(corpus_id) -> memory_id map via direct enumeration —
    # NOT via the search result page, which only covers the top-50 fused
    # results and would silently fail to resolve exactly the "never
    # surfaced" cases this script exists to look at.
    all_memories = await client.list_all_memories(project_name, batch_size=500)
    id_to_subject = {m["id"]: m["subject"] for m in all_memories}

    result = await client.search(miss["question"], project=project_name, limit=50, debug=True)
    await cleanup_project(client, project_name)

    debug = result.get("debug")
    final_results = result.get("results", [])
    if not debug:
        return None

    channels = debug["channels"]
    gold_subject = miss["missing_turn"]
    winner_subject = final_results[0]["subject"] if final_results else None

    def rank_of(subject: str | None, channel_ids: list[str]) -> int | None:
        if subject is None:
            return None
        mid = next((m for m, s in id_to_subject.items() if s == subject), None)
        if mid is None:
            return None
        try:
            return channel_ids.index(mid) + 1
        except ValueError:
            return None

    gold_ranks = {ch: rank_of(gold_subject, ids) for ch, ids in channels.items()}
    winner_ranks = {ch: rank_of(winner_subject, ids) for ch, ids in channels.items()}

    good = [r for r in gold_ranks.values() if r is not None and r <= GOOD_RANK_THRESHOLD]
    bad_or_absent = [ch for ch, r in gold_ranks.items() if r is None or r > GOOD_RANK_THRESHOLD]
    classification = (
        "single_channel_casualty" if good and bad_or_absent
        else "uniformly_weak" if not good
        else "uniformly_strong_but_still_lost"
    )

    return {
        "question_id": miss["question_id"],
        "question_type": miss["question_type"],
        "gold_subject": gold_subject,
        "winner_subject": winner_subject,
        "gold_ranks": gold_ranks,
        "winner_ranks": winner_ranks,
        "classification": classification,
    }


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-file", default="benchmarks/results_engram_lme_v700-baseline_20260904.jsonl")
    ap.add_argument("--data-file", default="benchmarks/data/longmemeval_s_cleaned.json")
    ap.add_argument("--engram-url", default="http://localhost:8000")
    ap.add_argument("--sample", type=int, default=20)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--types", default="multi-session,temporal-reasoning")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    wanted_types = set(args.types.split(","))
    misses = [m for m in find_misses(args.results_file, args.data_file) if m["question_type"] in wanted_types]
    print(f"Total misses available in {wanted_types}: {len(misses)}")

    sample = random.Random(args.seed).sample(misses, min(args.sample, len(misses)))

    with open(args.data_file) as f:
        dataset = {e["question_id"]: e for e in json.load(f)}

    client = EngramClient(base_url=args.engram_url)
    results = []
    for i, miss in enumerate(sample):
        qid = miss["question_id"]
        base_qid = qid.replace("_abs", "")
        entry = dataset.get(qid) or dataset.get(base_qid)
        r = await diagnose_one(client, entry, miss)
        if r:
            results.append(r)
            print(f"[{i+1}/{len(sample)}] {qid:20} {miss['question_type']:20} -> {r['classification']}")
        else:
            print(f"[{i+1}/{len(sample)}] {qid:20} {miss['question_type']:20} -> (unresolved, skipped)")
    await client.close()

    if args.out:
        with open(args.out, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")

    print("\n" + "=" * 78)
    print(f"  CLASSIFICATION SUMMARY (n={len(results)})")
    print("=" * 78)
    from collections import Counter
    counts = Counter(r["classification"] for r in results)
    for label, n in counts.most_common():
        print(f"  {label:32} {n:3}  ({n/len(results)*100:.1f}%)" if results else "")

    print("\n--- Per-case detail ---")
    for r in results:
        print(f"\n{r['question_id']} ({r['question_type']}) — {r['classification']}")
        print(f"  gold   ({r['gold_subject']}): {r['gold_ranks']}")
        print(f"  winner ({r['winner_subject']}): {r['winner_ranks']}")


if __name__ == "__main__":
    asyncio.run(main())
