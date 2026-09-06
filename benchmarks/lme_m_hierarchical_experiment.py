#!/usr/bin/env python3
"""Phase 5 time-boxed investigation (IMPLEMENTATION_PLAN.md).

Question: does two-stage retrieval (stage 1: rank *sessions* by aggregated
chunk scores; stage 2: rank chunks within winning sessions) beat flat chunk
search at LME-M scale (haystack >> LME-S — 501 sessions/question vs ~40)?

Design: ingest each haystack once, then do ONE over-fetched search per
question (fetch_limit, default 200). Both "flat" and "hierarchical" final
top-K lists are computed from that SAME candidate pool — no schema change,
no new production retrieval signal, no extra network round trips. This is a
benchmark-side prototype only; see the plan for the go/no-go bar
(productionize only if LME-M R@1 gains >= 3pp with LME-S neutral).

Usage:
    python benchmarks/lme_m_hierarchical_experiment.py benchmarks/data/longmemeval_m --limit 20
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from epimneme_client import EngramClient
from metrics import evaluate_retrieval, evidence_completeness, session_id_from_corpus_id
from longmemeval_bench import build_corpus, ingest_corpus, cleanup_project, load_data

KS = [1, 5, 10]


def hierarchical_rerank(results: list[dict], k: int, top_sessions: int) -> list[str]:
    """Stage 1: rank sessions by their best chunk score. Stage 2: within the
    top-N sessions, take chunks ranked by score. Returns corpus_ids, top-k."""
    session_best: dict[str, float] = {}
    by_session: dict[str, list[tuple[float, str]]] = defaultdict(list)
    for r in results:
        cid = r.get("subject", "")
        if not cid:
            continue
        sid = session_id_from_corpus_id(cid)
        score = r.get("score", 0)
        by_session[sid].append((score, cid))
        if sid not in session_best or score > session_best[sid]:
            session_best[sid] = score

    ranked_sessions = sorted(session_best, key=lambda s: session_best[s], reverse=True)[:top_sessions]

    pool: list[tuple[float, str]] = []
    for sid in ranked_sessions:
        pool.extend(by_session[sid])
    pool.sort(key=lambda x: x[0], reverse=True)
    return [cid for _, cid in pool[:k]]


async def process_question(
    entry: dict,
    client: EngramClient,
    project_name: str,
    top_sessions_variants: list[int],
    fetch_limit: int,
) -> dict:
    corpus, corpus_ids, corpus_timestamps = build_corpus(entry, granularity="turn-pair")

    t0 = time.monotonic()
    stored = await ingest_corpus(client, project_name, corpus, corpus_ids, corpus_timestamps)
    ingest_elapsed = time.monotonic() - t0

    t0 = time.monotonic()
    result = await client.search(entry["question"], project=project_name, limit=fetch_limit)
    query_elapsed = time.monotonic() - t0
    results = result.get("results", [])

    flat_ids = [r.get("subject", "") for r in results]
    answer_sids = set(entry["answer_session_ids"])

    configs: dict[str, list[str]] = {"flat": flat_ids}
    for ts in top_sessions_variants:
        configs[f"hierarchical_top{ts}"] = hierarchical_rerank(results, k=max(KS), top_sessions=ts)

    metrics: dict[str, dict[int, dict[str, float]]] = {}
    for label, ranked_ids in configs.items():
        session_level_ids = [session_id_from_corpus_id(cid) for cid in ranked_ids if cid]
        metrics[label] = {}
        for k in KS:
            ra, rl, nd = evaluate_retrieval(session_level_ids, answer_sids, k)
            ec = evidence_completeness(session_level_ids, answer_sids, k)
            metrics[label][k] = {"recall_any": ra, "recall_all": rl, "ndcg": nd, "evidence_completeness": ec}

    await cleanup_project(client, project_name)

    return {
        "question_id": entry["question_id"],
        "question_type": entry["question_type"],
        "stored": stored,
        "candidates_fetched": len(results),
        "ingest_time": round(ingest_elapsed, 2),
        "query_time": round(query_elapsed, 3),
        "metrics": metrics,
    }


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("data_file")
    ap.add_argument("--limit", type=int, default=20, help="Number of questions (time-boxed — default 20)")
    ap.add_argument("--engram-url", default="http://localhost:8000")
    ap.add_argument("--fetch-limit", type=int, default=200, help="Over-fetch size for the shared candidate pool")
    ap.add_argument("--top-sessions", default="3,5,10", help="Comma-separated stage-1 session counts to compare")
    ap.add_argument("--out", default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42, help="Random sample seed (the dataset is grouped by question_type, not shuffled)")
    args = ap.parse_args()

    top_sessions_variants = [int(x) for x in args.top_sessions.split(",")]

    data = load_data(args.data_file)
    if args.limit:
        # The dataset is grouped by question_type, not shuffled — a plain
        # slice would sample only one (often the easiest) category.
        data = random.Random(args.seed).sample(data, min(args.limit, len(data)))

    client = EngramClient(base_url=args.engram_url)
    sem = asyncio.Semaphore(args.workers)

    async def _bounded(entry: dict, i: int) -> dict:
        async with sem:
            project_name = f"_lme_m_hier_{entry['question_id']}"
            await client.create_project(project_name)
            row = await process_question(entry, client, project_name, top_sessions_variants, args.fetch_limit)
            print(
                f"[{i + 1:3}/{len(data)}] {row['question_id']:20} "
                f"stored={row['stored']:4} candidates={row['candidates_fetched']:3} "
                f"ingest={row['ingest_time']:6.1f}s query={row['query_time']:5.2f}s",
                flush=True,
            )
            return row

    t_start = time.monotonic()
    rows = await asyncio.gather(*[_bounded(e, i) for i, e in enumerate(data)])
    total_elapsed = time.monotonic() - t_start
    await client.close()

    if args.out:
        with open(args.out, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

    labels = ["flat"] + [f"hierarchical_top{ts}" for ts in top_sessions_variants]

    print("\n" + "=" * 78)
    print(f"  LME-M HIERARCHICAL EXPERIMENT — n={len(rows)}  total={total_elapsed:.1f}s")
    print("=" * 78)
    for label in labels:
        print(f"\n  {label}:")
        n = len(rows)
        for k in KS:
            ra = sum(r["metrics"][label][k]["recall_any"] for r in rows) / n
            rl = sum(r["metrics"][label][k]["recall_all"] for r in rows) / n
            ec = sum(r["metrics"][label][k]["evidence_completeness"] for r in rows) / n
            print(f"    R@{k:<2}: recall_any={ra:.3f}  recall_all={rl:.3f}  evidence_completeness={ec:.3f}")

    best_hier = max(labels[1:], key=lambda label: sum(r["metrics"][label][1]["recall_any"] for r in rows))
    flat_r1 = sum(r["metrics"]["flat"][1]["recall_any"] for r in rows) / len(rows)
    best_r1 = sum(r["metrics"][best_hier][1]["recall_any"] for r in rows) / len(rows)
    delta = (best_r1 - flat_r1) * 100
    print(f"\n  Best hierarchical config: {best_hier} (R@1 delta vs flat: {delta:+.1f}pp)")
    print(f"  Go/no-go bar: >= +3.0pp to productionize. {'GO' if delta >= 3.0 else 'NO-GO'}")


if __name__ == "__main__":
    asyncio.run(main())
