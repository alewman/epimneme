#!/usr/bin/env python3
"""
Engram × LongMemEval Benchmark
=================================

Evaluates engram's retrieval against the LongMemEval benchmark,
using the same methodology and metrics as MemPalace's benchmark
for like-for-like comparison.

For each of the 500 questions:
1. Ingest all haystack sessions into engram under a unique project
2. Query engram with the question
3. Score retrieval against ground-truth answer sessions
4. Clean up the project's data

Metrics: Recall@k, NDCG@k at session and turn level, per-type breakdown.

Usage:
    python benchmarks/longmemeval_bench.py data/longmemeval_s_cleaned.json
    python benchmarks/longmemeval_bench.py data/longmemeval_s_cleaned.json --limit 5
    python benchmarks/longmemeval_bench.py data/longmemeval_s_cleaned.json --granularity turn
    python benchmarks/longmemeval_bench.py data/longmemeval_s_cleaned.json --engram-url http://localhost:8000
"""

import asyncio
import json
import argparse
import sys
import os
import time
from pathlib import Path
from collections import defaultdict
from datetime import datetime

# Add benchmarks to path for sibling imports
sys.path.insert(0, str(Path(__file__).parent))

from epimneme_client import EngramClient
from metrics import evaluate_retrieval, session_id_from_corpus_id


# =============================================================================
# DATA LOADING
# =============================================================================


def download_dataset(dest_path: str) -> str:
    """Download LongMemEval dataset from HuggingFace if not present."""
    dest = Path(dest_path)
    if dest.exists():
        return str(dest)

    dest.parent.mkdir(parents=True, exist_ok=True)

    print("  Downloading LongMemEval dataset from HuggingFace...")
    print("  (This is a one-time download, ~50MB)")

    try:
        from datasets import load_dataset
        ds = load_dataset("dt-lindberg/LongMemEval", split="test")
        # Convert to the same JSON format MemPalace expects
        records = []
        for row in ds:
            records.append({
                "question_id": row.get("question_id", ""),
                "question_type": row.get("question_type", ""),
                "question": row.get("question", ""),
                "answer": row.get("answer", ""),
                "haystack_sessions": row.get("haystack_sessions", []),
                "haystack_session_ids": row.get("haystack_session_ids", []),
                "haystack_dates": row.get("haystack_dates", []),
                "answer_session_ids": row.get("answer_session_ids", []),
            })
        with open(dest, "w") as f:
            json.dump(records, f)
        print(f"  Saved {len(records)} questions → {dest}")
        return str(dest)
    except ImportError:
        print("  ERROR: 'datasets' package not installed.")
        print("  Install: pip install datasets")
        print("  Or manually download the dataset and pass its path.")
        sys.exit(1)


def load_data(data_file: str) -> list[dict]:
    """Load LongMemEval dataset from JSON file."""
    with open(data_file) as f:
        return json.load(f)


# =============================================================================
# CORPUS BUILDING
# =============================================================================


def build_corpus(entry: dict, granularity: str = "session") -> tuple[list[str], list[str], list[str]]:
    """Build corpus from haystack sessions.

    Returns (corpus_docs, corpus_ids, corpus_timestamps).
    Mirrors MemPalace's build_palace_and_retrieve() logic exactly.
    """
    corpus = []
    corpus_ids = []
    corpus_timestamps = []

    sessions = entry["haystack_sessions"]
    session_ids = entry["haystack_session_ids"]
    dates = entry["haystack_dates"]

    for sess_idx, (session, sess_id, date) in enumerate(zip(sessions, session_ids, dates)):
        if granularity == "session":
            # One document per session: join all user content
            user_turns = [t["content"] for t in session if t["role"] == "user"]
            if user_turns:
                doc = "\n".join(user_turns)
                corpus.append(doc)
                corpus_ids.append(sess_id)
                corpus_timestamps.append(date)
        elif granularity == "turn-pair":
            # One document per user+assistant exchange.
            # Includes assistant turns so assistant-authored facts are searchable.
            # Each chunk: header + user turn + assistant response (if any).
            turn_num = 0
            turns = session
            i = 0
            while i < len(turns):
                turn = turns[i]
                if turn["role"] == "user":
                    user_text = turn["content"]
                    asst_text = ""
                    if i + 1 < len(turns) and turns[i + 1]["role"] == "assistant":
                        asst_text = turns[i + 1]["content"]
                        i += 1  # consume assistant turn
                    header = f"[Date: {date}]"
                    if asst_text:
                        doc = f"{header}\n[USER]: {user_text}\n[ASSISTANT]: {asst_text}"
                    else:
                        doc = f"{header}\n[USER]: {user_text}"
                    corpus.append(doc)
                    corpus_ids.append(f"{sess_id}_turn_{turn_num}")
                    corpus_timestamps.append(date)
                    turn_num += 1
                elif turn["role"] == "assistant" and turn_num == 0:
                    # Session starts with an assistant turn (no preceding user turn)
                    doc = f"[Date: {date}]\n[ASSISTANT]: {turn['content']}"
                    corpus.append(doc)
                    corpus_ids.append(f"{sess_id}_turn_{turn_num}")
                    corpus_timestamps.append(date)
                    turn_num += 1
                i += 1
        else:
            # One document per user turn
            turn_num = 0
            for turn in session:
                if turn["role"] == "user":
                    corpus.append(turn["content"])
                    corpus_ids.append(f"{sess_id}_turn_{turn_num}")
                    corpus_timestamps.append(date)
                    turn_num += 1

    return corpus, corpus_ids, corpus_timestamps


# =============================================================================
# ENGRAM INGEST + RETRIEVE
# =============================================================================


async def ingest_corpus(
    client: EngramClient,
    project_name: str,
    corpus: list[str],
    corpus_ids: list[str],
    corpus_timestamps: list[str],
    batch_size: int = 100,
) -> int:
    """Ingest corpus into engram under project_name. Returns count of stored memories."""
    # Create project first
    await client.create_project(project_name)

    stored = 0
    # Batch ingest (engram limit is 100 per call)
    for i in range(0, len(corpus), batch_size):
        batch_docs = corpus[i : i + batch_size]
        batch_ids = corpus_ids[i : i + batch_size]
        batch_ts = corpus_timestamps[i : i + batch_size]

        memories = []
        for doc, cid, ts in zip(batch_docs, batch_ids, batch_ts):
            memories.append({
                "content": doc,
                "kind": "fact",
                "subject": cid,  # Store corpus_id as subject for traceability
                "tags": ["benchmark", "lme", ts],
            })

        result = await client.bulk_create(memories, project=project_name)
        stored += result.get("stored", 0)

    return stored


async def retrieve(
    client: EngramClient,
    project_name: str,
    query: str,
    limit: int = 50,
    reference_date: str | None = None,
) -> list[dict]:
    """Query engram and return results."""
    result = await client.search(query, project=project_name, limit=limit, reference_date=reference_date)
    return result.get("results", [])


async def cleanup_project(client: EngramClient, project_name: str) -> int:
    """Delete all memories in a benchmark project."""
    return await client.clear_project(project_name)


# =============================================================================
# BENCHMARK RUNNER
# =============================================================================


async def process_question(
    entry: dict,
    index: int,
    total: int,
    client: EngramClient,
    granularity: str,
    n_results: int,
    cleanup: bool,
    semaphore: asyncio.Semaphore,
    ks: list[int],
    existing_projects: set[str] | None = None,
) -> dict | None:
    """Process a single LME question. Returns result dict or None if skipped."""
    async with semaphore:
        qid = entry["question_id"]
        qtype = entry["question_type"]
        question = entry["question"]
        answer_sids = set(entry["answer_session_ids"])
        project_name = f"_lme_bench_{qid}"

        corpus, corpus_ids, corpus_timestamps = build_corpus(entry, granularity)
        if not corpus:
            print(f"  [{index:4}/{total}] {qid[:30]:30} SKIP (empty corpus)")
            return None

        # Ingest — skip if project is already staged (--skip-ingest mode)
        cached = existing_projects is not None and project_name in existing_projects
        if cached:
            stored = 0
            ingest_elapsed = 0.0
        else:
            t0 = time.monotonic()
            stored = await ingest_corpus(client, project_name, corpus, corpus_ids, corpus_timestamps)
            ingest_elapsed = time.monotonic() - t0

        # Query — pass question_date as reference_date so relative date expressions
        # ("last Friday", "5 days ago") resolve correctly against the haystack timeline
        # rather than the current server date (2026 vs 2023 benchmark dates).
        t0 = time.monotonic()
        question_date = entry.get("question_date", "")
        results = await retrieve(client, project_name, question, limit=n_results,
                                 reference_date=question_date or None)
        query_elapsed = time.monotonic() - t0

        # Map results back to corpus IDs
        ranked_ids = []
        for r in results:
            cid = r.get("subject", "")
            if cid:
                ranked_ids.append(cid)

        seen = set(ranked_ids)
        for cid in corpus_ids:
            if cid not in seen:
                ranked_ids.append(cid)

        session_level_ids = [session_id_from_corpus_id(cid) for cid in ranked_ids]
        session_correct = answer_sids

        turn_correct = set()
        for cid in corpus_ids:
            sid = session_id_from_corpus_id(cid)
            if sid in answer_sids:
                turn_correct.add(cid)

        session_metrics = {}
        turn_metrics = {}
        for k in ks:
            ra, rl, nd = evaluate_retrieval(session_level_ids, session_correct, k)
            session_metrics[k] = (ra, rl, nd)
            ra_t, rl_t, nd_t = evaluate_retrieval(ranked_ids, turn_correct, k)
            turn_metrics[k] = (ra_t, rl_t, nd_t)

        ranked_items = []
        for cid in ranked_ids[:50]:
            matching = [r for r in results if r.get("subject") == cid]
            ranked_items.append({
                "corpus_id": cid,
                "text": matching[0]["content"][:2000] if matching else "",
                "score": matching[0].get("score", 0) if matching else 0,
            })

        r1 = session_metrics[1][0]
        r5 = session_metrics[5][0]
        r10 = session_metrics[10][0]
        status = "HIT" if r1 > 0 else "miss"
        dedup_note = f" (stored {stored}/{len(corpus)})" if not cached and stored < len(corpus) else ""
        cache_note = " [CACHED]" if cached else ""
        print(
            f"  [{index:4}/{total}] {qid[:30]:30} "
            f"R@1={r1:.0f} R@5={r5:.0f} R@10={r10:.0f}  {status}  "
            f"[ingest {ingest_elapsed:.1f}s, query {query_elapsed:.2f}s{dedup_note}]{cache_note}"
        )

        cleanup_elapsed = 0.0
        if cleanup:
            t0 = time.monotonic()
            await cleanup_project(client, project_name)
            cleanup_elapsed = time.monotonic() - t0

        return {
            "question_id": qid,
            "question_type": qtype,
            "question": question,
            "answer": entry["answer"],
            "stored": stored,
            "ingest_time": round(ingest_elapsed, 2),
            "query_time": round(query_elapsed, 3),
            "cleanup_time": round(cleanup_elapsed, 3),
            "session_metrics": session_metrics,
            "turn_metrics": turn_metrics,
            "retrieval_results": {
                "query": question,
                "ranked_items": ranked_items,
                "metrics": {
                    "session": {f"recall_any@{k}": session_metrics[k][0] for k in ks},
                    "turn": {f"recall_any@{k}": turn_metrics[k][0] for k in ks},
                },
            },
        }


async def run_benchmark(
    data_file: str,
    engram_url: str = "http://localhost:8000",
    granularity: str = "session",
    limit: int = 0,
    skip: int = 0,
    out_file: str | None = None,
    cleanup: bool = True,
    n_results: int = 50,
    workers: int = 1,
    filter_ids: list[str] | None = None,
    skip_ingest: bool = False,
):
    """Run the full LongMemEval benchmark against engram."""
    data = load_data(data_file)

    if filter_ids:
        id_set = set(filter_ids)
        data = [e for e in data if e["question_id"] in id_set]
        print(f"  Filtered to {len(data)} questions by --filter-ids")
    if limit > 0:
        data = data[:limit]
    if skip > 0:
        print(f"  Skipping first {skip} questions (resume mode)")
        data = data[skip:]

    client = EngramClient(base_url=engram_url)

    # --skip-ingest: fetch existing projects once, disable cleanup to protect staged data
    existing_projects: set[str] | None = None
    if skip_ingest:
        if cleanup:
            print("  WARNING: --skip-ingest with cleanup enabled would delete staged data — disabling cleanup.")
            cleanup = False
        projects = await client.list_projects()
        existing_projects = {p["name"] for p in projects}
        staged = sum(1 for e in data if f"_lme_bench_{e['question_id']}" in existing_projects)
        print(f"  Skip-ingest: {staged}/{len(data)} questions already staged")

    print(f"\n{'=' * 60}")
    print("  Engram × LongMemEval Benchmark")
    print(f"{'=' * 60}")
    print(f"  Data:        {Path(data_file).name}")
    print(f"  Questions:   {len(data)}")
    print(f"  Granularity: {granularity}")
    print(f"  Workers:     {workers}")
    print(f"  Engram:      {engram_url}")
    print(f"  Cleanup:     {'yes' if cleanup else 'no (data persists)'}")
    if skip_ingest:
        print(f"  Skip ingest: yes ({staged}/{len(data)} cached)")
    print(f"{'─' * 60}\n")

    ks = [1, 3, 5, 10, 30, 50]
    start_time = datetime.now()

    semaphore = asyncio.Semaphore(workers)
    tasks = [
        process_question(
            entry, i + 1, len(data), client, granularity,
            n_results, cleanup, semaphore, ks, existing_projects,
        )
        for i, entry in enumerate(data)
    ]

    try:
        raw_results = await asyncio.gather(*tasks)
    finally:
        await client.close()

    # Filter out skipped (None) entries and sort by original index
    question_results = [r for r in raw_results if r is not None]

    # Aggregate metrics
    metrics_session = {f"recall_any@{k}": [] for k in ks}
    metrics_session.update({f"recall_all@{k}": [] for k in ks})
    metrics_session.update({f"ndcg_any@{k}": [] for k in ks})

    metrics_turn = {f"recall_any@{k}": [] for k in ks}
    metrics_turn.update({f"recall_all@{k}": [] for k in ks})
    metrics_turn.update({f"ndcg_any@{k}": [] for k in ks})

    per_type: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    results_log = []
    total_ingest_time = 0.0
    total_query_time = 0.0
    total_cleanup_time = 0.0

    for qr in question_results:
        total_ingest_time += qr["ingest_time"]
        total_query_time += qr["query_time"]
        total_cleanup_time += qr["cleanup_time"]
        qtype = qr["question_type"]

        for k in ks:
            ra, rl, nd = qr["session_metrics"][k]
            metrics_session[f"recall_any@{k}"].append(ra)
            metrics_session[f"recall_all@{k}"].append(rl)
            metrics_session[f"ndcg_any@{k}"].append(nd)

            ra_t, rl_t, nd_t = qr["turn_metrics"][k]
            metrics_turn[f"recall_any@{k}"].append(ra_t)
            metrics_turn[f"recall_all@{k}"].append(rl_t)
            metrics_turn[f"ndcg_any@{k}"].append(nd_t)

        per_type[qtype]["recall_any@1"].append(qr["session_metrics"][1][0])
        per_type[qtype]["recall_any@5"].append(qr["session_metrics"][5][0])
        per_type[qtype]["recall_any@10"].append(qr["session_metrics"][10][0])
        per_type[qtype]["ndcg_any@10"].append(qr["session_metrics"][10][2])

        results_log.append({
            "question_id": qr["question_id"],
            "question_type": qr["question_type"],
            "question": qr["question"],
            "answer": qr["answer"],
            "stored": qr["stored"],
            "ingest_time": qr["ingest_time"],
            "query_time": qr["query_time"],
            "retrieval_results": qr["retrieval_results"],
        })

    elapsed = (datetime.now() - start_time).total_seconds()
    n = len(metrics_session["recall_any@5"])

    if n == 0:
        print("  No questions evaluated.")
        return

    # Print results
    print(f"\n{'=' * 60}")
    print(f"  RESULTS — Engram (hybrid retrieval, {granularity} granularity)")
    print(f"{'=' * 60}")
    print(f"  Time: {elapsed:.1f}s ({elapsed / n:.2f}s per question)")
    print(f"    Ingest: {total_ingest_time:.1f}s  Query: {total_query_time:.1f}s  Cleanup: {total_cleanup_time:.1f}s\n")

    print("  SESSION-LEVEL METRICS:")
    for k in ks:
        ra = sum(metrics_session[f"recall_any@{k}"]) / n
        nd = sum(metrics_session[f"ndcg_any@{k}"]) / n
        print(f"    Recall@{k:2}: {ra:.3f}    NDCG@{k:2}: {nd:.3f}")

    print("\n  TURN-LEVEL METRICS:")
    for k in ks:
        ra = sum(metrics_turn[f"recall_any@{k}"]) / n
        nd = sum(metrics_turn[f"ndcg_any@{k}"]) / n
        print(f"    Recall@{k:2}: {ra:.3f}    NDCG@{k:2}: {nd:.3f}")

    print("\n  PER-TYPE BREAKDOWN (session recall_any@1 / @10):")
    for qtype, vals in sorted(per_type.items()):
        r1 = sum(vals["recall_any@1"]) / len(vals["recall_any@1"])
        r10 = sum(vals["recall_any@10"]) / len(vals["recall_any@10"])
        count = len(vals["recall_any@10"])
        print(f"    {qtype:35} R@1={r1:.3f}  R@10={r10:.3f}  (n={count})")

    print(f"\n{'=' * 60}\n")

    # Save results
    if out_file:
        with open(out_file, "w") as f:
            for entry in results_log:
                f.write(json.dumps(entry) + "\n")
        print(f"  Results saved to: {out_file}")


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Engram × LongMemEval Benchmark")
    parser.add_argument("data_file", help="Path to longmemeval_s_cleaned.json")
    parser.add_argument(
        "--granularity",
        choices=["session", "turn", "turn-pair"],
        default="session",
        help="Retrieval granularity (default: session). turn-pair indexes each user+assistant exchange as one document.",
    )
    parser.add_argument(
        "--filter-ids",
        nargs="+",
        default=None,
        metavar="QID",
        help="Only run questions with these IDs (space-separated). Useful for targeted validation.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Limit to N questions (0 = all)")
    parser.add_argument(
        "--skip", type=int, default=0, help="Skip first N questions (resume mode)"
    )
    parser.add_argument("--out", default=None, help="Output JSONL file path")
    parser.add_argument(
        "--engram-url",
        default="http://localhost:8000",
        help="Engram API base URL (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--no-cleanup",
        action="store_true",
        default=False,
        help="Don't delete benchmark data after each question (useful for debugging)",
    )
    parser.add_argument(
        "--skip-ingest",
        action="store_true",
        default=False,
        help=(
            "Skip ingest for questions whose _lme_bench_<qid> project is already staged. "
            "Use with --no-cleanup to pre-stage once, then iterate queries in ~2min instead of ~40min. "
            "Automatically disables cleanup to protect staged data."
        ),
    )
    parser.add_argument(
        "--n-results",
        type=int,
        default=50,
        help="Number of results to retrieve per query (default: 50)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of concurrent workers (default: 8)",
    )
    args = parser.parse_args()

    if not args.out:
        args.out = (
            f"benchmarks/results_engram_lme_{args.granularity}"
            f"_{datetime.now().strftime('%Y%m%d_%H%M')}.jsonl"
        )

    asyncio.run(
        run_benchmark(
            data_file=args.data_file,
            engram_url=args.engram_url,
            granularity=args.granularity,
            limit=args.limit,
            skip=args.skip,
            out_file=args.out,
            cleanup=not args.no_cleanup,
            n_results=args.n_results,
            workers=args.workers,
            filter_ids=args.filter_ids,
            skip_ingest=args.skip_ingest,
        )
    )
