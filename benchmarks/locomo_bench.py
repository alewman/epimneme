#!/usr/bin/env python3
"""
Engram × LoCoMo Benchmark
============================

Evaluates engram's retrieval against the LoCoMo benchmark,
using the same methodology and metrics as MemPalace's benchmark
for like-for-like comparison.

10 conversations, ~200 QA pairs across 5 categories:
  1. Single-hop
  2. Temporal
  3. Temporal-inference
  4. Open-domain
  5. Adversarial

For each conversation:
1. Ingest all sessions into engram under a unique project
2. For each QA pair, query engram
3. Score retrieval recall against evidence dialog IDs

Usage:
    python benchmarks/locomo_bench.py data/locomo10.json
    python benchmarks/locomo_bench.py data/locomo10.json --limit 2
    python benchmarks/locomo_bench.py data/locomo10.json --top-k 10
"""

import asyncio
import json
import re
import argparse
import sys
import time
from pathlib import Path
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from engram_client import EngramClient

CATEGORIES = {
    1: "Single-hop",
    2: "Temporal",
    3: "Temporal-inference",
    4: "Open-domain",
    5: "Adversarial",
}


# =============================================================================
# DATA LOADING
# =============================================================================


def load_conversation_sessions(conversation: dict, session_summaries: dict | None = None) -> list[dict]:
    """Extract sessions from a LoCoMo conversation dict."""
    sessions = []
    session_num = 1
    while True:
        key = f"session_{session_num}"
        date_key = f"session_{session_num}_date_time"
        if key not in conversation:
            break
        dialogs = conversation[key]
        date = conversation.get(date_key, "")
        summary = ""
        if session_summaries:
            summary = session_summaries.get(f"session_{session_num}_summary", "")
        sessions.append({
            "session_num": session_num,
            "date": date,
            "dialogs": dialogs,
            "summary": summary,
        })
        session_num += 1
    return sessions


def build_corpus_from_sessions(
    sessions: list[dict], granularity: str = "session"
) -> tuple[list[str], list[str], list[str]]:
    """Build retrieval corpus from conversation sessions.

    granularity:
        'dialog'  — one doc per dialog turn (matches evidence format D1:3)
        'session' — one doc per session (all dialog text joined)
    """
    corpus = []
    corpus_ids = []
    corpus_timestamps = []

    for sess in sessions:
        if granularity == "session":
            texts = []
            for d in sess["dialogs"]:
                speaker = d.get("speaker", "?")
                text = d.get("text", "")
                texts.append(f'{speaker} said, "{text}"')
            doc = "\n".join(texts)
            corpus.append(doc)
            corpus_ids.append(f"session_{sess['session_num']}")
            corpus_timestamps.append(sess["date"])
        else:
            for d in sess["dialogs"]:
                dia_id = d.get("dia_id", f"D{sess['session_num']}:?")
                speaker = d.get("speaker", "?")
                text = d.get("text", "")
                doc = f'{speaker} said, "{text}"'
                corpus.append(doc)
                corpus_ids.append(dia_id)
                corpus_timestamps.append(sess["date"])

    return corpus, corpus_ids, corpus_timestamps


# =============================================================================
# EVIDENCE HELPERS
# =============================================================================


def evidence_to_dialog_ids(evidence: list[str]) -> set[str]:
    return set(evidence)


def evidence_to_session_ids(evidence: list[str]) -> set[str]:
    sessions = set()
    for eid in evidence:
        match = re.match(r"D(\d+):", eid)
        if match:
            sessions.add(f"session_{match.group(1)}")
    return sessions


def compute_retrieval_recall(retrieved_ids: list[str], evidence_ids: set[str]) -> float:
    """What fraction of evidence dialog IDs were retrieved?"""
    if not evidence_ids:
        return 1.0
    found = sum(1 for eid in evidence_ids if eid in retrieved_ids)
    return found / len(evidence_ids)


# =============================================================================
# ENGRAM INGEST + RETRIEVE
# =============================================================================


async def ingest_corpus(
    client: EngramClient,
    project_name: str,
    corpus: list[str],
    corpus_ids: list[str],
    corpus_timestamps: list[str],
    batch_size: int = 50,
) -> int:
    """Ingest corpus into engram. Returns count of stored memories."""
    await client.create_project(project_name)

    stored = 0
    for i in range(0, len(corpus), batch_size):
        batch_docs = corpus[i : i + batch_size]
        batch_ids = corpus_ids[i : i + batch_size]
        batch_ts = corpus_timestamps[i : i + batch_size]

        memories = []
        for doc, cid, ts in zip(batch_docs, batch_ids, batch_ts):
            memories.append({
                "content": doc,
                "kind": "fact",
                "subject": cid,
                "tags": ["benchmark", "locomo", ts] if ts else ["benchmark", "locomo"],
            })

        result = await client.bulk_create(memories, project=project_name)
        stored += result.get("stored", 0)

    return stored


async def retrieve(
    client: EngramClient,
    project_name: str,
    query: str,
    limit: int = 50,
) -> list[dict]:
    """Query engram and return results."""
    result = await client.search(query, project=project_name, limit=limit)
    return result.get("results", [])


# =============================================================================
# BENCHMARK RUNNER
# =============================================================================


async def process_qa(
    client: EngramClient,
    project_name: str,
    sample_id: str,
    qa: dict,
    top_k: int,
    granularity: str,
    semaphore: asyncio.Semaphore,
) -> dict:
    """Process a single QA pair. Returns result dict."""
    async with semaphore:
        question = qa["question"]
        answer = qa.get("answer", qa.get("adversarial_answer", ""))
        category = qa["category"]
        evidence = qa.get("evidence", [])

        t0 = time.monotonic()
        results = await retrieve(client, project_name, question, limit=top_k)
        query_elapsed = time.monotonic() - t0

        retrieved_ids = []
        for r in results:
            cid = r.get("subject", "")
            if cid:
                retrieved_ids.append(cid)

        if granularity == "dialog":
            evidence_set = evidence_to_dialog_ids(evidence)
        else:
            evidence_set = evidence_to_session_ids(evidence)

        recall = compute_retrieval_recall(retrieved_ids, evidence_set)

        return {
            "sample_id": sample_id,
            "question": question,
            "answer": answer,
            "category": category,
            "evidence": evidence,
            "retrieved_ids": retrieved_ids[:top_k],
            "recall": recall,
            "query_time": round(query_elapsed, 3),
        }


async def run_benchmark(
    data_file: str,
    engram_url: str = "http://localhost:8000",
    top_k: int = 50,
    limit: int = 0,
    granularity: str = "session",
    out_file: str | None = None,
    cleanup: bool = True,
    workers: int = 8,
    skip_ingest: bool = False,
):
    """Run the LoCoMo retrieval benchmark against engram."""
    with open(data_file) as f:
        data = json.load(f)

    if limit > 0:
        data = data[:limit]

    client = EngramClient(base_url=engram_url)

    # --skip-ingest: fetch existing projects once, disable cleanup to protect staged data
    existing_projects: set[str] | None = None
    if skip_ingest:
        if cleanup:
            print("  WARNING: --skip-ingest with cleanup enabled would delete staged data — disabling cleanup.")
            cleanup = False
        projects = await client.list_projects()
        existing_projects = {p["name"] for p in projects}
        staged = sum(1 for s in data if f"_locomo_bench_{s.get('sample_id', '')}" in existing_projects)
        print(f"  Skip-ingest: {staged}/{len(data)} conversations already staged")

    print(f"\n{'=' * 60}")
    print("  Engram × LoCoMo Benchmark")
    print(f"{'=' * 60}")
    print(f"  Data:          {Path(data_file).name}")
    print(f"  Conversations: {len(data)}")
    print(f"  Top-k:         {top_k}")
    print(f"  Granularity:   {granularity}")
    print(f"  Workers:       {workers}")
    print(f"  Engram:        {engram_url}")
    if skip_ingest:
        print(f"  Skip ingest:   yes ({staged}/{len(data)} cached)")
    print(f"{'─' * 60}\n")

    all_recall: list[float] = []
    per_category: dict[int, list[float]] = defaultdict(list)
    results_log = []
    total_qa = 0
    total_ingest_time = 0.0
    total_query_time = 0.0

    start_time = datetime.now()
    semaphore = asyncio.Semaphore(workers)

    try:
        for conv_idx, sample in enumerate(data):
            sample_id = sample.get("sample_id", f"conv-{conv_idx}")
            conversation = sample["conversation"]
            qa_pairs = sample["qa"]
            session_summaries = sample.get("session_summary", {})

            sessions = load_conversation_sessions(conversation, session_summaries)
            corpus, corpus_ids, corpus_timestamps = build_corpus_from_sessions(
                sessions, granularity=granularity
            )

            project_name = f"_locomo_bench_{sample_id}"

            print(
                f"  [{conv_idx + 1}/{len(data)}] {sample_id}: "
                f"{len(sessions)} sessions, {len(corpus)} docs, {len(qa_pairs)} questions"
            )

            # Ingest conversation into engram — skip if already staged
            cached = existing_projects is not None and project_name in existing_projects
            if cached:
                stored = 0
                ingest_elapsed = 0.0
                print(f"    Ingested: [CACHED]")
            else:
                t0 = time.monotonic()
                stored = await ingest_corpus(
                    client, project_name, corpus, corpus_ids, corpus_timestamps
                )
                ingest_elapsed = time.monotonic() - t0
                print(f"    Ingested: {stored}/{len(corpus)} memories in {ingest_elapsed:.1f}s")
            total_ingest_time += ingest_elapsed

            # Run all QA pairs concurrently
            qa_tasks = [
                process_qa(client, project_name, sample_id, qa, top_k, granularity, semaphore)
                for qa in qa_pairs
            ]
            qa_results = await asyncio.gather(*qa_tasks)

            for qr in qa_results:
                all_recall.append(qr["recall"])
                per_category[qr["category"]].append(qr["recall"])
                total_qa += 1
                total_query_time += qr["query_time"]
                results_log.append(qr)

            # Cleanup this conversation's data
            if cleanup:
                await client.clear_project(project_name)

    finally:
        await client.close()

    elapsed = (datetime.now() - start_time).total_seconds()

    if not all_recall:
        print("  No questions evaluated.")
        return

    avg_recall = sum(all_recall) / len(all_recall)

    print(f"\n{'=' * 60}")
    print(f"  RESULTS — Engram (hybrid retrieval, {granularity}, top-{top_k})")
    print(f"{'=' * 60}")
    print(f"  Time:        {elapsed:.1f}s ({elapsed / max(total_qa, 1):.2f}s per question)")
    print(f"    Ingest: {total_ingest_time:.1f}s  Query: {total_query_time:.1f}s")
    print(f"  Questions:   {total_qa}")
    print(f"  Avg Recall:  {avg_recall:.3f}")

    print("\n  PER-CATEGORY RECALL:")
    for cat in sorted(per_category.keys()):
        vals = per_category[cat]
        avg = sum(vals) / len(vals)
        name = CATEGORIES.get(cat, f"Cat-{cat}")
        print(f"    {name:25} R={avg:.3f}  (n={len(vals)})")

    perfect = sum(1 for r in all_recall if r >= 1.0)
    partial = sum(1 for r in all_recall if 0 < r < 1.0)
    zero = sum(1 for r in all_recall if r == 0)
    print("\n  RECALL DISTRIBUTION:")
    print(f"    Perfect (1.0):  {perfect:4} ({perfect / len(all_recall) * 100:.1f}%)")
    print(f"    Partial (0-1):  {partial:4} ({partial / len(all_recall) * 100:.1f}%)")
    print(f"    Zero (0.0):     {zero:4} ({zero / len(all_recall) * 100:.1f}%)")

    print(f"\n{'=' * 60}\n")

    if out_file:
        with open(out_file, "w") as f:
            json.dump(results_log, f, indent=2)
        print(f"  Results saved to: {out_file}")


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Engram × LoCoMo Benchmark")
    parser.add_argument("data_file", help="Path to locomo10.json")
    parser.add_argument("--top-k", type=int, default=50, help="Top-k retrieval (default: 50)")
    parser.add_argument(
        "--granularity",
        choices=["dialog", "session"],
        default="session",
        help="Corpus granularity: dialog (per turn) or session (per session)",
    )
    parser.add_argument("--limit", type=int, default=0, help="Limit to N conversations")
    parser.add_argument("--out", default=None, help="Output JSON file path")
    parser.add_argument(
        "--engram-url",
        default="http://localhost:8000",
        help="Engram API base URL",
    )
    parser.add_argument(
        "--no-cleanup",
        action="store_true",
        default=False,
        help="Don't delete benchmark data after each conversation",
    )
    parser.add_argument(
        "--skip-ingest",
        action="store_true",
        default=False,
        help=(
            "Skip ingest for conversations whose _locomo_bench_<id> project is already staged. "
            "Use with --no-cleanup to pre-stage once, then iterate queries only. "
            "Automatically disables cleanup to protect staged data."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of concurrent query workers (default: 8)",
    )
    args = parser.parse_args()

    if not args.out:
        args.out = (
            f"benchmarks/results_engram_locomo_{args.granularity}_top{args.top_k}"
            f"_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
        )

    asyncio.run(
        run_benchmark(
            data_file=args.data_file,
            engram_url=args.engram_url,
            top_k=args.top_k,
            limit=args.limit,
            granularity=args.granularity,
            out_file=args.out,
            cleanup=not args.no_cleanup,
            workers=args.workers,
            skip_ingest=args.skip_ingest,
        )
    )
