#!/usr/bin/env python3
"""
Engram × BEIR Benchmark
========================

Evaluates Engram's retrieval against BEIR (Benchmarking Information Retrieval)
datasets (Thakur et al. 2021, arXiv:2104.08663).

Default dataset: SciFact — scientific claim fact-verification corpus
  - 5,183 documents (scientific abstracts)
  - 300 test queries (scientific claims)
  - Binary relevance judgments (0/1)

Other supported datasets (via --dataset):
  scifact, nfcorpus, arguana, scidocs, trec-covid, fiqa

For each run:
  1. Load corpus, queries, and qrels from HuggingFace
  2. Ingest corpus documents into Engram under project _beir_{dataset}
     Each document: subject=doc_id, content="[Title: ...]\n{text}"
  3. For each test query, retrieve top-100 documents from Engram
  4. Compute nDCG@10 and recall@100 following BEIR paper conventions

Skip-ingest mode: if the project already exists, skip step 2.
  Stage once:  python beir_bench.py --dataset scifact --no-cleanup
  Rerun fast:  python beir_bench.py --dataset scifact --skip-ingest

Usage:
    # Full run (ingest + query)
    python benchmarks/beir_bench.py \\
        --dataset scifact \\
        --engram-url http://192.168.90.45:8000

    # With output file
    python benchmarks/beir_bench.py --dataset scifact \\
        --out benchmarks/results_engram_beir_scifact_v402.json \\
        --engram-url http://192.168.90.45:8000

    # Smoke test (limit corpus to 200 docs, 20 queries)
    python benchmarks/beir_bench.py --dataset scifact --limit 200 --query-limit 20

    # Skip ingest if already staged
    python benchmarks/beir_bench.py --dataset scifact --skip-ingest
"""

from __future__ import annotations

import asyncio
import json
import argparse
import math
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

from epimneme_client import EngramClient

# ── Dataset registry ──────────────────────────────────────────────────────────
# HuggingFace dataset names for corpus, queries, and qrels.
# Qrels are in a separate {dataset}-qrels repo on HF.
DATASETS: dict[str, dict[str, str]] = {
    "scifact": {
        "corpus_hf":  "BeIR/scifact",
        "queries_hf": "BeIR/scifact",
        "qrels_hf":   "BeIR/scifact-qrels",
        "qrels_split": "test",
        "description": "Scientific claim verification (5,183 docs, 300 queries)",
    },
    "nfcorpus": {
        "corpus_hf":  "BeIR/nfcorpus",
        "queries_hf": "BeIR/nfcorpus",
        "qrels_hf":   "BeIR/nfcorpus-qrels",
        "qrels_split": "test",
        "description": "Nutrition/health information retrieval (3,633 docs, 323 queries)",
    },
    "arguana": {
        "corpus_hf":  "BeIR/arguana",
        "queries_hf": "BeIR/arguana",
        "qrels_hf":   "BeIR/arguana-qrels",
        "qrels_split": "test",
        "description": "Counter-argument retrieval (8,674 docs, 1,406 queries)",
    },
    "scidocs": {
        "corpus_hf":  "BeIR/scidocs",
        "queries_hf": "BeIR/scidocs",
        "qrels_hf":   "BeIR/scidocs-qrels",
        "qrels_split": "test",
        "description": "Scientific document similarity (25,657 docs, 1,000 queries)",
    },
    "fiqa": {
        "corpus_hf":  "BeIR/fiqa",
        "queries_hf": "BeIR/fiqa",
        "qrels_hf":   "BeIR/fiqa-qrels",
        "qrels_split": "test",
        "description": "Financial opinion QA (57,638 docs, 648 queries)",
    },
    "trec-covid": {
        "corpus_hf":  "BeIR/trec-covid",
        "queries_hf": "BeIR/trec-covid",
        "qrels_hf":   "BeIR/trec-covid-qrels",
        "qrels_split": "test",
        "description": "COVID-19 literature retrieval (171,332 docs, 50 queries)",
    },
}

RETRIEVE_LIMIT = 100  # standard BEIR retrieval depth


# =============================================================================
# DATA LOADING
# =============================================================================


def load_beir_dataset(
    dataset: str,
    corpus_limit: int = 0,
    query_limit: int = 0,
) -> tuple[list[dict], list[dict], dict[str, dict[str, int]]]:
    """Load corpus, queries, and qrels from HuggingFace.

    Returns:
        corpus:  list of {"_id": str, "title": str, "text": str}
        queries: list of {"_id": str, "text": str}
        qrels:   {query_id: {doc_id: relevance_score}}
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: 'datasets' package required. Run: pip install datasets")
        sys.exit(1)

    cfg = DATASETS[dataset]
    print(f"  Loading corpus from {cfg['corpus_hf']} …", flush=True)
    corpus_ds = load_dataset(cfg["corpus_hf"], "corpus", split="corpus", trust_remote_code=True)
    corpus = [{"_id": r["_id"], "title": r.get("title", ""), "text": r["text"]} for r in corpus_ds]
    if corpus_limit > 0:
        corpus = corpus[:corpus_limit]
    print(f"  Corpus: {len(corpus):,} documents", flush=True)

    print(f"  Loading queries from {cfg['queries_hf']} …", flush=True)
    queries_ds = load_dataset(cfg["queries_hf"], "queries", split="queries", trust_remote_code=True)
    queries = [{"_id": r["_id"], "text": r["text"]} for r in queries_ds]

    print(f"  Loading qrels from {cfg['qrels_hf']} …", flush=True)
    qrels_ds = load_dataset(cfg["qrels_hf"], split=cfg["qrels_split"], trust_remote_code=True)

    # Build qrel map: {query_id: {doc_id: score}}
    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    for row in qrels_ds:
        qid = str(row["query-id"])
        did = str(row["corpus-id"])
        score = int(row["score"])
        qrels[qid][did] = score

    # Filter queries to those with qrels (test set)
    queries = [q for q in queries if q["_id"] in qrels]
    if query_limit > 0:
        queries = queries[:query_limit]
    print(f"  Queries with qrels: {len(queries)}", flush=True)

    return corpus, queries, dict(qrels)


# =============================================================================
# METRICS
# =============================================================================


def ndcg_at_k(retrieved: list[str], qrels: dict[str, int], k: int) -> float:
    """Compute nDCG@k (standard BEIR metric)."""
    dcg = sum(
        qrels.get(doc_id, 0) / math.log2(rank + 2)
        for rank, doc_id in enumerate(retrieved[:k])
    )
    ideal = sorted(qrels.values(), reverse=True)[:k]
    idcg = sum(rel / math.log2(rank + 2) for rank, rel in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(retrieved: list[str], qrels: dict[str, int], k: int) -> float:
    """Compute recall@k."""
    relevant = {did for did, score in qrels.items() if score > 0}
    if not relevant:
        return 0.0
    return len(set(retrieved[:k]) & relevant) / len(relevant)


def precision_at_k(retrieved: list[str], qrels: dict[str, int], k: int) -> float:
    """Compute precision@k."""
    relevant = {did for did, score in qrels.items() if score > 0}
    if not relevant:
        return 0.0
    return len(set(retrieved[:k]) & relevant) / k


# =============================================================================
# INGEST
# =============================================================================


async def ingest_corpus(
    client: EngramClient,
    project_name: str,
    corpus: list[dict],
    batch_size: int = 100,
) -> int:
    """Ingest BEIR corpus documents into Engram. Returns count stored."""
    await client.create_project(project_name)

    stored = 0
    t0 = time.monotonic()

    for i in range(0, len(corpus), batch_size):
        batch = corpus[i : i + batch_size]
        memories = []
        for doc in batch:
            title = doc.get("title", "").strip()
            text = doc.get("text", "").strip()
            content = f"[Title: {title}]\n{text}" if title else text
            memories.append({"content": content, "subject": doc["_id"], "kind": "fact"})

        result = await client.bulk_create(memories, project=project_name)
        stored += result.get("stored", len(batch))

        pct = min(100, int((i + len(batch)) / len(corpus) * 100))
        elapsed = time.monotonic() - t0
        print(
            f"  Ingest {i + len(batch):,}/{len(corpus):,}  ({pct}%)  {elapsed:.1f}s",
            end="\r",
            flush=True,
        )

    elapsed = time.monotonic() - t0
    print(f"  Ingested {stored:,}/{len(corpus):,} documents in {elapsed:.1f}s      ", flush=True)
    return stored


# =============================================================================
# RETRIEVAL
# =============================================================================


async def retrieve_query(
    client: EngramClient,
    project_name: str,
    query_text: str,
    limit: int = RETRIEVE_LIMIT,
) -> list[str]:
    """Query Engram and return retrieved doc IDs (subject field), deduplicated."""
    result = await client.search(query_text, project=project_name, limit=limit)
    seen: set[str] = set()
    retrieved: list[str] = []
    for r in result.get("results", []):
        doc_id = r.get("subject", "")
        if doc_id and doc_id not in seen:
            seen.add(doc_id)
            retrieved.append(doc_id)
    return retrieved


async def run_queries(
    client: EngramClient,
    project_name: str,
    queries: list[dict],
    qrels: dict[str, dict[str, int]],
    workers: int = 4,
) -> list[dict]:
    """Run all queries concurrently. Returns per-query result dicts."""
    semaphore = asyncio.Semaphore(workers)
    results: list[dict] = []

    async def run_one(q: dict, idx: int) -> dict:
        async with semaphore:
            t0 = time.monotonic()
            try:
                retrieved = await retrieve_query(client, project_name, q["text"])
            except Exception as exc:
                print(
                    f"  [{idx+1:3d}/{len(queries)}]  !  ERROR: {exc}  "
                    f"{q['text'][:60]}",
                    flush=True,
                )
                return {
                    "query_id": q["_id"],
                    "query_text": q["text"],
                    "retrieved_ids": [],
                    "n_relevant": sum(1 for s in qrels.get(q["_id"], {}).values() if s > 0),
                    "n_found_at_100": 0,
                    "ndcg@10": 0.0,
                    "recall@100": 0.0,
                    "precision@10": 0.0,
                    "query_time": time.monotonic() - t0,
                    "error": str(exc),
                }
            elapsed = time.monotonic() - t0

            q_qrels = qrels.get(q["_id"], {})
            ndcg10 = ndcg_at_k(retrieved, q_qrels, 10)
            r100 = recall_at_k(retrieved, q_qrels, 100)
            p10 = precision_at_k(retrieved, q_qrels, 10)

            n_rel = sum(1 for s in q_qrels.values() if s > 0)
            n_found = sum(1 for did in retrieved[:100] if q_qrels.get(did, 0) > 0)
            status = "✓" if n_found > 0 else "✗"

            print(
                f"  [{idx+1:3d}/{len(queries)}]  {status}  "
                f"nDCG@10={ndcg10:.3f}  R@100={r100:.3f}  "
                f"({n_found}/{n_rel} rel)  {elapsed:.2f}s  "
                f"{q['text'][:60]}",
                flush=True,
            )

            return {
                "query_id": q["_id"],
                "query_text": q["text"],
                "retrieved_ids": retrieved,
                "n_relevant": n_rel,
                "n_found_at_100": n_found,
                "ndcg@10": ndcg10,
                "recall@100": r100,
                "precision@10": p10,
                "query_time": elapsed,
            }

    tasks = [run_one(q, i) for i, q in enumerate(queries)]
    results = await asyncio.gather(*tasks)
    return list(results)


# =============================================================================
# MAIN
# =============================================================================


async def main(args: argparse.Namespace) -> None:
    dataset = args.dataset
    if dataset not in DATASETS:
        print(f"ERROR: Unknown dataset '{dataset}'. Choose from: {', '.join(DATASETS)}")
        sys.exit(1)

    cfg = DATASETS[dataset]
    project_name = f"_beir_{dataset.replace('-', '_')}"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    print("=" * 60)
    print("  Engram × BEIR Benchmark")
    print("=" * 60)
    print(f"  Dataset:   {dataset}  ({cfg['description']})")
    print(f"  Project:   {project_name}")
    print(f"  Engram:    {args.engram_url}")
    print(f"  Workers:   {args.workers}")
    if args.skip_ingest:
        print("  Ingest:    SKIP (using staged data)")
    if args.no_cleanup:
        print("  Cleanup:   disabled (staging mode)")
    print("-" * 60)

    # ── Load data ─────────────────────────────────────────────────────────────
    corpus, queries, qrels = load_beir_dataset(
        dataset,
        corpus_limit=args.limit,
        query_limit=args.query_limit,
    )

    # ── Connect ───────────────────────────────────────────────────────────────
    client = EngramClient(base_url=args.engram_url)

    try:
        # ── Ingest ────────────────────────────────────────────────────────────
        if args.skip_ingest:
            projects = await client.list_projects()
            existing = {p["name"] for p in projects}
            if project_name in existing:
                print(f"  Skip-ingest: project '{project_name}' already staged", flush=True)
            else:
                print(
                    f"  WARNING: --skip-ingest specified but project '{project_name}' "
                    f"not found — ingesting now.",
                    flush=True,
                )
                await ingest_corpus(client, project_name, corpus)
        else:
            t_ingest = time.monotonic()
            stored = await ingest_corpus(client, project_name, corpus)
            print(f"  Ingest time: {time.monotonic() - t_ingest:.1f}s", flush=True)

        # ── Query ─────────────────────────────────────────────────────────────
        print(f"\n  Querying {len(queries)} test queries (workers={args.workers}) …\n", flush=True)
        t_query = time.monotonic()
        per_query = await run_queries(client, project_name, queries, qrels, workers=args.workers)
        query_time = time.monotonic() - t_query

        # ── Aggregate metrics ─────────────────────────────────────────────────
        n = len(per_query)
        avg_ndcg10  = sum(r["ndcg@10"]     for r in per_query) / n
        avg_r100    = sum(r["recall@100"]  for r in per_query) / n
        avg_p10     = sum(r["precision@10"] for r in per_query) / n
        n_any_found = sum(1 for r in per_query if r["n_found_at_100"] > 0)

        print()
        print("=" * 60)
        print(f"\n  Dataset:      {dataset}  ({n} queries)")
        print(f"  nDCG@10:      {avg_ndcg10:.4f}")
        print(f"  Recall@100:   {avg_r100:.4f}")
        print(f"  Precision@10: {avg_p10:.4f}")
        print(f"  Any-hit@100:  {n_any_found}/{n}  ({n_any_found/n*100:.1f}%)")
        print(f"  Query time:   {query_time:.1f}s  ({query_time/n:.2f}s/query)")
        print()

        # ── Save results ──────────────────────────────────────────────────────
        output = {
            "meta": {
                "benchmark": "beir",
                "dataset": dataset,
                "engram_url": args.engram_url,
                "timestamp": timestamp,
                "n_corpus": len(corpus),
                "n_queries": n,
                "ndcg@10": avg_ndcg10,
                "recall@100": avg_r100,
                "precision@10": avg_p10,
                "any_hit_at_100": n_any_found / n,
                "query_time_total": query_time,
            },
            "per_query": per_query,
        }

        out_path = args.out or f"benchmarks/results_engram_beir_{dataset}_{timestamp}.json"
        Path(out_path).write_text(json.dumps(output, indent=2))
        print(f"  Results saved to: {out_path}", flush=True)

    finally:
        if not args.no_cleanup and not args.skip_ingest:
            print(f"\n  Cleaning up project '{project_name}' …", flush=True)
            deleted = await client.clear_project(project_name)
            print(f"  Deleted {deleted} memories", flush=True)

        await client.close()


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Engram × BEIR retrieval benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--dataset", default="scifact",
        choices=list(DATASETS),
        help="BEIR dataset to run (default: scifact)",
    )
    p.add_argument(
        "--engram-url", default="http://localhost:8000",
        help="Engram API base URL",
    )
    p.add_argument(
        "--out", default="",
        help="Output JSON file path (default: auto-named in benchmarks/)",
    )
    p.add_argument(
        "--workers", type=int, default=4,
        help="Concurrent query workers (default: 4)",
    )
    p.add_argument(
        "--limit", type=int, default=0,
        help="Limit corpus to first N documents (0=all, for smoke testing)",
    )
    p.add_argument(
        "--query-limit", type=int, default=0,
        help="Limit to first N queries (0=all, for smoke testing)",
    )
    p.add_argument(
        "--skip-ingest", action="store_true",
        help="Skip corpus ingest if project already staged (fast re-run mode). "
             "Implies --no-cleanup.",
    )
    p.add_argument(
        "--no-cleanup", action="store_true",
        help="Keep project data after benchmark (enables future --skip-ingest)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.skip_ingest:
        args.no_cleanup = True
    asyncio.run(main(args))
