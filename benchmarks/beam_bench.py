#!/usr/bin/env python3
"""
Engram × BEAM Benchmark
========================

Evaluates Engram's retrieval against the BEAM benchmark (arXiv:2510.27246,
ICLR 2026). BEAM: 100 diverse conversations × 20 probing questions = 2000
questions spanning 10 memory abilities.

Conversation sizes available from HuggingFace (Mohammadta/BEAM):
  100K — 20 conversations
  500K — 35 conversations
  1M   — 35 conversations
  (10M — separate dataset: Mohammadta/BEAM-10M, excluded by default)

For each conversation:
  1. Flatten all chat turns (sequential id 0, 1, 2, …)
  2. Ingest each turn into Engram as a memory (subject = str(turn_id))
  3. For each probing question with source evidence (source_chat_ids):
       - Extract target turn IDs
       - Query Engram with the probing question
       - Compute recall: fraction of evidence turns found in top-K
  4. Abstention questions (no source_chat_ids) are excluded from recall scoring

Metric: Retrieval recall per ability and overall, following LoCoMo convention.

Usage:
    # Run full benchmark (100K + 500K + 1M splits)
    python benchmarks/beam_bench.py --engram-url http://192.168.90.45:8000

    # Single split, quick smoke-test
    python benchmarks/beam_bench.py --engram-url http://192.168.90.45:8000 \\
        --split 100K --limit 3

    # Save results
    python benchmarks/beam_bench.py --engram-url http://192.168.90.45:8000 \\
        --split 100K \\
        --out benchmarks/results_engram_beam_100k_v120_$(date +%Y%m%d_%H%M).json
"""

import asyncio
import json
import argparse
import sys
import time
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))

# Ensure stdout is line-buffered so tee/pipe shows progress as it happens
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

from epimneme_client import EngramClient

# ── HuggingFace dataset IDs ───────────────────────────────────────────────────
HF_DATASET = "Mohammadta/BEAM"
HF_DATASET_10M = "Mohammadta/BEAM-10M"

# ── 10 BEAM memory abilities ──────────────────────────────────────────────────
ABILITIES = [
    "abstention",
    "contradiction_resolution",
    "event_ordering",
    "information_extraction",
    "instruction_following",
    "knowledge_update",
    "multi_session_reasoning",
    "preference_following",
    "summarization",
    "temporal_reasoning",
]

# Abilities where source_chat_ids may be absent (no supporting evidence by design)
EVIDENCE_FREE_ABILITIES = {"abstention"}


# =============================================================================
# DATA LOADING
# =============================================================================


def load_beam_dataset(split: str = "100K", limit: int = 0) -> list[dict]:
    """Download BEAM from HuggingFace and return as a list of row dicts."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: 'datasets' package not installed. Run: pip install datasets")
        sys.exit(1)

    print(f"  Loading BEAM split '{split}' from {HF_DATASET} …", flush=True)
    ds = load_dataset(HF_DATASET, split=split)
    rows = list(ds)
    if limit > 0:
        rows = rows[:limit]
    return rows


def flatten_chat(chat: list) -> dict[int, dict]:
    """Flatten nested chat (list-of-sessions, each a list-of-turns) into
    a dict keyed by turn id.

    Each turn looks like:
        {"id": 28, "role": "user", "content": "...", "time_anchor": "...", ...}
    """
    by_id: dict[int, dict] = {}
    for session in chat:
        if isinstance(session, list):
            turns = session
        else:
            turns = [session]
        for turn in turns:
            if isinstance(turn, dict):
                tid = turn.get("id")
                if tid is not None:
                    by_id[tid] = turn
    return by_id


def extract_source_ids(source_chat_ids: Any) -> set[int]:
    """Normalise source_chat_ids from any of its observed shapes:
        - list[int]:  [28]                  → {28}
        - dict:       {"first": [58], ...}  → {58, ...}
        - None/empty: None or []            → {}
    """
    if not source_chat_ids:
        return set()
    if isinstance(source_chat_ids, list):
        return {int(v) for v in source_chat_ids if isinstance(v, (int, float))}
    if isinstance(source_chat_ids, dict):
        ids: set[int] = set()
        for v in source_chat_ids.values():
            if isinstance(v, list):
                ids.update(int(x) for x in v if isinstance(x, (int, float)))
            elif isinstance(v, (int, float)):
                ids.add(int(v))
        return ids
    return set()


def iter_probing_questions(pq: Any) -> list[tuple[str, dict]]:
    """Yield (ability, question_dict) pairs from the probing_questions field."""
    pairs: list[tuple[str, dict]] = []
    if isinstance(pq, str):
        import ast
        try:
            pq = ast.literal_eval(pq)
        except Exception:
            return pairs
    if not isinstance(pq, dict):
        return pairs

    for ability, questions in pq.items():
        if isinstance(questions, str):
            import ast
            try:
                questions = ast.literal_eval(questions)
            except Exception:
                questions = []
        if isinstance(questions, list):
            for q in questions:
                if isinstance(q, dict):
                    pairs.append((ability, q))
    return pairs


# =============================================================================
# ENGRAM INGEST
# =============================================================================


async def ingest_turns(
    client: EngramClient,
    project_name: str,
    turn_map: dict[int, dict],
    batch_size: int = 50,
) -> int:
    """Ingest chat turns into Engram. Returns count of stored memories."""
    await client.create_project(project_name)

    turns = sorted(turn_map.items())  # sorted by id for predictable ingest order
    stored = 0

    for i in range(0, len(turns), batch_size):
        batch = turns[i : i + batch_size]
        memories = []
        for tid, turn in batch:
            role = turn.get("role", "")
            content = turn.get("content", "")
            time_anchor = turn.get("time_anchor", "")
            text = f"{role}: {content}" if role else content

            tags = ["benchmark", "beam"]
            if time_anchor:
                tags.append(time_anchor)

            memories.append({
                "content": text,
                "kind": "fact",
                "subject": str(tid),
                "tags": tags,
            })

        result = await client.bulk_create(memories, project=project_name)
        stored += result.get("stored", 0)

    return stored


# =============================================================================
# BENCHMARK QUERY
# =============================================================================


async def query_beam(
    client: EngramClient,
    project_name: str,
    question: str,
    limit: int,
) -> list[str]:
    """Query Engram and return retrieved turn IDs (as strings), deduplicated by subject."""
    result = await client.search(question, project=project_name, limit=limit)
    retrieved = []
    seen: set[str] = set()
    for r in result.get("results", []):
        subj = r.get("subject", "")
        if subj and subj not in seen:
            seen.add(subj)
            retrieved.append(subj)
    return retrieved


def compute_recall(retrieved_ids: list[str], target_ids: set[int]) -> float:
    """Fraction of target turn IDs found anywhere in retrieved_ids."""
    if not target_ids:
        return 1.0
    retrieved_set = {s for s in retrieved_ids}
    found = sum(1 for tid in target_ids if str(tid) in retrieved_set)
    return found / len(target_ids)


async def process_question(
    client: EngramClient,
    project_name: str,
    conversation_id: str,
    ability: str,
    question_dict: dict,
    top_k: int,
    semaphore: asyncio.Semaphore,
) -> dict:
    """Process one probing question. Returns result dict."""
    async with semaphore:
        question = question_dict.get("question", "")
        source_ids = extract_source_ids(question_dict.get("source_chat_ids"))
        rubric = question_dict.get("rubric", [])

        # Skip retrieval scoring for evidence-free abilities
        if ability in EVIDENCE_FREE_ABILITIES or not source_ids:
            return {
                "conversation_id": conversation_id,
                "ability": ability,
                "question": question,
                "source_ids": sorted(source_ids),
                "retrieved_ids": [],
                "recall": None,  # N/A — no expected evidence
                "query_time": 0.0,
                "rubric": rubric,
            }

        t0 = time.monotonic()
        retrieved = await query_beam(client, project_name, question, limit=top_k)
        elapsed = time.monotonic() - t0

        recall = compute_recall(retrieved, source_ids)

        return {
            "conversation_id": conversation_id,
            "ability": ability,
            "question": question,
            "source_ids": sorted(source_ids),
            "retrieved_ids": retrieved[:top_k],
            "recall": recall,
            "query_time": round(elapsed, 3),
            "rubric": rubric,
        }


# =============================================================================
# BENCHMARK RUNNER
# =============================================================================


async def run_benchmark(
    split: str = "100K",
    engram_url: str = "http://localhost:8000",
    token: str = "",
    top_k: int = 10,
    limit: int = 0,
    out_file: str | None = None,
    cleanup: bool = True,
    workers: int = 8,
    skip_ingest: bool = False,
) -> None:
    """Run the BEAM retrieval benchmark against Engram."""

    rows = load_beam_dataset(split, limit)
    client = EngramClient(base_url=engram_url, token=token)

    # --skip-ingest: fetch existing projects once, disable cleanup to protect staged data
    existing_projects: set[str] | None = None
    if skip_ingest:
        if cleanup:
            print("  WARNING: --skip-ingest with cleanup enabled would delete staged data — disabling cleanup.")
            cleanup = False
        projects = await client.list_projects()
        existing_projects = {p["name"] for p in projects}
        staged = sum(1 for r in rows if f"_beam_bench_{r.get('conversation_id', '')}" in existing_projects)
        print(f"  Skip-ingest: {staged}/{len(rows)} conversations already staged")

    print(f"\n{'=' * 60}")
    print("  Engram × BEAM Benchmark")
    print(f"{'=' * 60}")
    print(f"  Split:         {split}")
    print(f"  Conversations: {len(rows)}")
    print(f"  Top-k:         {top_k}")
    print(f"  Workers:       {workers}")
    print(f"  Engram:        {engram_url}")
    if skip_ingest:
        print(f"  Skip ingest:   yes ({staged}/{len(rows)} cached)")
    print(f"{'─' * 60}\n")

    all_recall: list[float] = []
    per_ability: dict[str, list[float]] = defaultdict(list)
    results_log: list[dict] = []
    total_ingest_time = 0.0
    total_query_time = 0.0

    start_time = datetime.now()
    semaphore = asyncio.Semaphore(workers)

    try:
        for conv_idx, row in enumerate(rows):
            conv_id = row.get("conversation_id", f"conv-{conv_idx}")
            chat = row.get("chat", [])
            pq = row.get("probing_questions", {})

            turn_map = flatten_chat(chat)
            qa_pairs = iter_probing_questions(pq)

            # Count scoreable questions (those with evidence)
            scoreable = [
                (ab, q) for ab, q in qa_pairs
                if ab not in EVIDENCE_FREE_ABILITIES and extract_source_ids(q.get("source_chat_ids"))
            ]
            skipped = len(qa_pairs) - len(scoreable)

            project_name = f"_beam_bench_{conv_id}"

            seed = row.get("conversation_seed", {})
            title = seed.get("title", "") if isinstance(seed, dict) else ""
            title_short = (title[:45] + "…") if len(title) > 45 else title

            print(
                f"  [{conv_idx + 1}/{len(rows)}] {conv_id}: "
                f"{len(turn_map)} turns, {len(qa_pairs)} questions "
                f"({skipped} skipped)"
            )
            if title_short:
                print(f"    Topic: {title_short}")

            # Ingest — skip if already staged
            cached = existing_projects is not None and project_name in existing_projects
            if cached:
                stored = 0
                ingest_elapsed = 0.0
                print(f"    Ingested: [CACHED]")
            else:
                t0 = time.monotonic()
                stored = await ingest_turns(client, project_name, turn_map)
                ingest_elapsed = time.monotonic() - t0
                print(f"    Ingested: {stored}/{len(turn_map)} turns in {ingest_elapsed:.1f}s")
            total_ingest_time += ingest_elapsed

            # Run queries concurrently
            tasks = [
                process_question(
                    client, project_name, conv_id, ab, q, top_k, semaphore
                )
                for ab, q in qa_pairs
            ]
            results = await asyncio.gather(*tasks)

            for r in results:
                results_log.append(r)
                if r["recall"] is not None:
                    all_recall.append(r["recall"])
                    per_ability[r["ability"]].append(r["recall"])
                    total_query_time += r["query_time"]

            if cleanup:
                await client.clear_project(project_name)

    finally:
        await client.close()

    elapsed = (datetime.now() - start_time).total_seconds()

    if not all_recall:
        print("\n  No scoreable questions found.")
        return

    avg_recall = sum(all_recall) / len(all_recall)
    perfect = sum(1 for r in all_recall if r >= 1.0)
    partial = sum(1 for r in all_recall if 0 < r < 1.0)
    zero = sum(1 for r in all_recall if r == 0.0)
    n = len(all_recall)

    print(f"\n{'=' * 60}")
    print(f"  RESULTS — Engram × BEAM  (split={split}, top-{top_k})")
    print(f"{'=' * 60}")
    print(f"  Time:          {elapsed:.1f}s ({elapsed / max(n, 1):.2f}s per question)")
    print(f"    Ingest: {total_ingest_time:.1f}s  Query: {total_query_time:.1f}s")
    print(f"  Conversations: {len(rows)}")
    print(f"  Questions:     {n}")
    print(f"  Avg Recall:    {avg_recall:.3f}")

    print("\n  PER-ABILITY RECALL (questions with evidence only):")
    for ab in ABILITIES:
        vals = per_ability.get(ab)
        if vals:
            avg = sum(vals) / len(vals)
            label = ab.replace("_", " ").title()
            print(f"    {label:30} R={avg:.3f}  (n={len(vals)})")
        else:
            label = ab.replace("_", " ").title()
            print(f"    {label:30} — (excluded: no evidence)")

    print("\n  RECALL DISTRIBUTION:")
    print(f"    Perfect (1.0):  {perfect:4} ({perfect / n * 100:.1f}%)")
    print(f"    Partial (0-1):  {partial:4} ({partial / n * 100:.1f}%)")
    print(f"    Zero (0.0):     {zero:4}  ({zero / n * 100:.1f}%)")

    print(f"\n{'=' * 60}\n")

    if out_file:
        out = {
            "meta": {
                "split": split,
                "top_k": top_k,
                "conversations": len(rows),
                "questions": n,
                "avg_recall": round(avg_recall, 4),
                "perfect_pct": round(perfect / n * 100, 1),
                "engram_url": engram_url,
                "timestamp": datetime.now().isoformat(),
            },
            "per_ability": {
                ab: {
                    "recall": round(sum(v) / len(v), 4) if (v := per_ability.get(ab)) else None,
                    "n": len(per_ability.get(ab, [])),
                }
                for ab in ABILITIES
            },
            "results": results_log,
        }
        with open(out_file, "w") as f:
            json.dump(out, f, indent=2)
        print(f"  Results saved to: {out_file}")


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Engram × BEAM Benchmark (arXiv:2510.27246)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--split",
        choices=["100K", "500K", "1M"],
        default="100K",
        help="BEAM conversation-length split to evaluate (default: 100K)",
    )
    parser.add_argument(
        "--all-splits",
        action="store_true",
        help="Run all three splits sequentially (100K → 500K → 1M)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Top-k retrieval results to score against (default: 10)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit to N conversations per split (0 = all)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output JSON file path (per-split if --all-splits)",
    )
    parser.add_argument(
        "--engram-url",
        default="http://localhost:8000",
        help="Engram API base URL (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--token",
        default="",
        help="Engram API key (if auth enabled)",
    )
    parser.add_argument(
        "--no-cleanup",
        action="store_true",
        default=False,
        help="Keep benchmark project data in Engram after run",
    )
    parser.add_argument(
        "--skip-ingest",
        action="store_true",
        default=False,
        help=(
            "Skip ingest for conversations whose _beam_bench_<id> project is already staged. "
            "Use with --no-cleanup to pre-stage once, then iterate queries only. "
            "Automatically disables cleanup to protect staged data."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Concurrent query workers (default: 8)",
    )

    args = parser.parse_args()

    splits_to_run = ["100K", "500K", "1M"] if args.all_splits else [args.split]

    for split in splits_to_run:
        out_file = args.out
        if args.all_splits and args.out:
            stem = Path(args.out).stem
            suffix = Path(args.out).suffix or ".json"
            out_file = str(Path(args.out).parent / f"{stem}_{split}{suffix}")

        asyncio.run(
            run_benchmark(
                split=split,
                engram_url=args.engram_url,
                token=args.token,
                top_k=args.top_k,
                limit=args.limit,
                out_file=out_file,
                cleanup=not args.no_cleanup,
                workers=args.workers,
                skip_ingest=args.skip_ingest,
            )
        )
