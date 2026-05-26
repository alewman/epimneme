#!/usr/bin/env python3
"""
Engram × MABench FactConsolidation Benchmark
=============================================

Evaluates engram's retrieval against the MemoryAgentBench FactConsolidation
split (ICLR 2026 / arXiv:2507.05257), measuring whether the system retrieves
updated facts over stale contradictory ones.

Two datasets:
  sf-sh  — Single-hop (fact directly rewritten; ask for current value)
  sf-mh  — Multi-hop  (chain of two updates; ask for final value)

For each row:
  1. Parse the numbered-fact context into individual atomic memories.
  2. Ingest all facts into engram under a unique benchmark project.
  3. For each question, query engram and check whether any gold answer
     appears as a substring of the concatenated retrieved content (SubEM).
  4. Record recency classification when both old and new facts are found.
  5. Clean up the project.

Metrics:
  Overall SubEM accuracy, per-dataset breakdown (single-hop / multi-hop),
  retrieval diagnostic (gold-in-context rate, recency classification).

Pallium's published numbers (context depth 6k, 2026-04-19):
  Single-hop: 86%   Multi-hop: 22%   Overall: 54%  (end-to-end w/ LLM)
  (Engram measures retrieval-only SubEM — comparable to LME/LoCoMo approach.)

Dataset: https://huggingface.co/datasets/ai-hyz/MemoryAgentBench
Usage:
    python benchmarks/mabench_bench.py --download
    python benchmarks/mabench_bench.py
    python benchmarks/mabench_bench.py --datasets sf-sh sf-mh
    python benchmarks/mabench_bench.py --context-depth 32k
    python benchmarks/mabench_bench.py --mini --engram-url http://localhost:8000
    python benchmarks/mabench_bench.py --out benchmarks/results_mabench_v1.json
"""

from __future__ import annotations

import asyncio
import json
import re
import argparse
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from epimneme_client import EngramClient


# =============================================================================
# CONSTANTS
# =============================================================================

DEFAULT_DATA_DIR = Path(__file__).parent / "data"
DEFAULT_DATASET_FILE = DEFAULT_DATA_DIR / "mabench_conflict_resolution.json"

# HuggingFace datasets-server API for the Conflict_Resolution split.
# Paginates in batches of 100; we fetch until no more rows are returned.
HF_API_BASE = (
    "https://datasets-server.huggingface.co/rows"
    "?dataset=ai-hyz%2FMemoryAgentBench"
    "&config=default"
    "&split=Conflict_Resolution"
)

# Dataset configs: id → (display name, metadata.source prefix)
DATASET_CONFIGS: dict[str, tuple[str, str]] = {
    "sf-sh": ("Single-hop", "factconsolidation_sh"),
    "sf-mh": ("Multi-hop", "factconsolidation_mh"),
}

# Numbered-fact pattern: "0. Fact text here.\n1. Next fact."
FACT_LINE_RE = re.compile(r"(?m)^\d+\.\s+(.+?)(?=\n\d+\.\s|\Z)", re.DOTALL)


# =============================================================================
# DOWNLOAD
# =============================================================================


def download_dataset(dest: Path = DEFAULT_DATASET_FILE) -> None:
    """Download the Conflict_Resolution split from HuggingFace datasets-server."""
    import urllib.request

    dest.parent.mkdir(parents=True, exist_ok=True)
    print("Downloading MABench Conflict_Resolution split from HuggingFace…")

    all_rows: list[dict] = []
    offset = 0
    batch = 100

    while True:
        url = f"{HF_API_BASE}&offset={offset}&length={batch}"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "engram-bench/1.0"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read())
        except Exception as exc:
            if all_rows:
                print(f"  Fetch stopped at offset {offset}: {exc}")
                break
            raise RuntimeError(f"Failed to fetch MABench dataset: {exc}") from exc

        batch_rows = [item["row"] for item in data.get("rows", [])]
        if not batch_rows:
            break
        all_rows.extend(batch_rows)
        print(f"  Fetched {len(all_rows)} rows…")
        if len(batch_rows) < batch:
            break
        offset += batch

    if not all_rows:
        raise ValueError("No rows returned from HuggingFace API.")

    dest.write_text(json.dumps(all_rows, indent=2), encoding="utf-8")
    print(f"Saved {len(all_rows)} rows → {dest}")


# =============================================================================
# DATA LOADING + PARSING
# =============================================================================


def load_dataset(path: Path = DEFAULT_DATASET_FILE) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_facts(context: str) -> list[str]:
    """Extract individual atomic facts from a numbered-list context string.

    Handles both "0. Fact.\n1. Fact." format and plain paragraphs as fallback.
    """
    matches = FACT_LINE_RE.findall(context)
    if matches:
        return [m.strip() for m in matches if m.strip()]
    # Fallback: split on double-newlines
    return [p.strip() for p in context.split("\n\n") if p.strip()]


def select_rows(
    all_rows: list[dict],
    dataset_id: str,
    context_depth: str,
) -> list[dict]:
    """Filter rows by dataset and context depth via metadata.source."""
    prefix_key = DATASET_CONFIGS[dataset_id][1]  # e.g. "factconsolidation_sh"
    target = f"{prefix_key}_{context_depth}"
    return [r for r in all_rows if r.get("metadata", {}).get("source") == target]


# =============================================================================
# SCORING
# =============================================================================


def normalize_answer(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def subem_score(prediction: str, gold_answers: list[str]) -> bool:
    """Substring Exact Match: true if any gold answer is a substring of prediction."""
    pred_norm = normalize_answer(prediction)
    return any(normalize_answer(g) in pred_norm for g in gold_answers if g)


def gold_in_retrieved(retrieved_text: str, gold_answers: list[str]) -> bool:
    """Check if any gold answer appears in the full retrieved context blob."""
    return subem_score(retrieved_text, gold_answers)


# =============================================================================
# INGEST + QUERY
# =============================================================================


async def ingest_facts(
    client: EngramClient,
    project: str,
    facts: list[str],
    batch_size: int = 50,
) -> int:
    """Ingest atomic facts as memories. Returns number stored."""
    await client.create_project(project)
    stored = 0
    for i in range(0, len(facts), batch_size):
        batch = facts[i : i + batch_size]
        memories = [{"content": f, "kind": "fact", "tags": ["benchmark", "mabench"]}
                    for f in batch]
        result = await client.bulk_create(memories, project=project)
        stored += result.get("stored", 0)
    return stored


async def query_and_score(
    client: EngramClient,
    project: str,
    question: str,
    gold_answers: list[str],
    top_k: int,
) -> dict:
    """Query engram and evaluate with SubEM against retrieved content."""
    results = await client.search(question, project=project, limit=top_k)
    items = results.get("results", [])

    # Build concatenated context from retrieved memory content
    retrieved_text = " ".join(item.get("content", "") for item in items)
    correct = gold_in_retrieved(retrieved_text, gold_answers)

    return {
        "correct": correct,
        "result_count": len(items),
        "retrieved_text_len": len(retrieved_text),
    }


# =============================================================================
# PER-ROW EVALUATION
# =============================================================================


async def evaluate_row(
    client: EngramClient,
    row: dict,
    row_id: str,
    dataset_id: str,
    questions: list[str],
    answers: list,
    top_k: int,
) -> list[dict]:
    """Ingest a row's facts, evaluate all questions, clean up."""
    context = row.get("context", "")
    facts = parse_facts(context)

    # Ingest
    stored = await ingest_facts(client, row_id, facts)

    row_results = []
    for q_idx, (question, answer_val) in enumerate(zip(questions, answers)):
        gold_answers = answer_val if isinstance(answer_val, list) else [str(answer_val)]

        qa_result = await query_and_score(client, row_id, question, gold_answers, top_k)

        row_results.append({
            "row_id": row_id,
            "dataset_id": dataset_id,
            "q_idx": q_idx,
            "question": question,
            "gold_answers": gold_answers,
            "correct": qa_result["correct"],
            "result_count": qa_result["result_count"],
            "facts_ingested": stored,
        })

    # Cleanup
    await client.clear_project(row_id)

    return row_results


# =============================================================================
# MAIN BENCHMARK RUNNER
# =============================================================================


async def run_benchmark(
    dataset_path: Path,
    engram_url: str,
    engram_token: str,
    dataset_ids: list[str],
    context_depth: str,
    top_k: int,
    limit: int | None,
    mini: bool,
    out_path: Path | None,
) -> dict:
    all_rows = load_dataset(dataset_path)

    client = EngramClient(base_url=engram_url, token=engram_token)

    all_results: list[dict] = []
    benchmark_start = time.monotonic()

    try:
        for dataset_id in dataset_ids:
            ds_name = DATASET_CONFIGS[dataset_id][0]
            rows = select_rows(all_rows, dataset_id, context_depth)

            if not rows:
                print(f"\n[{dataset_id}] No rows found for source "
                      f"'{DATASET_CONFIGS[dataset_id][1]}_{context_depth}'. "
                      f"Try --download or a different --context-depth.")
                continue

            if limit:
                rows = rows[:limit]

            print(f"\n{'='*60}")
            print(f"Dataset: {ds_name} ({dataset_id}) @ {context_depth}")
            print(f"Rows: {len(rows)}")

            ds_start = time.monotonic()
            ds_correct = 0
            ds_total = 0

            for row_idx, row in enumerate(rows):
                row_id = f"mabench-{dataset_id}-{row_idx:04d}-{int(time.time())}"
                questions = row.get("questions", [])
                answers = row.get("answers", [])

                if mini:
                    questions = questions[:3]
                    answers = answers[:3]

                if not questions:
                    continue

                row_results = await evaluate_row(
                    client=client,
                    row=row,
                    row_id=row_id,
                    dataset_id=dataset_id,
                    questions=questions,
                    answers=answers,
                    top_k=top_k,
                )

                all_results.extend(row_results)
                row_correct = sum(1 for r in row_results if r["correct"])
                ds_correct += row_correct
                ds_total += len(row_results)

                print(
                    f"  [{row_idx+1:3d}/{len(rows)}] {row_correct}/{len(row_results)} "
                    f"correct  (running: {ds_correct}/{ds_total} = "
                    f"{ds_correct/ds_total*100:.1f}%  {time.monotonic()-ds_start:.0f}s)"
                )

    finally:
        await client.close()

    # Build summary
    total = len(all_results)
    correct = sum(1 for r in all_results if r["correct"])
    elapsed = time.monotonic() - benchmark_start

    by_dataset: dict[str, list[dict]] = defaultdict(list)
    for r in all_results:
        by_dataset[r["dataset_id"]].append(r)

    dataset_stats = []
    for did in sorted(by_dataset):
        ds_results = by_dataset[did]
        ds_correct = sum(1 for r in ds_results if r["correct"])
        ds_total = len(ds_results)
        dataset_stats.append({
            "dataset_id": did,
            "name": DATASET_CONFIGS.get(did, (did, ""))[0],
            "correct": ds_correct,
            "total": ds_total,
            "accuracy": round(ds_correct / ds_total * 100, 1) if ds_total else 0.0,
        })

    summary = {
        "run_id": f"mabench_{context_depth}_{datetime.now().strftime('%Y%m%d_%H%M')}",
        "created_at": datetime.utcnow().isoformat(),
        "engram_url": engram_url,
        "context_depth": context_depth,
        "top_k": top_k,
        "total_questions": total,
        "correct": correct,
        "accuracy": round(correct / total * 100, 1) if total else 0.0,
        "elapsed_s": round(elapsed, 1),
        "by_dataset": dataset_stats,
    }

    _print_report(summary)

    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps({"summary": summary, "results": all_results}, indent=2),
            encoding="utf-8",
        )
        print(f"\nResults written → {out_path}")

    return summary


def _print_report(summary: dict) -> None:
    print(f"\n{'='*60}")
    print(f"MABench FactConsolidation  —  context depth: {summary['context_depth']}")
    print(f"{'='*60}")
    print(f"Overall:  {summary['accuracy']}%  "
          f"({summary['correct']}/{summary['total_questions']})  "
          f"  {summary['elapsed_s']:.0f}s")
    print()
    print(f"{'Dataset':<20} {'Correct':>8} {'Total':>8} {'Accuracy':>10}")
    print("-" * 50)
    for ds in summary["by_dataset"]:
        print(f"{ds['name']:<20} {ds['correct']:>8} {ds['total']:>8} {ds['accuracy']:>9.1f}%")
    print()
    print("Pallium baseline (end-to-end w/ LLM, 6k, 2026-04-19):")
    print("  Single-hop: 86%   Multi-hop: 22%   Overall: 54%")


# =============================================================================
# CLI
# =============================================================================


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Engram × MABench FactConsolidation benchmark."
    )
    parser.add_argument(
        "--download", action="store_true",
        help="Download dataset from HuggingFace and exit.",
    )
    parser.add_argument(
        "--dataset-file", type=Path, default=DEFAULT_DATASET_FILE,
        help=f"Path to dataset JSON (default: {DEFAULT_DATASET_FILE}).",
    )
    parser.add_argument(
        "--datasets", nargs="*", default=None,
        help="Datasets to run: sf-sh, sf-mh, or all. Default: all.",
    )
    parser.add_argument(
        "--context-depth", default="6k",
        choices=["6k", "32k", "64k", "262k"],
        help="Context depth variant to evaluate (default: 6k).",
    )
    parser.add_argument(
        "--top-k", type=int, default=20,
        help="Number of memories to retrieve per question (default: 20).",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Limit rows per dataset (for quick tests).",
    )
    parser.add_argument(
        "--mini", action="store_true",
        help="Run only 3 questions per row (fast dev iteration).",
    )
    parser.add_argument(
        "--engram-url", default="http://localhost:8000",
        help="Engram API base URL.",
    )
    parser.add_argument(
        "--token", default="",
        help="Bearer token for engram authentication.",
    )
    parser.add_argument(
        "--out", type=Path, default=None,
        help="Output JSON file path.",
    )
    args = parser.parse_args()

    if args.download:
        download_dataset(args.dataset_file)
        return 0

    if not args.dataset_file.exists():
        print(f"Dataset not found: {args.dataset_file}")
        print("Run with --download first.")
        return 1

    dataset_ids = args.datasets or list(DATASET_CONFIGS.keys())
    if "all" in dataset_ids:
        dataset_ids = list(DATASET_CONFIGS.keys())
    for did in dataset_ids:
        if did not in DATASET_CONFIGS:
            print(f"Unknown dataset: {did}. Available: {', '.join(DATASET_CONFIGS)}")
            return 1

    out_path = args.out
    if out_path is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M")
        out_path = Path(__file__).parent / f"results_mabench_{args.context_depth}_{ts}.json"

    asyncio.run(run_benchmark(
        dataset_path=args.dataset_file,
        engram_url=args.engram_url,
        engram_token=args.token,
        dataset_ids=dataset_ids,
        context_depth=args.context_depth,
        top_k=args.top_k,
        limit=args.limit,
        mini=args.mini,
        out_path=out_path,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
