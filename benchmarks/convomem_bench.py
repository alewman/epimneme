#!/usr/bin/env python3
"""
Engram × ConvoMem Benchmark
===============================

Evaluates engram's retrieval against the ConvoMem benchmark
(arXiv:2511.10523) — 75,336 QA pairs across 6 evidence categories.

Categories:
  1. User Facts       — recall of user-stated information
  2. Assistant Facts   — recall of assistant-stated information
  3. Changing Facts    — tracking evolving information
  4. Abstention        — recognizing absent information
  5. Preferences       — understanding user preferences
  6. Implicit          — multi-hop reasoning across messages

For each evidence item:
1. Ingest all conversation messages into engram under a unique project
2. Query engram with the question
3. Score retrieval: did we find the evidence message(s)?
4. Clean up the project's data

Metrics: Per-category recall, overall recall, per-evidence-count breakdown.

Usage:
    python benchmarks/convomem_bench.py
    python benchmarks/convomem_bench.py --sample 30
    python benchmarks/convomem_bench.py --categories user assistant
    python benchmarks/convomem_bench.py --evidence-counts 1 2 3
    python benchmarks/convomem_bench.py --engram-url http://localhost:8000
"""

import asyncio
import json
import argparse
import sys
import os
import time
import random
from pathlib import Path
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from epimneme_client import EngramClient

# =============================================================================
# CONSTANTS
# =============================================================================

DATASET_REPO = "Salesforce/ConvoMem"
EVIDENCE_BASE = "core_benchmark/evidence_questions"

# Map short names → HuggingFace directory names
CATEGORY_DIRS = {
    "user":      "user_evidence",
    "assistant": "assistant_facts_evidence",
    "changing":  "changing_evidence",
    "abstention": "abstention_evidence",
    "preference": "preference_evidence",
    "implicit":  "implicit_connection_evidence",
}

CATEGORY_LABELS = {
    "user":      "User Facts",
    "assistant": "Assistant Facts",
    "changing":  "Changing Facts",
    "abstention": "Abstention",
    "preference": "Preferences",
    "implicit":  "Implicit Connections",
}

# Valid evidence counts per category (from the paper's Table 2)
VALID_EVIDENCE_COUNTS = {
    "user":      [1, 2, 3, 4, 5, 6],
    "assistant": [1, 2, 3, 4, 5, 6],
    "changing":  [2, 3, 4, 5, 6],
    "abstention": [1, 2, 3],
    "preference": [1, 2],
    "implicit":  [1, 2, 3],
}


# =============================================================================
# DATA DOWNLOAD & LOADING
# =============================================================================


def get_data_dir() -> Path:
    """Return path to local ConvoMem data cache."""
    data_dir = Path(__file__).parent / "data" / "convomem"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def download_evidence_files(
    category: str,
    evidence_count: int,
    data_dir: Path | None = None,
    max_files: int = 5,
) -> list[Path]:
    """Download evidence JSON files from HuggingFace for a category/count.

    Returns list of local file paths. Downloads at most max_files files
    to keep bandwidth reasonable.
    """
    if data_dir is None:
        data_dir = get_data_dir()

    cat_dir_name = CATEGORY_DIRS[category]
    hf_path = f"{EVIDENCE_BASE}/{cat_dir_name}/{evidence_count}_evidence"
    local_dir = data_dir / cat_dir_name / f"{evidence_count}_evidence"
    local_dir.mkdir(parents=True, exist_ok=True)

    # Check if we already have files cached
    existing = list(local_dir.glob("*.json"))
    if existing:
        return existing[:max_files]

    # List files in the HF repo
    try:
        from huggingface_hub import HfApi, hf_hub_download
        api = HfApi()
        files = api.list_repo_tree(
            DATASET_REPO,
            path_in_repo=hf_path,
            repo_type="dataset",
        )
        json_files = [f for f in files if hasattr(f, 'rfilename') and f.rfilename.endswith(".json")]
        if not json_files:
            # Try alternate attribute
            json_files = [f for f in files if hasattr(f, 'path') and f.path.endswith(".json")]
    except Exception as e:
        print(f"  ERROR listing HF files for {hf_path}: {e}")
        return []

    # Download up to max_files
    downloaded = []
    for i, f_info in enumerate(json_files[:max_files]):
        fname = getattr(f_info, 'rfilename', None) or getattr(f_info, 'path', '')
        if not fname:
            continue
        local_path = local_dir / Path(fname).name
        if local_path.exists():
            downloaded.append(local_path)
            continue

        try:
            result = hf_hub_download(
                repo_id=DATASET_REPO,
                filename=fname,
                repo_type="dataset",
                local_dir=str(data_dir),
            )
            # hf_hub_download puts files in local_dir mirroring repo structure
            result_path = Path(result)
            downloaded.append(result_path)
            print(f"    Downloaded: {Path(fname).name} ({i + 1}/{min(len(json_files), max_files)})")
        except Exception as e:
            print(f"    ERROR downloading {fname}: {e}")

    return downloaded


def load_evidence_items(file_path: Path) -> list[dict]:
    """Load evidence items from a ConvoMem JSON file."""
    with open(file_path) as f:
        data = json.load(f)
    return data.get("evidence_items", [])


def download_filler_files(
    data_dir: Path | None = None,
    max_files: int = 2,
) -> list[Path]:
    """Download filler conversation files from HuggingFace.

    Filler conversations are used to increase the retrieval haystack,
    making the benchmark more challenging and realistic.
    """
    if data_dir is None:
        data_dir = get_data_dir()

    hf_path = "core_benchmark/filler_conversations"
    local_dir = data_dir / "filler_conversations"
    local_dir.mkdir(parents=True, exist_ok=True)

    existing = list(local_dir.glob("*.json"))
    if existing:
        return existing[:max_files]

    try:
        from huggingface_hub import HfApi, hf_hub_download
        api = HfApi()
        files = api.list_repo_tree(
            DATASET_REPO,
            path_in_repo=hf_path,
            repo_type="dataset",
        )
        json_files = [f for f in files if hasattr(f, 'rfilename') and f.rfilename.endswith(".json")]
        if not json_files:
            json_files = [f for f in files if hasattr(f, 'path') and f.path.endswith(".json")]
    except Exception as e:
        print(f"  ERROR listing filler files: {e}")
        return []

    downloaded = []
    for i, f_info in enumerate(json_files[:max_files]):
        fname = getattr(f_info, 'rfilename', None) or getattr(f_info, 'path', '')
        if not fname:
            continue
        try:
            result = hf_hub_download(
                repo_id=DATASET_REPO,
                filename=fname,
                repo_type="dataset",
                local_dir=str(data_dir),
            )
            downloaded.append(Path(result))
            print(f"    Downloaded filler: {Path(fname).name} ({i + 1}/{min(len(json_files), max_files)})")
        except Exception as e:
            print(f"    ERROR downloading filler {fname}: {e}")

    return downloaded


def load_filler_messages(
    max_conversations: int = 100,
    max_files: int = 2,
    seed: int = 42,
) -> list[list[dict]]:
    """Load filler conversations as lists of messages.

    Returns list of conversations, each conversation is a list of
    message dicts (same format as evidence conversations).
    """
    files = download_filler_files(max_files=max_files)
    if not files:
        return []

    all_convos = []
    for fp in files:
        with open(fp) as f:
            data = json.load(f)
        # Filler files use same format as evidence:
        # {"evidence_items": [{"conversations": [{"messages": [...]}]}]}
        for item in data.get("evidence_items", []):
            for conv in item.get("conversations", []):
                msgs = conv.get("messages", [])
                if msgs:
                    all_convos.append(msgs)

    rng = random.Random(seed)
    if max_conversations > 0 and len(all_convos) > max_conversations:
        all_convos = rng.sample(all_convos, max_conversations)

    return all_convos


def load_category_data(
    category: str,
    evidence_count: int,
    sample_size: int = 50,
    max_files: int = 5,
    seed: int = 42,
) -> list[dict]:
    """Load and sample evidence items for a category.

    Downloads from HuggingFace if needed, then samples N items.
    """
    files = download_evidence_files(category, evidence_count, max_files=max_files)
    if not files:
        print(f"  WARNING: No files found for {category}/{evidence_count}_evidence")
        return []

    all_items = []
    for fp in files:
        items = load_evidence_items(fp)
        for item in items:
            item["_category"] = category
            item["_evidence_count"] = evidence_count
        all_items.extend(items)

    if not all_items:
        return []

    # Deterministic sample
    rng = random.Random(seed)
    if sample_size > 0 and len(all_items) > sample_size:
        all_items = rng.sample(all_items, sample_size)

    return all_items


# =============================================================================
# CORPUS BUILDING
# =============================================================================


def build_corpus(
    item: dict,
    filler_conversations: list[list[dict]] | None = None,
) -> tuple[list[dict], set[str]]:
    """Build memory corpus from an evidence item's conversations + optional filler.

    Args:
        item: Evidence item with conversations and message_evidences
        filler_conversations: Optional list of filler conversations to mix in

    Returns:
        memories: list of dicts ready for bulk_create
        evidence_ids: set of subject IDs that are evidence messages
    """
    memories = []
    evidence_ids = set()

    # Collect evidence texts for matching (evidence text is often
    # embedded within a longer conversational message, with potential
    # minor variations like extra words or capitalization differences)
    evidence_texts = []
    for ev in item.get("message_evidences", []):
        t = ev["text"].strip()
        if t:
            evidence_texts.append(t)

    for conv_idx, conv in enumerate(item.get("conversations", [])):
        for msg_idx, msg in enumerate(conv.get("messages", [])):
            msg_id = f"c{conv_idx}_m{msg_idx}"
            speaker = msg.get("speaker", "Unknown")
            text = msg.get("text", "").strip()

            if not text:
                continue

            # Check if this message contains any evidence text
            # (case-insensitive substring match, with word-overlap fallback)
            text_lower = text.lower()
            for ev_text in evidence_texts:
                if ev_text.lower() in text_lower:
                    evidence_ids.add(msg_id)
                    break
                # Fallback: check if 90%+ evidence words appear in message
                ev_words = ev_text.lower().split()
                if len(ev_words) >= 5:
                    matched_words = sum(1 for w in ev_words if w in text_lower)
                    if matched_words / len(ev_words) >= 0.9:
                        evidence_ids.add(msg_id)
                        break

            memories.append({
                "content": f'{speaker}: {text}',
                "kind": "fact",
                "subject": msg_id,
                "tags": ["benchmark", "convomem"],
            })

    # Add filler conversations to increase haystack size
    if filler_conversations:
        n_evidence_convs = len(item.get("conversations", []))
        for filler_idx, filler_msgs in enumerate(filler_conversations):
            conv_idx = n_evidence_convs + filler_idx
            for msg_idx, msg in enumerate(filler_msgs):
                speaker = msg.get("speaker", "Unknown")
                text = msg.get("text", "").strip()
                if not text:
                    continue
                msg_id = f"f{filler_idx}_m{msg_idx}"
                memories.append({
                    "content": f'{speaker}: {text}',
                    "kind": "fact",
                    "subject": msg_id,
                    "tags": ["benchmark", "convomem"],
                })

    return memories, evidence_ids


# =============================================================================
# ENGRAM INGEST + RETRIEVE
# =============================================================================


async def ingest_corpus(
    client: EngramClient,
    project_name: str,
    memories: list[dict],
    batch_size: int = 50,
) -> int:
    """Ingest memory corpus into engram. Returns count of stored memories."""
    await client.create_project(project_name)

    stored = 0
    for i in range(0, len(memories), batch_size):
        batch = memories[i : i + batch_size]
        for attempt in range(3):
            try:
                result = await client.bulk_create(batch, project=project_name)
                batch_stored = result.get("stored", 0)
                if batch_stored > 0 or result.get("total", 0) > 0:
                    stored += batch_stored
                    break
                # Result has no 'stored' — might be an error response
                if "detail" in result or "error" in result:
                    if attempt < 2:
                        await asyncio.sleep(1 * (attempt + 1))
                        continue
                break
            except Exception:
                if attempt < 2:
                    await asyncio.sleep(1 * (attempt + 1))
                    continue
                break

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
# RECALL COMPUTATION
# =============================================================================


def compute_recall(retrieved_ids: list[str], evidence_ids: set[str]) -> float:
    """What fraction of evidence IDs appear in retrieved results?"""
    if not evidence_ids:
        return 1.0
    found = sum(1 for eid in evidence_ids if eid in retrieved_ids)
    return found / len(evidence_ids)


def compute_recall_any(retrieved_ids: list[str], evidence_ids: set[str]) -> float:
    """Did we find at least one evidence message?"""
    if not evidence_ids:
        return 1.0
    return float(any(eid in retrieved_ids for eid in evidence_ids))


# =============================================================================
# BENCHMARK RUNNER
# =============================================================================


async def process_item(
    item: dict,
    index: int,
    total: int,
    client: EngramClient,
    top_k: int,
    cleanup: bool,
    semaphore: asyncio.Semaphore,
    filler_conversations: list[list[dict]] | None = None,
) -> dict:
    """Process a single ConvoMem evidence item. Returns result dict."""
    async with semaphore:
        category = item["_category"]
        evidence_count = item["_evidence_count"]
        question = item["question"]
        answer = item.get("answer", "")
        project_name = f"_convomem_{category}_{index}_{int(time.time() * 1000) % 100000}"

        # Build corpus
        memories, evidence_ids = build_corpus(item, filler_conversations)

        if not memories:
            print(f"  [{index:4}/{total}] SKIP (empty corpus)")
            return {
                "index": index,
                "category": category,
                "evidence_count": evidence_count,
                "question": question,
                "answer": answer,
                "recall": 0.0,
                "recall_any": 0.0,
                "evidence_found": 0,
                "evidence_total": 0,
                "corpus_size": 0,
                "stored": 0,
                "ingest_time": 0.0,
                "query_time": 0.0,
                "skipped": True,
            }

        # Ingest
        t0 = time.monotonic()
        stored = await ingest_corpus(client, project_name, memories)
        ingest_elapsed = time.monotonic() - t0

        # Query
        t0 = time.monotonic()
        results = await retrieve(client, project_name, question, limit=top_k)
        query_elapsed = time.monotonic() - t0

        # Extract retrieved subject IDs
        retrieved_ids = [r.get("subject", "") for r in results if r.get("subject")]

        recall = compute_recall(retrieved_ids, evidence_ids)
        recall_any = compute_recall_any(retrieved_ids, evidence_ids)
        evidence_found = sum(1 for eid in evidence_ids if eid in retrieved_ids)

        status = "HIT" if recall_any > 0 else "miss"
        print(
            f"  [{index:4}/{total}] {CATEGORY_LABELS.get(category, category):20} "
            f"ev={evidence_count} R={recall:.2f} {status}  "
            f"[{stored}/{len(memories)} msgs, ingest {ingest_elapsed:.1f}s, query {query_elapsed:.2f}s]"
        )

        # Cleanup
        cleanup_elapsed = 0.0
        if cleanup:
            t0 = time.monotonic()
            await client.clear_project(project_name)
            cleanup_elapsed = time.monotonic() - t0

        return {
            "index": index,
            "category": category,
            "evidence_count": evidence_count,
            "question": question,
            "answer": answer,
            "recall": recall,
            "recall_any": recall_any,
            "evidence_found": evidence_found,
            "evidence_total": len(evidence_ids),
            "corpus_size": len(memories),
            "stored": stored,
            "ingest_time": round(ingest_elapsed, 2),
            "query_time": round(query_elapsed, 3),
            "cleanup_time": round(cleanup_elapsed, 3),
            "skipped": False,
        }


async def run_benchmark(
    engram_url: str = "http://localhost:8000",
    categories: list[str] | None = None,
    evidence_counts: list[int] | None = None,
    sample: int = 50,
    top_k: int = 50,
    max_files: int = 5,
    filler: int = 0,
    out_file: str | None = None,
    cleanup: bool = True,
    workers: int = 8,
    seed: int = 42,
):
    """Run the ConvoMem retrieval benchmark against engram."""
    if categories is None:
        categories = list(CATEGORY_DIRS.keys())
    if evidence_counts is None:
        evidence_counts = [1]

    # Load and sample data for each category
    print(f"\n{'=' * 60}")
    print("  Engram × ConvoMem Benchmark")
    print(f"{'=' * 60}")
    print(f"  Categories:      {', '.join(categories)}")
    print(f"  Evidence counts: {evidence_counts}")
    print(f"  Sample/cat:      {sample}")
    print(f"  Top-k:           {top_k}")
    print(f"  Filler convos:   {filler if filler > 0 else 'none'}")
    print(f"  Workers:         {workers}")
    print(f"  Engram:          {engram_url}")
    print(f"  Cleanup:         {'yes' if cleanup else 'no'}")
    print(f"{'─' * 60}")
    print("  Downloading evidence data from HuggingFace...")

    all_items = []
    for cat in categories:
        valid_counts = VALID_EVIDENCE_COUNTS.get(cat, [1])
        for ec in evidence_counts:
            if ec not in valid_counts:
                print(f"  Skipping {cat}/{ec}_evidence (not available)")
                continue
            items = load_category_data(cat, ec, sample_size=sample, max_files=max_files, seed=seed)
            all_items.extend(items)
            print(f"  Loaded {len(items):4} items: {CATEGORY_LABELS[cat]} ({ec}-evidence)")

    if not all_items:
        print("  ERROR: No data loaded. Check HuggingFace connectivity.")
        return

    # Load filler conversations if requested
    filler_conversations = None
    if filler > 0:
        print(f"  Loading {filler} filler conversations...")
        filler_conversations = load_filler_messages(
            max_conversations=filler, seed=seed
        )
        if filler_conversations:
            filler_msgs = sum(len(c) for c in filler_conversations)
            print(f"  Loaded {len(filler_conversations)} filler conversations ({filler_msgs} messages)")
        else:
            print("  WARNING: No filler conversations loaded")

    total = len(all_items)
    print(f"\n  Total items: {total}")
    print(f"{'─' * 60}\n")

    client = EngramClient(base_url=engram_url)
    semaphore = asyncio.Semaphore(workers)
    start_time = datetime.now()

    tasks = [
        process_item(
            item, i + 1, total, client, top_k, cleanup, semaphore,
            filler_conversations=filler_conversations,
        )
        for i, item in enumerate(all_items)
    ]

    try:
        results = await asyncio.gather(*tasks)
    finally:
        await client.close()

    # Filter skipped
    results = [r for r in results if not r.get("skipped")]

    if not results:
        print("  No items evaluated.")
        return

    elapsed = (datetime.now() - start_time).total_seconds()

    # Aggregate metrics
    per_category: dict[str, list[dict]] = defaultdict(list)
    per_evidence_count: dict[int, list[dict]] = defaultdict(list)
    total_ingest_time = 0.0
    total_query_time = 0.0

    for r in results:
        per_category[r["category"]].append(r)
        per_evidence_count[r["evidence_count"]].append(r)
        total_ingest_time += r["ingest_time"]
        total_query_time += r["query_time"]

    overall_recall = sum(r["recall"] for r in results) / len(results)
    overall_recall_any = sum(r["recall_any"] for r in results) / len(results)

    # Print results
    print(f"\n{'=' * 60}")
    print(f"  RESULTS — Engram × ConvoMem (hybrid retrieval, top-{top_k})")
    print(f"{'=' * 60}")
    print(f"  Time:          {elapsed:.1f}s ({elapsed / len(results):.2f}s per item)")
    print(f"    Ingest: {total_ingest_time:.1f}s  Query: {total_query_time:.1f}s")
    print(f"  Items:         {len(results)}")
    print(f"  Overall R:     {overall_recall:.3f}")
    print(f"  Overall R_any: {overall_recall_any:.3f}")

    print("\n  PER-CATEGORY RECALL:")
    for cat in categories:
        cat_results = per_category.get(cat, [])
        if not cat_results:
            continue
        avg_r = sum(r["recall"] for r in cat_results) / len(cat_results)
        avg_r_any = sum(r["recall_any"] for r in cat_results) / len(cat_results)
        label = CATEGORY_LABELS.get(cat, cat)
        print(f"    {label:25} R={avg_r:.3f}  R_any={avg_r_any:.3f}  (n={len(cat_results)})")

    if len(per_evidence_count) > 1:
        print("\n  PER-EVIDENCE-COUNT RECALL:")
        for ec in sorted(per_evidence_count.keys()):
            ec_results = per_evidence_count[ec]
            avg_r = sum(r["recall"] for r in ec_results) / len(ec_results)
            avg_r_any = sum(r["recall_any"] for r in ec_results) / len(ec_results)
            print(f"    {ec}-evidence:  R={avg_r:.3f}  R_any={avg_r_any:.3f}  (n={len(ec_results)})")

    # Recall distribution
    perfect = sum(1 for r in results if r["recall"] >= 1.0)
    partial = sum(1 for r in results if 0 < r["recall"] < 1.0)
    zero = sum(1 for r in results if r["recall"] == 0)
    print("\n  RECALL DISTRIBUTION:")
    print(f"    Perfect (1.0):  {perfect:4} ({perfect / len(results) * 100:.1f}%)")
    print(f"    Partial (0-1):  {partial:4} ({partial / len(results) * 100:.1f}%)")
    print(f"    Zero (0.0):     {zero:4} ({zero / len(results) * 100:.1f}%)")

    print(f"\n{'=' * 60}\n")

    # Save results
    if out_file:
        with open(out_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Results saved to: {out_file}")


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Engram × ConvoMem Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Categories: user, assistant, changing, abstention, preference, implicit
Evidence counts vary by category (see --help-categories).

Examples:
  %(prog)s                              # All categories, 1-evidence, 50 samples each
  %(prog)s --sample 20                  # Quick run: 20 per category
  %(prog)s --categories user implicit   # Only user facts and implicit connections
  %(prog)s --evidence-counts 1 2 3      # Test multi-evidence retrieval
""",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        choices=list(CATEGORY_DIRS.keys()),
        default=None,
        help="Evidence categories to evaluate (default: all)",
    )
    parser.add_argument(
        "--evidence-counts",
        nargs="+",
        type=int,
        default=None,
        help="Evidence counts to evaluate (default: [1])",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=50,
        help="Number of items to sample per category/count (default: 50, 0=all)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Top-k retrieval (default: 50)",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=5,
        help="Max evidence files to download per category/count (default: 5)",
    )
    parser.add_argument(
        "--filler",
        type=int,
        default=0,
        help="Number of filler conversations to mix in (default: 0, try 10-50 for harder test)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output JSON file path",
    )
    parser.add_argument(
        "--engram-url",
        default="http://localhost:8000",
        help="Engram API base URL",
    )
    parser.add_argument(
        "--no-cleanup",
        action="store_true",
        default=False,
        help="Don't delete benchmark data after each item",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of concurrent workers (default: 8)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling (default: 42)",
    )
    args = parser.parse_args()

    if not args.out:
        args.out = (
            f"benchmarks/results_engram_convomem"
            f"_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
        )

    asyncio.run(
        run_benchmark(
            engram_url=args.engram_url,
            categories=args.categories,
            evidence_counts=args.evidence_counts,
            sample=args.sample,
            top_k=args.top_k,
            max_files=args.max_files,
            filler=args.filler,
            out_file=args.out,
            cleanup=not args.no_cleanup,
            workers=args.workers,
            seed=args.seed,
        )
    )
