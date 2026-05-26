#!/usr/bin/env python3
"""Diagnose ConvoMem misses — replay specific items and show what went wrong."""

import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from engram_client import EngramClient
from convomem_bench import (
    build_corpus,
    load_category_data,
    load_filler_messages,
    ingest_corpus,
    retrieve,
    CATEGORY_DIRS,
    CATEGORY_LABELS,
    VALID_EVIDENCE_COUNTS,
)


async def diagnose_item(client, item, filler_conversations, top_k=50):
    """Replay a single item and print detailed diagnostics."""
    category = item["_category"]
    ec = item["_evidence_count"]
    question = item["question"]
    answer = item.get("answer", "")

    print(f"\n{'='*80}")
    print(f"CATEGORY: {category} | EVIDENCE COUNT: {ec}")
    print(f"QUESTION: {question}")
    print(f"ANSWER: {answer[:300]}")
    print(f"{'='*80}")

    # Show all evidence texts from the item
    print(f"\n--- Evidence texts from item ---")
    for i, ev in enumerate(item.get("message_evidences", [])):
        print(f"  Evidence #{i+1}: {ev['text'][:200]}")

    # Build corpus
    memories, evidence_ids = build_corpus(item, filler_conversations)
    print(f"\nCorpus: {len(memories)} messages, Evidence IDs: {evidence_ids}")

    if not evidence_ids:
        print("  *** NO EVIDENCE IDS MATCHED — evidence text could not be located in messages ***")
        # Try to find what messages are closest
        ev_texts = [ev["text"].strip().lower() for ev in item.get("message_evidences", [])]
        for ev_text in ev_texts:
            ev_words = set(ev_text.split())
            print(f"\n  Looking for evidence: '{ev_text[:100]}...'")
            best_overlap = 0
            best_msg = ""
            best_id = ""
            for m in memories:
                msg_words = set(m["content"].lower().split())
                overlap = len(ev_words & msg_words) / len(ev_words) if ev_words else 0
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_msg = m["content"][:200]
                    best_id = m["subject"]
            print(f"  Best match: {best_id} ({best_overlap:.1%} word overlap)")
            print(f"  Content: {best_msg}")
        return

    # Show what the evidence messages actually contain
    ev_memories = [m for m in memories if m["subject"] in evidence_ids]
    print(f"\n--- Evidence message content ---")
    for m in ev_memories:
        print(f"  [{m['subject']}]: {m['content'][:200]}")

    # Ingest
    project_name = f"_convomem_diag_{int(time.time()*1000) % 100000}"
    stored = await ingest_corpus(client, project_name, memories)
    print(f"\nIngested: {stored}/{len(memories)} messages")

    # Query
    results = await retrieve(client, project_name, question, limit=top_k)
    retrieved_ids = [r.get("subject", "") for r in results if r.get("subject")]

    # Check if evidence was found
    found = [eid for eid in evidence_ids if eid in retrieved_ids]
    missed = [eid for eid in evidence_ids if eid not in retrieved_ids]
    print(f"\nRetrieval: Found {len(found)}/{len(evidence_ids)} evidence IDs in top-{top_k}")
    if found:
        for eid in found:
            pos = retrieved_ids.index(eid) + 1
            print(f"  FOUND: {eid} at rank {pos}")
    if missed:
        print(f"  MISSED: {missed}")

    # Show top 10 retrieved
    print(f"\n--- Top 10 retrieved ---")
    for i, r in enumerate(results[:10]):
        content = r.get("content", "")[:120]
        subj = r.get("subject", "?")
        score = r.get("score", 0)
        is_ev = "*** EVIDENCE ***" if subj in evidence_ids else ""
        print(f"  [{i+1:2}] subj={subj:12} score={score:.4f} {is_ev}")
        print(f"       {content}")

    # If missed, check if evidence is anywhere in the full result set
    if missed:
        print(f"\n--- Scanning all {len(results)} results for evidence ---")
        for eid in missed:
            positions = [i+1 for i, r in enumerate(results) if r.get("subject") == eid]
            if positions:
                print(f"  {eid} found at rank(s): {positions}")
            else:
                print(f"  {eid} NOT IN any of the {len(results)} results")

    # Semantic similarity check — query the evidence text directly
    if missed:
        print(f"\n--- Querying evidence text directly (sanity check) ---")
        for m in ev_memories:
            # Use first 100 chars of evidence content as query
            ev_query = m["content"][:150]
            ev_results = await retrieve(client, project_name, ev_query, limit=5)
            ev_retrieved = [r.get("subject", "") for r in ev_results]
            found_self = m["subject"] in ev_retrieved
            print(f"  Query: '{ev_query[:80]}...'")
            print(f"  Found self: {found_self}")
            if found_self:
                pos = ev_retrieved.index(m["subject"]) + 1
                print(f"  Self at rank {pos}")
            else:
                print(f"  Top results: {[(r.get('subject','?'), r.get('score',0)) for r in ev_results[:3]]}")

    # Cleanup
    await client.clear_project(project_name)
    print(f"\nCleaned up project: {project_name}")


async def main():
    # Load results to find misses
    results_file = Path(__file__).parent / "results_engram_convomem_20260408_0934.json"
    with open(results_file) as f:
        results = json.load(f)

    miss_indices = {r["index"] for r in results if r["recall"] == 0.0}
    miss_categories = {r["category"] for r in results if r["recall"] == 0.0}
    print(f"Found {len(miss_indices)} misses at indices: {sorted(miss_indices)}")
    print(f"Categories with misses: {miss_categories}")

    # Download data
    # Load all items using the same params as the benchmark
    all_items = []
    for cat in sorted(CATEGORY_DIRS.keys()):
        valid_counts = VALID_EVIDENCE_COUNTS.get(cat, [1])
        for ec in [1]:
            if ec not in valid_counts:
                continue
            items = load_category_data(cat, ec, sample_size=10, max_files=1, seed=42)
            all_items.extend(items)
            print(f"  Loaded {len(items):4} items: {CATEGORY_LABELS.get(cat, cat)} ({ec}-evidence)")

    # Load filler
    filler_conversations = load_filler_messages(max_conversations=20, seed=42)
    print(f"Loaded {len(filler_conversations)} filler conversations "
          f"({sum(len(c) for c in filler_conversations)} messages)")

    # Re-index to match benchmark
    indexed = list(enumerate(all_items, 1))

    # Find the missed items
    missed_items = [(idx, item) for idx, item in indexed if idx in miss_indices]
    print(f"\nReplaying {len(missed_items)} missed items...")

    client = EngramClient("http://localhost:8000")
    for idx, item in missed_items:
        await diagnose_item(client, item, filler_conversations, top_k=50)


if __name__ == "__main__":
    asyncio.run(main())
