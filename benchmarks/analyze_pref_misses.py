#!/usr/bin/env python3
"""Analyze preference question misses from LME benchmark results."""
import json, sys

path = sys.argv[1] if len(sys.argv) > 1 else "benchmarks/results_engram_lme_rrf_final.jsonl"
with open(path) as f:
    rs = [json.loads(l) for l in f]

# Also load the raw data to get gold session IDs
data_path = "benchmarks/data/longmemeval_s_cleaned.json"
with open(data_path) as f:
    raw_data = json.load(f)
gold_map = {e["question_id"]: e.get("answer_session_ids", []) for e in raw_data}

pref = [r for r in rs if r.get("question_type") == "single-session-preference"]
print(f"Total preference questions: {len(pref)}")

for p in pref:
    sm = p["retrieval_results"]["metrics"]["session"]
    if sm["recall_any@10"] < 1.0:
        qid = p["question_id"]
        q = p["question"]
        stored = p.get("stored", "?")
        items = p["retrieval_results"]["ranked_items"]
        gold_sids = gold_map.get(qid, [])
        
        print(f"\n{'='*70}")
        print(f"QID:    {qid}")
        print(f"Q:      {q[:140]}")
        print(f"Answer: {p.get('answer', '?')[:140]}")
        print(f"Stored: {stored}")
        print(f"Gold session IDs: {gold_sids}")
        
        print(f"\nTop 10 retrieved ({len(items)} total):")
        for i, it in enumerate(items[:10]):
            cid = it.get("corpus_id", "")
            score = it.get("score", 0)
            text = it.get("text", "")[:150]
            marker = " <-- GOLD" if any(g in cid for g in gold_sids) else ""
            print(f"  #{i+1} cid={cid[:50]} score={score:.4f}{marker}")
            print(f"       {text}")
        
        # Check where gold appears
        for gi, g in enumerate(gold_sids):
            found_at = None
            for i, it in enumerate(items):
                if g in it.get("corpus_id", ""):
                    found_at = i + 1
                    break
            status = f"at position #{found_at}" if found_at else "NOT FOUND"
            print(f"\n  Gold session '{g}' -> {status}")

total_misses = sum(1 for p in pref if p["retrieval_results"]["metrics"]["session"]["recall_any@10"] < 1.0)
print(f"\n\nTotal misses: {total_misses}/30")

