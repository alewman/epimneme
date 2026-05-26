#!/usr/bin/env python3
"""Compare preference R@10 across benchmark configs."""
import json

configs = [
    ("benchmarks/results_engram_lme_session_full.jsonl", "Pre-RRF"),
    ("benchmarks/results_engram_lme_rrf2.jsonl", "RRF 1.0/0.7"),
    ("benchmarks/results_engram_lme_rrf_final.jsonl", "RRF 1.0/0.5"),
]

all_results = {}
for path, label in configs:
    with open(path) as f:
        data = [json.loads(l) for l in f]
    all_results[label] = {r["question_id"]: r for r in data}

# Get misses from RRF 1.0/0.5
final = all_results["RRF 1.0/0.5"]
pref_qids = [qid for qid, r in final.items() if r["question_type"] == "single-session-preference"]

for label, data in all_results.items():
    pref = [data[qid] for qid in pref_qids if qid in data]
    misses = [p for p in pref if p["retrieval_results"]["metrics"]["session"]["recall_any@10"] < 1.0]
    r10 = sum(p["retrieval_results"]["metrics"]["session"]["recall_any@10"] for p in pref) / len(pref)
    r5 = sum(p["retrieval_results"]["metrics"]["session"]["recall_any@5"] for p in pref) / len(pref)
    print(f"{label:15s}: pref R@5={r5:.3f}  R@10={r10:.3f}  misses@10={len(misses)}")

# Detail: the 6 RRF-0.5 misses across configs
miss_qids = [qid for qid in pref_qids if final[qid]["retrieval_results"]["metrics"]["session"]["recall_any@10"] < 1.0]
print(f"\nThe 6 RRF-0.5 misses - cross-config comparison:")
for qid in miss_qids:
    q = final[qid]["question"][:80]
    print(f"\n  {qid[:8]}: {q}")
    for label, data in all_results.items():
        if qid in data:
            sm = data[qid]["retrieval_results"]["metrics"]["session"]
            print(f"    {label:15s}: R@5={sm['recall_any@5']:.0f}  R@10={sm['recall_any@10']:.0f}")
