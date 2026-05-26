#!/usr/bin/env python3
"""
LongMemEval results reporter.
Usage:
    python3 lme_report.py results_v120.jsonl
    python3 lme_report.py results_v205.jsonl --compare results_v120.jsonl
"""
import argparse
import json
import sys

KS = [1, 3, 5, 10, 30, 50]


def found_at(m):
    for k in KS:
        if m.get(f"recall_any@{k}", 0) >= 1.0:
            return k
    return None


def fmt_found(fa):
    if fa is None:
        return "not-found "
    if fa == 1:
        return "found@1   "
    return f"found@{fa:<3}  "


def load(path):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    result = {}
    for i, r in enumerate(rows, 1):
        m = r["retrieval_results"]["metrics"]["session"]
        result[r["question_id"]] = {
            "idx": i,
            "row": r,
            "r1": m["recall_any@1"],
            "fa": found_at(m),
        }
    return result


def report(path, compare_path=None):
    data = load(path)
    compare = load(compare_path) if compare_path else None

    tagged = sorted(data.values(), key=lambda x: x["idx"])
    misses = [t for t in tagged if t["r1"] < 1.0]
    hits   = [t for t in tagged if t["r1"] >= 1.0]
    total  = len(tagged)
    r1_avg = sum(t["r1"] for t in tagged) / total

    label = path.split("/")[-1].replace(".jsonl", "")
    print("=" * 90)
    print(f"  {label}  ·  R@1={r1_avg:.4f}  ({len(hits)} hits / {len(misses)} misses / {total})")
    if compare:
        clabel = compare_path.split("/")[-1].replace(".jsonl", "")
        c_r1 = sum(v["r1"] for v in compare.values()) / len(compare)
        flipped_good = [t for t in misses if compare.get(t["row"]["question_id"], {}).get("r1", 0) >= 1.0]
        flipped_bad  = [t for t in hits   if compare.get(t["row"]["question_id"], {}).get("r1", 0) < 1.0]
        print(f"  vs {clabel}  R@1={c_r1:.4f}  (Δ={r1_avg - c_r1:+.4f})")
        print(f"  Regressions (was hit, now miss): {len(flipped_good)}  |  Improvements (was miss, now hit): {len(flipped_bad)}")
    print("=" * 90)

    def print_section(title, rows):
        print(f"\n  ── {title} ({len(rows)}) ──\n")
        print(f"  {'#':>5}  {'question_id':<24} {'type':<28}  {'result':<12}  {'stored':>6}", end="")
        if compare:
            print(f"  {'vs':^12}", end="")
        print(f"  question")
        print("  " + "─" * 115)
        for t in rows:
            qid = t["row"]["question_id"]
            cmp_str = ""
            if compare and qid in compare:
                cv = compare[qid]
                marker = "▲" if t["r1"] > cv["r1"] else ("▼" if t["r1"] < cv["r1"] else " ")
                cmp_str = f"  {marker} was {fmt_found(cv['fa'])}"
            print(
                f"  [{t['idx']:4d}]  {qid:<24} {t['row']['question_type']:<28}  "
                f"{fmt_found(t['fa'])}  {t['row']['stored']:>6}"
                f"{cmp_str}  {t['row']['question'][:60]}"
            )

    print_section("MISSES", misses)
    print_section("HITS", hits)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results", help="JSONL results file to report on")
    parser.add_argument("--compare", default=None, help="Baseline JSONL to diff against")
    args = parser.parse_args()
    report(args.results, args.compare)
