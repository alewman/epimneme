#!/usr/bin/env python3
"""Compare R@k metrics across all LongMemEval benchmark runs."""

import json
import re
import sys
from datetime import datetime
from pathlib import Path

BENCH_DIR = Path(__file__).parent

# Map filename patterns → friendly version labels
VERSION_LABELS = [
    (r"rrf_final",               "rrf-final"),
    (r"v070",                    "v0.70"),
    (r"v080",                    "v0.80 (baseline)"),
    (r"9miss",                   "9miss"),
    (r"turn-pair_20260509_1049", "early-turnpair"),
    (r"v090",                    "v0.90"),
    (r"v091",                    "v0.91"),
    (r"v092_noboost_targeted",   "v0.92-noboost"),
    (r"v100",                    "v1.00"),
]

# Runs to skip (too small / not meaningful full runs)
SKIP_PATTERNS = [r"9miss", r"early-turnpair", r"session_"]

def label(filename):
    for pattern, name in VERSION_LABELS:
        if re.search(pattern, filename):
            return name
    return Path(filename).stem

def should_skip(filename):
    return any(re.search(p, filename) for p in SKIP_PATTERNS)

def r_at_k(d, k):
    return d["retrieval_results"]["metrics"]["session"].get(f"recall_any@{k}", 0)

def analyze(path):
    results = []
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    results.append(json.loads(line))
    except Exception as e:
        return None, str(e)
    if not results:
        return None, "empty"
    n = len(results)
    metrics = {}
    hits = {}
    for k in [1, 3, 5, 10]:
        h = sum(1 for d in results if r_at_k(d, k) == 1)
        hits[k] = h
        metrics[k] = h / n * 100
    return {"n": n, "metrics": metrics, "hits": hits}, None

def delta_str(val, prev):
    if prev is None:
        return ""
    d = val - prev
    if abs(d) < 0.05:
        return "  —  "
    sign = "+" if d > 0 else ""
    return f"{sign}{d:.1f}"

def main():
    # Collect turn-pair runs
    files = [
        p for p in BENCH_DIR.glob("*.jsonl")
        if "turnpair" in p.name or "turn-pair" in p.name
        or "rrf_final" in p.name or "v070" in p.name
    ]
    # Allow extra files from command line
    if len(sys.argv) > 1:
        files += [Path(a) for a in sys.argv[1:]]

    # Sort by file modification time (chronological)
    files = sorted(set(files), key=lambda p: p.stat().st_mtime)

    rows = []
    for path in files:
        ver = label(path.name)
        skip = should_skip(path.name)
        mtime = datetime.fromtimestamp(path.stat().st_mtime).strftime("%b %d")
        data, err = analyze(path)
        if data is None:
            rows.append({"ver": ver, "date": mtime, "n": 0, "metrics": None,
                         "hits": None, "partial": False, "skip": skip, "err": err})
        else:
            rows.append({"ver": ver, "date": mtime, "n": data["n"],
                         "metrics": data["metrics"], "hits": data["hits"],
                         "partial": data["n"] < 500, "skip": skip, "err": None})

    # Print full table (all runs including partial)
    col_ver = max(len(r["ver"]) for r in rows) + 2
    ks = [1, 5, 10]

    print()
    print("  All runs (chronological):")
    print(f"  {'Date':<8} {'Version':<{col_ver}} {'N':>5}   "
          + "  ".join(f"{'R@'+str(k):>6}" for k in ks)
          + "  Note")
    print("  " + "-" * (8 + col_ver + 5 + 4 + len(ks) * 10 + 6))
    for r in rows:
        m = r["metrics"]
        if m is None:
            vals = "  ".join(f"{'ERR':>6}" for k in ks)
        else:
            vals = "  ".join(f"{m[k]:>6.1f}%" for k in ks)
        note = "(partial)" if r["partial"] else ""
        print(f"  {r['date']:<8} {r['ver']:<{col_ver}} {r['n']:>5}   {vals}  {note}")

    # Print delta table for full-500 runs only
    full_rows = [r for r in rows if not r["partial"] and r["metrics"] is not None and not r["skip"]]
    if len(full_rows) > 1:
        print()
        print("  Full-run deltas (vs prior full run):")
        print(f"  {'Date':<8} {'Version':<{col_ver}} {'N':>5}   "
              + "  ".join(f"{'R@'+str(k):>9}" for k in ks))
        print("  " + "-" * (8 + col_ver + 5 + 4 + len(ks) * 13))
        prev = None
        for r in full_rows:
            m = r["metrics"]
            parts = []
            for k in ks:
                d = delta_str(m[k], prev[k] if prev else None)
                parts.append(f"{m[k]:>5.1f}% {d:>4}")
            print(f"  {r['date']:<8} {r['ver']:<{col_ver}} {r['n']:>5}   " + "  ".join(parts))
            prev = m
    print()

if __name__ == "__main__":
    main()
