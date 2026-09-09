#!/usr/bin/env python3
"""Per-category comparison of LME e2e result files.

Usage: compare_e2e.py LABEL=path.jsonl [LABEL=path.jsonl ...]

Prints hit rate per question type for each file (paired on the question IDs
present in *every* file, so partial runs compare fairly) plus the delta of
each column against the first one. Also reports mean reader latency.
"""
import json
import sys
from collections import defaultdict

TYPES = [
    "single-session-user", "single-session-assistant", "single-session-preference",
    "multi-session", "knowledge-update", "temporal-reasoning",
]


def load(path):
    rows = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                rows[r["question_id"]] = r
    return rows


def main(argv):
    if not argv:
        print(__doc__)
        return 1
    runs = []
    for arg in argv:
        label, _, path = arg.partition("=")
        if not path:
            label, path = arg.rsplit("/", 1)[-1], arg
        runs.append((label, load(path)))
    common = set.intersection(*(set(r.keys()) for _, r in runs))
    print(f"paired on {len(common)} questions present in all {len(runs)} files\n")

    def stats(rows):
        per = defaultdict(lambda: [0, 0, 0.0])
        for qid in common:
            r = rows[qid]
            t = per[r["question_type"]]
            t[0] += 1
            t[1] += 1 if r["hit"] else 0
            t[2] += r.get("gen_time", 0.0)
        return per

    table = [(lab, stats(rows)) for lab, rows in runs]
    w = max(len(lab) for lab, _ in table)
    hdr = f"{'type':28s} {'n':>4s} " + " ".join(f"{lab:>{max(w, 12)}s}" for lab, _ in table)
    print(hdr)
    print("-" * len(hdr))
    for qt in TYPES + ["overall"]:
        cells, n = [], 0
        for i, (lab, per) in enumerate(table):
            if qt == "overall":
                tot = sum(v[0] for v in per.values())
                hit = sum(v[1] for v in per.values())
            else:
                tot, hit, _ = per.get(qt, [0, 0, 0.0])
            n = tot
            rate = hit / tot if tot else float("nan")
            if i == 0:
                base = rate
                cells.append(f"{rate:.3f}")
            else:
                cells.append(f"{rate:.3f} ({(rate - base) * 100:+.1f}pp)")
        print(f"{qt:28s} {n:4d} " + " ".join(f"{c:>{max(w, 12)}s}" for c in cells))
    print()
    for lab, per in table:
        tot = sum(v[0] for v in per.values())
        secs = sum(v[2] for v in per.values())
        judged = sum(1 for qid in common if runs[[l for l, _ in runs].index(lab)][1][qid].get("judge_match"))
        print(f"{lab:>{w}s}: mean reader time {secs / tot:.1f}s, judge rescues {judged}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
