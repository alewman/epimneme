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
from math import comb


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value for discordant counts b (base hit, other
    miss) and c (base miss, other hit): binomial(b+c, 0.5) on min(b, c)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)

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

    # A results file straight off a no-judge generation run scores substring-only.
    # Comparing one against a judged file silently understates it by ~10pp and
    # invents one-directional "regressions" — warn loudly instead.
    for lab, rows in runs:
        judged = sum(1 for r in rows.values() if r.get("judge_match"))
        misses = sum(1 for r in rows.values() if not r.get("hit"))
        if judged == 0 and misses:
            print(f"  !! WARNING: '{lab}' has no judge_match rows ({misses} misses) — "
                  f"it looks UNJUDGED. Run --rescore-only --judge and compare the "
                  f".rescored.jsonl, or this comparison is meaningless.\n")

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
    base_lab, base_rows = runs[0]
    print(f"paired vs {base_lab}: b = {base_lab} hit & other miss, c = the reverse, p = exact McNemar")
    for lab, rows in runs[1:]:
        for qt in TYPES + ["overall"]:
            ids = [q for q in common if qt == "overall" or base_rows[q]["question_type"] == qt]
            b = sum(1 for q in ids if base_rows[q]["hit"] and not rows[q]["hit"])
            cc = sum(1 for q in ids if rows[q]["hit"] and not base_rows[q]["hit"])
            if b + cc:
                print(f"  {lab:>{w}s} {qt:28s} b={b:3d} c={cc:3d} p={mcnemar_exact(b, cc):.3f}")
    print()
    for lab, per in table:
        tot = sum(v[0] for v in per.values())
        secs = sum(v[2] for v in per.values())
        judged = sum(1 for qid in common if runs[[l for l, _ in runs].index(lab)][1][qid].get("judge_match"))
        print(f"{lab:>{w}s}: mean reader time {secs / tot:.1f}s, judge rescues {judged}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
