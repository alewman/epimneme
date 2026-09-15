#!/usr/bin/env python3
"""Offline RRF channel ablation over a capture from `capture_channels.py`.

One expensive retrieval pass is captured once; this re-fuses any subset of the
channels in seconds and scores it, so "does BM25 earn its place?" costs no
ingest and no server.

WHAT THIS MEASURES — read before quoting a number
-------------------------------------------------
The capture records the *inputs* to RRF: each channel's pre-fusion ranked list.
This script reproduces the RRF stage exactly (same formula, same k, same
per-channel weights) and scores its output.

It does NOT reproduce the post-fusion stages of `MemoryManager.recall`:
proper-noun boost, decay scoring, keyword rerank, session-recency/vague-query
boosts, temporal boost, MMR diversification, gap-aware tiebreak and temporal
partition all run after fusion and are not replayable from the capture. So the
absolute R@k here is NOT the pipeline's R@k. The `baseline_gap` row printed by
`--check` quantifies the difference against the live final ranking stored in
the same capture.

Use it for RELATIVE comparison between channel subsets, then confirm the one or
two configurations that look best with a real retrieval run. That is the same
screen-then-confirm pattern the reader-side ablations used.

Usage:
    python benchmarks/ablate_channels.py --capture benchmarks/channels_v700.jsonl
    python benchmarks/ablate_channels.py --capture ... --check      # fidelity report
    python benchmarks/ablate_channels.py --capture ... --by-type
    python benchmarks/ablate_channels.py --capture ... --only semantic,bm25
    python benchmarks/ablate_channels.py --capture A.jsonl --compare B.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from math import comb
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from metrics import evaluate_retrieval, evidence_completeness, session_id_from_corpus_id  # noqa: E402

RRF_K = 60  # mirrors epimneme.fusion.RRF_K
KS = (1, 3, 5, 10, 30, 50)


def rrf(channel_lists: dict[str, list[str]], weights: dict[str, float]) -> list[str]:
    """Reciprocal Rank Fusion over corpus_id lists — mirrors fusion.rrf_fuse.

    score(doc) = sum over channels containing it of w / (k + rank), rank 1-based.
    Ties broken by first appearance, matching dict insertion order in the
    original (which iterates channels in order and preserves first-seen).
    """
    scores: dict[str, float] = {}
    order: dict[str, int] = {}
    seq = 0
    for name, ids in channel_lists.items():
        w = weights.get(name, 1.0)
        for rank, cid in enumerate(ids):
            scores[cid] = scores.get(cid, 0.0) + w / (RRF_K + rank + 1)
            if cid not in order:
                order[cid] = seq
                seq += 1
    return sorted(scores, key=lambda c: (-scores[c], order[c]))


def score_ranking(ranked: list[str], row: dict) -> dict[str, float]:
    """Session- and turn-level metrics for one ranked corpus_id list."""
    # Match longmemeval_bench: pad with unretrieved corpus ids so recall@k is
    # computed over a full-length list rather than a truncated one.
    seen = set(ranked)
    padded = ranked + [c for c in row["corpus_ids"] if c not in seen]
    session_ids = [session_id_from_corpus_id(c) for c in padded]
    session_correct = set(row["session_correct"])
    turn_correct = set(row["turn_correct"])

    out: dict[str, float] = {}
    for k in KS:
        ra, rl, nd = evaluate_retrieval(session_ids, session_correct, k)
        out[f"s_any@{k}"] = ra
        out[f"s_all@{k}"] = rl
        out[f"s_ndcg@{k}"] = nd
        out[f"s_ec@{k}"] = evidence_completeness(session_ids, session_correct, k)
        ra_t, rl_t, nd_t = evaluate_retrieval(padded, turn_correct, k)
        out[f"t_any@{k}"] = ra_t
        out[f"t_all@{k}"] = rl_t
        out[f"t_ec@{k}"] = evidence_completeness(padded, turn_correct, k)
    return out


def mean(rows: list[dict], key: str) -> float:
    vals = [r[key] for r in rows if key in r]
    return sum(vals) / len(vals) if vals else float("nan")


def load(path: str) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def run_config(capture: list[dict], drop: set[str], only: set[str] | None) -> list[dict]:
    scored = []
    for row in capture:
        chans = {
            n: ids
            for n, ids in row["channels"].items()
            if n not in drop and (only is None or n in only)
        }
        if not chans:
            continue
        ranked = rrf(chans, row.get("channel_weights") or {})
        s = score_ranking(ranked, row)
        s["question_type"] = row["question_type"]
        scored.append(s)
    return scored


def exact_mcnemar(wins: int, losses: int) -> float:
    """Two-sided exact binomial test on the discordant pairs."""
    n = wins + losses
    if n == 0:
        return 1.0
    lo = min(wins, losses)
    return min(1.0, 2 * sum(comb(n, i) for i in range(lo + 1)) / 2**n)


def compare_live(base_path: str, other_path: str) -> int:
    """Paired comparison of two captures on their LIVE final rankings.

    The offline replay above only reaches the fusion stage. When a config change
    is made at the server and a second capture taken, `final_ranked` holds the
    real pipeline's answer for both, on the same questions — so the two can be
    compared directly, with no replay and no missing post-fusion stages.
    """
    a = {r["question_id"]: r for r in load(base_path)}
    b = {r["question_id"]: r for r in load(other_path)}
    common = sorted(set(a) & set(b))
    if not common:
        print("ERROR: the two captures share no question ids", file=sys.stderr)
        return 2

    print(f"paired live comparison — {len(common)} shared questions")
    print(f"  A (baseline): {base_path}")
    print(f"  B (variant):  {other_path}")
    wa, wb = a[common[0]].get("channel_weights"), b[common[0]].get("channel_weights")
    if wa != wb:
        diff = {k: (wa.get(k), wb.get(k)) for k in set(wa or {}) | set(wb or {}) if (wa or {}).get(k) != (wb or {}).get(k)}
        print(f"  channel weights differ: {diff}")
    else:
        print("  !! both captures record the SAME channel weights — is B really a variant?")

    # A capture is only comparable if the two runs saw the same corpus and gold.
    bad = [q for q in common if set(a[q]["session_correct"]) != set(b[q]["session_correct"])]
    if bad:
        print(f"  !! {len(bad)} questions have differing gold sets — not comparable")
    drift = sum(1 for q in common if a[q]["channels"] != b[q]["channels"])
    print(f"  pre-fusion channel lists identical on {len(common) - drift}/{len(common)} questions")
    moved = sum(1 for q in common if a[q]["final_ranked"][:10] != b[q]["final_ranked"][:10])
    print(f"  live top-10 ordering changed on {moved}/{len(common)} questions"
          f" — if this is ~0 the knob never reached the pipeline\n")

    sa = {q: score_ranking(a[q]["final_ranked"], a[q]) for q in common}
    sb = {q: score_ranking(b[q]["final_ranked"], b[q]) for q in common}
    n = len(common)

    print(f"  {'metric':10s} {'A':>7s} {'B':>7s} {'delta':>8s} {'B fixes':>8s} {'B breaks':>9s} {'p':>7s}")
    for k in ("s_any@1", "s_any@3", "s_any@5", "s_any@10"):
        ma = sum(sa[q][k] for q in common) / n
        mb = sum(sb[q][k] for q in common) / n
        wins = sum(1 for q in common if sb[q][k] > sa[q][k])
        losses = sum(1 for q in common if sa[q][k] > sb[q][k])
        p = exact_mcnemar(wins, losses)
        print(f"  {k:10s} {ma:7.3f} {mb:7.3f} {mb - ma:+8.3f} {wins:8d} {losses:9d} {p:7.3f}")
    for k in ("t_all@10", "t_ec@10"):
        ma = sum(sa[q][k] for q in common) / n
        mb = sum(sb[q][k] for q in common) / n
        print(f"  {k:10s} {ma:7.3f} {mb:7.3f} {mb - ma:+8.3f} {'—':>8s} {'—':>9s} {'—':>7s}")
    print("\n  p is a two-sided exact binomial test over the discordant pairs. Recall@k\n"
          "  is binary per question so it admits one; the evidence-completeness rows\n"
          "  are continuous and report means only.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--capture", default="benchmarks/channels_v700.jsonl")
    ap.add_argument("--only", default="", help="Comma-separated channels to keep (default: all)")
    ap.add_argument("--metric", default="s_any@5", help="Metric column to rank configs by")
    ap.add_argument("--by-type", action="store_true", help="Break the baseline down per question type")
    ap.add_argument("--check", action="store_true", help="Report fidelity vs the live final ranking")
    ap.add_argument("--compare", default="",
                    help="Second capture: compare LIVE final rankings pairwise against --capture")
    args = ap.parse_args()

    if not Path(args.capture).exists():
        print(f"ERROR: capture not found: {args.capture}\n"
              f"Run: python benchmarks/capture_channels.py --out {args.capture}", file=sys.stderr)
        return 2

    if args.compare:
        if not Path(args.compare).exists():
            print(f"ERROR: capture not found: {args.compare}", file=sys.stderr)
            return 2
        return compare_live(args.capture, args.compare)

    capture = load(args.capture)
    if not capture:
        print(f"ERROR: {args.capture} is empty", file=sys.stderr)
        return 2

    all_channels: list[str] = []
    for row in capture:
        for n in row["channels"]:
            if n not in all_channels:
                all_channels.append(n)
    only = {c.strip() for c in args.only.split(",") if c.strip()} or None

    print(f"capture: {args.capture}  ({len(capture)} questions)")
    print(f"channels present: {', '.join(all_channels)}")
    coverage = {n: sum(1 for r in capture if r['channels'].get(n)) for n in all_channels}
    print(f"questions where the channel fired: { {n: coverage[n] for n in all_channels} }")
    unmapped = sum(r.get("unmapped_channel_entries", 0) for r in capture)
    if unmapped:
        print(f"  !! {unmapped} channel entries could not be mapped to a corpus_id")
    print()

    base = run_config(capture, set(), only)
    if args.check:
        live = []
        for row in capture:
            s = score_ranking(row["final_ranked"], row)
            s["question_type"] = row["question_type"]
            live.append(s)
        print("fidelity — RRF-only replay vs the live pipeline's final ranking:")
        print(f"  {'metric':12s} {'replay':>8s} {'live':>8s} {'gap':>8s}")
        for k in ("s_any@1", "s_any@5", "s_any@10", "t_all@10", "t_ec@10"):
            r, l = mean(base, k), mean(live, k)
            print(f"  {k:12s} {r:8.3f} {l:8.3f} {r - l:+8.3f}")
        print("  (post-fusion boosts, rerank, MMR and tiebreak are not replayable —\n"
              "   treat the gap as the error bar on absolute numbers, not on deltas.)\n")

    if args.by_type:
        print(f"baseline (all channels) by question type, {args.metric}:")
        per = defaultdict(list)
        for s in base:
            per[s["question_type"]].append(s)
        for t in sorted(per):
            print(f"  {t:28s} n={len(per[t]):3d}  {mean(per[t], args.metric):.3f}")
        print()

    cols = ("s_any@1", "s_any@5", "s_any@10", "t_all@10", "t_ec@10")
    print(f"leave-one-out ablation (metric ordering by {args.metric}):")
    header = f"  {'config':22s}" + "".join(f"{c:>10s}" for c in cols) + f"{'delta':>9s}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    base_m = mean(base, args.metric)
    print(f"  {'all channels':22s}" + "".join(f"{mean(base, c):10.3f}" for c in cols) + f"{0.0:+9.3f}")
    rows_out = []
    for ch in all_channels:
        if only and ch not in only:
            continue
        cfg = run_config(capture, {ch}, only)
        if not cfg:
            continue
        rows_out.append((mean(cfg, args.metric) - base_m, ch, cfg))
    for delta, ch, cfg in sorted(rows_out):
        print(f"  {'− ' + ch:22s}" + "".join(f"{mean(cfg, c):10.3f}" for c in cols) + f"{delta:+9.3f}")
    print("\n  A channel whose removal has delta >= 0 is not paying for itself on this\n"
          "  metric. Confirm any candidate for removal with a live retrieval run.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
