#!/usr/bin/env python3
"""Third pass — uses the dataset's real per-turn `has_answer` ground-truth
flag instead of substring matching. No heuristic, no fallback, no
circularity: gold = turn-pairs where the underlying user or assistant
message has has_answer=True. Questions with zero such turns (true
abstention-adjacent cases) are excluded and reported as unmeasured, not
scored as failures.

Reports, per type, with n stated:
  - binary recall_all@10 AND evidence_completeness@10 side by side (same
    gold, same k, same 500-question run) to settle whether they actually
    reconcile.
  - simulated session-cap sweep on the same real gold.
  - answer-POSITION test (not total length): for has_answer turn-pairs,
    where in the combined [USER]/[ASSISTANT] text does the answer-bearing
    message start? Compares that offset between turns that were retrieved
    (found in top-50) vs missed, against the embedder's ~1300-char window.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from metrics import session_id_from_corpus_id

EMBED_WINDOW_CHARS = 1300  # ~256 tokens at ~5 chars/token, all-MiniLM-L6-v2


def build_corpus_with_has_answer(entry: dict) -> tuple[list[str], list[str], list[int]]:
    """Reimplements longmemeval_bench.build_corpus's turn-pair walk, but
    also tracks (a) whether the resulting turn-pair contains a has_answer=True
    message, and (b) the character offset within the combined text where
    that answer-bearing message's content begins (-1 if not has_answer)."""
    corpus, corpus_ids, offsets = [], [], []
    for sess_idx, (session, sess_id, date) in enumerate(
        zip(entry["haystack_sessions"], entry["haystack_session_ids"], entry["haystack_dates"])
    ):
        turn_num = 0
        i = 0
        while i < len(session):
            turn = session[i]
            if turn["role"] == "user":
                user_text = turn["content"]
                user_has_answer = bool(turn.get("has_answer"))
                asst_text, asst_has_answer = "", False
                if i + 1 < len(session) and session[i + 1]["role"] == "assistant":
                    asst_text = session[i + 1]["content"]
                    asst_has_answer = bool(session[i + 1].get("has_answer"))
                    i += 1
                header = f"[Date: {date}]"
                user_prefix = f"{header}\n[USER]: "
                doc = user_prefix + user_text
                user_offset = len(user_prefix)
                asst_offset = -1
                if asst_text:
                    asst_prefix = f"\n[ASSISTANT]: "
                    asst_offset = len(doc) + len(asst_prefix)
                    doc = doc + asst_prefix + asst_text

                if user_has_answer or asst_has_answer:
                    offset = user_offset if user_has_answer else asst_offset
                else:
                    offset = -1

                corpus.append(doc)
                corpus_ids.append(f"{sess_id}_turn_{turn_num}")
                offsets.append(offset)
                turn_num += 1
            i += 1
    return corpus, corpus_ids, offsets


def capped_top_k(ranked_ids: list[str], k: int, session_cap: int) -> list[str]:
    out: list[str] = []
    counts: dict[str, int] = defaultdict(int)
    for cid in ranked_ids:
        if len(out) >= k:
            break
        sid = session_id_from_corpus_id(cid)
        if counts[sid] < session_cap:
            out.append(cid)
            counts[sid] += 1
    if len(out) < k:
        for cid in ranked_ids:
            if len(out) >= k:
                break
            if cid not in out:
                out.append(cid)
    return out


def main() -> None:
    results_file = sys.argv[1] if len(sys.argv) > 1 else "benchmarks/results_engram_lme_v700-baseline_20260904.jsonl"
    data_file = sys.argv[2] if len(sys.argv) > 2 else "benchmarks/data/longmemeval_s_cleaned.json"

    with open(data_file) as f:
        dataset = {e["question_id"]: e for e in json.load(f)}
    rows = [json.loads(l) for l in open(results_file)]

    cache: dict[str, dict] = {}
    unmeasured = Counter()
    unmeasured_total = Counter()

    for row in rows:
        # _abs variants are DISTINCT dataset entries (different haystack,
        # different answer_session_ids) — look up the exact question_id
        # first. Previously this fell back to the base (non-abs) entry
        # whenever it existed, silently scoring all _abs rows against the
        # wrong haystack.
        base_qid = row["question_id"].replace("_abs", "")
        entry = dataset.get(row["question_id"]) or dataset.get(base_qid)
        if entry is None:
            continue

        corpus, corpus_ids, offsets = build_corpus_with_has_answer(entry)
        answer_sids = set(entry["answer_session_ids"])
        gold = {
            cid for cid, sess_cid_offset in zip(corpus_ids, offsets)
            if sess_cid_offset >= 0 and session_id_from_corpus_id(cid) in answer_sids
        }

        unmeasured_total[row["question_type"]] += 1
        if not gold:
            unmeasured[row["question_type"]] += 1
            continue

        ranked = [it["corpus_id"] for it in row["retrieval_results"]["ranked_items"]]
        offset_by_id = {cid: off for cid, off in zip(corpus_ids, offsets)}
        text_by_id = {cid: text for cid, text in zip(corpus_ids, corpus)}

        cache[row["question_id"]] = {
            "qtype": row["question_type"],
            "gold": gold,
            "ranked": ranked,
            "offset_by_id": offset_by_id,
            "text_by_id": text_by_id,
        }

    print("=" * 78)
    print("  MEASURABILITY — questions with >=1 has_answer=True turn in the answer session")
    print("=" * 78)
    for qt in sorted(unmeasured_total):
        u, t = unmeasured[qt], unmeasured_total[qt]
        print(f"  {qt:28} measurable={t - u:3}/{t:3}   unmeasured={u:3}")

    # ── (A)/(B) reconciliation: binary recall_all@10 vs evidence_completeness@10,
    #    SAME gold, SAME k, printed together, per type, with n. ──────────────
    print("\n" + "=" * 78)
    print("  BINARY recall_all@10  vs  evidence_completeness@10  (same real gold, k=10)")
    print("=" * 78)
    by_type = defaultdict(list)
    for qid, c in cache.items():
        top10 = set(c["ranked"][:10])
        exact = c["gold"].issubset(top10)
        ec = len(c["gold"] & top10) / len(c["gold"])
        by_type[c["qtype"]].append((exact, ec, len(c["gold"])))

    for qt in sorted(by_type):
        vals = by_type[qt]
        n = len(vals)
        exact_rate = sum(1 for e, _, _ in vals if e) / n
        avg_ec = sum(ec for _, ec, _ in vals) / n
        avg_gold_size = sum(g for _, _, g in vals) / n
        se = (exact_rate * (1 - exact_rate) / n) ** 0.5
        print(f"  {qt:28} n={n:3}  recall_all@10={exact_rate*100:5.1f}% (+/-{se*100:.1f}pp)  "
              f"evidence_completeness@10={avg_ec*100:5.1f}%  avg_gold_size={avg_gold_size:.2f}")

    # ── Session-cap sweep on the SAME real gold ──────────────────────────
    print("\n" + "=" * 78)
    print("  SESSION-CAP SWEEP — evidence_completeness@10, real gold, per type")
    print("=" * 78)
    for cap in (1, 2, 3, 99):
        print(f"\n  --- session_cap={cap} ---")
        for qt in sorted(by_type):
            vals = []
            for qid, c in cache.items():
                if c["qtype"] != qt:
                    continue
                capped = set(capped_top_k(c["ranked"], 10, cap))
                vals.append(len(c["gold"] & capped) / len(c["gold"]))
            print(f"    {qt:28} {sum(vals)/len(vals)*100:5.1f}%  (n={len(vals)})")

    # ── Answer-position test (not total length) ──────────────────────────
    print("\n" + "=" * 78)
    print(f"  ANSWER-POSITION TEST — offset of the has_answer message within its turn-pair text")
    print(f"  (embedder window ~{EMBED_WINDOW_CHARS} chars — content past this is invisible to the vector)")
    print("=" * 78)

    found_offsets, missed_offsets = [], []
    for qid, c in cache.items():
        ranked_all50 = set(c["ranked"])
        top10 = set(c["ranked"][:10])
        for cid in c["gold"]:
            off = c["offset_by_id"][cid]
            if cid in ranked_all50:
                found_offsets.append(off)
            else:
                missed_offsets.append(off)

    def stats(label, offs):
        if not offs:
            print(f"  {label}: n=0")
            return
        past_window = sum(1 for o in offs if o > EMBED_WINDOW_CHARS)
        print(f"  {label}: n={len(offs)}  median_offset={sorted(offs)[len(offs)//2]}  "
              f"mean_offset={sum(offs)/len(offs):.0f}  "
              f"past_{EMBED_WINDOW_CHARS}char_window={past_window} ({past_window/len(offs)*100:.1f}%)")

    stats("FOUND  (in top-50)", found_offsets)
    stats("MISSED (absent from top-50)", missed_offsets)

    # ── Scope: how much of the gap is even reachable by a rerank-stage fix? ──
    # A cap/penalty at final reranking can only reorder items already in the
    # top-50 pool. If a gold turn was never fetched into that pool at all,
    # no rerank-stage intervention could ever have surfaced it.
    print("\n" + "=" * 78)
    print("  SCOPE — of MISSING gold turns (real has_answer gold), how many were")
    print("  never in the top-50 pool at all vs. present but ranked 11-50?")
    print("=" * 78)
    never_in_pool = 0
    ranked_low = 0
    for qid, c in cache.items():
        top10 = set(c["ranked"][:10])
        pool50 = set(c["ranked"])
        missing = c["gold"] - top10
        for mid in missing:
            if mid in pool50:
                ranked_low += 1
            else:
                never_in_pool += 1
    total_missing = never_in_pool + ranked_low
    if total_missing:
        print(f"  never in top-50 pool:  {never_in_pool:4} ({never_in_pool / total_missing * 100:.1f}%)")
        print(f"  in pool, ranked 11-50: {ranked_low:4} ({ranked_low / total_missing * 100:.1f}%)")
        print(f"  => a rerank-stage fix (cap/penalty) is structurally capped at addressing "
              f"{ranked_low / total_missing * 100:.1f}% of missing turns, at best")

    # ── Null-model check on same-session vs other-session crowding ──────────
    # Raw counts of "how many missing turns had >=1 same-session competitor
    # in top-10" vs "...>=1 different-session competitor" are not comparable
    # without a base rate: with dozens of distinct sessions in a haystack,
    # "at least one OTHER session in top-10" is true almost by construction,
    # while "the ONE specific correct session is in top-10" is a much rarer
    # event under a random-assignment null. Compare against that null instead.
    print("\n" + "=" * 78)
    print("  NULL-MODEL CHECK — is the missing turn's own session over- or under-")
    print("  represented in top-10, relative to a random-session baseline?")
    print("=" * 78)
    obs_same_session_present = 0
    expected_same_session_present = 0.0
    n_missing_checked = 0
    for qid, c in cache.items():
        top10 = c["ranked"][:10]
        top10_sessions = [session_id_from_corpus_id(cid) for cid in top10]
        all_sessions_in_pool = {session_id_from_corpus_id(cid) for cid in c["ranked"]}
        n_distinct_sessions = max(len(all_sessions_in_pool), 1)
        top10 = set(top10)
        missing = c["gold"] - top10
        for mid in missing:
            msid = session_id_from_corpus_id(mid)
            n_missing_checked += 1
            if msid in top10_sessions:
                obs_same_session_present += 1
            # Null: probability that a FIXED session appears at least once
            # among 10 draws without replacement from n_distinct_sessions,
            # approximated as 1 - C(n-1,10)/C(n,10) = 10/n (for n>10).
            expected_same_session_present += min(1.0, 10 / n_distinct_sessions)

    if n_missing_checked:
        obs_rate = obs_same_session_present / n_missing_checked
        exp_rate = expected_same_session_present / n_missing_checked
        print(f"  n missing gold turns checked: {n_missing_checked}")
        print(f"  observed: correct session present somewhere in top-10: {obs_rate*100:.1f}%")
        print(f"  null baseline (random session draw, same pool sizes):   {exp_rate*100:.1f}%")
        print(f"  enrichment: {obs_rate / exp_rate:.2f}x over chance" if exp_rate > 0 else "")


if __name__ == "__main__":
    main()
