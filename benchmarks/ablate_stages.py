#!/usr/bin/env python3
"""Live leave-one-out ablation of the POST-fusion recall stages.

Why this is a separate script from `ablate_channels.py`
------------------------------------------------------
`ablate_channels.py` ablates the RRF *channels* by replaying a capture offline.
That replay stops at fusion, and the September 2026 keyword-weight result showed
what that costs: a 21.4pp fusion-stage gain measured exactly zero on the live
pipeline, because the post-fusion stack had been repairing the damage all along.
Post-fusion stages cannot be replayed from a capture, so they are ablated here
the only way that is trustworthy — live, against the real server.

What makes that affordable
--------------------------
Per question, ingest costs ~9s and a query ~0.3s: the corpus, not the query,
is the expense. So each haystack is ingested ONCE and then queried once per
configuration via the `skip` parameter on `/api/memories/search`. A 14-stage
leave-one-out therefore costs roughly one capture run (~90 min for 500
questions), not fourteen.

Two hygiene rules make the shared corpus sound, and both are enforced:

  * `update_access=false` on every query. `recall` normally fires decay/access
    writes for its top 5 results; across N queries on one corpus, config 1 would
    otherwise change the decay state config 2 is scored under.
  * The baseline is re-run LAST as `baseline_check` as well as first. If the two
    disagree on any question, the corpus did not stay constant across the sweep
    and the run says so instead of reporting a clean-looking table.

Usage:
    EPIMNEME_TOKEN=... python benchmarks/ablate_stages.py --limit 25   # smoke
    EPIMNEME_TOKEN=... python benchmarks/ablate_stages.py --out benchmarks/stages_v700.jsonl
    python benchmarks/ablate_stages.py --score benchmarks/stages_v700.jsonl  # offline
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from ablate_channels import exact_mcnemar, mean, score_ranking  # noqa: E402
from epimneme_client import EngramClient  # noqa: E402
from longmemeval_bench import (  # noqa: E402
    build_corpus,
    has_answer_gold,
    cleanup_project,
    ingest_corpus,
    load_data,
)
from metrics import session_id_from_corpus_id  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from epimneme.manager import POSTFUSION_STAGES, RETRIEVAL_CHANNELS  # noqa: E402


class FatalAblationError(RuntimeError):
    """A condition that invalidates the whole run, not just one question."""


# Stages the 2026-09-15 live leave-one-out found changed no answer in 500
# questions. Removing them one at a time is measured; removing them together is
# not, and a leave-one-out cannot see interactions between them.
INERT_STAGES = ["tiebreak", "proper_noun", "temporal_boost", "temporal_partition"]


def build_configs(
    stages: list[str], combos: bool = True
) -> list[tuple[str, list[str]]]:
    """(label, skip-list) pairs.

    baseline, each stage/channel removed alone, any combination probes, then the
    baseline again as `baseline_check`. Order matters only in that the two
    baselines bracket the sweep — everything between them must be scored against
    a corpus that did not move.
    """
    cfgs: list[tuple[str, list[str]]] = [("baseline", [])]
    cfgs += [(f"-{s}", [s]) for s in stages]
    if combos:
        inert = [s for s in INERT_STAGES if s in stages]
        if len(inert) > 1:
            cfgs.append(("-ALL_INERT", inert))
        chans = [c for c in sorted(RETRIEVAL_CHANNELS) if c in stages]
        # The offline replay claimed dropping fulltext was worth +21.4pp and it
        # was worth zero live. Keep the direct live equivalent in the sweep.
        if "fulltext" in chans and "date_proximity" in chans:
            cfgs.append(("-fulltext-date_proximity", ["fulltext", "date_proximity"]))
    cfgs.append(("baseline_check", []))
    return cfgs


async def preflight(client: EngramClient, project_name: str) -> None:
    """Fail loudly if the server predates the `skip` parameter.

    FastAPI silently ignores query parameters it does not declare. Against an
    old build every config would therefore return the *baseline* ranking and the
    leave-one-out table would read as a tidy "no stage matters" — the most
    plausible-looking wrong answer this harness could produce. So we send a
    deliberately invalid stage name and require the server to reject it.
    """
    result = await client.search(
        "preflight", project=project_name, limit=1,
        skip="__nosuchstage__", update_access=False,
    )
    detail = str(result.get("detail", ""))
    # Distinguish an auth failure from a stale build. Both leave `skip`
    # unhonoured, but the fix is completely different and the wrong message
    # sends you rebuilding a container that was fine.
    if "API key" in detail or "Authentication" in detail or "not have access" in detail:
        raise FatalAblationError(
            f"the server rejected the token before `skip` was ever evaluated: {detail}. "
            f"Check EPIMNEME_TOKEN — keys expire (`manage.py list-keys` shows expires_at) "
            f"and must be scoped to the --project namespace."
        )
    if "unknown recall stage" not in detail:
        raise FatalAblationError(
            "server did not reject an invalid --skip stage name, so it is "
            "ignoring `skip` entirely — rebuild and restart the container "
            "(docker compose build engram && docker compose up -d --no-deps engram) "
            f"before running this. Response was: {str(result)[:200]}"
        )


async def ablate_one(
    client: EngramClient,
    entry: dict,
    configs: list[tuple[str, list[str]]],
    limit: int,
    cleanup: bool,
    project_name: str,
) -> dict | None:
    """Ingest one haystack once, then query it once per configuration."""
    qid = entry["question_id"]
    corpus, corpus_ids, corpus_ts = build_corpus(entry, granularity="turn-pair")

    t0 = time.monotonic()
    stored = await ingest_corpus(client, project_name, corpus, corpus_ids, corpus_ts)
    ingest_elapsed = time.monotonic() - t0

    try:
        rankings: dict[str, list[str]] = {}
        t0 = time.monotonic()
        for label, skip in configs:
            result = await client.search(
                entry["question"],
                project=project_name,
                limit=limit,
                reference_date=entry.get("question_date"),
                skip=",".join(skip) or None,
                update_access=False,
            )
            rankings[label] = [
                r.get("subject") for r in result.get("results", []) if r.get("subject")
            ]
        query_elapsed = time.monotonic() - t0

        if not rankings.get("baseline"):
            raise FatalAblationError(
                f"{qid}: baseline query returned nothing — is the server current "
                f"and the corpus ingested?"
            )

        answer_sids = set(entry["answer_session_ids"]) if isinstance(
            entry["answer_session_ids"], list
        ) else set(json.loads(str(entry["answer_session_ids"]).replace("'", '"')))
        # Real evidence turns. The old definition here ("every turn of a gold
        # session") inflates the gold ~6x and turns t_ec@10 into session recall
        # wearing a different name — it reported that removing keyword_rerank
        # IMPROVED evidence by +3.5pp when on true gold it is -0.4pp and not
        # significant. Questions with no flagged turn are excluded from turn
        # averages by the scorer.
        turn_correct = sorted(has_answer_gold(entry))

        return {
            "question_id": qid,
            "question_type": entry.get("question_type", "unknown"),
            "question": entry["question"],
            "question_date": entry.get("question_date"),
            "stored": stored,
            "corpus_size": len(corpus_ids),
            "rankings": rankings,
            "corpus_ids": corpus_ids,
            "session_correct": sorted(answer_sids),
            "turn_correct": turn_correct,
            "ingest_time": round(ingest_elapsed, 2),
            "query_time_total": round(query_elapsed, 2),
        }
    finally:
        if cleanup:
            await cleanup_project(client, project_name)
            leftover = await client.list_all_memories(project_name)
            if leftover:
                raise FatalAblationError(
                    f"{project_name} still holds {len(leftover)} memories after "
                    f"cleanup following {qid} — aborting rather than contaminating "
                    f"the next haystack."
                )


def score(path: str, metric: str = "s_any@5") -> int:
    """Score a completed run: leave-one-out table with paired significance."""
    rows = [json.loads(l) for l in open(path) if l.strip()]
    if not rows:
        print(f"ERROR: {path} is empty", file=sys.stderr)
        return 2

    labels = list(rows[0]["rankings"].keys())
    print(f"capture: {path}  ({len(rows)} questions, {len(labels)} configs)\n")

    # Integrity gate first — everything below is meaningless if it fails.
    drift = [
        r["question_id"] for r in rows
        if r["rankings"].get("baseline") != r["rankings"].get("baseline_check")
    ]
    if drift:
        print(f"  !! CORPUS DRIFT: baseline and baseline_check disagree on "
              f"{len(drift)}/{len(rows)} questions.")
        print(f"     e.g. {drift[:5]}")
        print("     The corpus did not stay constant across each question's sweep,\n"
              "     so per-config differences are not attributable to the stage.\n")
    else:
        print(f"  integrity: baseline == baseline_check on all {len(rows)} questions\n")

    scored: dict[str, list[dict]] = {}
    for label in labels:
        out = []
        for r in rows:
            s = score_ranking(r["rankings"].get(label, []), r)
            s["question_type"] = r["question_type"]
            s["question_id"] = r["question_id"]
            out.append(s)
        scored[label] = out

    base = scored["baseline"]
    base_by_q = {s["question_id"]: s for s in base}
    cols = ("s_any@1", "s_any@5", "s_any@10", "t_all@10", "t_ec@10")

    print(f"live leave-one-out over post-fusion stages (ordered by {metric}):")
    header = (f"  {'config':22s}" + "".join(f"{c:>10s}" for c in cols)
              + f"{'delta':>9s}{'fixes':>7s}{'breaks':>8s}{'p':>8s}")
    print(header)
    print("  " + "-" * (len(header) - 2))

    base_m = mean(base, metric)
    print(f"  {'baseline':22s}" + "".join(f"{mean(base, c):10.3f}" for c in cols)
          + f"{0.0:+9.3f}{'—':>7s}{'—':>8s}{'—':>8s}")

    out_rows = []
    for label in labels:
        if label in ("baseline", "baseline_check"):
            continue
        cfg = scored[label]
        wins = sum(1 for s in cfg if s[metric] > base_by_q[s["question_id"]][metric])
        losses = sum(1 for s in cfg if s[metric] < base_by_q[s["question_id"]][metric])
        out_rows.append((mean(cfg, metric) - base_m, label, cfg, wins, losses))

    for delta, label, cfg, wins, losses in sorted(out_rows):
        p = exact_mcnemar(wins, losses)
        print(f"  {label:22s}" + "".join(f"{mean(cfg, c):10.3f}" for c in cols)
              + f"{delta:+9.3f}{wins:7d}{losses:8d}{p:8.3f}")

    print("\n  Removing a stage that MATTERS should hurt (delta < 0). A stage whose\n"
          "  removal is +0.000 with zero discordant questions never fired or never\n"
          "  changed an answer; one with delta >= 0 is not paying for itself.\n"
          "  p is a two-sided exact binomial over the discordant pairs.")
    return 0


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-file", default="benchmarks/data/longmemeval_s_cleaned.json")
    ap.add_argument("--engram-url", default="http://localhost:8000")
    ap.add_argument("--out", default="benchmarks/stages_v700.jsonl")
    ap.add_argument("--limit", type=int, default=0, help="Only first N questions (0=all)")
    ap.add_argument("--search-limit", type=int, default=50, help="Final list depth to record")
    ap.add_argument("--project", default="_chancap",
                    help="Reused benchmark namespace. `AuthContext.can_access_project` "
                         "matches exact strings with no globbing, so the benchmark key "
                         "is scoped to this one name — a new name needs a new key.")
    ap.add_argument("--token", default="", help="Bearer token (or set EPIMNEME_TOKEN)")
    ap.add_argument("--stages", default="",
                    help="Comma-separated stages to ablate (default: all)")
    ap.add_argument("--no-cleanup", action="store_true")
    ap.add_argument("--no-combos", action="store_true",
                    help="Leave-one-out only; skip the combination probes")
    ap.add_argument("--combo", default="",
                    help="Also score one explicit combination, comma-separated. Use this "
                         "to measure the exact configuration you intend to ship: a "
                         "leave-one-out proves each stage is individually free, never "
                         "that removing several together is.")
    ap.add_argument("--score", default="",
                    help="Score an existing run file and exit (no server needed)")
    ap.add_argument("--metric", default="s_any@5")
    args = ap.parse_args()

    if args.score:
        return score(args.score, args.metric)

    token = args.token or os.environ.get("EPIMNEME_TOKEN", "")
    if not token:
        print("ERROR: pass --token or set EPIMNEME_TOKEN", file=sys.stderr)
        return 2

    valid = POSTFUSION_STAGES | RETRIEVAL_CHANNELS
    stages = ([x.strip() for x in args.stages.split(",") if x.strip()]
              if args.stages else sorted(valid))
    unknown = sorted(set(stages) - valid)
    if unknown:
        print(f"ERROR: unknown stage(s): {unknown}\n"
              f"valid: {sorted(valid)}", file=sys.stderr)
        return 2
    configs = build_configs(stages, combos=not args.no_combos)
    if args.combo:
        combo = [x.strip() for x in args.combo.split(",") if x.strip()]
        unknown_c = sorted(set(combo) - valid)
        if unknown_c:
            print(f"ERROR: unknown stage(s) in --combo: {unknown_c}", file=sys.stderr)
            return 2
        # insert before the trailing baseline_check so the two baselines still
        # bracket every scored configuration
        configs.insert(len(configs) - 1, ("-COMBO:" + "+".join(combo), combo))

    entries = load_data(args.data_file)
    if args.limit:
        entries = entries[: args.limit]

    print(f"{len(entries)} questions x {len(configs)} configs "
          f"= {len(entries) * len(configs)} queries over {len(entries)} ingests")
    print(f"stages: {', '.join(stages)}\n")

    written = failures = 0
    t_start = time.monotonic()
    client = EngramClient(args.engram_url, token=token)
    try:
        try:
            await preflight(client, args.project)
        except FatalAblationError as exc:
            print(f"FATAL: {exc}", file=sys.stderr)
            return 1
        print("preflight: server honours `skip`\n")

        with open(args.out, "w") as fh:
            for i, entry in enumerate(entries, 1):
                try:
                    row = await ablate_one(
                        client, entry, configs, args.search_limit,
                        not args.no_cleanup, args.project,
                    )
                except FatalAblationError as exc:
                    print(f"\nFATAL: {exc}", file=sys.stderr)
                    return 1
                except Exception as exc:  # noqa: BLE001
                    failures += 1
                    print(f"  !! {entry['question_id']}: {type(exc).__name__}: {exc}")
                    continue
                if row is None:
                    failures += 1
                    continue
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                written += 1
                rate = (time.monotonic() - t_start) / i
                eta = rate * (len(entries) - i) / 60
                print(f"  [{i:4d}/{len(entries)}] {row['question_id']:12s} "
                      f"ingest {row['ingest_time']:5.1f}s  "
                      f"{len(configs)} queries {row['query_time_total']:5.1f}s  "
                      f"ETA={eta:.0f}m")
    finally:
        await client.close()

    print(f"\nDone. {written} written, {failures} failures. Output: {args.out}")
    if written:
        print()
        score(args.out, args.metric)
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
