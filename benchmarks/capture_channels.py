#!/usr/bin/env python3
"""Capture every RRF channel's pre-fusion ranked list, once, for offline ablation.

Why this exists
---------------
The reader-side ablations (Phase 3) were cheap because a *single* expensive
retrieval pass was saved to disk and every later experiment replayed it. The
retrieval side has no such artefact: each channel ablation would otherwise mean
re-ingesting 500 haystacks (~3.5 h) just to answer "does BM25 earn its place?".

This script does the expensive pass exactly once. For each LongMemEval question
it ingests the haystack, enumerates the project to build a reliable
memory_id -> corpus_id map, queries with `debug=true`, and writes every
channel's full pre-fusion ranked list translated into corpus_ids, along with the
channel weights, the live pipeline's final ranked list, and the gold sets needed
to score. `ablate_channels.py` then re-fuses any subset of channels in seconds.

What it captures is the RRF *input*. Post-fusion stages (proper-noun boost,
decay, keyword rerank, recency/vague boosts, temporal boost, MMR, tiebreak,
temporal partition) are NOT replayable from this file — see the honesty note in
`ablate_channels.py` about what that means for the numbers.

Usage:
    EPIMNEME_TOKEN=... python benchmarks/capture_channels.py --out benchmarks/channels_v700.jsonl
    python benchmarks/capture_channels.py --limit 25          # smoke test
    python benchmarks/capture_channels.py --no-cleanup        # keep projects staged
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

from epimneme_client import EngramClient  # noqa: E402
from longmemeval_bench import (  # noqa: E402
    build_corpus,
    cleanup_project,
    ingest_corpus,
    load_data,
)
from metrics import session_id_from_corpus_id  # noqa: E402


class FatalCaptureError(RuntimeError):
    """A condition that invalidates the whole run, not just one question.

    Deliberately distinct from the per-question exceptions the main loop
    tolerates: a namespace that will not empty, or a capture that records no
    usable channel data, corrupts everything after it and must stop the run.
    """


async def capture_one(
    client: EngramClient,
    entry: dict,
    limit: int,
    cleanup: bool,
    project_name: str,
) -> dict | None:
    """Ingest one haystack, query with debug, return the channel snapshot.

    One project namespace is reused for every question (emptied between them)
    rather than a project per question. `AuthContext.can_access_project` does
    exact string matching with no globbing, so per-question names would force
    the benchmark key to carry the `*` wildcard — i.e. access to every project
    on the server. A single fixed namespace lets the key be scoped to exactly
    that one name.
    """
    qid = entry["question_id"]

    corpus, corpus_ids, corpus_ts = build_corpus(entry, granularity="turn-pair")

    t0 = time.monotonic()
    stored = await ingest_corpus(client, project_name, corpus, corpus_ids, corpus_ts)
    ingest_elapsed = time.monotonic() - t0

    try:
        # Reliable memory_id -> corpus_id map. A relevance-scored search only
        # returns the top N, so channel entries outside that window would be
        # unmappable; direct enumeration visits every memory exactly once.
        all_memories = await client.list_all_memories(project_name)
        id_to_subject = {m["id"]: m.get("subject") for m in all_memories}

        t0 = time.monotonic()
        result = await client.search(
            entry["question"],
            project=project_name,
            limit=limit,
            reference_date=entry.get("question_date"),
            debug=True,
        )
        query_elapsed = time.monotonic() - t0

        debug = result.get("debug") or {}
        raw_channels = debug.get("channels") or {}
        if not raw_channels:
            print(f"  !! {qid}: no debug channels in response — is this build current?")
            return None

        # Translate every channel list from memory_id to corpus_id. Anything
        # that fails to map is recorded so a silent gap can't masquerade as a
        # channel that simply ranks poorly.
        channels: dict[str, list[str]] = {}
        unmapped = 0
        for name, ids in raw_channels.items():
            out = []
            for mid in ids:
                subj = id_to_subject.get(mid)
                if subj is None:
                    unmapped += 1
                    continue
                out.append(subj)
            channels[name] = out

        final_ranked = [
            r.get("subject") for r in result.get("results", []) if r.get("subject")
        ]

        answer_sids = set(entry["answer_session_ids"]) if isinstance(
            entry["answer_session_ids"], list
        ) else set(json.loads(str(entry["answer_session_ids"]).replace("'", '"')))

        turn_correct = [
            cid for cid in corpus_ids if session_id_from_corpus_id(cid) in answer_sids
        ]

        return {
            "question_id": qid,
            "question_type": entry.get("question_type", "unknown"),
            "question": entry["question"],
            "question_date": entry.get("question_date"),
            "stored": stored,
            "corpus_size": len(corpus_ids),
            "channels": channels,
            "channel_weights": debug.get("channel_weights") or {},
            "final_ranked": final_ranked,
            "corpus_ids": corpus_ids,
            "session_correct": sorted(answer_sids),
            "turn_correct": turn_correct,
            "unmapped_channel_entries": unmapped,
            "ingest_time": round(ingest_elapsed, 2),
            "query_time": round(query_elapsed, 3),
        }
    finally:
        if cleanup:
            await cleanup_project(client, project_name)
            # A reused namespace is only safe if it is actually empty again:
            # leftovers from question N become distractors in question N+1 and
            # would silently corrupt every metric downstream.
            leftover = await client.list_all_memories(project_name)
            if leftover:
                raise FatalCaptureError(
                    f"{project_name} still holds {len(leftover)} memories after "
                    f"cleanup following {qid} — aborting rather than contaminating "
                    f"the next haystack."
                )


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-file", default="benchmarks/data/longmemeval_s_cleaned.json")
    ap.add_argument("--engram-url", default="http://localhost:8000")
    ap.add_argument("--out", default="benchmarks/channels_v700.jsonl")
    ap.add_argument("--limit", type=int, default=0, help="Only first N questions (0=all)")
    ap.add_argument("--search-limit", type=int, default=50, help="Final list depth to record")
    ap.add_argument("--project", default="_chancap",
                    help="Single project namespace reused for every question; scope the "
                         "benchmark API key to exactly this name")
    ap.add_argument("--token", default="", help="Bearer token (or set EPIMNEME_TOKEN)")
    ap.add_argument("--no-cleanup", action="store_true",
                    help="Leave the last haystack staged (disables the empty-namespace guard)")
    args = ap.parse_args()

    data = load_data(args.data_file)
    if args.limit:
        data = data[: args.limit]

    out_path = Path(args.out)
    done: set[str] = set()
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        done.add(json.loads(line)["question_id"])
                    except Exception:
                        pass
        if done:
            print(f"Resuming: {len(done)} questions already captured, skipping.")

    remaining = [e for e in data if e["question_id"] not in done]
    print(f"Capturing channels for {len(remaining)} questions -> {out_path}")

    token = args.token or os.environ.get("EPIMNEME_TOKEN", "")
    if not token:
        print("ERROR: no API token. Pass --token or set EPIMNEME_TOKEN.\n"
              "  An agent-role key scoped to just this namespace is enough:\n"
              f"    python -m epimneme.manage create-key --name bench-chancap \\\n"
              f"        --role agent --projects {args.project} --expires-in-days 7",
              file=sys.stderr)
        return 2

    client = EngramClient(base_url=args.engram_url, token=token)

    # Pre-flight: prove the key can actually enumerate the namespace before
    # spending hours on a run whose cleanup and id-mapping both depend on it.
    probe = await client.list_all_memories(args.project)
    if probe:
        print(f"ERROR: {args.project} is not empty ({len(probe)} memories). "
              f"Clear it before capturing.", file=sys.stderr)
        await client.close()
        return 2

    fh = open(out_path, "a")
    t_start = time.monotonic()
    failures = 0
    try:
        for i, entry in enumerate(remaining, 1):
            try:
                row = await capture_one(
                    client, entry, args.search_limit, not args.no_cleanup, args.project
                )
            except FatalCaptureError:
                raise
            except Exception as exc:  # keep going; one bad haystack shouldn't kill the pass
                print(f"  [{i}/{len(remaining)}] {entry['question_id']}: FAILED {exc!r}")
                failures += 1
                continue
            if row is None:
                failures += 1
                continue

            # Fail fast on the first row. A silently-empty enumeration makes
            # every channel list empty AND makes the cleanup guard vacuous, so
            # the run looks healthy while producing nothing usable. Ten hours
            # were lost to exactly this; check it once, immediately.
            if i == 1 and not done:
                # An individual empty channel is legitimate — fulltext in
                # particular returns nothing when the query has no lexical
                # match. The fatal signals are entries that would not map to a
                # corpus_id, or every channel coming back empty.
                empty = [k for k, v in row["channels"].items() if not v]
                if row["unmapped_channel_entries"] or len(empty) == len(row["channels"]):
                    raise FatalCaptureError(
                        f"first capture is unusable: {row['unmapped_channel_entries']} "
                        f"unmapped entries, empty channels {empty}. An empty "
                        f"memory_id->corpus_id map means /api/memories/recent "
                        f"returned nothing for this key — and that is what "
                        f"clear_project enumerates, so cleanup is a no-op too. "
                        f"Check the key's project scope."
                    )
                if empty:
                    print(f"  (note: {empty} empty on the first question — "
                          f"legitimate when the query has no match in that channel)")
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            elapsed = time.monotonic() - t_start
            eta = (len(remaining) - i) * (elapsed / i) / 60
            print(
                f"  [{i:4}/{len(remaining)}] {row['question_id'][:28]:28} "
                f"channels={len(row['channels'])} "
                f"sizes={ {k: len(v) for k, v in row['channels'].items()} } "
                f"unmapped={row['unmapped_channel_entries']} "
                f"[ingest {row['ingest_time']:.1f}s query {row['query_time']:.2f}s] "
                f"ETA={eta:.0f}m",
                flush=True,
            )
    finally:
        fh.close()
        await client.close()

    print(f"\nDone. {failures} failures. Output: {out_path}")
    return 1 if failures and failures == len(remaining) else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
