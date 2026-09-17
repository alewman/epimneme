"""Semantic-channel-only simulation of chunking strategies on LME-S.

For every question, rank that question's own haystack turn-pair docs by cosine
similarity to the question under several *indexing* strategies, using the
production embedder (all-MiniLM-L6-v2, max_seq_length=256):

  head      current behaviour: one vector per doc, model truncates to 256 tokens
  win       sliding 254-token windows over the doc, doc score = max over windows
  role      {whole-doc head, [USER] turn alone, [ASSISTANT] turn alone}, max
  role_win  role chunks, each windowed, max

Metrics per question type: turn-level found@10/@50 of has_answer gold turns,
session-level recall_any@1/@10. Whole population (hits AND misses), so this is
a fair pre-test of the mechanism — not fit to a miss-selected sample.

Also the embedder-swap screen: `--model` runs any SentenceTransformer through the
identical path, so a candidate embedder is measured against the current one with
chunking, gold and metrics held fixed. Establish the reference on the production
model first, then re-run with `--model <candidate>` and compare like for like.

Usage (runs inside the app image so the production embedder + versions are used):

    docker run --rm --no-healthcheck \
      -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
      -v "$PWD/benchmarks/data:/d:ro" -v "$PWD/benchmarks/sem_chunking_sim.py:/s.py:ro" \
      engram:latest python /s.py /d/longmemeval_s_cleaned.json <limit> <every> <offset> \
        [--model NAME] [--max-seq N] [--variants head,win,...]

  <limit> 0 = all selected questions; <every>/<offset> stratify (the dataset is
  grouped by question type, so e.g. every=5 offset=0 gives ~100 questions spanning
  all six types). Pass --no-healthcheck or the host's autoheal restarts the
  container (the image's healthcheck probes :8000, which a script never serves).

Gold: turns flagged `has_answer` — the dataset's real evidence annotation, ~2
turns per question. Not "every turn of a gold session", which the retrieval
harnesses used until 2026-09-16 and which inflates the gold ~6x. Metrics are
micro-averaged over gold turns, so the 21/500 questions with no flagged turn
contribute nothing rather than a free perfect score.

Cost: ~2,000 encodes of up to 256 tokens per question. Needs an idle CPU — on a
loaded host (2026-09-06: 30 pypy3 z80 conformance jobs) the embedder fell to
~4 long-encodes/s and two passes made no progress in 4 hours.
"""
import json, sys, time
from collections import defaultdict
import numpy as np
from sentence_transformers import SentenceTransformer

import argparse

ap = argparse.ArgumentParser()
ap.add_argument("data_path")
ap.add_argument("limit", type=int)
ap.add_argument("every", type=int)
ap.add_argument("offset", type=int)
ap.add_argument("--model", default="all-MiniLM-L6-v2",
                help="SentenceTransformer name. The comparison axis for an embedder swap.")
ap.add_argument("--max-seq", type=int, default=256,
                help="Tokens per encode. 256 is the production setting for MiniLM; "
                     "a model with a longer window should be given its own, and the "
                     "window/stride below scale with it.")
ap.add_argument("--variants", default="head,win,role,role_win",
                help="Indexing strategies to score. `head` alone is production behaviour "
                     "and is ~4x cheaper.")
ap.add_argument("--query-prefix", default="",
                help="Prepended to the QUESTION only. Instruction-tuned embedders "
                     "(Qwen3-Embedding, E5, BGE) expect one and score materially worse "
                     "without it, so omitting it understates the candidate.")
ap.add_argument("--doc-prefix", default="",
                help="Prepended to every DOCUMENT chunk (e.g. E5's 'passage: ').")
ap.add_argument("--rows", default="",
                help="Write one JSON object per question per variant to this path. "
                     "Required for a paired significance test between two models — the "
                     "aggregate report cannot support one.")
ap.add_argument("--compare", nargs=2, metavar=("BASE", "CAND"),
                help="Paired comparison of two --rows files. Loads no model; every other "
                     "argument is ignored.")
args = ap.parse_args()
data_path, limit, every, offset = args.data_path, args.limit, args.every, args.offset


def _load_rows(path):
    out = {}
    with open(path) as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            out[(r["question_id"], r["variant"])] = r
    return out


def compare(base_path, cand_path):
    """Paired significance test between two --rows files.

    Questions are matched by id and variant, so a model change is measured on
    exactly the same questions under exactly the same indexing strategy.
    """
    from math import comb

    A, B = _load_rows(base_path), _load_rows(cand_path)
    keys = sorted(set(A) & set(B))
    if not keys:
        sys.exit("the two row files share no (question_id, variant) pairs")

    ma = {A[k]["model"] for k in keys}
    mb = {B[k]["model"] for k in keys}
    print(f"paired comparison over {len(keys)} (question, variant) pairs")
    print(f"  BASE {base_path}: model={sorted(ma)}")
    print(f"  CAND {cand_path}: model={sorted(mb)}")
    if ma == mb:
        print("  !! both files record the SAME model — is CAND really a different embedder?")
    for field in ("max_seq", "query_prefix", "doc_prefix"):
        va = {A[k].get(field) for k in keys}
        vb = {B[k].get(field) for k in keys}
        if va != vb:
            print(f"  note: {field} differs — base={sorted(map(str, va))} cand={sorted(map(str, vb))}")
    bad = [k for k in keys if A[k]["gold_ids"] != B[k]["gold_ids"]]
    if bad:
        print(f"  !! {len(bad)} pairs disagree on the gold set — not comparable")
    print()

    variants = sorted({k[1] for k in keys})
    for v in variants:
        vk = [k for k in keys if k[1] == v and A[k]["gold_n"]]
        if not vk:
            continue
        print(f"variant {v}  (n={len(vk)} questions with gold)")
        print(f"  {'metric':10s} {'base':>8s} {'cand':>8s} {'delta':>8s} "
              f"{'better':>7s} {'worse':>6s} {'p':>8s}")
        for metric, cont in (("ec@10", True), ("all10", False), ("hit1", False),
                             ("ec@50", True), ("s1", False), ("s10", False)):
            va = [A[k][metric] for k in vk]
            vb = [B[k][metric] for k in vk]
            mean_a, mean_b = sum(va) / len(va), sum(vb) / len(vb)
            w = sum(1 for x, y in zip(va, vb) if y > x)
            l = sum(1 for x, y in zip(va, vb) if y < x)
            n = w + l
            p = (min(1.0, 2 * sum(comb(n, i) for i in range(min(w, l) + 1)) / 2**n)
                 if n else 1.0)
            star = " *" if p < 0.05 else ""
            print(f"  {metric:10s} {mean_a:8.3f} {mean_b:8.3f} {mean_b - mean_a:+8.3f} "
                  f"{w:7d} {l:6d} {p:8.3f}{star}")
        print()
    print("  p is a two-sided exact binomial over the discordant questions. ec@10 and\n"
          "  ec@50 are continuous per question, so the test is on whether the candidate\n"
          "  retrieved more of that question's evidence, not on the size of the change.")


if args.compare:
    compare(*args.compare)
    sys.exit(0)


model = SentenceTransformer(args.model)
model.max_seq_length = args.max_seq
tok = model.tokenizer
# Window just under the encode limit, with ~12% overlap so a fact spanning a
# boundary is whole in some window.
WIN = args.max_seq - 2
STRIDE = max(1, int(WIN * 0.875))
# ST 6.x renamed this; keep both so the script runs on either version — an
# embedder comparison is worthless if the two runs used different libraries.
DIM = (model.get_embedding_dimension() if hasattr(model, "get_embedding_dimension")
       else model.get_sentence_embedding_dimension())

cache: dict[str, np.ndarray] = {}

def embed_all(texts):
    """Encode DOCUMENT chunks. Cache keys are the raw text; the doc prefix is
    constant for a run, so it cannot collide across prefixes within one file."""
    todo = [t for t in set(texts) if t not in cache]
    if todo:
        vecs = model.encode([args.doc_prefix + t for t in todo],
                            normalize_embeddings=True, batch_size=128,
                            show_progress_bar=False)
        for t, v in zip(todo, vecs): cache[t] = v
    return np.stack([cache[t] for t in texts])


def embed_query(text):
    """Encode the QUESTION, which may take a different prefix from documents.
    Cached under a reserved key so it cannot be confused with the same string
    encoded as a document."""
    key = "\x00query\x00" + text
    if key not in cache:
        cache[key] = model.encode([args.query_prefix + text],
                                  normalize_embeddings=True,
                                  show_progress_bar=False)[0]
    return cache[key]

def windows(text):
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True, truncation=False)
    offs = enc["offset_mapping"]; n = len(offs)
    if n <= WIN: return [text]
    out, start = [], 0
    while start < n:
        end = min(start + WIN, n)
        out.append(text[offs[start][0]:offs[end-1][1]])
        if end == n: break
        start += STRIDE
    return out

ALL_VARIANTS = ["head", "win", "role", "role_win"]
VARIANTS = [v.strip() for v in args.variants.split(",") if v.strip()]
if set(VARIANTS) - set(ALL_VARIANTS):
    sys.exit(f"unknown variant(s): {sorted(set(VARIANTS) - set(ALL_VARIANTS))}")
agg = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))  # qtype -> variant -> metric -> sum
nq = defaultdict(int)
data = json.load(open(data_path))
sel = [e for i, e in enumerate(data) if i % every == offset][: limit or None]
t0 = time.time(); n_enc = 0
rows_fh = open(args.rows, "w") if args.rows else None

def report(final=False):
    print(("FINAL " if final else "PARTIAL ")
          + f"model={args.model} dim={DIM} max_seq={args.max_seq} "
            f"qprefix={args.query_prefix!r} dprefix={args.doc_prefix!r} "
            f"questions={sum(nq.values())} unique_texts_embedded={len(cache)} "
            f"elapsed={time.time()-t0:.0f}s")
    print(f"{'qtype':28s} {'n':>4s} {'variant':>9s} {'turn@10':>8s} {'turn@50':>8s} "
          f"{'tAll@10':>8s} {'tHit@1':>7s} {'sessR@1':>8s} {'sessR@10':>9s}")
    for qt in sorted(agg):
        for v in VARIANTS:
            m = agg[qt][v]; n = nq[qt]; g = m["gold_n"] or 1
            qg = m["q_withgold"] or 1
            print(f"{qt:28s} {n:4d} {v:>9s} {m['t10']/g:8.1%} {m['t50']/g:8.1%} "
                  f"{m['all10']/qg:8.1%} {m['hit1']/qg:7.1%} {m['s1']/n:8.1%} {m['s10']/n:9.1%}")
    # micro-average across types — the single number an embedder swap moves
    print(f"{'ALL':28s} {sum(nq.values()):4d}", end="")
    for v in VARIANTS:
        G = sum(agg[qt][v]["gold_n"] for qt in agg) or 1
        QG = sum(agg[qt][v]["q_withgold"] for qt in agg) or 1
        N = sum(nq.values()) or 1
        t10 = sum(agg[qt][v]["t10"] for qt in agg); t50 = sum(agg[qt][v]["t50"] for qt in agg)
        a10 = sum(agg[qt][v]["all10"] for qt in agg); h1 = sum(agg[qt][v]["hit1"] for qt in agg)
        s1 = sum(agg[qt][v]["s1"] for qt in agg); s10 = sum(agg[qt][v]["s10"] for qt in agg)
        print(f"{'' if v == VARIANTS[0] else ' ' * 33}{v:>9s} {t10/G:8.1%} {t50/G:8.1%} "
              f"{a10/QG:8.1%} {h1/QG:7.1%} {s1/N:8.1%} {s10/N:9.1%}")
    sys.stdout.flush()

for qi, e in enumerate(sel):
    qt = e["question_type"]; gold_sess = set(e["answer_session_ids"])
    docs = []  # (cid, sid, doc_text, user_chunk, asst_chunk, is_gold_turn)
    for session, sid, date in zip(e["haystack_sessions"], e["haystack_session_ids"], e["haystack_dates"]):
        i = 0; tn = 0
        while i < len(session):
            t = session[i]
            if t["role"] == "user":
                u = t["content"]; a = ""; ha = bool(t.get("has_answer")); aha = False
                if i+1 < len(session) and session[i+1]["role"] == "assistant":
                    a = session[i+1]["content"]; aha = bool(session[i+1].get("has_answer")); i += 1
                header = f"[Date: {date}]"
                doc = f"{header}\n[USER]: {u}" + (f"\n[ASSISTANT]: {a}" if a else "")
                docs.append((f"{sid}_turn_{tn}", sid, doc, f"{header}\n[USER]: {u}", f"{header}\n[ASSISTANT]: {a}" if a else None, (sid in gold_sess) and (ha or aha)))
                tn += 1
            elif t["role"] == "assistant" and tn == 0:
                doc = f"[Date: {date}]\n[ASSISTANT]: {t['content']}"
                docs.append((f"{sid}_turn_{tn}", sid, doc, None, doc, (sid in gold_sess) and bool(t.get("has_answer"))))
                tn += 1
            i += 1
    if not docs: continue
    # chunk sets per variant
    chunks = {v: [] for v in VARIANTS}
    for cid, sid, doc, uc, ac, g in docs:
        role_parts = [doc] + [c for c in (uc, ac) if c and c != doc]
        # Only build what was asked for — windowing is the expensive part, and
        # --variants head must not pay for the three it did not select.
        if "head" in chunks: chunks["head"].append([doc])
        if "win" in chunks: chunks["win"].append(windows(doc))
        if "role" in chunks: chunks["role"].append(role_parts)
        if "role_win" in chunks:
            rw = []
            for c in role_parts: rw.extend(windows(c))
            chunks["role_win"].append(list(dict.fromkeys(rw)))
    all_texts = []
    for v in VARIANTS:
        for cl in chunks[v]: all_texts.extend(cl)
    n_enc += len(set(all_texts) - set(cache))
    embed_all(all_texts)
    q = embed_query(e["question"])
    gold_ids = {d[0] for d in docs if d[5]}
    for v in VARIANTS:
        scores = np.array([max(float(q @ cache[c]) for c in cl) for cl in chunks[v]])
        order = np.argsort(-scores, kind="stable")
        ranked = [docs[j][0] for j in order]; ranked_sess = [docs[j][1] for j in order]
        m = agg[qt][v]
        m["gold_n"] += len(gold_ids)
        m["t10"] += len(gold_ids & set(ranked[:10])); m["t50"] += len(gold_ids & set(ranked[:50]))
        if gold_ids:
            m["q_withgold"] += 1
            m["all10"] += gold_ids <= set(ranked[:10])
            m["hit1"] += ranked[0] in gold_ids
        m["s1"] += ranked_sess[0] in gold_sess
        m["s10"] += any(s in gold_sess for s in ranked_sess[:10])
        if rows_fh is not None:
            top10, top50 = set(ranked[:10]), set(ranked[:50])
            rows_fh.write(json.dumps({
                "question_id": e["question_id"],
                "question_type": qt,
                "model": args.model,
                "max_seq": args.max_seq,
                "query_prefix": args.query_prefix,
                "doc_prefix": args.doc_prefix,
                "variant": v,
                "gold_n": len(gold_ids),
                "gold_ids": sorted(gold_ids),
                # Per-question values the paired test consumes. ec@k is this
                # question's own completeness, not a running total.
                "ec@10": (len(gold_ids & top10) / len(gold_ids)) if gold_ids else 0.0,
                "ec@50": (len(gold_ids & top50) / len(gold_ids)) if gold_ids else 0.0,
                "all10": float(bool(gold_ids) and gold_ids <= top10),
                "hit1": float(bool(gold_ids) and ranked[0] in gold_ids),
                "s1": float(ranked_sess[0] in gold_sess),
                "s10": float(any(x in gold_sess for x in ranked_sess[:10])),
                # Kept so a future metric can be recomputed without re-encoding.
                "ranked_top50": ranked[:50],
            }) + "\n")
            rows_fh.flush()
    nq[qt] += 1
    if (qi + 1) % 25 == 0:
        print(f"progress {qi+1}/{len(sel)} elapsed={time.time()-t0:.0f}s cache={len(cache)}", file=sys.stderr, flush=True)
    if (qi + 1) % 100 == 0: report()
report(final=True)
if rows_fh is not None:
    rows_fh.close()
    print(f"per-question rows written to {args.rows}")
