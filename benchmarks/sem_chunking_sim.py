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

Usage (runs inside the app image so the production embedder + versions are used):

    docker run --rm --no-healthcheck \
      -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
      -v "$PWD/benchmarks/data:/d:ro" -v "$PWD/benchmarks/sem_chunking_sim.py:/s.py:ro" \
      engram:latest python /s.py /d/longmemeval_s_cleaned.json <limit> <every> <offset>

  <limit> 0 = all selected questions; <every>/<offset> stratify (the dataset is
  grouped by question type, so e.g. every=5 offset=0 gives ~100 questions spanning
  all six types). Pass --no-healthcheck or the host's autoheal restarts the
  container (the image's healthcheck probes :8000, which a script never serves).

Cost: ~2,000 encodes of up to 256 tokens per question. Needs an idle CPU — on a
loaded host (2026-09-06: 30 pypy3 z80 conformance jobs) the embedder fell to
~4 long-encodes/s and two passes made no progress in 4 hours.
"""
import json, sys, time
from collections import defaultdict
import numpy as np
from sentence_transformers import SentenceTransformer

data_path, limit, every, offset = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
model = SentenceTransformer("all-MiniLM-L6-v2")
model.max_seq_length = 256
tok = model.tokenizer
WIN, STRIDE = 254, 222

cache: dict[str, np.ndarray] = {}
def embed_all(texts):
    todo = [t for t in set(texts) if t not in cache]
    if todo:
        vecs = model.encode(todo, normalize_embeddings=True, batch_size=128, show_progress_bar=False)
        for t, v in zip(todo, vecs): cache[t] = v
    return np.stack([cache[t] for t in texts])

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

VARIANTS = ["head", "win", "role", "role_win"]
agg = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))  # qtype -> variant -> metric -> sum
nq = defaultdict(int)
data = json.load(open(data_path))
sel = [e for i, e in enumerate(data) if i % every == offset][: limit or None]
t0 = time.time(); n_enc = 0

def report(final=False):
    print(("FINAL " if final else "PARTIAL ") + f"questions={sum(nq.values())} unique_texts_embedded={len(cache)} elapsed={time.time()-t0:.0f}s")
    print(f"{'qtype':28s} {'n':>4s} {'variant':>9s} {'turn@10':>8s} {'turn@50':>8s} {'sessR@1':>8s} {'sessR@10':>9s}")
    for qt in sorted(agg):
        for v in VARIANTS:
            m = agg[qt][v]; n = nq[qt]; g = m["gold_n"] or 1
            print(f"{qt:28s} {n:4d} {v:>9s} {m['t10']/g:8.1%} {m['t50']/g:8.1%} {m['s1']/n:8.1%} {m['s10']/n:9.1%}")
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
        chunks["head"].append([doc])
        chunks["win"].append(windows(doc))
        chunks["role"].append(role_parts)
        rw = []
        for c in role_parts: rw.extend(windows(c))
        chunks["role_win"].append(list(dict.fromkeys(rw)))
    all_texts = [e["question"]]
    for v in VARIANTS:
        for cl in chunks[v]: all_texts.extend(cl)
    n_enc += len(set(all_texts) - set(cache))
    embed_all(all_texts)
    q = cache[e["question"]]
    gold_ids = {d[0] for d in docs if d[5]}
    for v in VARIANTS:
        scores = np.array([max(float(q @ cache[c]) for c in cl) for cl in chunks[v]])
        order = np.argsort(-scores, kind="stable")
        ranked = [docs[j][0] for j in order]; ranked_sess = [docs[j][1] for j in order]
        m = agg[qt][v]
        m["gold_n"] += len(gold_ids)
        m["t10"] += len(gold_ids & set(ranked[:10])); m["t50"] += len(gold_ids & set(ranked[:50]))
        m["s1"] += ranked_sess[0] in gold_sess
        m["s10"] += any(s in gold_sess for s in ranked_sess[:10])
    nq[qt] += 1
    if (qi + 1) % 25 == 0:
        print(f"progress {qi+1}/{len(sel)} elapsed={time.time()-t0:.0f}s cache={len(cache)}", file=sys.stderr, flush=True)
    if (qi + 1) % 100 == 0: report()
report(final=True)
