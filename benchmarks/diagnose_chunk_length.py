"""Join an LME-S retrieval results file against per-gold-turn token length: does
the embedder's 256-token window predict whether a has_answer turn is found?

Usage (inside the app image, which has the `tokenizers` package):

    docker run --rm --no-healthcheck -v "$PWD/benchmarks:/b:ro" \
      -v "$HOME/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/<hash>/tokenizer.json:/tok.json:ro" \
      -v "$PWD/benchmarks/diagnose_chunk_length.py:/s.py:ro" engram:latest \
      python /s.py /b/data/longmemeval_s_cleaned.json /b/results_engram_lme_<tag>.jsonl /tok.json

GOTCHA: all-MiniLM-L6-v2's tokenizer.json carries a baked-in truncation at 128
tokens (its training length). Loading it with `tokenizers.Tokenizer.from_file`
silently caps every count at 128 — call `tok.no_truncation()` (done below) or
every document looks short. sentence-transformers overrides this with its own
max_seq_length=256 at encode time, so the *production* window is 256, not 128.

Result on v700 (2026-09-04 baseline), gold turns found in top-10 by length:
  <=256 tok 97.6% | 257-512 88.3% | >512 77.9% (multi-session >512: 62.1%).
"""
import json, sys
from collections import defaultdict
from tokenizers import Tokenizer
tok = Tokenizer.from_file(sys.argv[3]); tok.no_truncation(); tok.no_padding()
W = 256
res = {}
for line in open(sys.argv[2]):
    r = json.loads(line)
    ids = [it["corpus_id"] for it in r["retrieval_results"]["ranked_items"]]
    res[r["question_id"]] = ids
data = json.load(open(sys.argv[1]))
buckets = ["<=128", "129-256", "257-512", ">512"]
def b(n): return buckets[0] if n <= 128 else buckets[1] if n <= W else buckets[2] if n <= 512 else buckets[3]
agg = defaultdict(lambda: defaultdict(lambda: [0, 0, 0]))  # [n, found@10, found@50]
rows = []
for e in data:
    qid = e["question_id"]
    if qid not in res: continue
    ranked = res[qid]; top10 = set(ranked[:10]); top50 = set(ranked)
    qt = e["question_type"]
    for session, sid, date in zip(e["haystack_sessions"], e["haystack_session_ids"], e["haystack_dates"]):
        if sid not in e["answer_session_ids"]: continue
        i = 0; tn = 0
        while i < len(session):
            t = session[i]
            if t["role"] == "user":
                u = t["content"]; a = ""; ha = bool(t.get("has_answer")); aha = False
                if i+1 < len(session) and session[i+1]["role"] == "assistant":
                    a = session[i+1]["content"]; aha = bool(session[i+1].get("has_answer")); i += 1
                if ha or aha:
                    doc = f"[Date: {date}]\n[USER]: {u}" + (f"\n[ASSISTANT]: {a}" if a else "")
                    n = len(tok.encode(doc).ids)
                    cid = f"{sid}_turn_{tn}"
                    where = "asst" if (aha and not ha) else "user"
                    for key in (("ALL", "all"), (qt, "all"), ("ALL", where)):
                        c = agg[key][b(n)]
                        c[0] += 1; c[1] += cid in top10; c[2] += cid in top50
                tn += 1
            elif t["role"] == "assistant" and i == 0:
                if t.get("has_answer"):
                    n = len(tok.encode(f"[Date: {date}]\n[ASSISTANT]: {t['content']}").ids)
                    cid = f"{sid}_turn_{tn}"
                    for key in (("ALL", "all"), (qt, "all"), ("ALL", "asst")):
                        c = agg[key][b(n)]; c[0] += 1; c[1] += cid in top10; c[2] += cid in top50
                tn += 1
            i += 1
print(f"{'group':38s} {'bucket':>8s} {'gold_n':>7s} {'found@10':>9s} {'found@50':>9s}")
for key in sorted(agg):
    for bk in buckets:
        n, f10, f50 = agg[key][bk]
        if n: print(f"{key[0]+'/'+key[1]:38s} {bk:>8s} {n:7d} {f10/n:9.1%} {f50/n:9.1%}")
