# Engram AI Toolbox

Reference for agents running benchmarks, tuning config, and interpreting results.

---

## Running Benchmarks

Always use the wrapper script — it captures git hash, container config, and version automatically:

```bash
cd /docker/appdata/engram

# LongMemEval (500q, ~40min with MiniLM, ~55hr with BGE-large)
nohup ./benchmarks/run_bench.sh lme <version_tag> > /dev/null 2>&1 &

# LME with pre-staged data (~2min query-only — see Pre-Staging section below)
nohup ./benchmarks/run_bench.sh lme <version_tag> --skip-ingest > /dev/null 2>&1 &

# LoCoMo (10 conversations, ~3min; ~30sec with --skip-ingest)
nohup ./benchmarks/run_bench.sh locomo <version_tag> > /dev/null 2>&1 &

# MABench FactConsolidation (~30sec)
./benchmarks/run_bench.sh mabench <version_tag>

# BEAM 100K (~10min; ~2min with --skip-ingest)
nohup ./benchmarks/run_bench.sh beam <version_tag> --split 100K > /dev/null 2>&1 &

# ConvoMem (sample=50 per category, ~20min)
nohup ./benchmarks/run_bench.sh convomem <version_tag> > /dev/null 2>&1 &

# BEIR SciFact (~46min full; ~46min query-only with --skip-ingest)
# Full corpus: 5,183 docs; 300 test queries; metric: nDCG@10
nohup ./benchmarks/run_bench.sh beir <version_tag> --dataset scifact --no-cleanup > /dev/null 2>&1 &

# BEIR skip-ingest (corpus already staged as _beir_scifact project)
nohup ./benchmarks/run_bench.sh beir <version_tag> --dataset scifact --skip-ingest > /dev/null 2>&1 &
```

Logs go to `/tmp/bench_<bench>_<tag>.log`. Results go to `benchmarks/results_engram_<bench>_<tag>_<timestamp>.*`.

All runs are appended to `benchmarks/results_history.tsv` for easy comparison.

---

## Pre-Staging Benchmark Data (Fast Iteration)

Ingest dominates benchmark runtime. Once staged, data lives in isolated projects with zero
leakage between questions/conversations. Subsequent runs skip ingest and go straight to queries.

| Benchmark | Normal runtime | Query-only runtime | Project pattern |
|---|---|---|---|
| LME (500q) | ~40min | ~2min | `_lme_bench_<question_id>` |
| LoCoMo (10 convs) | ~3min | ~30sec | `_locomo_bench_<sample_id>` |
| BEAM 100K | ~10min | ~2min | `_beam_bench_<conv_id>` |
| MABench | ~30sec | — | not worth staging |
| ConvoMem | ~20min | ❌ | uses timestamp in project name, can't match |
| BEIR SciFact | ~50min (ingest ~4min, queries ~46min) | ~46min | `_beir_scifact` |

**Workflow:**

```bash
cd /docker/appdata/engram

# Step 1: Stage each benchmark once (~40min for LME, then done).
# Run this after any model/chunk-size change that invalidates existing embeddings.
python3 -u benchmarks/longmemeval_bench.py \
  benchmarks/data/longmemeval_s_cleaned.json \
  --granularity turn-pair --workers 4 \
  --engram-url http://192.168.90.45:8000 --no-cleanup > /tmp/lme_prestage.log 2>&1 &

python3 -u benchmarks/locomo_bench.py \
  benchmarks/data/locomo10.json \
  --granularity dialog --engram-url http://192.168.90.45:8000 --no-cleanup

python3 -u benchmarks/beam_bench.py \
  --split 100K --engram-url http://192.168.90.45:8000 --no-cleanup

# Step 2: All subsequent runs — just add --skip-ingest.
# --skip-ingest checks which projects already exist, auto-disables cleanup.
nohup ./benchmarks/run_bench.sh lme    <version_tag> --skip-ingest > /dev/null 2>&1 &
nohup ./benchmarks/run_bench.sh locomo <version_tag> --skip-ingest > /dev/null 2>&1 &
nohup ./benchmarks/run_bench.sh beam   <version_tag> --skip-ingest > /dev/null 2>&1 &
```

**When to re-stage:**
- After changing `ENGRAM_EMBEDDING_MODEL` or `ENGRAM_CHUNK_SIZE` (old embeddings are stale)
- After a schema migration that drops the memories table
- To re-stage: delete staged projects first (`clear_all` or drop+recreate DB), then re-run Step 1

**Note:** `run_bench.sh` passes `--engram-url http://192.168.90.45:8000` and all extra args automatically.

---

## Checking Progress

```bash
tail -5 /tmp/bench_lme_<tag>.log
tail -5 /tmp/bench_locomo_<tag>.log
```

---

## Scoring Results

```bash
# Quick LME scorecard (compare multiple versions)
python3 << 'EOF'
import json, glob

def score(path):
    r = [json.loads(l) for l in open(path) if l.strip()]
    s = lambda k: sum(1 for x in r if x['retrieval_results']['metrics']['session'].get(k, 0) >= 1.0)
    n = len(r)
    return n, s('recall_any@1'), s('recall_any@5'), s('recall_any@10'), \
           sum(1 for x in r if x['retrieval_results']['metrics']['session'].get('recall_any@10', 0) < 1.0)

for path in sorted(glob.glob('benchmarks/results_engram_lme_*.jsonl')):
    n, h1, h5, h10, miss = score(path)
    tag = path.split('/')[-1]
    print(f"{tag}: R@1={h1}/{n}={h1/n:.4f}  R@5={h5/n:.4f}  R@10={h10/n:.4f}  misses={miss}")
EOF
```

---

## Key Config Variables (engram2.yml)

| Variable | Current | Notes |
|---|---|---|
| `ENGRAM_EMBEDDING_MODEL` | all-MiniLM-L6-v2 | Switch to BGE-large for higher quality |
| `ENGRAM_EMBEDDING_DIM` | 384 | Must match model (BGE-large=1024) |
| `ENGRAM_CHUNK_SIZE` | 1000 | Max chars per chunk (~1150 safe limit for MiniLM) |
| `ENGRAM_CHUNK_OVERLAP` | 300 | Higher = better boundary coverage, more chunks |
| `ENGRAM_HNSW_EF_SEARCH` | 200 | Higher = better recall, slower query |
| `ENGRAM_LLM_RERANK_TOP_N` | 20 | Candidates sent to LLM reranker |

### Pure-Math Recall Improvement Knobs (no new models)

| Variable | Default | Notes |
|---|---|---|
| `ENGRAM_BM25_SIGNAL_ENABLED` | `1` | In-process BM25 as additional RRF signal |
| `ENGRAM_BM25_SIGNAL_WEIGHT` | `0.5` | RRF weight for BM25 ranked list |
| `ENGRAM_ENTITY_SIGNAL_ENABLED` | `1` | Proper nouns + numbers overlap as RRF signal |
| `ENGRAM_ENTITY_SIGNAL_WEIGHT` | `0.3` | RRF weight for entity-overlap list |
| `ENGRAM_DATE_SIGNAL_WEIGHT` | `0.6` | RRF weight for date-proximity list (temporal queries) |
| `ENGRAM_RECENCY_SIGNAL_WEIGHT` | `0.2` | RRF weight for session-recency list (recency queries) |
| `ENGRAM_TURN_PAIR_SIGNAL_WEIGHT` | `0.15` | RRF weight for turn-pair-completeness list |
| `ENGRAM_MAXSIM_ENABLED` | `0` | Token-level MaxSim rerank (ColBERT-style, same model) |
| `ENGRAM_MAXSIM_TOP_N` | `20` | Candidates to rerank with MaxSim |
| `ENGRAM_MAXSIM_CACHE_SIZE` | `2048` | LRU doc-embedding cache entries |
| `ENGRAM_PRF_ENABLED` | `0` | Pseudo-relevance feedback (Rocchio, vague queries only) |
| `ENGRAM_PRF_TOP_K` | `5` | Top-K results used for PRF term extraction |
| `ENGRAM_PRF_N_TERMS` | `8` | Max expansion terms appended to FTS re-query |
| `ENGRAM_PRF_FTS_WEIGHT` | `0.3` | RRF weight for PRF result list |
| `ENGRAM_TIEBREAK_ENABLED` | `1` | Gap-aware tiebreaker (fires when top-2 gap ≤ eps) |
| `ENGRAM_TIEBREAK_EPS` | `0.005` | Score-gap threshold to trigger tiebreaker |
| `ENGRAM_MMR_ENABLED` | `1` | Session MMR diversification (counting queries only) |
| `ENGRAM_MMR_LAMBDA` | `0.7` | Relevance weight in MMR (0=diversity, 1=relevance) |
| `ENGRAM_MMR_SESSION_CAP` | `2` | Max chunks per session_id in output |
| `ENGRAM_TEMPORAL_HARD_FILTER` | `0` | Pre-filter candidates to target-date window |
| `ENGRAM_TEMPORAL_HARD_FILTER_SIGMA` | `3.5` | Window half-size in days |

Compose file: `/docker/compose/dev/engram2.yml`

After changing config, rebuild and restart:
```bash
cd /docker/compose/dev
docker compose -f engram2.yml --env-file /docker/compose/.env up -d --build
# Wait ~10s then verify
curl -s http://192.168.90.45:8000/health
```

---

## Benchmark Version History

| Version | Model | chunk/overlap/ef | LME R@1 | LoCoMo avg | BEAM 100K | BEIR SciFact nDCG@10 | Notes |
|---|---|---|---|---|---|---|---|
| v100/v110 | MiniLM | 800/200/ef50 | 0.846 | 0.906 | — | — | Baseline |
| v120 | BGE-large | 800/200/ef50 | 0.860 | — | 0.3274 | — | +7 LME R@1, worse LoCoMo |
| v301 | MiniLM | 900/100/ef50 | 0.852 | 0.763 | — | — | TF cap=2 fix |
| v302 | MiniLM | 1000/300/ef200 | 0.852 | 0.763 | 0.3917 | — | Plateau, best pre-v4 |
| **v402** | MiniLM | 1000/300/ef200 | **0.856** | **0.785** | **0.4167** | **0.6569** | Multi-signal RRF + MMR gate fix |

**BEIR SciFact v402 (2026-05-12):** nDCG@10=0.6569, Recall@100=0.9350, Precision@10=0.0887, Any-hit@100=94.0% (282/300)
For context: BM25 baseline ≈ 0.665, SentenceBERT ≈ 0.664, DPR ≈ 0.318 — Engram is at BM25 level on this domain-specific IR task.

**v402 signals (all enabled by default):**
- Phase A: BM25 (w=0.5) + entity overlap (w=0.3) fused via RRF
- Phase B: MaxSim rerank (disabled by default, `ENGRAM_MAXSIM_ENABLED=1`)
- Phase C: PRF expansion (disabled by default, `ENGRAM_PRF_ENABLED=1`)
- Phase D: Date-proximity Gaussian boost (w=0.6), session recency (w=0.2), turn-pair (w=0.15)
- Phase E: MMR diversification (enabled, gate: fires for non-counting queries)
- Phase F: Gap-aware tiebreaker

**v402 clean ablation (BEAM 100K, skip-ingest on staged data):**
| Variant | avg_recall | Δ vs v402 |
|---|---|---|
| v402 (all on) | 0.4167 | baseline |
| abl-all-off | 0.3917 | −0.025 |
| abl-no-bm25 | 0.4148 | −0.002 |
| abl-no-entity | 0.4165 | −0.000 |
| abl-no-date | 0.4167 | 0.000 |
| abl-no-recency-turnpair | 0.4181 | +0.001 (recency/turn-pair mildly hurt BEAM) |
| abl-no-tiebreak-mmr | 0.4043 | −0.012 (tiebreak+MMR are critical) |

**Current best**: v402 (git: `public-prep` branch, commit `25d6016`)

---

## Known Hard Misses (LME v402)

15 questions that fail R@1 consistently:
- **single-session-preference** (18/30 misses): soft/subjective queries ("Can you suggest accessories for my setup?") — correct session is in top-10 in 13/18 cases but ranking noise prevents rank-1. Small n=30 makes improvements unreliable.
- **temporal-reasoning** (~22/133 misses, R@1=0.835): time-dependent queries
- **multi-session** (~21/133 misses, R@1=0.842): 14 are close-rank misses (answer at rank 2-3); often generic sharegpt/ultrachat content outranks personal conversation turns via BM25

**Preference boost notes**: Broadening `apply_preference_signal_boost` to fire on advice-seeking queries regardless of vagueness HURT preference (-1 hit) because non-topic preference language in other sessions gets incorrectly boosted. The vague-query gate is intentional.

---

## End-to-End LME (Retrieval + Generation)

`lme_e2e_bench.py` loads retrieval results from a prior LME run, feeds top-K chunks to
Gemma4:31b via Ollama, and evaluates generated answers against gold.

```bash
cd /docker/appdata/engram

# Full 500q E2E (top-10 chunks → Gemma4, ~55min)
python3 benchmarks/lme_e2e_bench.py \
  --retrieval-results benchmarks/results_engram_lme_v402-mmr-fix_20260512_1106.jsonl \
  --out benchmarks/results_engram_lme_e2e_v402.jsonl

# Judge pass on misses (adds LLM rescoring of substring-match failures)
python3 benchmarks/lme_e2e_bench.py \
  --retrieval-results benchmarks/results_engram_lme_v402-mmr-fix_20260512_1106.jsonl \
  --judge --out benchmarks/results_engram_lme_e2e_v402_judged.jsonl
```

**v402 E2E results (retrieval + Gemma4:31b, top-K=10, corrected scoring):**

| Type | N | Exact | Smart-Norm | Judge | **Final** | Retrieval R@1 |
|---|---|---|---|---|---|---|
| single-session-user       | 70  | 0.800 | +2  | +4  | **0.843** | 0.957 |
| knowledge-update          | 78  | 0.654 | +5  | +2  | **0.821** | 0.990 |
| single-session-assistant  | 56  | 0.268 | +1  | +3  | **0.429** | 0.929 |
| multi-session             | 133 | 0.293 | +10 | +15 | **0.451** | 0.985 |
| temporal-reasoning        | 133 | 0.195 | +5  | +25 | **0.429** | 0.962 |
| single-session-preference | 30  | 0.000 | +1  | +6  | **0.067** | 0.933 |
| **TOTAL**                 | **500** | **0.374** | **+24** | **+55** | **0.532** | **0.856** |

**Key finding:** 174/313 misses (56%) are "Unknown" responses — the LLM cannot extract the
answer from the provided chunks even though retrieval is excellent. This is an extraction
problem, not a retrieval problem. The assistant (50% Unknown) and preference (73% Unknown)
categories are hardest; user-facts (84.3%) and knowledge-updates (82.1%) are our strong suit.

**Judge details:** Gemma4:31b acting as LLM judge with leniency (paraphrases, number formats,
minor wording). 55/115 judgeable misses rescued. "Unknown" responses skipped (can't judge).

---

## Important Code Locations

| File | What's there |
|---|---|
| `src/engram/rerank.py` | TF cap (`_TF_CAP = 2`), keyword reranker |
| `src/engram/manager.py` | `apply_turn_pair_boost()`, embedding prefix handling |
| `src/engram/config.py` | All config defaults |
| `benchmarks/run_bench.sh` | Benchmark launcher (use this!) |
| `benchmarks/lme_e2e_bench.py` | End-to-end LME (retrieval + generation + judge) |
| `benchmarks/results_history.tsv` | One-line summary of every run |
| `/docker/compose/dev/engram2.yml` | Container config (model, chunk size, etc.) |
| `/tmp/bench_<bench>_<tag>.log` | Live benchmark log |
