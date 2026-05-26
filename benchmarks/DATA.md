# Engram Benchmarks

This directory contains the harness used to produce the numbers in [`BENCHMARK_RESULTS.md`](BENCHMARK_RESULTS.md). The raw benchmark datasets are **not** redistributed in this repository; they are downloaded on demand from their original sources.

## Datasets

| Dataset | Size | License | Upstream |
|---|---|---|---|
| LongMemEval | ~130 MB | see upstream repo | https://github.com/xiaowu0162/LongMemEval |
| LoCoMo      | ~1 MB   | CC BY-NC 4.0 (research only) | https://github.com/snap-research/locomo |

**Note on LoCoMo**: CC BY-NC 4.0 prohibits commercial use. Use Engram's LoCoMo benchmark only for non-commercial research.

## Getting the data

```bash
./download_data.sh
```

This fetches the raw datasets into `benchmarks/data/`. That directory is gitignored — only the small `test_*_fixture.json` files used by CI are tracked.

## Running a benchmark

```bash
# Start engram locally (see repo README)
pip install -r benchmarks/requirements.txt

# Create a scoped API key for the benchmark:
docker exec engram python -m engram.manage create-key \
  --name bench --role agent --projects benchmarks

export ENGRAM_API_KEY="engram_..."

# LongMemEval
python benchmarks/longmemeval_bench.py \
  benchmarks/data/longmemeval_s_cleaned.json \
  --engram-url http://localhost:8000 \
  --token "$ENGRAM_API_KEY" \
  --output benchmarks/results_engram_lme.jsonl

# LoCoMo
python benchmarks/locomo_bench.py \
  benchmarks/data/locomo10.json \
  --engram-url http://localhost:8000 \
  --token "$ENGRAM_API_KEY" \
  --top-k 10 \
  --output benchmarks/results_engram_locomo.json
```

## Result files in the repo

Only the headline runs are kept:

| File | Used for |
|---|---|
| `results_engram_lme_rrf_final.jsonl` | LongMemEval R@K table in `BENCHMARK_RESULTS.md` |
| `results_engram_locomo_top10_final.json` | LoCoMo per-category breakdown |

Intermediate tuning-sweep results were pruned before publication.
