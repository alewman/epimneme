#!/usr/bin/env bash
# =============================================================================
# run_bench.sh — Consistent benchmark launcher for engram
# =============================================================================
# Usage:
#   ./benchmarks/run_bench.sh <benchmark> <version_tag> [extra args...]
#
# Benchmarks:
#   lme         LongMemEval 500-question retrieval benchmark
#   locomo      LoCoMo multi-session conversation benchmark
#   mabench     MABench FactConsolidation (conflict resolution)
#   beam        BEAM arXiv benchmark (--split 100K|500K|1M)
#   convomem    ConvoMem category recall benchmark
#   beir        BEIR document retrieval (--dataset scifact|nfcorpus|arguana|scidocs|fiqa)
#
# Examples:
#   ./benchmarks/run_bench.sh lme v303
#   ./benchmarks/run_bench.sh lme v303 --workers 8
#   ./benchmarks/run_bench.sh locomo v303
#   ./benchmarks/run_bench.sh beam v303 --split 100K
#   ./benchmarks/run_bench.sh mabench v303 --context-depth 6k
#   ./benchmarks/run_bench.sh convomem v303 --sample 50
#   ./benchmarks/run_bench.sh beir v303 --dataset scifact
# =============================================================================

set -euo pipefail

BENCH=${1:-}
TAG=${2:-}
shift 2 2>/dev/null || true
EXTRA_ARGS=("$@")

EPIMNEME_URL="${EPIMNEME_URL:-http://192.168.90.45:8000}"
BENCH_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="$BENCH_DIR/data"
LOG_DIR="/tmp"
HISTORY="$BENCH_DIR/results_history.tsv"
TIMESTAMP=$(date +%Y%m%d_%H%M)

# ── Validate args ─────────────────────────────────────────────────────────────
if [[ -z "$BENCH" || -z "$TAG" ]]; then
  echo "Usage: $0 <benchmark> <version_tag> [extra args...]"
  echo "       benchmarks: lme | locomo | mabench | beam | convomem | beir"
  exit 1
fi

# ── Resolve output paths ──────────────────────────────────────────────────────
OUT_FILE="$BENCH_DIR/results_engram_${BENCH}_${TAG}_${TIMESTAMP}.jsonl"
LOG_FILE="$LOG_DIR/bench_${BENCH}_${TAG}.log"

# BEAM, BEIR, MABench use .json not .jsonl
[[ "$BENCH" == "beam" || "$BENCH" == "mabench" || "$BENCH" == "beir" ]] && OUT_FILE="${OUT_FILE%.jsonl}.json"

# ── Gather metadata ───────────────────────────────────────────────────────────
GIT_HASH=$(git -C "$BENCH_DIR/.." log -1 --format="%h" 2>/dev/null || echo "unknown")
GIT_MSG=$(git -C "$BENCH_DIR/.." log -1 --format="%s" 2>/dev/null || echo "unknown")
EPIMNEME_VERSION=$(curl -s "$EPIMNEME_URL/health" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('version','?'))" 2>/dev/null || echo "unknown")

# Pull current EPIMNEME_* env vars from running container
CONTAINER_CONFIG=$(docker inspect engram2 --format='{{range .Config.Env}}{{println .}}{{end}}' 2>/dev/null \
  | grep "^EPIMNEME_" | grep -v "PASSWORD\|SECRET\|TOKEN" | sort || echo "(docker inspect unavailable)")

# ── Print + log header ────────────────────────────────────────────────────────
print_header() {
  echo "============================================================"
  echo "  Engram Benchmark Runner"
  echo "============================================================"
  echo "  Benchmark:   $BENCH"
  echo "  Tag:         $TAG"
  echo "  Timestamp:   $TIMESTAMP"
  echo "  Engram:      $EPIMNEME_URL"
  echo "  Version:     $EPIMNEME_VERSION"
  echo "  Git:         $GIT_HASH  $GIT_MSG"
  echo "  Output:      $(basename "$OUT_FILE")"
  echo "────────────────────────────────────────────────────────────"
  echo "  Container config:"
  echo "$CONTAINER_CONFIG" | sed 's/^/    /'
  echo "────────────────────────────────────────────────────────────"
  [[ ${#EXTRA_ARGS[@]} -gt 0 ]] && echo "  Extra args:  ${EXTRA_ARGS[*]}"
  echo ""
}

print_header | tee "$LOG_FILE"

# ── Build benchmark command ───────────────────────────────────────────────────
case "$BENCH" in
  lme)
    CMD=(python3 -u "$BENCH_DIR/longmemeval_bench.py"
      "$DATA_DIR/longmemeval_s_cleaned.json"
      --granularity turn-pair
      --workers 4
      --engram-url "$EPIMNEME_URL"
      --out "$OUT_FILE"
      "${EXTRA_ARGS[@]}")
    ;;
  locomo)
    CMD=(python3 -u "$BENCH_DIR/locomo_bench.py"
      "$DATA_DIR/locomo10.json"
      --granularity dialog
      --workers 4
      --engram-url "$EPIMNEME_URL"
      --out "$OUT_FILE"
      "${EXTRA_ARGS[@]}")
    ;;
  mabench)
    CMD=(python3 -u "$BENCH_DIR/mabench_bench.py"
      --dataset-file "$DATA_DIR/mabench_conflict_resolution.json"
      --context-depth 6k
      --engram-url "$EPIMNEME_URL"
      --out "$OUT_FILE"
      "${EXTRA_ARGS[@]}")
    ;;
  beam)
    CMD=(python3 -u "$BENCH_DIR/beam_bench.py"
      --split 100K
      --engram-url "$EPIMNEME_URL"
      --out "$OUT_FILE"
      "${EXTRA_ARGS[@]}")
    ;;
  convomem)
    CMD=(python3 -u "$BENCH_DIR/convomem_bench.py"
      --sample 50
      --workers 4
      --engram-url "$EPIMNEME_URL"
      --out "$OUT_FILE"
      "${EXTRA_ARGS[@]}")
    ;;
  beir)
    CMD=(python3 -u "$BENCH_DIR/beir_bench.py"
      --dataset scifact
      --workers 4
      --engram-url "$EPIMNEME_URL"
      --out "$OUT_FILE"
      "${EXTRA_ARGS[@]}")
    ;;
  *)
    echo "Unknown benchmark: $BENCH"
    echo "Valid: lme | locomo | mabench | beam | convomem | beir"
    exit 1
    ;;
esac

# ── Run ───────────────────────────────────────────────────────────────────────
echo "Running: ${CMD[*]}" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

"${CMD[@]}" 2>&1 | tee -a "$LOG_FILE"
EXIT_CODE=${PIPESTATUS[0]}

# ── Append one-line summary to history ───────────────────────────────────────
if [[ ! -f "$HISTORY" ]]; then
  echo -e "timestamp\tbench\ttag\tversion\tgit_hash\tchunk_size\tchunk_overlap\tef_search\tmodel\tresult_file" > "$HISTORY"
fi

CHUNK_SIZE=$(echo "$CONTAINER_CONFIG" | grep "EPIMNEME_CHUNK_SIZE" | cut -d= -f2 | tr -d '"' || echo "?")
CHUNK_OVERLAP=$(echo "$CONTAINER_CONFIG" | grep "EPIMNEME_CHUNK_OVERLAP" | cut -d= -f2 | tr -d '"' || echo "?")
EF_SEARCH=$(echo "$CONTAINER_CONFIG" | grep "EPIMNEME_HNSW_EF_SEARCH" | cut -d= -f2 | tr -d '"' || echo "?")
MODEL=$(echo "$CONTAINER_CONFIG" | grep "EPIMNEME_EMBEDDING_MODEL" | cut -d= -f2 | tr -d '"' || echo "?")

echo -e "$TIMESTAMP\t$BENCH\t$TAG\t$EPIMNEME_VERSION\t$GIT_HASH\t$CHUNK_SIZE\t$CHUNK_OVERLAP\t$EF_SEARCH\t$MODEL\t$(basename "$OUT_FILE")" >> "$HISTORY"

echo ""
echo "Log:     $LOG_FILE"
echo "History: $HISTORY"

exit $EXIT_CODE
