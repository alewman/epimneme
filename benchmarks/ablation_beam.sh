#!/usr/bin/env bash
# =============================================================================
# ablation_beam.sh — BEAM signal ablation grid
# =============================================================================
# Runs BEAM 100K with individual signals disabled to isolate regression cause.
# Requires staged BEAM data (run_bench.sh beam <tag> first without --skip-ingest).
#
# Usage:
#   ./benchmarks/ablation_beam.sh [--dry-run]
#
# Each variant restarts the engram2 container with an override compose file,
# runs BEAM 100K --skip-ingest (~30s), then restores normal config.
# =============================================================================

set -euo pipefail

BENCH_DIR="$(cd "$(dirname "$0")" && pwd)"
COMPOSE_DIR="/docker/compose/dev"
COMPOSE_FILE="$COMPOSE_DIR/engram2.yml"
ENV_FILE="/docker/compose/.env"
EPIMNEME_URL="${EPIMNEME_URL:-http://192.168.90.45:8000}"
DRY_RUN=false

[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

# ── Ablation variants ─────────────────────────────────────────────────────────
# Format: "tag|KEY1=V1;KEY2=V2;..."
# Empty overrides string = all signals on (v402 default, sanity check)
VARIANTS=(
  "abl-all-off|EPIMNEME_BM25_SIGNAL_ENABLED=0;EPIMNEME_ENTITY_SIGNAL_ENABLED=0;EPIMNEME_DATE_SIGNAL_WEIGHT=0;EPIMNEME_RECENCY_SIGNAL_WEIGHT=0;EPIMNEME_TURN_PAIR_SIGNAL_WEIGHT=0;EPIMNEME_TIEBREAK_ENABLED=0;EPIMNEME_MMR_ENABLED=0"
  "abl-no-bm25|EPIMNEME_BM25_SIGNAL_ENABLED=0"
  "abl-no-entity|EPIMNEME_ENTITY_SIGNAL_ENABLED=0"
  "abl-no-date|EPIMNEME_DATE_SIGNAL_WEIGHT=0"
  "abl-no-recency-turnpair|EPIMNEME_RECENCY_SIGNAL_WEIGHT=0;EPIMNEME_TURN_PAIR_SIGNAL_WEIGHT=0"
  "abl-no-tiebreak-mmr|EPIMNEME_TIEBREAK_ENABLED=0;EPIMNEME_MMR_ENABLED=0"
)

OVERRIDE_FILE="/tmp/engram2_ablation_override.yml"
RESULTS_TSV="/tmp/ablation_beam_results.tsv"

echo "Category	v302	variant	tag	delta_vs_v302" > "$RESULTS_TSV"

# ── Helper: write compose override ───────────────────────────────────────────
write_override() {
  local overrides="$1"
  {
    echo "services:"
    echo "  engram2:"
    echo "    environment:"
    IFS=';' read -ra PAIRS <<< "$overrides"
    for pair in "${PAIRS[@]}"; do
      local key="${pair%%=*}"
      local val="${pair#*=}"
      echo "      $key: \"$val\""
    done
  } > "$OVERRIDE_FILE"
}

# ── Helper: restart engram2 with optional override ────────────────────────────
restart_with_override() {
  local overrides="${1:-}"
  echo ""
  echo "── Restarting engram2 ──"
  if [[ -n "$overrides" ]]; then
    write_override "$overrides"
    echo "   Override file:"
    cat "$OVERRIDE_FILE"
    cd "$COMPOSE_DIR"
    docker compose -f engram2.yml -f "$OVERRIDE_FILE" --env-file "$ENV_FILE" up -d engram2 2>&1
  else
    rm -f "$OVERRIDE_FILE"
    cd "$COMPOSE_DIR"
    docker compose -f engram2.yml --env-file "$ENV_FILE" up -d engram2 2>&1
  fi

  echo -n "   Waiting for health..."
  for i in $(seq 1 30); do
    status=$(docker inspect engram2 --format '{{.State.Health.Status}}' 2>/dev/null || echo "unknown")
    if [[ "$status" == "healthy" ]]; then
      echo " healthy"
      return 0
    fi
    sleep 2
    echo -n "."
  done
  echo " TIMEOUT — container may not be healthy"
  docker inspect engram2 --format '{{.State.Health.Status}}'
}

# ── Helper: run BEAM and extract avg_recall ───────────────────────────────────
run_beam() {
  local tag="$1"
  cd "$BENCH_DIR/.."
  echo ""
  echo "── Running BEAM 100K --skip-ingest (tag=$tag) ──"
  if $DRY_RUN; then
    echo "   [dry-run] skipping"
    echo "0.0000"
    return
  fi
  ./benchmarks/run_bench.sh beam "$tag" --skip-ingest --split 100K 2>&1
  # Find latest result file for this tag
  local result_file
  result_file=$(ls -t "$BENCH_DIR"/results_engram_beam_${tag}_*.json 2>/dev/null | head -1)
  if [[ -z "$result_file" ]]; then
    echo "   ERROR: no result file found for tag=$tag"
    echo "0.0000"
    return
  fi
  python3 -c "
import json, sys
d = json.load(open('$result_file'))
per = d.get('per_ability', {})
v302 = {
  'contradiction_resolution': 0.6637,
  'event_ordering': 0.0980,
  'information_extraction': 0.5917,
  'instruction_following': 0.0813,
  'knowledge_update': 0.6667,
  'multi_session_reasoning': 0.2895,
  'preference_following': 0.2308,
  'summarization': 0.1560,
  'temporal_reasoning': 0.7125,
}
print(f'  avg_recall={d[\"meta\"][\"avg_recall\"]:.4f}  perfect={d[\"meta\"][\"perfect_pct\"]}%')
print(f'  Category                       v302    this    delta')
print(f'  {\"─\"*56}')
for k in sorted(v302.keys()):
    r_this = per.get(k, {}).get('recall')
    r_v302 = v302[k]
    if r_this is not None:
        print(f'  {k:<30} {r_v302:.4f}  {r_this:.4f}  {r_this-r_v302:+.4f}')
print()
"
}

# ── Main ablation loop ────────────────────────────────────────────────────────
echo "============================================================"
echo "  BEAM 100K Ablation Grid"
echo "  Variants: ${#VARIANTS[@]}"
echo "  Date:     $(date)"
echo "============================================================"
echo ""

SUMMARY=()

for variant in "${VARIANTS[@]}"; do
  IFS='|' read -r tag overrides <<< "$variant"
  echo ""
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "  VARIANT: $tag"
  echo "  OVERRIDES: ${overrides:-<none>}"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

  if ! $DRY_RUN; then
    restart_with_override "$overrides"
  fi

  run_beam "$tag"

  # Grab avg_recall from latest result
  if ! $DRY_RUN; then
    result_file=$(ls -t "$BENCH_DIR"/results_engram_beam_${tag}_*.json 2>/dev/null | head -1)
    if [[ -n "$result_file" ]]; then
      avg=$(python3 -c "import json; d=json.load(open('$result_file')); print(f'{d[\"meta\"][\"avg_recall\"]:.4f}')")
      SUMMARY+=("$tag: avg_recall=$avg  (v402-clean=0.4167  v302=0.3917)")
    fi
  fi
done

# ── Restore normal config ─────────────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Restoring normal config (all signals on)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if ! $DRY_RUN; then
  restart_with_override ""
fi

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  Ablation Summary"
echo "  v302 baseline:  avg_recall=0.3917"
echo "  v402-clean (all on):  avg_recall=0.4167"
echo "────────────────────────────────────────────────────────────"
for s in "${SUMMARY[@]}"; do
  echo "  $s"
done
echo "============================================================"
