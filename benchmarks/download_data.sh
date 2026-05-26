#!/usr/bin/env bash
# Download benchmark datasets into benchmarks/data/ .
#
# Datasets (each under their own license — see benchmarks/DATA.md):
#   - LongMemEval (Xiao et al.)
#   - LoCoMo (Snap Research, CC BY-NC 4.0 — non-commercial research only)
#   - BEAM (Tavakoli et al., CC BY-SA 4.0 — loaded live from HuggingFace)

set -euo pipefail

cd "$(dirname "$0")/data"

if [[ ! -f longmemeval_s_cleaned.json ]]; then
    echo "→ LongMemEval: please fetch 'longmemeval_s_cleaned.json' from"
    echo "  https://github.com/xiaowu0162/LongMemEval"
    echo "  and place it at: $PWD/longmemeval_s_cleaned.json"
else
    echo "✓ LongMemEval already present"
fi

if [[ ! -f locomo10.json ]]; then
    echo "→ LoCoMo: please fetch 'locomo10.json' from"
    echo "  https://github.com/snap-research/locomo"
    echo "  and place it at: $PWD/locomo10.json"
    echo "  (CC BY-NC 4.0 — non-commercial research only)"
else
    echo "✓ LoCoMo already present"
fi

# BEAM is loaded directly from HuggingFace via the `datasets` library at runtime.
# No local download needed.  Dataset: Mohammadta/BEAM  (CC BY-SA 4.0)
# Run: python3 benchmarks/beam_bench.py --engram-url http://... --split 100K
echo "✓ BEAM: loaded from HuggingFace at runtime (Mohammadta/BEAM, CC BY-SA 4.0)"

echo ""
echo "Fixture files used by CI are tracked in git:"
ls -1 test_*_fixture.json 2>/dev/null || true
