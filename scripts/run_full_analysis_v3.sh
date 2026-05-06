#!/usr/bin/env bash
# Run the complete v3 analysis pipeline on a generated story set.
#
# Sequence (one model load each, runs sequentially):
#   1. extract_v3_compare.py  → 4 method dirs with concept_vectors_modeA.npz, mean.npy
#   2. sanity_check_v3.py     → consistency report
#   3. plot_v3.py             → per-method figures (cosine, PCA, layer scan, arithmetic)
#   4. run_v2_analyses_v3.py  → var probe, staining, steering on Methods C and D
#
# Usage:
#   bash scripts/run_full_analysis_v3.sh runs/cognitive_v3_midscale 30
#
# Args:
#   $1 — run directory (e.g., runs/cognitive_v3_midscale)
#   $2 — primary layer for v2 analyses (default: 30)
#   $3 — comma-separated layers for extraction (default: 10,20,30,36)

set -euo pipefail

RUN_DIR="${1:-runs/cognitive_v3_midscale}"
LAYER="${2:-30}"
LAYERS="${3:-10,20,30,36}"

PY=/workspace/venv/bin/python
TASK_NAME=$(basename "$RUN_DIR")

echo "=== [1/4] extract_v3_compare on $RUN_DIR ==="
$PY scripts/extract_v3_compare.py --run-dir "$RUN_DIR" --layers "$LAYERS"

echo ""
echo "=== [2/4] sanity_check_v3 (re-validate stories on disk) ==="
$PY scripts/sanity_check_v3.py "$RUN_DIR"

echo ""
echo "=== [3/4] plot_v3 for all 4 methods ==="
for m in methodA_v2style methodB_isolation methodC_incontext methodD_contrast; do
  echo "--- plotting $m ---"
  $PY scripts/plot_v3.py \
    --run-dir "$RUN_DIR/extractions/$m" \
    --layers "$LAYERS" \
    --output-dir "outputs/$TASK_NAME/comparison/$m"
done

echo ""
echo "=== [4/4] v2 analyses (var probe, staining, steering) on Method C and D ==="
for m in methodC_incontext methodD_contrast; do
  echo "--- v2 analyses on $m at layer $LAYER ---"
  $PY scripts/run_v2_analyses_v3.py \
    --vec-dir "$RUN_DIR/extractions/$m" \
    --layer "$LAYER" \
    --output-dir "outputs/$TASK_NAME/v2_analyses_$m"
done

echo ""
echo "=== ALL DONE ==="
echo "Stories:    $RUN_DIR/stories/"
echo "Vectors:    $RUN_DIR/extractions/"
echo "Figures:    outputs/$TASK_NAME/comparison/"
echo "V2 probes:  outputs/$TASK_NAME/v2_analyses_method[CD]/"
