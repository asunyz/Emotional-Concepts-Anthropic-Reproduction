#!/usr/bin/env bash
# Cognitive v4 — Two-speaker dialogue pipeline (Anthropic Table 14 reproduction).
#
# Steps:
#   1. Generate dialogues (8x8 concept pairs × N per pair)
#   2. Extract 2x2 grid probes at target layer
#   3. Geometry analysis (cosine matrices + Table 14)
#
# Usage:
#   bash scripts/run_dialogue_pipeline.sh sanity    # 4 per pair, 256 dialogues
#   bash scripts/run_dialogue_pipeline.sh mid       # 8 per pair, 512 dialogues
#   bash scripts/run_dialogue_pipeline.sh full      # 16 per pair, 1024 dialogues

set -euo pipefail

PY="${PY:-/workspace/venv/bin/python}"
SCALE="${1:-sanity}"
LAYER="${LAYER:-30}"

case "$SCALE" in
  sanity) DPP=4 ;;
  mid)    DPP=8 ;;
  full)   DPP=16 ;;
  *) echo "unknown scale: $SCALE (use sanity|mid|full)"; exit 1 ;;
esac

LABEL="cognitive_v4_dialogue_${SCALE}"
RUN_DIR="runs/${LABEL}"

echo "============================================================"
echo "v4 dialogue pipeline — scale=${SCALE}, dialogues_per_pair=${DPP}"
echo "============================================================"

echo ""
echo "=== [1/3] generate dialogues → ${RUN_DIR}/dialogues/ ==="
$PY scripts/generate_dialogues_v4.py \
  --task-label "$LABEL" \
  --dialogues-per-pair "$DPP"

echo ""
echo "=== [2/3] extract 2x2 grid probes (layer ${LAYER}) ==="
$PY scripts/extract_dialogue_probes.py \
  --run-dir "$RUN_DIR" \
  --layer "$LAYER"

echo ""
echo "=== [3/3] geometry analysis (Table 14 reproduction) ==="
$PY scripts/analyze_dialogue_geometry.py \
  --vec-dir "${RUN_DIR}/extractions_dialogue/layer_${LAYER}"

echo ""
echo "=== ALL DONE ==="
echo "Run dir : ${RUN_DIR}"
echo "Vectors : ${RUN_DIR}/extractions_dialogue/layer_${LAYER}"
echo "Analysis: ${RUN_DIR}/extractions_dialogue/layer_${LAYER}/analysis"
