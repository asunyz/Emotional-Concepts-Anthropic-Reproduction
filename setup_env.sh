#!/usr/bin/env bash
# One-time setup: creates the `nnsight` conda env and installs dependencies.
# Usage:  bash setup_env.sh
#         conda activate nnsight
#         python nnsight_intro.py

set -euo pipefail

ENV_NAME="nnsight"
PY_VER="3.11"

# Create env (skip if it already exists)
if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    conda create -y -n "${ENV_NAME}" python="${PY_VER}"
fi

# Install into the env without needing `conda activate` in this shell
conda run -n "${ENV_NAME}" pip install --upgrade pip
conda run -n "${ENV_NAME}" pip install \
    "nnsight>=0.3" \
    "torch" \
    "transformers>=4.43" \
    "accelerate" \
    "bitsandbytes" \
    "sentencepiece" \
    "protobuf"

echo
echo "Done. Next:"
echo "  conda activate ${ENV_NAME}"
echo "  huggingface-cli login          # needed for gated Llama 3.1"
echo "  python nnsight_intro.py"
