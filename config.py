"""
Central config for every script in this repo.

Swap the constants below to target a different model / quantization / cache
location; every script picks them up on the next run. Local save dirs are
derived from (MODEL_ID, QUANTIZATION) so different combos don't clobber each
other on disk.

MODELS_ROOT can be overridden via the MODELS_ROOT env var so the same code
runs on different machines (e.g. /workspace on RunPod, /mnt/e/models on WSL).
"""
import os
from pathlib import Path

import torch
from transformers import BitsAndBytesConfig

# --- what to load --------------------------------------------------------
# Qwen3.6-35B-A3B: MoE, 35B total / 3B active, 40 layers, hidden=2048,
# 256 experts (8 routed + 1 shared), 262K native context (1M with YaRN).
# Goal: validate the concept-vector pipeline on MoE arch (Anthropic's
# emotion paper used dense Llama).
MODEL_ID = "Qwen/Qwen3.6-35B-A3B"

# "nf4" (4-bit, best for 11 GB GPUs), "int8", or "none" (full fp16)
QUANTIZATION = "nf4"

COMPUTE_DTYPE = torch.float16
DEVICE_MAP = "auto"

# --- where things live ---------------------------------------------------
# Default to /workspace/models (RunPod persistent volume). Override with
# MODELS_ROOT env var if running elsewhere (e.g. /mnt/e/models on WSL).
MODELS_ROOT = Path(os.environ.get("MODELS_ROOT", "/workspace/models"))
HF_CACHE = MODELS_ROOT / "hf_cache"


def local_model_dir() -> Path:
    """On-disk folder for the pre-quantized copy of (MODEL_ID, QUANTIZATION)."""
    stem = MODEL_ID.split("/")[-1].lower()
    return MODELS_ROOT / f"{stem}-{QUANTIZATION}"


def build_quant_config() -> BitsAndBytesConfig | None:
    if QUANTIZATION == "none":
        return None
    if QUANTIZATION == "int8":
        return BitsAndBytesConfig(load_in_8bit=True)
    if QUANTIZATION == "nf4":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=COMPUTE_DTYPE,
            bnb_4bit_use_double_quant=True,
        )
    raise ValueError(f"Unknown QUANTIZATION={QUANTIZATION!r}")


# --- generation defaults (used by scripts that sample) -------------------
GEN_MAX_NEW_TOKENS = 200
GEN_TEMPERATURE = 0.8
GEN_TOP_P = 0.9
