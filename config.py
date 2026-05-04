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

# Qwen3.5-35B-A3B-Base — used by SAE analysis on the `asuka` branch. Same
# architecture as Qwen3.6 (40 layers, hidden=2048, 256 experts top-8 + 1
# shared) but distinct weights. Pinned here because the Qwen-Scope SAE was
# trained on this model; running the SAE on Qwen3.6-Instruct activations
# would mix two distribution shifts (weight version + base→instruct).
# See docs/sae_surprise_analysis_plan.md.
BASE_MODEL_ID = "Qwen/Qwen3.5-35B-A3B-Base"

# "nf4" (4-bit, best for 11 GB GPUs), "int8", or "none" (full fp16)
QUANTIZATION = "nf4"

COMPUTE_DTYPE = torch.float16

# transformers 5.x's "auto" infer_auto_device_map is over-conservative for
# Qwen3 MoE in NF4 — it tries to offload to CPU/disk even when the quantized
# model (~18 GB) easily fits on a 48 GB card, then bnb 4-bit refuses to load
# because partial offload isn't supported. Force everything onto GPU 0.
# Override via DEVICE_MAP env var if you actually need multi-GPU.
DEVICE_MAP = os.environ.get("DEVICE_MAP", "cuda:0")

# --- where things live ---------------------------------------------------
# Default to /workspace/models (RunPod persistent volume). Override with
# MODELS_ROOT env var if running elsewhere (e.g. /mnt/e/models on WSL).
MODELS_ROOT = Path(os.environ.get("MODELS_ROOT", "/workspace/models"))
HF_CACHE = MODELS_ROOT / "hf_cache"


def local_model_dir(model_id: str | None = None) -> Path:
    """On-disk folder for the pre-quantized copy of (model_id, QUANTIZATION).

    Defaults to MODEL_ID if no override given, preserving the original
    single-model behavior. Pass BASE_MODEL_ID to address the SAE-companion
    base model without mutating global state.
    """
    mid = model_id if model_id is not None else MODEL_ID
    stem = mid.split("/")[-1].lower()
    return MODELS_ROOT / f"{stem}-{QUANTIZATION}"


# --- SAE (Qwen-Scope) ----------------------------------------------------
# Sparse Autoencoder for the BASE_MODEL_ID's residual stream. TopK SAE,
# k=100, width=131072, d_model=2048. One file per layer (0..39) named
# `layer{N}.sae.pt`. Each file is a torch dict with keys:
#   W_enc (131072, 2048), W_dec (2048, 131072), b_enc (131072,), b_dec (2048,)
# Encoding: acts = topk(x @ W_enc.T + b_enc, k=K), other dims zeroed.
SAE_REPO = "Qwen/SAE-Res-Qwen3.5-35B-A3B-Base-W128K-L0_100"
SAE_LAYERS = (30, 31, 35)         # Phase 1 scope
SAE_K = 100
SAE_LOCAL_DIR = MODELS_ROOT / "sae" / "qwen3.5-35b-a3b-base-w128k-l0_100"
# Hook point is set after Phase 1's smoke test. Valid values:
#   "post_block" — model.model.layers[L].output[0]   (current default in cv_utils)
#   "pre_block"  — model.model.layers[L].input[0]
#   "mid_block"  — between attention residual-add and MLP (post-attn, pre-MLP)
# Resolved by scripts/smoke_test_hook.py (2026-05-04). post_block matches
# `model.model.layers[L].output[0]` — i.e., the same hook the existing
# extract_layer_activations uses. Normalized recon MSE: 0.287 on a 49-token
# probe text (vs 0.338 mid_block, 0.345 pre_block).
SAE_HOOK_POINT = "post_block"


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
# Defaults follow Qwen3.6-A3B's official non-thinking-mode recommendation
# from https://huggingface.co/Qwen/Qwen3.6-35B-A3B. Qwen recommends
# presence_penalty=1.5, but HuggingFace's `model.generate()` doesn't accept
# that kwarg (it's a vLLM/SGLang-style param); the closest native equivalent
# is `repetition_penalty`, used here at a milder value to avoid degrading
# fluency while still suppressing the NF4-MoE repetition loops.
GEN_MAX_NEW_TOKENS = 200
GEN_TEMPERATURE = 0.7
GEN_TOP_P = 0.8
GEN_TOP_K = 20
GEN_REPETITION_PENALTY = 1.1
