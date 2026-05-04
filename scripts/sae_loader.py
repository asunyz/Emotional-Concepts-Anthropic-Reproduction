"""Load Qwen-Scope SAE weights and run the TopK encoder.

The SAE is a TopK sparse autoencoder:
    pre_acts = x @ W_enc.T + b_enc       # [..., d_sae]
    keep top-k pre_acts (per-position), zero the rest -> acts
    recon     = acts @ W_dec.T + b_dec   # [..., d_model]
    (W_dec stored as [d_model, d_sae], so we use W_dec.T or matmul accordingly)

Inspect each .pt to confirm the W_dec shape; this loader normalizes to
acts: [..., d_sae] and recon: [..., d_model] regardless.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

# Allow `python scripts/sae_loader.py` and direct imports both.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402


def sae_path(layer: int) -> Path:
    return config.SAE_LOCAL_DIR / f"layer{layer}.sae.pt"


def load_sae(layer: int, device: str | torch.device = "cuda:0",
             dtype: torch.dtype = torch.float16) -> dict[str, torch.Tensor]:
    """Load one layer's SAE state-dict to `device` in `dtype`.

    Returns dict with keys W_enc, W_dec, b_enc, b_dec, plus normalized
    metadata: d_model, d_sae, k.
    """
    p = sae_path(layer)
    if not p.exists():
        raise FileNotFoundError(
            f"SAE for layer {layer} not found at {p}. "
            f"Run scripts/download_sae.py first."
        )
    sd = torch.load(str(p), map_location="cpu", weights_only=True)
    out = {k: v.to(device=device, dtype=dtype).contiguous()
           for k, v in sd.items() if isinstance(v, torch.Tensor)}

    # Normalize shapes. Card says:
    #   W_enc (d_sae, d_model), W_dec (d_model, d_sae),
    #   b_enc (d_sae,),         b_dec (d_model,)
    W_enc = out["W_enc"]
    if W_enc.ndim != 2:
        raise ValueError(f"W_enc has unexpected shape {tuple(W_enc.shape)}")
    d_sae, d_model = W_enc.shape
    out["d_sae"] = d_sae
    out["d_model"] = d_model
    out["k"] = config.SAE_K
    return out


def encode_topk(x: torch.Tensor, sae: dict, k: int | None = None) -> torch.Tensor:
    """Encode `x` ([..., d_model]) through the SAE's TopK encoder.

    Returns [..., d_sae] with exactly `k` non-zero entries per position.
    """
    if k is None:
        k = sae["k"]
    W_enc, b_enc = sae["W_enc"], sae["b_enc"]
    # Match dtype/device of weights.
    x = x.to(device=W_enc.device, dtype=W_enc.dtype)
    pre = x @ W_enc.T + b_enc                          # [..., d_sae]
    topk_vals, topk_idx = pre.topk(k, dim=-1)
    acts = torch.zeros_like(pre)
    acts.scatter_(-1, topk_idx, topk_vals)
    return acts


def decode(acts: torch.Tensor, sae: dict) -> torch.Tensor:
    """Decode TopK feature acts back to residual space."""
    W_dec, b_dec = sae["W_dec"], sae["b_dec"]
    # W_dec is (d_model, d_sae); recon = acts @ W_dec.T + b_dec
    acts = acts.to(device=W_dec.device, dtype=W_dec.dtype)
    return acts @ W_dec.T + b_dec


def reconstruct(x: torch.Tensor, sae: dict, k: int | None = None) -> torch.Tensor:
    return decode(encode_topk(x, sae, k=k), sae)


def reconstruction_mse(x: torch.Tensor, sae: dict, k: int | None = None) -> dict:
    """Return absolute and ‖x‖²-normalized MSE for a quick hook-point sanity check."""
    recon = reconstruct(x, sae, k=k).to(dtype=torch.float32)
    x32 = x.to(device=recon.device, dtype=torch.float32)
    err = (x32 - recon).pow(2).sum(dim=-1)            # [...]
    norm_sq = x32.pow(2).sum(dim=-1).clamp_min(1e-9)
    return {
        "mse_per_token": err.mean().item(),
        "norm_sq_per_token": norm_sq.mean().item(),
        "normalized_mse": (err / norm_sq).mean().item(),
        "n_tokens": int(err.numel()),
    }
