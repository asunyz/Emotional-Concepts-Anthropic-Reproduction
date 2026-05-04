"""Phase 4: paired test on epistemic-violation prompts.

For a list of (control, violation) prompt pairs, forward through BASE_MODEL_ID,
encode through the layer-L SAE, and for each candidate feature report:
  - mean_delta  = mean over pairs of (act@violation_target - act@control_target)
  - paired_t    = paired t-statistic across pairs
  - hit_rate    = fraction of pairs with positive delta
  - per-subcategory hit rates (A_numeric / B_geographic / C_category /
    D_historical / E_physical) to flag domain-general vs narrow features
  - subcat_coverage = #subcategories with hit_rate >= 0.75 (max 5)

The "target token" is the substituted word's first token. We tokenize the
control and violation, find the first index where they diverge, and read the
SAE activation at that position.

Usage:
    python scripts/epistemic_probe.py \\
        --candidates outputs/sae_surprise/candidates_layer_30_surprise_vs__neutral.csv \\
        --prompts inputs/epistemic/prompts.tsv \\
        --layer 30 --hook-point post_block \\
        --top 50
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config                                                 # noqa: E402
from cv_utils import load_model, extract_per_token_residuals  # noqa: E402
from scripts.sae_loader import load_sae, encode_topk          # noqa: E402

SUBCATS = ["A_numeric", "B_geographic", "C_category", "D_historical", "E_physical"]


def first_divergent_idx(tok_c: list[int], tok_v: list[int]) -> int:
    """Return the first position where tok_c and tok_v differ. Caller uses
    this as the target token in BOTH sequences (they share a prefix up to
    this point by construction of the prompt pairs)."""
    n = min(len(tok_c), len(tok_v))
    for i in range(n):
        if tok_c[i] != tok_v[i]:
            return i
    return n  # full prefix match — divergence is at the next token


def load_prompts(path: Path) -> list[dict]:
    with path.open() as f:
        reader = csv.DictReader(f, delimiter="\t")
        return list(reader)


def feature_act_at(model, sae, text: str, layer: int, hook: str,
                   feature_ids: np.ndarray, target_idx: int) -> np.ndarray:
    """Forward `text`, encode at layer `layer`, return SAE acts for the
    given feature_ids at position `target_idx`. Shape [n_features], fp32."""
    res = extract_per_token_residuals(model, text, [layer], hook_point=hook)
    x = res[layer].to(torch.float16).cuda()  # [seq, d_model]
    acts = encode_topk(x, sae)               # [seq, d_sae]
    target_idx = min(target_idx, acts.shape[0] - 1)
    return acts[target_idx, feature_ids].to(torch.float32).cpu().numpy()


def paired_t_stat(deltas: np.ndarray) -> float:
    """One-sample paired t-statistic against zero. NaN-safe."""
    d = deltas[~np.isnan(deltas)]
    if d.size < 2:
        return float("nan")
    mean = d.mean()
    sd = d.std(ddof=1)
    if sd == 0:
        return float("inf") if mean > 0 else float("-inf") if mean < 0 else 0.0
    return float(mean / (sd / np.sqrt(d.size)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True, type=Path,
                    help="CSV from contrastive_features.py")
    ap.add_argument("--prompts", default="inputs/epistemic/prompts.tsv", type=Path)
    ap.add_argument("--layer", required=True, type=int)
    ap.add_argument("--hook-point", default=None)
    ap.add_argument("--top", type=int, default=50,
                    help="Take the top-N rows from candidates CSV by `delta`.")
    ap.add_argument("--out-dir", default="outputs/sae_surprise", type=Path)
    args = ap.parse_args()

    hook = args.hook_point or config.SAE_HOOK_POINT
    if hook == "auto":
        raise SystemExit("config.SAE_HOOK_POINT is still 'auto'.")

    # Load candidate feature ids (already sorted by delta desc).
    with args.candidates.open() as f:
        rows = list(csv.DictReader(f))[:args.top]
    feature_ids = np.array([int(r["feature_id"]) for r in rows], dtype=np.int64)
    print(f"Probing {len(feature_ids)} candidate features (layer {args.layer})")

    pairs = load_prompts(args.prompts)
    print(f"Loaded {len(pairs)} prompt pairs from {args.prompts}")

    print(f"Loading BASE model: {config.BASE_MODEL_ID}")
    model = load_model(model_id=config.BASE_MODEL_ID)
    sae = load_sae(args.layer)
    tok = model.tokenizer

    # Pre-tokenize all pairs, locate target indices.
    target_idx_per_pair = []
    for p in pairs:
        ids_c = tok(p["control"]).input_ids
        ids_v = tok(p["violation"]).input_ids
        idx = first_divergent_idx(ids_c, ids_v)
        target_idx_per_pair.append(idx)

    # Collect acts: [n_pairs, n_features] for control and violation.
    n_pairs = len(pairs)
    n_feat = len(feature_ids)
    acts_c = np.zeros((n_pairs, n_feat), dtype=np.float32)
    acts_v = np.zeros((n_pairs, n_feat), dtype=np.float32)
    for i, p in enumerate(tqdm(pairs, desc="probe", unit="pair")):
        idx = target_idx_per_pair[i]
        acts_c[i] = feature_act_at(model, sae, p["control"],   args.layer, hook, feature_ids, idx)
        acts_v[i] = feature_act_at(model, sae, p["violation"], args.layer, hook, feature_ids, idx)
    deltas = acts_v - acts_c   # [n_pairs, n_feat]

    # Per-subcategory masks.
    subcats = np.array([p["subcategory"] for p in pairs])
    submasks = {sc: (subcats == sc) for sc in SUBCATS}

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"epistemic_layer_{args.layer}.csv"
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        header = ["feature_id", "mean_delta", "paired_t", "hit_rate_overall"]
        for sc in SUBCATS:
            header.append(f"hit_rate_{sc}")
        header += ["subcat_coverage", "verdict"]
        w.writerow(header)
        for j, fid in enumerate(feature_ids):
            d = deltas[:, j]
            mean_delta = float(d.mean())
            t = paired_t_stat(d)
            hit_overall = float((d > 0).mean())
            sc_hits = {sc: float((d[submasks[sc]] > 0).mean()) for sc in SUBCATS}
            coverage = sum(1 for sc in SUBCATS if sc_hits[sc] >= 0.75)
            if coverage >= 4 and t > 2.5:
                verdict = "epistemic"
            elif coverage >= 2 and t > 2.0:
                verdict = "partial"
            else:
                verdict = "narrow_or_none"
            row = [int(fid), mean_delta, t, hit_overall]
            row += [sc_hits[sc] for sc in SUBCATS]
            row += [coverage, verdict]
            w.writerow(row)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
