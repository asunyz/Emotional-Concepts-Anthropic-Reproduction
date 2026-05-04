"""Phase 3: rank SAE features by (concept_A vs concept_B) mean activation diff.

Default: surprise vs neutral (the main hypothesis). Other contrasts (e.g.
afraid vs neutral) are useful as controls.

Story → concept mapping uses the filename convention:
    <concept>-<topic_idx>-<story_idx>.txt   for concept stories
    _neutral-<topic_idx>-<story_idx>.txt    for neutral

Outputs:
    outputs/sae_surprise/candidates_layer_<L>_<A>_vs_<B>.csv
    Columns: feature_id, mean_A, mean_B, delta, n_A, n_B,
             nonzero_rate_A, nonzero_rate_B

Usage:
    python scripts/contrastive_features.py \\
        --sae-acts-dir runs/emotions_qwen35_BASE \\
        --layer 30 --concept-a surprise --concept-b _neutral \\
        --top-k 50
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def stem_to_concept(stem: str) -> str:
    """`surprise-3-2` -> `surprise`; `_neutral-1-4` -> `_neutral`."""
    return stem.rsplit("-", 2)[0]


def load_acts(layer_dir: Path, agg: str = "mean") -> tuple[list[str], np.ndarray]:
    p = layer_dir / f"agg_{agg}.npz"
    with np.load(p) as d:
        names = sorted(d.files)
        mat = np.stack([d[k] for k in names])  # [n_stories, d_sae]
    return names, mat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sae-acts-dir", required=True, type=Path,
                    help="e.g. runs/emotions_qwen35_BASE — must contain layer_<L>/agg_mean.npz")
    ap.add_argument("--layer", required=True, type=int)
    ap.add_argument("--concept-a", default="surprise")
    ap.add_argument("--concept-b", default="_neutral")
    ap.add_argument("--agg", default="mean", choices=["mean", "max"])
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--out-dir", default="outputs/sae_surprise", type=Path)
    args = ap.parse_args()

    layer_dir = args.sae_acts_dir / f"layer_{args.layer}"
    names, mat = load_acts(layer_dir, agg=args.agg)
    concepts = np.array([stem_to_concept(n) for n in names])
    mat = mat.astype(np.float32)

    mask_a = concepts == args.concept_a
    mask_b = concepts == args.concept_b
    n_a, n_b = int(mask_a.sum()), int(mask_b.sum())
    if n_a == 0 or n_b == 0:
        raise SystemExit(f"No stories matched: concept_a={args.concept_a} -> {n_a}, "
                         f"concept_b={args.concept_b} -> {n_b}. "
                         f"Available concepts: {sorted(set(concepts))}")
    print(f"Layer {args.layer}: {args.concept_a}={n_a} vs {args.concept_b}={n_b} "
          f"({mat.shape[1]} features)")

    mean_a = mat[mask_a].mean(0)
    mean_b = mat[mask_b].mean(0)
    delta  = mean_a - mean_b
    nz_a = (mat[mask_a] > 0).mean(0)
    nz_b = (mat[mask_b] > 0).mean(0)

    top_idx = np.argsort(-delta)[:args.top_k]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"candidates_layer_{args.layer}_{args.concept_a}_vs_{args.concept_b}.csv"
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["feature_id", f"mean_{args.concept_a}", f"mean_{args.concept_b}",
                    "delta", f"nonzero_rate_{args.concept_a}",
                    f"nonzero_rate_{args.concept_b}", f"n_{args.concept_a}",
                    f"n_{args.concept_b}"])
        for i in top_idx:
            w.writerow([int(i), float(mean_a[i]), float(mean_b[i]), float(delta[i]),
                        float(nz_a[i]), float(nz_b[i]), n_a, n_b])
    print(f"Wrote top-{args.top_k} candidates -> {out}")


if __name__ == "__main__":
    main()
