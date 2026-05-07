"""
Re-plot var_probe results from saved `*_scores.npz` files.

Why this script exists
----------------------
The raw cosine-similarity matrix S (variants x concepts) is dominated by a
shared concept baseline: at the probe template's hidden state, some concepts
(curious, uncertain, confident) sit at +1.3 cosine and others (enlightened,
confused) sit at -0.95, regardless of which variant phrase was used.

ANOVA-style variance decomposition on var_reading:
    column (concept-baseline) effect: 99.4%
    row (variant) effect:              0.0%
    interaction (the actual signal):   0.6%

The interaction term is what tells us "did this variant phrase push concept c
above c's typical value?" — that is the experimentally meaningful quantity.

Two scientifically defensible normalizations
--------------------------------------------
1. Column centering    : S_c[i,j] = S[i,j] - col_mean[j]
   Units = raw cosine deviation. Preserves magnitude.

2. Column z-score      : S_z[i,j] = (S[i,j] - col_mean[j]) / col_std[j]
   Units = sigmas above/below this concept's typical value at this probe.
   Best for ranking and cross-concept comparison.

We render both, with bar charts as the primary view (one subplot per variant,
horizontal bars sorted by z-score, color-coded by sign).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_scores(npz_path: Path):
    d = dict(np.load(npz_path, allow_pickle=True))
    return d["scores"], list(d["values"]), list(d["concepts"])


def column_center(S: np.ndarray) -> np.ndarray:
    return S - S.mean(axis=0, keepdims=True)


def column_zscore(S: np.ndarray) -> np.ndarray:
    mu = S.mean(axis=0, keepdims=True)
    sd = S.std(axis=0, keepdims=True)
    return (S - mu) / (sd + 1e-9)


def plot_per_variant_bars(S_norm: np.ndarray, variants, concepts, title: str,
                           xlabel: str, out_path: Path):
    """Grid of horizontal bar charts: one subplot per variant.

    Bars are sorted by normalized score, colored red for positive (variant
    pushes concept up) and steel-blue for negative.
    """
    n = len(variants)
    n_cols = min(3, n)
    n_rows = int(np.ceil(n / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.2 * n_cols, 0.6 * len(concepts) * n_rows + 0.4))
    axes = np.atleast_2d(axes).reshape(n_rows, n_cols)

    vmax = float(np.abs(S_norm).max()) * 1.05

    for k, variant in enumerate(variants):
        r, c = divmod(k, n_cols)
        ax = axes[r, c]
        row = S_norm[k]
        order = np.argsort(row)
        sorted_vals = row[order]
        sorted_names = [concepts[i] for i in order]
        colors = ["#c0392b" if v > 0 else "#34495e" for v in sorted_vals]
        ax.barh(np.arange(len(sorted_vals)), sorted_vals, color=colors,
                edgecolor="black", linewidth=0.4)
        ax.set_yticks(np.arange(len(sorted_vals)))
        ax.set_yticklabels(sorted_names, fontsize=9)
        ax.axvline(0, color="black", linewidth=0.6)
        ax.set_xlim(-vmax, vmax)
        ax.set_title(f'"{variant}"', fontsize=10)
        ax.set_xlabel(xlabel, fontsize=8)
        ax.tick_params(axis="x", labelsize=8)
        ax.grid(axis="x", alpha=0.25, linewidth=0.5)
        # annotate top-1
        top_idx = order[-1]
        ax.text(row[top_idx], len(sorted_vals) - 1,
                f"  {row[top_idx]:+.2f}", va="center", ha="left",
                fontsize=8, color="#c0392b", weight="bold")

    # blank unused cells
    for k in range(n, n_rows * n_cols):
        r, c = divmod(k, n_cols)
        axes[r, c].axis("off")

    fig.suptitle(title, fontsize=13, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_heatmap(S_norm: np.ndarray, variants, concepts, title: str,
                  cbar_label: str, out_path: Path, fmt: str = "+.2f"):
    fig, ax = plt.subplots(figsize=(0.95 * len(concepts) + 1.5,
                                     0.45 * len(variants) + 1.5))
    vmax = float(np.abs(S_norm).max())
    im = ax.imshow(S_norm, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(np.arange(len(concepts)))
    ax.set_xticklabels(concepts, rotation=35, ha="right", fontsize=9)
    ax.set_yticks(np.arange(len(variants)))
    ax.set_yticklabels(variants, fontsize=9)
    for i in range(len(variants)):
        for j in range(len(concepts)):
            v = S_norm[i, j]
            color = "white" if abs(v) > 0.6 * vmax else "black"
            ax.text(j, i, f"{v:{fmt}}", ha="center", va="center",
                    color=color, fontsize=7)
        # gold border on top-1 concept per row
        top = int(np.argmax(S_norm[i]))
        ax.add_patch(plt.Rectangle((top - 0.5, i - 0.5), 1, 1,
                                     fill=False, edgecolor="gold", linewidth=2))
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label(cbar_label, fontsize=9)
    ax.set_title(title, fontsize=12, weight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"  wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores-dir", required=True,
                    help="dir containing *_scores.npz")
    ap.add_argument("--names", nargs="+",
                    default=["var_reading", "var_priors", "var_experiment", "var_gift"])
    ap.add_argument("--out-suffix", default="_v2")
    args = ap.parse_args()

    scores_dir = Path(args.scores_dir)
    out_dir = scores_dir / f"replot{args.out_suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)

    for name in args.names:
        npz = scores_dir / f"{name}_scores.npz"
        if not npz.exists():
            print(f"  skip {name} (no {npz.name})")
            continue
        S, V, C = load_scores(npz)
        print(f"\n=== {name} :: shape {S.shape} ===")

        S_centered = column_center(S)
        S_z = column_zscore(S)

        # Bar charts (primary)
        plot_per_variant_bars(
            S_z, V, C,
            title=f"{name}  (column z-score: σ above/below each concept's baseline at this probe)",
            xlabel="z-score (σ)",
            out_path=out_dir / f"{name}_bars_zscore.png",
        )
        plot_per_variant_bars(
            S_centered, V, C,
            title=f"{name}  (column-centered: cosine deviation from each concept's baseline)",
            xlabel="Δ cosine vs concept baseline",
            out_path=out_dir / f"{name}_bars_centered.png",
        )

        # Heatmaps (compact summary)
        plot_heatmap(
            S_z, V, C,
            title=f"{name}  (z-score)",
            cbar_label="σ vs concept baseline",
            out_path=out_dir / f"{name}_heatmap_zscore.png",
        )
        plot_heatmap(
            S_centered, V, C,
            title=f"{name}  (column-centered)",
            cbar_label="Δ cosine",
            out_path=out_dir / f"{name}_heatmap_centered.png",
            fmt="+.3f",
        )

    print(f"\nAll figures in {out_dir}")


if __name__ == "__main__":
    main()
