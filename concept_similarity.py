"""
Pairwise cosine-similarity heatmap between concept vectors.

One subplot per layer. Values are annotated in each cell.

Usage:
    python concept_similarity.py --concept-dir runs/emotions_8b_nf4 --layers 16,24
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concept-dir", required=True)
    ap.add_argument("--layers", required=True, help="comma-separated layer indices")
    ap.add_argument("--output", default="concept_similarity.png")
    args = ap.parse_args()

    root = Path(args.concept_dir)
    layers = [int(x) for x in args.layers.split(",")]

    fig, axes = plt.subplots(1, len(layers), figsize=(5 * len(layers), 5), squeeze=False)
    for ax, L in zip(axes[0], layers):
        cvecs = np.load(root / f"layer_{L}" / "concept_vectors.npz")
        names = list(cvecs.files)
        V = np.stack([cvecs[n] for n in names])
        V = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)
        S = V @ V.T  # cosine since V is unit-norm

        im = ax.imshow(S, vmin=-1, vmax=1, cmap="coolwarm")
        ax.set_xticks(range(len(names)), names, rotation=45, ha="right")
        ax.set_yticks(range(len(names)), names)
        ax.set_title(f"layer {L}")
        for i in range(len(names)):
            for j in range(len(names)):
                ax.text(j, i, f"{S[i, j]:+.2f}", ha="center", va="center",
                        fontsize=8, color="white" if abs(S[i, j]) > 0.5 else "black")
        plt.colorbar(im, ax=ax, fraction=0.046)

    fig.tight_layout()
    fig.savefig(args.output, dpi=120)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
