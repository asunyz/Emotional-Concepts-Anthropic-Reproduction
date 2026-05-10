"""
PCA scatter of concept vectors (2D or 3D), one subplot per layer.

Usage:
    python concept_cluster.py --concept-dir runs/emotions_8b_nf4 --layers 16,24
    python concept_cluster.py ... --n-components 3
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concept-dir", required=True)
    ap.add_argument("--layers", required=True)
    ap.add_argument("--n-components", type=int, default=2, choices=[2, 3])
    ap.add_argument("--output", default="concept_cluster.png")
    args = ap.parse_args()

    root = Path(args.concept_dir)
    layers = [int(x) for x in args.layers.split(",")]

    if args.n_components == 2:
        fig, axes = plt.subplots(1, len(layers), figsize=(5 * len(layers), 5), squeeze=False)
        for ax, L in zip(axes[0], layers):
            cvecs = np.load(root / f"layer_{L}" / "concept_vectors.npz")
            names = list(cvecs.files)
            V = np.stack([cvecs[n] for n in names])
            Z = PCA(n_components=2).fit_transform(V)
            ax.scatter(Z[:, 0], Z[:, 1], s=60)
            for i, n in enumerate(names):
                ax.annotate(n, Z[i], fontsize=10, xytext=(4, 4), textcoords="offset points")
            ax.axhline(0, color="gray", lw=0.5)
            ax.axvline(0, color="gray", lw=0.5)
            ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
            ax.set_title(f"layer {L}")
    else:
        fig = plt.figure(figsize=(6 * len(layers), 6))
        for i, L in enumerate(layers):
            ax = fig.add_subplot(1, len(layers), i + 1, projection="3d")
            cvecs = np.load(root / f"layer_{L}" / "concept_vectors.npz")
            names = list(cvecs.files)
            V = np.stack([cvecs[n] for n in names])
            Z = PCA(n_components=3).fit_transform(V)
            ax.scatter(Z[:, 0], Z[:, 1], Z[:, 2], s=60)
            for j, n in enumerate(names):
                ax.text(Z[j, 0], Z[j, 1], Z[j, 2], n, fontsize=9)
            ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("PC3")
            ax.set_title(f"layer {L}")

    fig.tight_layout()
    fig.savefig(args.output, dpi=120)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
