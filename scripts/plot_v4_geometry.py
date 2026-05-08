"""
Plot v4 dialogue probe geometry from sanity-run JSON outputs.

Generates:
  - cosine_self_self.png         (8x8 heatmap, diag=1, off-diag near 0)
  - cosine_other_other.png       (8x8 heatmap, diag=1, off-diag near 0)
  - cosine_self_other.png        (8x8 heatmap, diag=0.33 expected, off-diag ~0)
  - table14_bars.png             (per other-concept, top-3 present-concepts as bars)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def heatmap(mat_dict: dict, out_path: Path, title: str, cbar_label: str,
            vmin=-0.5, vmax=1.0, fmt="+.2f"):
    rows = list(mat_dict.keys())
    cols = list(mat_dict[rows[0]].keys())
    M = np.array([[mat_dict[r].get(c, np.nan) for c in cols] for r in rows])
    fig, ax = plt.subplots(figsize=(7, 5.5))
    im = ax.imshow(M, cmap="RdBu_r", vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=20, ha="right", fontsize=10)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows, fontsize=10)
    for i in range(len(rows)):
        for j in range(len(cols)):
            v = M[i, j]
            color = "white" if abs(v) > 0.55 else "black"
            ax.text(j, i, f"{v:{fmt}}", ha="center", va="center",
                    color=color, fontsize=9)
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    cbar.set_label(cbar_label, fontsize=9)
    ax.set_title(title, fontsize=12, weight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_table14(table14: dict, out_path: Path, title: str):
    """Per other-concept, plot top 4 present-concept similarities as horizontal bars."""
    n = len(table14)
    fig, axes = plt.subplots(2, 4, figsize=(16, 6.5))
    axes = axes.flatten()
    for k, (other_c, sims) in enumerate(table14.items()):
        ax = axes[k]
        # take top 4 by similarity
        top = sorted(sims, key=lambda x: -x[1])[:4]
        names = [c for c, _ in top]
        vals = [v for _, v in top]
        colors = ["#c0392b" if v > 0.2 else "#7f8c8d" if v > 0 else "#34495e"
                  for v in vals]
        # highlight if top is the SAME concept (mirror/contagion)
        edge = ["gold" if names[0] == other_c else "black"]
        edges = ["gold" if c == other_c else "black" for c in names]
        ax.barh(np.arange(len(top)), vals, color=colors,
                edgecolor=edges, linewidth=2.0)
        ax.set_yticks(np.arange(len(top)))
        ax.set_yticklabels(names, fontsize=10)
        ax.invert_yaxis()
        ax.axvline(0, color="black", linewidth=0.6)
        ax.set_xlim(-0.3, 0.7)
        ax.set_title(f"Other = {other_c}", fontsize=11, weight="bold")
        ax.tick_params(axis="x", labelsize=8)
        for i, v in enumerate(vals):
            ax.text(v + 0.01 * (1 if v >= 0 else -1), i, f"{v:+.2f}",
                    va="center", ha="left" if v >= 0 else "right", fontsize=9)
    fig.suptitle(title + "\n(gold border = mirror; red = positive ≥0.2)",
                 fontsize=13, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"  wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="/tmp/v4_data")
    ap.add_argument("--output-dir", default="outputs/cognitive_v4_dialogue_sanity")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # -- Cosine matrices --
    ss = json.loads((data_dir / "cosines_self_self.json").read_text())
    oo = json.loads((data_dir / "cosines_other_other.json").read_text())
    so = json.loads((data_dir / "cosines_self_other.json").read_text())

    heatmap(ss, out_dir / "cosine_self_self.png",
            title="self × self — cognitive concept vectors at speaker-self positions",
            cbar_label="cosine similarity",
            vmin=-0.5, vmax=1.0)
    heatmap(oo, out_dir / "cosine_other_other.png",
            title="other × other — cognitive concept vectors at speaker-other positions",
            cbar_label="cosine similarity",
            vmin=-0.5, vmax=1.0)
    heatmap(so, out_dir / "cosine_self_other.png",
            title="self × other — diag=0.33 (partial separation, weaker than Anthropic ≈0)",
            cbar_label="cosine similarity",
            vmin=-0.5, vmax=0.7)

    # -- Table 14 --
    t14 = json.loads((data_dir / "table14.json").read_text())
    plot_table14(t14, out_dir / "table14_bars.png",
                  title="Cognitive Table 14: closest present-speaker vector to each other-speaker vector")


if __name__ == "__main__":
    main()
