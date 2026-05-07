"""
Re-plot cross-method and cross-layer consistency from runs/.../consistency_report.md
as bar charts (with optional z-score normalization), per concept.

Read structured tables from the markdown report and render:
  - cross_method_bars.png       — 9 concepts × 6 method-pairs
  - cross_method_zscore.png     — z-score per row (highlights anomalies per concept)
  - cross_layer_bars.png        — 9 concepts × 6 layer-pairs
  - cross_layer_zscore.png      — z-score per row
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_md_table(text: str, anchor: str) -> tuple[list[str], list[str], np.ndarray]:
    """Find a markdown table after `anchor` heading. Returns (headers, row_labels, values)."""
    lines = text.splitlines()
    idx = next((i for i, l in enumerate(lines) if anchor in l), None)
    if idx is None:
        raise ValueError(f"anchor not found: {anchor!r}")
    # find table header (first line starting with '|' after idx)
    h = next(i for i in range(idx, len(lines)) if lines[i].lstrip().startswith("|"))
    headers = [c.strip() for c in lines[h].strip().strip("|").split("|")]
    row_labels: list[str] = []
    values: list[list[float]] = []
    for j in range(h + 2, len(lines)):
        line = lines[j].strip()
        if not line.startswith("|"):
            break
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        row_labels.append(cells[0])
        values.append([float(c) for c in cells[1:]])
    return headers[1:], row_labels, np.array(values)


def plot_bars(values: np.ndarray, row_labels: list[str], col_labels: list[str],
              title: str, ylabel: str, out_path: Path,
              *, hlines=((0.0, "black", "-", 0.5),)):
    n_rows = len(row_labels)
    n_cols = len(col_labels)
    width = 0.8 / n_cols
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(n_rows)
    cmap = plt.cm.viridis(np.linspace(0.15, 0.85, n_cols))
    for j, col in enumerate(col_labels):
        offset = j * width - width * (n_cols - 1) / 2
        ax.bar(x + offset, values[:, j], width, label=col,
               color=cmap[j], edgecolor="black", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(row_labels, rotation=20, ha="right", fontsize=10)
    for y, c, ls, alpha in hlines:
        ax.axhline(y, color=c, linestyle=ls, linewidth=0.6, alpha=alpha)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=12, weight="bold")
    ax.legend(loc="best", fontsize=8, ncols=2)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_heatmap(values: np.ndarray, row_labels: list[str], col_labels: list[str],
                 title: str, cbar_label: str, out_path: Path,
                 *, vmin=None, vmax=None, fmt="+.2f", cmap="RdBu_r"):
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    if vmax is None:
        vmax = float(np.abs(values).max())
        vmin = -vmax
    im = ax.imshow(values, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=15, ha="right", fontsize=9)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=10)
    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            v = values[i, j]
            color = "white" if abs(v) > 0.6 * vmax else "black"
            ax.text(j, i, f"{v:{fmt}}", ha="center", va="center",
                    color=color, fontsize=8)
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    cbar.set_label(cbar_label, fontsize=9)
    ax.set_title(title, fontsize=12, weight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"  wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", default="runs/cognitive_v3_full/consistency_report.md")
    ap.add_argument("--output-dir", default="outputs/cognitive_v3_full/analyses_methodC")
    args = ap.parse_args()

    text = Path(args.report).read_text()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # -------- Cross-method --------
    cols, rows, vals = parse_md_table(text, "Cross-method consistency")
    print(f"\n[cross-method] {len(rows)} concepts × {len(cols)} method-pairs")
    print(f"  global mean = {vals.mean():+.3f},  diag of robust pairs (B-D, C-D) typically high")

    plot_bars(
        vals, rows, cols,
        title="Cross-method cosine consistency — 4 extraction methods, layer 30",
        ylabel="cosine similarity (raw)",
        out_path=out_dir / "cross_method_bars.png",
        hlines=((0.0, "black", "-", 0.5), (0.5, "gray", "--", 0.4)),
    )

    # column z-score: for each method-pair, normalize across concepts
    # (factors out "this pair is just easier on average") — reveals per-concept structure
    mu = vals.mean(axis=0, keepdims=True)
    sd = vals.std(axis=0, keepdims=True) + 1e-9
    z = (vals - mu) / sd
    plot_heatmap(
        z, rows, cols,
        title="Cross-method consistency — per-pair z-score (which concepts are unusually robust?)",
        cbar_label="σ within method-pair",
        out_path=out_dir / "cross_method_zscore.png",
        vmin=-2, vmax=2,
    )

    # -------- Cross-layer --------
    cols, rows, vals = parse_md_table(text, "Cross-layer consistency")
    print(f"\n[cross-layer] {len(rows)} concepts × {len(cols)} layer-pairs")
    print(f"  global mean = {vals.mean():+.3f}")

    plot_bars(
        vals, rows, cols,
        title="Cross-layer cosine consistency — Method C, layers 10/20/30/36",
        ylabel="cosine similarity (raw)",
        out_path=out_dir / "cross_layer_bars.png",
        hlines=((0.0, "black", "-", 0.5), (0.5, "gray", "--", 0.4),
                (0.8, "gray", ":", 0.4)),
    )
    mu = vals.mean(axis=0, keepdims=True)
    sd = vals.std(axis=0, keepdims=True) + 1e-9
    z = (vals - mu) / sd
    plot_heatmap(
        z, rows, cols,
        title="Cross-layer consistency — per-pair z-score",
        cbar_label="σ within layer-pair",
        out_path=out_dir / "cross_layer_zscore.png",
        vmin=-2, vmax=2,
    )


if __name__ == "__main__":
    main()
