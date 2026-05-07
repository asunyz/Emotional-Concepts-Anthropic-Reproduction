"""
Cognitive v3 — Cross-template winner consistency.

Each var_probe template uses different variant phrases, but probes the same
9 cognitive concepts. The question: is each concept's vector reliably
activated by SOME variant in EVERY template?

For each concept c and each of the 4 templates, compute the **best z-score**
that c achieves across that template's variants (column z-score, normalizing
out each concept's baseline cosine with the probe template).

A concept that scores high across all 4 templates is robustly extractable;
one that scores high in only 1-2 templates is template-specific.

Outputs (next to the *_scores.npz files):
  cross_template_consistency.png    — 9×4 heatmap with z-scores
  cross_template_bars.png           — grouped bar chart
  cross_template_summary.json       — per-concept best-z + mean + range
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


TEMPLATES = ["var_reading", "var_priors", "var_experiment", "var_gift"]
COLORS = ["#3498db", "#2ecc71", "#e74c3c", "#f39c12"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores-dir", required=True,
                    help="dir containing var_*_scores.npz files")
    ap.add_argument("--templates", nargs="+", default=TEMPLATES)
    args = ap.parse_args()

    scores_dir = Path(args.scores_dir)

    # Build {concept: {template: max_z_across_variants}}
    data: dict[str, dict[str, float]] = {}
    all_concepts: list[str] | None = None

    for t in args.templates:
        npz_path = scores_dir / f"{t}_scores.npz"
        if not npz_path.exists():
            print(f"  skip {t}: {npz_path} not found")
            continue
        d = dict(np.load(npz_path, allow_pickle=True))
        S = d["scores"]                          # (n_variants, n_concepts)
        concepts = list(d["concepts"])
        # column z-score: subtract concept-baseline, divide by per-concept std
        mu = S.mean(axis=0, keepdims=True)
        sd = S.std(axis=0, keepdims=True) + 1e-9
        Z = (S - mu) / sd                        # (n_variants, n_concepts)
        best_z = Z.max(axis=0)                   # (n_concepts,)
        if all_concepts is None:
            all_concepts = concepts
        for i, c in enumerate(concepts):
            data.setdefault(c, {})[t] = float(best_z[i])

    if all_concepts is None:
        raise SystemExit("no scores files found")

    templates_used = [t for t in args.templates
                      if any(t in data[c] for c in all_concepts)]

    # ----- Heatmap -----
    fig, ax = plt.subplots(figsize=(8, 5))
    mat = np.array([[data[c].get(t, np.nan) for t in templates_used]
                    for c in all_concepts])
    im = ax.imshow(mat, cmap="Reds", vmin=0, vmax=3.5, aspect="auto")
    ax.set_xticks(range(len(templates_used)))
    ax.set_xticklabels(templates_used, rotation=20, ha="right", fontsize=10)
    ax.set_yticks(range(len(all_concepts)))
    ax.set_yticklabels(all_concepts, fontsize=10)
    for i, c in enumerate(all_concepts):
        for j, t in enumerate(templates_used):
            v = mat[i, j]
            if np.isnan(v):
                continue
            color = "white" if v > 2.0 else "black"
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                    color=color, fontsize=9)
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    cbar.set_label("best z-score across variants (σ vs concept baseline)", fontsize=9)
    ax.set_title("v3 cognitive: each concept's best activation per template",
                 fontsize=12, weight="bold")
    fig.tight_layout()
    out_heatmap = scores_dir / "cross_template_consistency.png"
    fig.savefig(out_heatmap, dpi=140)
    plt.close(fig)
    print(f"  wrote {out_heatmap}")

    # ----- Grouped bar chart -----
    fig, ax = plt.subplots(figsize=(11, 5))
    n_concepts = len(all_concepts)
    n_t = len(templates_used)
    width = 0.8 / max(n_t, 1)
    x = np.arange(n_concepts)
    for j, t in enumerate(templates_used):
        vals = [data[c].get(t, 0.0) for c in all_concepts]
        offset = j * width - width * (n_t - 1) / 2
        ax.bar(x + offset, vals, width, label=t,
               color=COLORS[j % len(COLORS)],
               edgecolor="black", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(all_concepts, rotation=20, ha="right", fontsize=10)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.4)
    ax.text(n_concepts - 0.5, 1.05, "1σ", color="gray", fontsize=8)
    ax.axhline(2.0, color="gray", linestyle=":", alpha=0.4)
    ax.text(n_concepts - 0.5, 2.05, "2σ", color="gray", fontsize=8)
    ax.set_ylabel("best z-score across variants (σ)")
    ax.set_title("v3 cognitive: cross-template concept activation",
                 fontsize=12, weight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    out_bars = scores_dir / "cross_template_bars.png"
    fig.savefig(out_bars, dpi=140)
    plt.close(fig)
    print(f"  wrote {out_bars}")

    # ----- Summary table -----
    summary = {}
    print("\n=== per-concept summary ===")
    header = f"  {'concept':<12} " + "".join(f"{t.replace('var_',''):>10}" for t in templates_used) + f"  {'mean':>7} {'range':>7}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for c in all_concepts:
        zs = [data[c].get(t, 0.0) for t in templates_used]
        mean_z = float(np.mean(zs))
        range_z = float(max(zs) - min(zs))
        summary[c] = {
            "per_template": {t: data[c].get(t, None) for t in templates_used},
            "mean": round(mean_z, 3),
            "range": round(range_z, 3),
            "min_template": templates_used[int(np.argmin(zs))],
            "max_template": templates_used[int(np.argmax(zs))],
        }
        line = f"  {c:<12} " + "".join(f"{z:>+10.2f}" for z in zs) + f"  {mean_z:>+7.2f} {range_z:>7.2f}"
        print(line)

    print("\n=== template-level summary ===")
    print(f"  {'template':<14}  {'mean':>7}  {'#concepts >1σ':>15}  {'#concepts >2σ':>15}")
    for t in templates_used:
        zs = [data[c].get(t, 0.0) for c in all_concepts]
        m = float(np.mean(zs))
        n1 = int(sum(1 for z in zs if z > 1.0))
        n2 = int(sum(1 for z in zs if z > 2.0))
        print(f"  {t:<14}  {m:>+7.2f}  {n1:>15}  {n2:>15}")

    out_json = scores_dir / "cross_template_summary.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"\n  wrote {out_json}")


if __name__ == "__main__":
    main()
