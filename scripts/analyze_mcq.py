"""
Analyze MCQ raw projections to find which concept vector(s) discriminate
between correct vs incorrect answers most strongly.

Reads `outputs/cognitive_v3_mcq/raw_projections.json` and produces:

  strip_plot_per_concept_<scope>.png   one panel per concept, dots colored by
                                       correct/incorrect, x = projection
  cohen_d_summary.png                  bar chart of Cohen's d per concept
                                       (effect size of correct vs incorrect)
  summary.json                         numeric summary

Two scopes: "last" (last token of answer) and "pool" (last 8 tokens averaged).
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


CONCEPTS_ORDER = ["surprised", "confused", "uncertain", "stubborn", "enlightened",
                  "confirmed", "curious", "confident", "bored"]


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d effect size: (mean_a - mean_b) / pooled_std."""
    if len(a) < 2 or len(b) < 2:
        return 0.0
    n_a, n_b = len(a), len(b)
    var_a, var_b = a.var(ddof=1), b.var(ddof=1)
    pooled = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
    if pooled < 1e-9:
        return 0.0
    return float((a.mean() - b.mean()) / pooled)


def strip_plot_per_concept(data: dict, scope: str, out_path: Path):
    """One subplot per concept; x = projection; jittered y; color by correct."""
    concepts = [c for c in CONCEPTS_ORDER if c in data]
    n = len(concepts)
    n_cols = 3
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 2.6 * n_rows))
    axes = np.atleast_2d(axes).reshape(n_rows, n_cols)

    rng = np.random.default_rng(42)
    for k, c in enumerate(concepts):
        r, col = divmod(k, n_cols)
        ax = axes[r, col]
        cor = np.array(data[c]["correct"])
        inc = np.array(data[c]["incorrect"])
        d = cohens_d(cor, inc)

        ax.scatter(cor, rng.uniform(0.55, 0.95, size=len(cor)),
                   color="#27ae60", s=20, alpha=0.65, label=f"correct (n={len(cor)})")
        ax.scatter(inc, rng.uniform(0.05, 0.45, size=len(inc)),
                   color="#c0392b", s=20, alpha=0.45, label=f"incorrect (n={len(inc)})")
        ax.axvline(cor.mean(), color="#27ae60", linestyle="--", alpha=0.7, linewidth=1)
        ax.axvline(inc.mean(), color="#c0392b", linestyle="--", alpha=0.7, linewidth=1)
        ax.set_xlim(-0.2, 0.6)
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_title(f"{c}  (Cohen's d = {d:+.2f})", fontsize=10,
                     weight="bold" if abs(d) > 0.3 else "normal")
        ax.set_xlabel("cosine projection", fontsize=9)
        ax.legend(fontsize=7, loc="upper right" if d <= 0 else "upper left",
                  framealpha=0.85)
        ax.grid(axis="x", alpha=0.2)

    # Hide unused panels
    for k in range(n, n_rows * n_cols):
        r, col = divmod(k, n_cols)
        axes[r, col].axis("off")

    fig.suptitle(f"MCQ: cognitive vector activation on correct vs wrong answer  "
                 f"(scope={scope})", fontsize=12, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"  wrote {out_path}")


def cohen_d_bar(data: dict, scope: str, out_path: Path) -> dict:
    """Bar chart of Cohen's d per concept; sorted desc by |d|."""
    concepts = [c for c in CONCEPTS_ORDER if c in data]
    ds = []
    means_c = []
    means_i = []
    for c in concepts:
        cor = np.array(data[c]["correct"])
        inc = np.array(data[c]["incorrect"])
        d = cohens_d(cor, inc)
        ds.append(d)
        means_c.append(float(cor.mean()))
        means_i.append(float(inc.mean()))

    order = sorted(range(len(concepts)), key=lambda i: -abs(ds[i]))
    sorted_concepts = [concepts[i] for i in order]
    sorted_ds = [ds[i] for i in order]

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    colors = ["#27ae60" if d > 0 else "#c0392b" for d in sorted_ds]
    bars = ax.bar(np.arange(len(sorted_concepts)), sorted_ds,
                  color=colors, edgecolor="black", linewidth=0.4)
    ax.set_xticks(np.arange(len(sorted_concepts)))
    ax.set_xticklabels(sorted_concepts, rotation=20, ha="right", fontsize=10)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.axhline(0.2, color="gray", linestyle=":", alpha=0.5)
    ax.axhline(-0.2, color="gray", linestyle=":", alpha=0.5)
    ax.text(len(sorted_concepts) - 0.5, 0.21, "small effect (0.2)",
            color="gray", fontsize=8, ha="right")
    ax.text(len(sorted_concepts) - 0.5, 0.51, "medium effect (0.5)",
            color="gray", fontsize=8, ha="right")
    ax.axhline(0.5, color="gray", linestyle=":", alpha=0.5)
    ax.axhline(-0.5, color="gray", linestyle=":", alpha=0.5)
    for i, (b, d) in enumerate(zip(bars, sorted_ds)):
        ax.text(i, d + (0.02 if d >= 0 else -0.04), f"{d:+.2f}",
                ha="center", va="bottom" if d >= 0 else "top", fontsize=9)
    ax.set_ylabel("Cohen's d  (positive = vector higher on correct answers)")
    ax.set_title(f"MCQ: which concept discriminates correct vs wrong?  (scope={scope})",
                 fontsize=12, weight="bold")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"  wrote {out_path}")

    return {c: {"d": ds[i], "mean_correct": means_c[i], "mean_incorrect": means_i[i]}
            for i, c in enumerate(concepts)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="outputs/cognitive_v3_mcq/raw_projections.json")
    ap.add_argument("--output-dir", default="outputs/cognitive_v3_mcq")
    args = ap.parse_args()

    raw = json.loads(Path(args.input).read_text())
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary: dict = {"n_records": len(raw),
                     "n_correct": sum(1 for r in raw if r["is_correct"]),
                     "n_incorrect": sum(1 for r in raw if not r["is_correct"])}
    print(f"loaded {summary['n_records']} records "
          f"({summary['n_correct']} correct, {summary['n_incorrect']} incorrect)")

    for scope_key in ("proj_last", "proj_pool"):
        scope_label = "last" if scope_key == "proj_last" else "pool"
        # group: data[concept] = {"correct": [...], "incorrect": [...]}
        data: dict = defaultdict(lambda: {"correct": [], "incorrect": []})
        for r in raw:
            for c, v in r[scope_key].items():
                key = "correct" if r["is_correct"] else "incorrect"
                data[c][key].append(v)

        strip_plot_per_concept(data, scope_label,
                               out_dir / f"strip_plot_per_concept_{scope_label}.png")
        s = cohen_d_bar(data, scope_label,
                        out_dir / f"cohen_d_summary_{scope_label}.png")
        summary[scope_label] = s

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"  wrote {out_dir / 'summary.json'}")

    # Stdout: top 3 winners per scope
    print("\n=== top discriminator concepts ===")
    for scope in ("last", "pool"):
        ranked = sorted(summary[scope].items(), key=lambda x: -abs(x[1]["d"]))
        print(f"  scope={scope}:")
        for c, s in ranked[:5]:
            print(f"    {c:<12}: d={s['d']:+.2f}  "
                  f"(correct={s['mean_correct']:+.3f}, incorrect={s['mean_incorrect']:+.3f})")


if __name__ == "__main__":
    main()
