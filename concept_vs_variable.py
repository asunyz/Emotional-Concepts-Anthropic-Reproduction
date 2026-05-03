"""
Sweep a variable inside a prompt and plot each concept's activation strength
at a chosen layer as the variable changes.

The per-token score is either:
  - "projection" (default): (act − mean) · v̂  — signed projection onto the
    unit concept direction, in activation units. Matches label_text.py and
    the rest of the repo; magnitudes scale with ‖act − mean‖ so cross-concept
    comparisons can be biased by residual norm.
  - "cosine": (act − mean) · v̂ / ‖act − mean‖ — true cosine similarity in
    [-1, 1]. Use this when comparing magnitudes across concepts or against
    papers that report cosine (e.g. Anthropic's emotion work).
The script averages the per-token score across all non-BOS tokens.

Usage:
    python concept_vs_variable.py \
        --prompt "I took {x} mg of tylenol." \
        --values 0,250,500,1000,2000 \
        --concept-dir runs/emotions_8b_nf4 \
        --layer 16 \
        --concepts happy,sad \
        --plot line \
        --xlabel "dose (mg)" \
        --output tylenol.png
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from cv_utils import load_model, extract_layer_activations


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", required=True, help="prompt string with a {x} placeholder")
    ap.add_argument("--values", required=True, help="comma-separated values to plug into {x}")
    ap.add_argument("--concept-dir", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--concepts", required=True, help="comma-separated concept names")
    ap.add_argument("--score", choices=["projection", "cosine"], default="cosine",
                    help="projection: (a−μ)·v̂ in activation units; cosine: divides by ‖a−μ‖, bounded [-1,1]")
    ap.add_argument("--plot", choices=["line", "bar"], default="line")
    ap.add_argument("--xlabel", default="x")
    ap.add_argument("--output", default="concept_vs_variable.png")
    args = ap.parse_args()

    values = [v.strip() for v in args.values.split(",")]
    concepts = [c.strip() for c in args.concepts.split(",")]
    ldir = Path(args.concept_dir) / f"layer_{args.layer}"
    mean = np.load(ldir / "mean.npy")
    cvecs = np.load(ldir / "concept_vectors.npz")
    missing = [c for c in concepts if c not in cvecs.files]
    if missing:
        raise KeyError(f"concepts not found {missing}. available: {list(cvecs.files)}")
    cv_units = {c: cvecs[c] / (np.linalg.norm(cvecs[c]) + 1e-9) for c in concepts}

    model = load_model()
    tok = model.tokenizer

    scores = np.zeros((len(values), len(concepts)))
    for i, v in enumerate(values):
        text = args.prompt.format(v)
        h = extract_layer_activations(model, text, [args.layer])[args.layer].numpy()  # [seq, d]
        ids = tok(text, return_tensors="pt").input_ids[0]
        if tok.bos_token_id is not None and int(ids[0]) == tok.bos_token_id:
            h = h[1:]
        H = h - mean  # [seq, d] — centered activations for all non-BOS tokens
        if args.score == "cosine":
            H_norm = np.linalg.norm(H, axis=-1, keepdims=True) + 1e-9
            H_scored = H / H_norm
        else:
            H_scored = H
        for j, c in enumerate(concepts):
            scores[i, j] = (H_scored @ cv_units[c]).mean()
        print(f"{v:>12}  " + "  ".join(f"{c}={scores[i, j]:+.3f}" for j, c in enumerate(concepts)))

    fig, ax = plt.subplots(figsize=(8, 5))
    x_idx = np.arange(len(values))
    if args.plot == "line":
        for j, c in enumerate(concepts):
            ax.plot(x_idx, scores[:, j], marker="o", label=c)
    else:
        width = 0.8 / len(concepts)
        for j, c in enumerate(concepts):
            ax.bar(x_idx + j * width - 0.4 + width / 2, scores[:, j], width, label=c)
    ax.set_xticks(x_idx, values)
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_xlabel(args.xlabel)
    ylabel = "cosine similarity" if args.score == "cosine" else "projection onto concept"
    ax.set_ylabel(f"{ylabel} (layer {args.layer})")
    ax.set_title(args.prompt)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output, dpi=120)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
