"""
Run V2's three downstream analyses on v3 concept vectors, sharing one
model session so we only pay the load cost once.

Analyses:

  1. concept_vs_variable — sweep a sentence variable, project all 9 concept
     vectors onto each variant. Replicates v2's var_reading.png test.
     Diagnostic: does each variant fire its expected concept (e.g.,
     "saw the connection" → enlightened; "felt lost" → confused)?

  2. label_text — color each token of a held-out passage by its projection
     onto every concept vector. HTML output. Diagnostic: do tokens of
     the right "stage" actually fire the right concept?

  3. steer — add the concept vector to the residual stream during
     generation, sweep strengths. Diagnostic: does steering with v_uncertain
     produce uncertain-sounding text? This is the strongest causal test.

Uses Method C (in-context paragraph) vectors as the source — picked by the
4-method comparison as the cleanest extraction.

Outputs:
  outputs/cognitive_v3_sanity/v2_analyses/
    var_reading.png       — sentence selectivity
    var_priors.png        — prior-stage variant test
    stained_*.html        — token coloring (per concept)
    steer_*.txt           — steered completions

Usage:
    python scripts/run_v2_analyses_v3.py \\
        --vec-dir runs/cognitive_v3_sanity/extractions/methodC_incontext \\
        --layer 30
"""
from __future__ import annotations

import argparse
import html as htmllib
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from cv_utils import load_model, extract_layer_activations  # noqa: E402


# ---------------------------------------------------------------------------
# Test inputs
# ---------------------------------------------------------------------------

# var_reading-style: each variant should fire one specific concept.
VAR_READING = {
    "prompt": "After reading the paper, I {x}.",
    "values": [
        "saw the connection",         # → enlightened
        "felt lost",                  # → confused
        "couldn't tell what to think", # → uncertain
        "felt sure of my view",       # → confident
        "kept thinking",              # → curious
        "wanted to know more",        # → curious
        "had no reaction",            # → bored
        "realized I was wrong",       # → enlightened
        "refused to update my view",  # → stubborn
        "was right after all",        # → confirmed
    ],
}

# var_priors: testing prior-stage selectivity in a different prompt frame.
VAR_PRIORS = {
    "prompt": "Before opening the data, I {x}.",
    "values": [
        "had no idea what I'd find",      # → curious
        "wasn't sure of my prediction",   # → uncertain
        "knew exactly what I'd see",      # → confident
        "expected the result clearly",    # → confident
        "was open to anything",           # → curious
        "had a guess but doubted it",     # → uncertain
    ],
}

# A held-out passage for token-level staining.
HELD_OUT_PASSAGE = (
    "Dr. Chen pulled the printout from the tray and laid it on the desk. "
    "She had run this assay three times now and each time the band pattern "
    "had been the same. Today she pushed the gel under the imager almost "
    "as a formality, expecting the same result. The image came up dark in "
    "the corner where it should have been bright, and bright where it had "
    "always been faint. She frowned, leaned closer, and ran her finger "
    "along the lane. Something had changed. She pulled up the previous "
    "three runs side by side. The pattern she was seeing now didn't match "
    "any of them — and she had no idea what was going on."
)

# Steering test prompts (kept generic so the steering signal isn't drowned
# by topic content).
STEER_PROMPTS = [
    "I just got the result of the experiment.",
    "I am about to open the file.",
]

# Concepts to actually steer with — picking representatives across stages.
STEER_CONCEPTS = ["confident", "uncertain", "surprised", "stubborn",
                  "enlightened", "confused", "confirmed"]
STEER_STRENGTHS = [-3.0, 0.0, 3.0]


# ---------------------------------------------------------------------------
# Analysis 1: concept_vs_variable
# ---------------------------------------------------------------------------

def run_var_probe(
    model, mean: np.ndarray, cv_units: dict[str, np.ndarray],
    prompt: str, values: list[str], layer: int, out_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    tok = model.tokenizer
    concepts = list(cv_units.keys())
    scores = np.zeros((len(values), len(concepts)))

    for i, v in enumerate(values):
        text = prompt.replace("{x}", v)
        h = extract_layer_activations(model, text, [layer])[layer].numpy()
        ids = tok(text, return_tensors="pt").input_ids[0]
        if tok.bos_token_id is not None and int(ids[0]) == tok.bos_token_id:
            h = h[1:]
        H = h - mean
        for j, c in enumerate(concepts):
            scores[i, j] = (H @ cv_units[c]).mean()
        top = concepts[int(np.argmax(scores[i]))]
        print(f"  '{v[:35]}' -> top: {top} ({scores[i, np.argmax(scores[i])]:+.3f})")

    fig_w = max(11, 1.4 * len(values))
    fig, ax = plt.subplots(figsize=(fig_w, 6))
    x_idx = np.arange(len(values))
    for j, c in enumerate(concepts):
        ax.plot(x_idx, scores[:, j], marker="o", label=c, linewidth=1.5)
    ax.set_xticks(x_idx)
    ax.set_xticklabels(values, rotation=45, ha="right", rotation_mode="anchor", fontsize=9)
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_xlabel("variant")
    ax.set_ylabel(f"projection onto concept (layer {layer})")
    ax.set_title(prompt, fontsize=10)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


# ---------------------------------------------------------------------------
# Analysis 2: label_text (token staining)
# ---------------------------------------------------------------------------

PALETTE = [
    "rgba(31, 119, 180, {a})",   # blue
    "rgba(255, 127, 14, {a})",   # orange
    "rgba(44, 160, 44, {a})",    # green
    "rgba(214, 39, 40, {a})",    # red
    "rgba(148, 103, 189, {a})",  # purple
    "rgba(140, 86, 75, {a})",    # brown
    "rgba(227, 119, 194, {a})",  # pink
    "rgba(127, 127, 127, {a})",  # gray
    "rgba(188, 189, 34, {a})",   # olive
]


def run_staining(
    model, mean: np.ndarray, cv_units: dict[str, np.ndarray],
    text: str, layer: int, out_dir: Path,
) -> None:
    """Per-concept token coloring. One HTML file per concept, intensity = projection."""
    out_dir.mkdir(parents=True, exist_ok=True)
    tok = model.tokenizer

    h = extract_layer_activations(model, text, [layer])[layer].numpy()  # [seq, d]
    ids = tok(text, return_tensors="pt").input_ids[0].tolist()
    if tok.bos_token_id is not None and ids and ids[0] == tok.bos_token_id:
        ids = ids[1:]
        h = h[1:]
    H = h - mean

    # Decode each token to display-friendly text
    decoded = [tok.decode([i]) for i in ids]

    for ci, (cname, cvec) in enumerate(cv_units.items()):
        scores = H @ cvec  # [seq]
        # rescale: clip negative to 0, normalize positive to [0, 1]
        pos = np.clip(scores, 0, None)
        if pos.max() > 0:
            pos = pos / pos.max()

        color_template = PALETTE[ci % len(PALETTE)]
        spans = []
        for tok_text, p, raw in zip(decoded, pos, scores):
            t = htmllib.escape(tok_text).replace("\n", "<br>")
            color = color_template.format(a=f"{p:.3f}")
            spans.append(
                f'<span style="background:{color};" title="{cname}: {raw:+.3f}">{t}</span>'
            )
        body = (
            f"<html><body style='font-family:Georgia,serif; font-size:18px; "
            f"line-height:1.7; padding:24px;'>\n"
            f"<h2>Token staining — concept: <code>{cname}</code> at layer {layer}</h2>\n"
            f"<p>{''.join(spans)}</p>\n"
            f"<p style='color:#666; font-size:13px;'>Raw projection ranges: "
            f"min={scores.min():+.3f}, max={scores.max():+.3f}. "
            f"Hover any token to see its raw projection.</p>\n"
            f"</body></html>"
        )
        out_path = out_dir / f"stained_{cname}.html"
        out_path.write_text(body)
    print(f"  wrote {len(cv_units)} stained HTML files to {out_dir}")


# ---------------------------------------------------------------------------
# Analysis 3: steering
# ---------------------------------------------------------------------------

def run_steer(
    model, cv_units: dict[str, np.ndarray],
    prompts: list[str], concepts: list[str], strengths: list[float],
    layer: int, out_dir: Path,
    max_new_tokens: int = 80,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    tok = model.tokenizer

    target_param = next(p for p in model.parameters() if p.device.type == "cuda")

    for cname in concepts:
        cv_unit = cv_units[cname]
        cv_t = torch.tensor(cv_unit, dtype=target_param.dtype, device=target_param.device)
        for prompt in prompts:
            input_text = tok.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
            prompt_len = tok(input_text, return_tensors="pt").input_ids.shape[1]

            sections = [f"=== PROMPT ===\n{prompt}\n",
                        f"=== CONCEPT: {cname} @ layer {layer} ===\n"]
            for s in strengths:
                delta = s * cv_t
                with model.generate(
                    input_text, max_new_tokens=max_new_tokens,
                    do_sample=True, temperature=0.7, top_p=0.8, top_k=20,
                    repetition_penalty=1.1,
                    pad_token_id=tok.eos_token_id,
                    stop_strings=["<think>", "<|im_start|>"], tokenizer=tok,
                ):
                    model.model.layers[layer].all()
                    h = model.model.layers[layer].output[0]
                    model.model.layers[layer].output[0][:] = h + delta
                    out = model.generator.output.save()
                completion = tok.decode(out[0, prompt_len:].cpu(), skip_special_tokens=True).strip()
                sections.append(f"\n--- strength = {s:+g} ---\n{completion}\n")

            stem = f"steer_{cname}_{prompt[:20].replace(' ', '_').replace('.', '')}"
            out_path = out_dir / f"{stem}.txt"
            out_path.write_text("".join(sections))
            print(f"  wrote {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vec-dir", required=True,
                    help="e.g. runs/cognitive_v3_sanity/extractions/methodC_incontext")
    ap.add_argument("--layer", type=int, default=30)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--skip-steer", action="store_true",
                    help="skip steering analysis (slowest)")
    args = ap.parse_args()

    vec_dir = Path(args.vec_dir)
    ldir = vec_dir / f"layer_{args.layer}"
    cvecs = np.load(ldir / "concept_vectors.npz")
    mean = np.load(ldir / "mean.npy") if (ldir / "mean.npy").exists() else None
    if mean is None:
        v0 = next(iter(cvecs.values()))
        mean = np.zeros(v0.shape[0], dtype=np.float32)

    cv_units = {c: cvecs[c] / (np.linalg.norm(cvecs[c]) + 1e-9)
                for c in cvecs.files}
    print(f"Loaded {len(cv_units)} concept vectors at layer {args.layer}: "
          f"{list(cv_units.keys())}")

    out_root = Path(args.output_dir or
                    f"outputs/{vec_dir.parents[1].name}/v2_analyses_layer{args.layer}")
    out_root.mkdir(parents=True, exist_ok=True)

    print("Loading model...")
    model = load_model(args.model_path)
    print("Model loaded.")

    print("\n[1/3] var probe — VAR_READING")
    run_var_probe(model, mean, cv_units, VAR_READING["prompt"], VAR_READING["values"],
                  args.layer, out_root / "var_reading.png")

    print("\n[1/3 cont] var probe — VAR_PRIORS")
    run_var_probe(model, mean, cv_units, VAR_PRIORS["prompt"], VAR_PRIORS["values"],
                  args.layer, out_root / "var_priors.png")

    print("\n[2/3] token staining")
    run_staining(model, mean, cv_units, HELD_OUT_PASSAGE, args.layer,
                 out_root / "stained")

    if not args.skip_steer:
        print(f"\n[3/3] steering — {len(STEER_CONCEPTS)} concepts x "
              f"{len(STEER_PROMPTS)} prompts x {len(STEER_STRENGTHS)} strengths")
        run_steer(model, cv_units, STEER_PROMPTS, STEER_CONCEPTS, STEER_STRENGTHS,
                  args.layer, out_root / "steer")

    print(f"\nAll outputs in {out_root}")


if __name__ == "__main__":
    main()
