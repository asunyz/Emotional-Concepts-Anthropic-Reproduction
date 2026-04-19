"""
Color-code tokens of a text by their activation intensity along a concept vector.

For each token t and layer L (field-standard: signed projection onto the unit
concept direction; only the concept vector is normalized, not the activation):
    score(t) = ( act[L, t] - mean[L] ) · concept_vec[L] / ‖concept_vec[L]‖
Display rescales per-text by max(score) and clips at 0, so only tokens that
align *positively* with the concept are tinted (negative alignment = no color).
Raw scores remain visible in each token's hover tooltip.

Usage:
    python label_text.py \
        --text "some text or path/to/file.txt" \
        --layers 16,24 \
        --concept-dir runs/emotions_v1 \
        --concepts happy,sad,angry \
        --colors red,blue,orange \
        --output labeled.html

`--concepts` and `--colors` must have the same length (one color per concept).
Omit both to auto-render every concept in the folder using a rotating palette.
"""
import argparse
import html as htmllib
from pathlib import Path

import numpy as np

from cv_utils import load_model, extract_layer_activations

COLOR_MAP = {
    "red":    (220,  50,  50),
    "green":  ( 60, 180,  75),
    "blue":   ( 60, 120, 220),
    "orange": (240, 140,  40),
    "purple": (160,  80, 200),
    "yellow": (240, 200,  40),
}


def unit_projection(H: np.ndarray, v: np.ndarray) -> np.ndarray:
    """H: [seq, d], v: [d] -> [seq] signed projection onto the unit v.

    Only v is normalized (not H): residual-stream magnitudes carry real signal
    that full cosine would cancel out — see RepE (Zou et al.) and activation-
    steering literature.
    """
    return H @ (v / (np.linalg.norm(v) + 1e-9))


def render_html(tokens: list[str], raw: np.ndarray, disp: np.ndarray,
                rgb: tuple[int, int, int], title: str) -> str:
    r, g, b = rgb
    spans = []
    for tok, s_raw, s_disp in zip(tokens, raw, disp):
        alpha = max(0.0, min(1.0, float(s_disp)))
        bg = f"rgba({r},{g},{b},{alpha:.3f})"
        safe = htmllib.escape(tok.replace("\n", "\u21b5\n"))
        spans.append(
            f'<span title="{s_raw:+.3f}" style="background:{bg};padding:0 1px;">{safe}</span>'
        )
    return (f"<h3>{htmllib.escape(title)}</h3>"
            f"<pre style='white-space:pre-wrap;font:14px monospace;'>{''.join(spans)}</pre>")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True, help="string, or path to a .txt file")
    ap.add_argument("--layers", required=True, help="comma-separated layer indices")
    ap.add_argument("--concept-dir", required=True,
                    help="task-label folder produced by extract_concepts.py")
    ap.add_argument("--concepts", default=None,
                    help="comma-separated concept names; omit to render all in the folder")
    ap.add_argument("--colors", default=None,
                    help=f"comma-separated colors, one per concept. "
                         f"choices: {','.join(COLOR_MAP)}. omit to auto-rotate.")
    ap.add_argument("--output", default="labeled.html")
    ap.add_argument("--model-path", default=None)
    args = ap.parse_args()

    text = Path(args.text).read_text() if Path(args.text).exists() else args.text
    layers = [int(x) for x in args.layers.split(",")]
    root = Path(args.concept_dir)

    concept_names = (
        [c.strip() for c in args.concepts.split(",") if c.strip()] if args.concepts else None
    )
    if args.colors:
        color_names = [c.strip() for c in args.colors.split(",") if c.strip()]
        bad = [c for c in color_names if c not in COLOR_MAP]
        if bad:
            raise ValueError(f"unknown color(s) {bad}. choices: {list(COLOR_MAP)}")
        if concept_names is None:
            raise ValueError("--colors requires --concepts")
        if len(color_names) != len(concept_names):
            raise ValueError(f"--concepts ({len(concept_names)}) and --colors "
                             f"({len(color_names)}) must have the same length")
    else:
        color_names = None  # fall back to palette rotation

    model = load_model(args.model_path)
    tok = model.tokenizer
    ids = tok(text, return_tensors="pt").input_ids[0]
    tokens = [tok.decode([i]) for i in ids]

    acts = extract_layer_activations(model, text, layers)  # {L: [seq, d]}

    # Drop the BOS token (<|begin_of_text|>) if the tokenizer prepended one.
    # It isn't part of the visible text, and its residual norm is typically
    # an order of magnitude larger than real tokens, which would dominate the
    # per-text rescaling and squash every real signal to near-zero alpha.
    if tok.bos_token_id is not None and int(ids[0]) == tok.bos_token_id:
        tokens = tokens[1:]
        acts = {L: h[1:] for L, h in acts.items()}

    palette = list(COLOR_MAP)  # used when --colors is omitted

    sections = []
    for L in layers:
        ldir = root / f"layer_{L}"
        mean = np.load(ldir / "mean.npy")
        cvecs = np.load(ldir / "concept_vectors.npz")
        if concept_names is not None:
            names = concept_names
            missing = [c for c in names if c not in cvecs.files]
            if missing:
                raise KeyError(f"layer {L}: concepts not found {missing}. "
                               f"available: {list(cvecs.files)}")
            colors = color_names if color_names is not None \
                else [palette[i % len(palette)] for i in range(len(names))]
        else:
            names = list(cvecs.files)
            colors = [palette[i % len(palette)] for i in range(len(names))]

        H = acts[L].numpy() - mean  # [seq, d]
        for name, color in zip(names, colors):
            cv = cvecs[name]
            raw = unit_projection(H, cv)                           # signed, unbounded
            scale = raw.max() + 1e-9                       # per-text rescale
            disp = np.clip(raw / scale, 0.0, 1.0)                  # positive alignment only
            title = (f"layer {L} · '{name}' ({color})   "
                     f"raw ∈ [{raw.min():+.3f}, {raw.max():+.3f}]  mean={raw.mean():+.3f}")
            sections.append(render_html(tokens, raw, disp, COLOR_MAP[color], title))

    Path(args.output).write_text(
        "<html><body style='background:#fff;color:#111;padding:16px;'>"
        + "\n<hr/>\n".join(sections)
        + "</body></html>"
    )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
