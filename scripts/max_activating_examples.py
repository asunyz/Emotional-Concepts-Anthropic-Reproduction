"""For a list of feature ids, find the top-K stories where each feature
activates most, and emit token-level highlighted HTML for inspection.

Usage:
    python scripts/max_activating_examples.py \\
        --sae-acts-dir runs/emotions_qwen35_BASE \\
        --stories-dir runs/emotions_qwen35_nf4/stories \\
        --layer 30 --features 12345,67890 --top-k 5 \\
        --hook-point post_block \\
        --out outputs/sae_surprise/examples_layer_30.html
"""
from __future__ import annotations

import argparse
import html
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config                                                 # noqa: E402
from cv_utils import load_model, extract_per_token_residuals  # noqa: E402
from scripts.sae_loader import load_sae, encode_topk          # noqa: E402


def render_tokens(tokens: list[str], scores: np.ndarray, color: str) -> str:
    if scores.max() <= 0:
        norm = np.zeros_like(scores)
    else:
        norm = np.clip(scores / scores.max(), 0, 1)
    parts = []
    for tok, s in zip(tokens, norm):
        alpha = float(s)
        bg = f"background:rgba({color}, {alpha:.2f})"
        parts.append(f'<span style="{bg}">{html.escape(tok)}</span>')
    return "".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sae-acts-dir", required=True, type=Path)
    ap.add_argument("--stories-dir", required=True, type=Path)
    ap.add_argument("--layer", required=True, type=int)
    ap.add_argument("--features", required=True,
                    help="Comma-separated SAE feature ids")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--hook-point", default=None)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--color", default="255,140,0",
                    help="RGB triple used for the per-token highlight")
    args = ap.parse_args()

    hook = args.hook_point or config.SAE_HOOK_POINT
    if hook == "auto":
        raise SystemExit("config.SAE_HOOK_POINT is still 'auto'.")

    feature_ids = [int(x) for x in args.features.split(",")]

    # Find each feature's top-K stories from the agg_max table.
    layer_dir = args.sae_acts_dir / f"layer_{args.layer}"
    with np.load(layer_dir / "agg_max.npz") as d:
        names = sorted(d.files)
        mat = np.stack([d[k] for k in names]).astype(np.float32)  # [n_stories, d_sae]

    print(f"Loading BASE model: {config.BASE_MODEL_ID}")
    model = load_model(model_id=config.BASE_MODEL_ID)
    sae = load_sae(args.layer)
    tok = model.tokenizer

    args.out.parent.mkdir(parents=True, exist_ok=True)
    sections = []
    for fid in feature_ids:
        col = mat[:, fid]
        top_idx = np.argsort(-col)[:args.top_k]
        section = [f"<h2>Feature {fid}</h2>"]
        for rank, i in enumerate(top_idx, 1):
            stem = names[i]
            text_path = args.stories_dir / f"{stem}.txt"
            if not text_path.exists():
                section.append(f"<p>missing: {stem}</p>")
                continue
            text = text_path.read_text()
            res = extract_per_token_residuals(model, text, [args.layer],
                                              hook_point=hook)
            x = res[args.layer].to(torch.float16).cuda()
            acts = encode_topk(x, sae)                         # [seq, d_sae]
            scores = acts[:, fid].to(torch.float32).cpu().numpy()
            ids = tok(text).input_ids
            tokens = [tok.decode([t]) for t in ids][:scores.shape[0]]
            scores = scores[:len(tokens)]
            section.append(
                f"<h3>#{rank} — {stem} (max={float(col[i]):.3f})</h3>"
                f"<p style='font-family:monospace;line-height:1.6;'>"
                f"{render_tokens(tokens, scores, args.color)}</p>"
            )
        sections.append("\n".join(section))

    head = ("<html><head><meta charset='utf-8'>"
            "<title>Max-activating examples</title></head><body>")
    args.out.write_text(head + "\n<hr/>\n".join(sections) + "</body></html>")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
