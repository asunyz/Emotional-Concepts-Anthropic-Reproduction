"""Phase 2: extract SAE feature activations for every story.

For each story under runs/<source>/stories/, forward through BASE_MODEL_ID,
take per-token residuals at each requested layer, encode through the layer's
TopK SAE, and persist per-story aggregates (mean and max over tokens >=
AVG_FROM_TOKEN, with the same `len//2` fallback as extract_concepts.py).

Outputs (per layer):
    <out_dir>/layer_<L>/agg_mean.npz   {story_stem: float16[d_sae]}
    <out_dir>/layer_<L>/agg_max.npz    {story_stem: float16[d_sae]}

Resume rule: stories already aggregated under both files are skipped. Delete
the files to force re-extraction.

Usage:
    python scripts/extract_sae_acts.py \\
        --stories-dir runs/emotions_qwen35_nf4/stories \\
        --layers 30,31,35 \\
        --hook-point post_block \\
        --out-dir runs/emotions_qwen35_BASE
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config                                                 # noqa: E402
from cv_utils import load_model, extract_per_token_residuals  # noqa: E402
from scripts.sae_loader import load_sae, encode_topk          # noqa: E402

AVG_FROM_TOKEN = 50  # match extract_concepts.py


def aggregate(acts: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """acts: [seq, d_sae] dense (mostly zero). Return (mean, max) over the
    chosen token slice as float16 numpy arrays of shape [d_sae].
    """
    seq = acts.shape[0]
    offset = AVG_FROM_TOKEN if seq > AVG_FROM_TOKEN else seq // 2
    if seq - offset < 1:
        offset = 0
    sub = acts[offset:]
    mean = sub.mean(dim=0).to(torch.float16).cpu().numpy()
    mx = sub.max(dim=0).values.to(torch.float16).cpu().numpy()
    return mean, mx


def load_existing(p: Path) -> dict[str, np.ndarray]:
    if not p.exists():
        return {}
    with np.load(p) as d:
        return {k: d[k] for k in d.files}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stories-dir", required=True, type=Path)
    ap.add_argument("--layers", default=",".join(str(L) for L in config.SAE_LAYERS))
    ap.add_argument("--hook-point", default=None,
                    help="post_block / pre_block / mid_block. "
                         "Defaults to config.SAE_HOOK_POINT.")
    ap.add_argument("--out-dir", required=True, type=Path,
                    help="e.g. runs/emotions_qwen35_BASE")
    ap.add_argument("--limit", type=int, default=0,
                    help="If >0, only process this many stories (smoke test).")
    ap.add_argument("--include-glob", default="*.txt",
                    help="Glob inside stories-dir (default: all .txt).")
    args = ap.parse_args()

    layers = [int(x) for x in args.layers.split(",")]
    hook = args.hook_point or config.SAE_HOOK_POINT
    if hook == "auto":
        raise SystemExit("config.SAE_HOOK_POINT is still 'auto' — run "
                         "scripts/smoke_test_hook.py first and update config.py.")

    stories = sorted(p for p in args.stories_dir.glob(args.include_glob)
                     if p.parent == args.stories_dir)
    if args.limit > 0:
        stories = stories[:args.limit]
    print(f"Found {len(stories)} stories under {args.stories_dir}")

    print(f"Loading BASE model: {config.BASE_MODEL_ID}")
    model = load_model(model_id=config.BASE_MODEL_ID)
    saes = {L: load_sae(L) for L in layers}

    layer_dirs = {L: args.out_dir / f"layer_{L}" for L in layers}
    for d in layer_dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    # Load existing aggregates so we can resume.
    means = {L: load_existing(layer_dirs[L] / "agg_mean.npz") for L in layers}
    maxes = {L: load_existing(layer_dirs[L] / "agg_max.npz") for L in layers}

    pbar = tqdm(stories, desc=f"sae acts (hook={hook})", unit="story")
    n_skipped = 0
    n_done_since_save = 0
    SAVE_EVERY = 100  # checkpoint to disk every N stories

    for path in pbar:
        stem = path.stem
        if all(stem in means[L] and stem in maxes[L] for L in layers):
            n_skipped += 1
            continue
        text = path.read_text()
        try:
            res = extract_per_token_residuals(model, text, layers, hook_point=hook)
        except Exception as e:
            pbar.write(f"  WARN: forward failed on {stem}: {type(e).__name__}: {e}")
            continue
        for L in layers:
            x = res[L].to(torch.float16).cuda()
            acts = encode_topk(x, saes[L])  # [seq, d_sae] dense
            mean_v, max_v = aggregate(acts)
            means[L][stem] = mean_v
            maxes[L][stem] = max_v
            del acts, x
        n_done_since_save += 1
        if n_done_since_save >= SAVE_EVERY:
            for L in layers:
                np.savez(layer_dirs[L] / "agg_mean.npz", **means[L])
                np.savez(layer_dirs[L] / "agg_max.npz",  **maxes[L])
            n_done_since_save = 0
            pbar.set_postfix_str(f"saved (n_skip={n_skipped})")

    # Final flush.
    for L in layers:
        np.savez(layer_dirs[L] / "agg_mean.npz", **means[L])
        np.savez(layer_dirs[L] / "agg_max.npz",  **maxes[L])
    print(f"Done. Skipped {n_skipped} already-cached stories. "
          f"Wrote agg_{{mean,max}}.npz under {args.out_dir}/layer_*/.")


if __name__ == "__main__":
    main()
