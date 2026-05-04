"""Phase 1 smoke test: detect which residual-stream hook point matches the SAE.

The Qwen-Scope SAE card says "Hook point: Residual stream" without specifying
pre / mid / post-block. We try all three on a small text sample and pick the
one with the lowest normalized reconstruction MSE — that's the hook the SAE
was trained on.

Outputs:
  outputs/sae_surprise/hook_probe_layer{L}.json  — per-hook MSE
  Stdout summary table (also logged to changelog by the operator).

Usage:
  python scripts/smoke_test_hook.py                           # layer 30 default
  python scripts/smoke_test_hook.py --layers 30,31,35
  python scripts/smoke_test_hook.py --text "..."              # custom probe text
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config                                                 # noqa: E402
from cv_utils import load_model, extract_per_token_residuals  # noqa: E402
from scripts.sae_loader import load_sae, reconstruction_mse   # noqa: E402

DEFAULT_TEXT = (
    "The Industrial Revolution was a period of major industrialization and "
    "innovation that took place during the late 18th and early 19th centuries. "
    "It transformed largely agrarian, rural societies in Europe and America "
    "into industrialized, urban ones."
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="30",
                    help="Comma-separated SAE layer indices (default: 30 only — "
                         "one is enough to pick the hook point).")
    ap.add_argument("--text", default=DEFAULT_TEXT)
    ap.add_argument("--out-dir", default="outputs/sae_surprise")
    args = ap.parse_args()

    layers = [int(x) for x in args.layers.split(",")]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading BASE model: {config.BASE_MODEL_ID}")
    model = load_model(model_id=config.BASE_MODEL_ID)

    saes = {L: load_sae(L) for L in layers}
    for L, sae in saes.items():
        print(f"  layer {L} SAE: d_sae={sae['d_sae']}, d_model={sae['d_model']}, "
              f"k={sae['k']}")

    results: dict = {"text": args.text, "layers": {}}
    hooks = ["post_block", "pre_block", "mid_block"]
    for hook in hooks:
        print(f"\n--- hook: {hook} ---")
        try:
            residuals = extract_per_token_residuals(model, args.text, layers,
                                                    hook_point=hook)
        except Exception as e:
            print(f"  FAILED to extract: {type(e).__name__}: {e}")
            for L in layers:
                results["layers"].setdefault(str(L), {})[hook] = {"error": str(e)}
            continue
        for L in layers:
            x = residuals[L].to(torch.float16).cuda()
            stats = reconstruction_mse(x, saes[L])
            print(f"  layer {L}: norm_sq={stats['norm_sq_per_token']:.3f}  "
                  f"mse={stats['mse_per_token']:.3f}  "
                  f"normalized_mse={stats['normalized_mse']:.4f}  "
                  f"n_tokens={stats['n_tokens']}")
            results["layers"].setdefault(str(L), {})[hook] = stats

    # Pick best hook per layer (lowest normalized_mse).
    print("\n=== summary ===")
    best_hooks: dict = {}
    for L in layers:
        per_hook = results["layers"][str(L)]
        ranked = sorted(
            ((h, v.get("normalized_mse", float("inf")))
             for h, v in per_hook.items() if "error" not in v),
            key=lambda kv: kv[1],
        )
        if not ranked:
            print(f"layer {L}: ALL HOOKS FAILED")
            continue
        best, best_mse = ranked[0]
        best_hooks[L] = best
        print(f"layer {L}: best={best} (normalized_mse={best_mse:.4f})  "
              f"others={[(h, round(v,4)) for h,v in ranked[1:]]}")
    results["best_hooks"] = best_hooks

    out_path = out_dir / f"hook_probe_layers_{'_'.join(str(L) for L in layers)}.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
