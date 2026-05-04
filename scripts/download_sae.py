"""Download the Qwen-Scope SAE files for the layers listed in config.SAE_LAYERS.

Pulls only `layer{L}.sae.pt` for L in `config.SAE_LAYERS` (~1 GB each),
not the full 40-layer set. Files land under `config.SAE_LOCAL_DIR`.

Usage:
    python scripts/download_sae.py
    python scripts/download_sae.py --layers 30,31,35   # override config
"""
import argparse
import os
import sys
from pathlib import Path

# Make sibling modules importable when run as `python scripts/download_sae.py`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402

# HF cache & token must be set before importing huggingface_hub.
os.environ.setdefault("HF_HOME", str(config.HF_CACHE))
config.HF_CACHE.mkdir(parents=True, exist_ok=True)
config.SAE_LOCAL_DIR.mkdir(parents=True, exist_ok=True)

from huggingface_hub import snapshot_download  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default=",".join(str(L) for L in config.SAE_LAYERS),
                    help="Comma-separated layer indices to fetch (default: config.SAE_LAYERS)")
    ap.add_argument("--repo", default=config.SAE_REPO)
    ap.add_argument("--out", default=str(config.SAE_LOCAL_DIR))
    args = ap.parse_args()

    layers = [int(x) for x in args.layers.split(",") if x.strip()]
    patterns = [f"layer{L}.sae.pt" for L in layers]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Skip already-downloaded layers (resume rule).
    missing = [L for L in layers if not (out_dir / f"layer{L}.sae.pt").exists()]
    if not missing:
        print(f"All layers {layers} already on disk at {out_dir}")
        return

    print(f"Downloading SAE layers {missing} from {args.repo} -> {out_dir}")
    snapshot_download(
        repo_id=args.repo,
        local_dir=str(out_dir),
        allow_patterns=[f"layer{L}.sae.pt" for L in missing],
    )

    # Quick post-check.
    for L in layers:
        p = out_dir / f"layer{L}.sae.pt"
        if p.exists():
            size_mb = p.stat().st_size / 1e6
            print(f"  layer {L}: {p}  ({size_mb:.0f} MB)")
        else:
            print(f"  layer {L}: MISSING after download — check repo file list")


if __name__ == "__main__":
    main()
