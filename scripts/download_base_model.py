"""Trigger download + NF4 quantization of config.BASE_MODEL_ID.

Idempotent — does nothing if the local quantized copy already exists.
Useful to run in background so Phase 1 smoke test isn't blocked on the
~17 GB download.

Usage:
    python scripts/download_base_model.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config                              # noqa: E402
from cv_utils import _materialize_local_copy  # noqa: E402


def main():
    target = config.local_model_dir(config.BASE_MODEL_ID)
    if (target / "config.json").exists():
        print(f"Already on disk at {target}")
        return
    print(f"Materializing {config.BASE_MODEL_ID} -> {target}")
    _materialize_local_copy(config.BASE_MODEL_ID)
    print(f"Done. Saved to {target}")


if __name__ == "__main__":
    main()
