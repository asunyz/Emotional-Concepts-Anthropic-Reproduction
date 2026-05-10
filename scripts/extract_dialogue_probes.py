"""
Cognitive v4 — 2x2 dialogue probe extraction.

For each dialogue with assigned (p1_state, p2_state):
  - Tokenize the full dialogue with per-token speaker labels
  - Forward pass at target layer (default 30)
  - Pool hidden states by speaker → p1_pool, p2_pool (each: (H,))
  - Aggregate into 4 buckets:
        bucket[(p1_state, "self")]   += p1_pool   (P1's state on P1's tokens)
        bucket[(p2_state, "self")]   += p2_pool   (P2's state on P2's tokens)
        bucket[(p2_state, "other")]  += p1_pool   (P2's state observed on P1's tokens)
        bucket[(p1_state, "other")]  += p2_pool   (P1's state observed on P2's tokens)
  - Average each bucket → final vector per (concept, role)

Centering: subtract mean across concepts within each role separately
(self has its own mean, other has its own).

Output:
    runs/<task>/extractions_dialogue/layer_<L>/
        concept_vectors_self.npz   { concept: (H,) float32 }
        concept_vectors_other.npz  { concept: (H,) float32 }
        mean_self.npy              # (H,) — what was subtracted from self vectors
        mean_other.npy             # (H,) — what was subtracted from other vectors
        per_dialogue_pools.npz     # debug: (n_dialogue, 2, H) raw pools

Usage:
    python scripts/extract_dialogue_probes.py \\
        --run-dir runs/cognitive_v4_dialogue_sanity \\
        --layer 30
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cv_utils import load_model  # noqa: E402
from scripts.v4_dialogue_validate import parse_turns  # noqa: E402


def parse_dialogue_file(path: Path) -> tuple[dict, list[dict]]:
    """Return (metadata, turns) from a dialogue file."""
    text = path.read_text()
    metadata: dict = {}
    if "---" in text:
        header, body = text.split("---", 1)
        for line in header.strip().splitlines():
            if line.startswith("# ") and ":" in line:
                k, v = line[2:].split(":", 1)
                metadata[k.strip()] = v.strip()
    else:
        body = text
    turns, _ = parse_turns(body)
    return metadata, turns


def tokenize_with_speaker_mask(tok, turns: list[dict]) -> tuple[torch.Tensor, list[str]]:
    """Tokenize dialogue piece-by-piece with speaker tracking.

    Returns:
        input_ids: (1, T) torch.long
        speaker_mask: list[str] of length T, values in {"BOS", "P1", "P2", "SEP"}
    """
    all_ids: list[int] = []
    all_mask: list[str] = []

    if tok.bos_token_id is not None:
        all_ids.append(int(tok.bos_token_id))
        all_mask.append("BOS")

    sep = "\n\n"
    sep_ids = tok(sep, add_special_tokens=False).input_ids

    for i, t in enumerate(turns):
        speaker_label = "Person 1" if t["speaker"] == "P1" else "Person 2"
        turn_text = f"{speaker_label}: {t['text']}"
        ids = tok(turn_text, add_special_tokens=False).input_ids
        all_ids.extend(ids)
        all_mask.extend([t["speaker"]] * len(ids))
        if i < len(turns) - 1:
            all_ids.extend(sep_ids)
            all_mask.extend(["SEP"] * len(sep_ids))

    input_ids = torch.tensor(all_ids, dtype=torch.long).unsqueeze(0)
    return input_ids, all_mask


def forward_capture_layer(model, hf_model, layer_idx: int,
                           input_ids: torch.Tensor) -> torch.Tensor:
    """Forward `input_ids` through hf_model, capture residual at `layer_idx`.

    Returns hidden states (T, H) on CPU as float32.
    """
    captured: list[torch.Tensor] = []

    def hook(_module, _inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        captured.append(h.detach())

    target = model.model.layers[layer_idx]
    handle = target.register_forward_hook(hook)
    try:
        with torch.no_grad():
            input_ids_dev = input_ids.to(next(model.parameters()).device)
            attn = torch.ones_like(input_ids_dev)
            hf_model(input_ids_dev, attention_mask=attn)
    finally:
        handle.remove()

    h = captured[0]  # (1, T, H)
    if h.ndim == 3:
        h = h[0]
    return h.cpu().float()  # (T, H)


def resolve_hf_model(model):
    """nnsight wraps HF model — locate the inner PreTrainedModel that has .generate()."""
    for attr in ("_model", "_module", "module"):
        cand = getattr(model, attr, None)
        if cand is not None and hasattr(cand, "generate"):
            return cand
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True,
                    help="e.g. runs/cognitive_v4_dialogue_sanity")
    ap.add_argument("--layer", type=int, default=30)
    ap.add_argument("--output-dir", default=None,
                    help="default: <run-dir>/extractions_dialogue/layer_<L>")
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--limit", type=int, default=None,
                    help="(debug) only process first N dialogues")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.output_dir) if args.output_dir else \
        run_dir / "extractions_dialogue" / f"layer_{args.layer}"
    out_dir.mkdir(parents=True, exist_ok=True)

    dlg_files = sorted(p for p in (run_dir / "dialogues").glob("*.txt")
                        if not p.name.startswith("_"))
    if args.limit:
        dlg_files = dlg_files[:args.limit]
    print(f"found {len(dlg_files)} dialogue files in {run_dir}/dialogues/")

    print("loading model ...")
    model = load_model(args.model_path)
    hf_model = resolve_hf_model(model)
    tok = model.tokenizer
    print(f"  model layers: {len(model.model.layers)}")

    # 4 buckets keyed by (concept, role) → list of pooled vectors
    buckets: dict[tuple[str, str], list[np.ndarray]] = defaultdict(list)
    skipped = 0
    per_dialogue: list[dict] = []  # for debug: each dialogue's pools

    for path in tqdm(dlg_files, desc="extract"):
        metadata, turns = parse_dialogue_file(path)
        if not turns or not metadata.get("p1_state") or not metadata.get("p2_state"):
            skipped += 1
            continue
        p1_state = metadata["p1_state"]
        p2_state = metadata["p2_state"]

        input_ids, speaker_mask = tokenize_with_speaker_mask(tok, turns)
        # Forward pass + capture at target layer
        h = forward_capture_layer(model, hf_model, args.layer, input_ids)  # (T, H)
        T = h.shape[0]
        if len(speaker_mask) != T:
            # tokenizer may add some special tokens we didn't account for; fall back
            # to truncating to min length
            T = min(T, len(speaker_mask))
            h = h[:T]
            speaker_mask = speaker_mask[:T]

        mask = np.array(speaker_mask)
        p1_idx = np.where(mask == "P1")[0]
        p2_idx = np.where(mask == "P2")[0]

        if len(p1_idx) == 0 or len(p2_idx) == 0:
            skipped += 1
            continue

        h_np = h.numpy()
        p1_pool = h_np[p1_idx].mean(axis=0)
        p2_pool = h_np[p2_idx].mean(axis=0)

        # 2x2 grid:
        buckets[(p1_state, "self")].append(p1_pool)   # P1's state on P1's tokens
        buckets[(p2_state, "self")].append(p2_pool)   # P2's state on P2's tokens
        buckets[(p2_state, "other")].append(p1_pool)  # P2's state observed on P1's tokens
        buckets[(p1_state, "other")].append(p2_pool)  # P1's state observed on P2's tokens

        per_dialogue.append({
            "stem": path.stem,
            "p1_state": p1_state,
            "p2_state": p2_state,
            "n_tokens": T,
            "n_p1": len(p1_idx),
            "n_p2": len(p2_idx),
        })

    print(f"\nbucket counts:")
    for (c, r), vs in sorted(buckets.items()):
        print(f"  ({c:<11}, {r:<5}): {len(vs):4d} samples")
    print(f"  skipped: {skipped} dialogues (missing metadata or empty turn)")

    # Aggregate per (concept, role)
    self_vecs: dict[str, np.ndarray] = {}
    other_vecs: dict[str, np.ndarray] = {}
    for (c, r), vs in buckets.items():
        avg = np.stack(vs).mean(axis=0)
        if r == "self":
            self_vecs[c] = avg
        else:
            other_vecs[c] = avg

    # Center within each role: subtract mean across concepts
    if self_vecs:
        mean_self = np.stack(list(self_vecs.values())).mean(axis=0)
        self_centered = {c: v - mean_self for c, v in self_vecs.items()}
    else:
        mean_self, self_centered = None, {}
    if other_vecs:
        mean_other = np.stack(list(other_vecs.values())).mean(axis=0)
        other_centered = {c: v - mean_other for c, v in other_vecs.items()}
    else:
        mean_other, other_centered = None, {}

    # Save
    if self_centered:
        np.savez(out_dir / "concept_vectors_self.npz", **self_centered)
        np.save(out_dir / "mean_self.npy", mean_self)
    if other_centered:
        np.savez(out_dir / "concept_vectors_other.npz", **other_centered)
        np.save(out_dir / "mean_other.npy", mean_other)

    (out_dir / "per_dialogue.json").write_text(json.dumps(per_dialogue, indent=2))
    (out_dir / "summary.json").write_text(json.dumps({
        "n_dialogues_processed": len(per_dialogue),
        "n_skipped": skipped,
        "concepts_self": sorted(self_centered.keys()),
        "concepts_other": sorted(other_centered.keys()),
        "layer": args.layer,
        "hidden_dim": int(next(iter(self_centered.values())).shape[0]) if self_centered else None,
    }, indent=2))

    print(f"\nWrote vectors to {out_dir}")
    print(f"  self : {len(self_centered)} concepts")
    print(f"  other: {len(other_centered)} concepts")


if __name__ == "__main__":
    main()
