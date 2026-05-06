"""
Cognitive v3 — Stage-localized concept-vector extraction.

For each story produced by `generate_trajectories_v3.py`:
  1. parse the metadata header + <P1>/<P2>/<P3> blocks
  2. run each paragraph through the model independently
     (paragraph in isolation: avoids late-layer leakage from other stages)
  3. average activations across all tokens of that paragraph

Then aggregate two ways and write per-layer output:

  Mode A (stage-wise concept vectors, 9 per layer):
    v_<concept> = mean( <P_stage> raw acts across stories where stage_concept == concept )
    e.g. v_uncertain = mean of P1 acts from trajectories #9, #10, #11, #16
    e.g. v_confirmed = mean of P3 acts from trajectory #16 only

  Mode B (trajectory vectors, 9 per layer):
    v_traj_<id> = mean( P1+P2+P3 raw acts across stories of this trajectory )

NEG baseline is built per paragraph slot (separate PCA basis for P1, P2, P3) and
projected off the corresponding stage-concept vectors.

Usage:
    python scripts/extract_v3.py \\
        --run-dir runs/cognitive_v3_sanity \\
        --layers 10,20,30,36

Resume: any raw activation file already on disk is reused.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from tqdm.auto import tqdm

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cv_utils import load_model, extract_layer_activations  # noqa: E402
from scripts.v3_validate import parse_blocks  # noqa: E402


def parse_metadata_header(text: str) -> tuple[dict, str]:
    """Files written by generate_trajectories_v3.py start with '# key: value'
    lines, then '---', then the body. Returns (metadata, body)."""
    meta: dict = {}
    body_lines: list[str] = []
    in_body = False
    for line in text.splitlines():
        if not in_body:
            if line.strip() == "---":
                in_body = True
                continue
            if line.startswith("#"):
                k, _, v = line[1:].strip().partition(":")
                meta[k.strip()] = v.strip()
        else:
            body_lines.append(line)
    return meta, "\n".join(body_lines)


def collect_stories(stories_dir: Path) -> list[dict]:
    out = []
    for path in sorted(stories_dir.glob("*.txt")):
        if path.parent != stories_dir:
            continue
        text = path.read_text()
        meta, body = parse_metadata_header(text)
        blocks = parse_blocks(body)
        if blocks is None:
            print(f"  SKIP {path.name}: failed to parse <P1>/<P2>/<P3>")
            continue
        out.append({"path": path, "meta": meta, "blocks": blocks})
    return out


def extract_paragraph_vec(model, text: str, layers: list[int]) -> dict[int, np.ndarray]:
    """Mean-pool activations across all tokens of `text` for each layer.

    Per-paragraph passes are short (<= ~100 tokens), so no token-skip is
    used here (unlike v2's whole-story AVG_FROM_TOKEN=50 heuristic).
    """
    acts = extract_layer_activations(model, text, layers)
    return {L: h.mean(0).numpy() for L, h in acts.items()}


def ensure_raw(model, story: dict, layers: list[int],
               layer_dirs: dict[int, Path]) -> None:
    """Compute and save per-paragraph raw vectors for one story, skipping
    paragraphs already on disk for a given layer."""
    name = story["path"].stem
    is_neg = story["meta"].get("type") == "NEG"
    subdir = "raw_neutral" if is_neg else "raw_concept"
    for tag in ("P1", "P2", "P3"):
        targets = {L: layer_dirs[L] / subdir / f"{name}_{tag}.npy" for L in layers}
        missing = [L for L, p in targets.items() if not p.exists()]
        if not missing:
            continue
        for p in targets.values():
            p.parent.mkdir(parents=True, exist_ok=True)
        vecs = extract_paragraph_vec(model, story["blocks"][tag], missing)
        for L, v in vecs.items():
            np.save(targets[L], v)


def load_raws_grouped(layer_dir: Path, subdir: str) -> dict[str, np.ndarray]:
    """Load all .npy files in <layer_dir>/<subdir>/ and key by stem."""
    out = {}
    files_dir = layer_dir / subdir
    if not files_dir.exists():
        return out
    for f in sorted(files_dir.glob("*.npy")):
        out[f.stem] = np.load(f)
    return out


def pca_variance_basis(X: np.ndarray, variance_fraction: float = 0.5) -> np.ndarray:
    if len(X) < 2:
        return np.zeros((X.shape[1], 0))
    pca = PCA().fit(X)
    cum = np.cumsum(pca.explained_variance_ratio_)
    k = int(np.searchsorted(cum, variance_fraction) + 1)
    return pca.components_[:k].T


def project_off(v: np.ndarray, basis: np.ndarray) -> np.ndarray:
    if basis.shape[1] == 0:
        return v
    return v - basis @ (basis.T @ v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True,
                    help="Directory produced by generate_trajectories_v3.py "
                         "(contains stories/ and trajectories.json)")
    ap.add_argument("--layers", default="10,20,30,36")
    ap.add_argument("--model-path", default=None)
    args = ap.parse_args()

    root = Path(args.run_dir)
    stories_dir = root / "stories"
    layers = [int(x) for x in args.layers.split(",")]
    layer_dirs = {L: root / f"layer_{L}" for L in layers}
    for d in layer_dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    traj_cfg = json.loads((root / "trajectories.json").read_text())
    trajectories = {t["id"]: t for t in traj_cfg["trajectories"]}

    print(f"Reading stories from {stories_dir}")
    stories = collect_stories(stories_dir)
    n_pos = sum(1 for s in stories if s["meta"].get("type") == "POS")
    n_neg = sum(1 for s in stories if s["meta"].get("type") == "NEG")
    print(f"Loaded {len(stories)} stories: {n_pos} POS, {n_neg} NEG")

    print("Loading model...")
    model = load_model(args.model_path)
    print("Model loaded.")

    # =====================================================================
    # Phase 1: per-paragraph raw activations
    # =====================================================================
    pbar = tqdm(stories, desc="extract", unit="story")
    for s in pbar:
        pbar.set_postfix_str(s["path"].stem)
        ensure_raw(model, s, layers, layer_dirs)

    # =====================================================================
    # Phase 2: aggregate per layer
    # =====================================================================
    summary = {}
    for L in layers:
        ldir = layer_dirs[L]
        raw_concept = load_raws_grouped(ldir, "raw_concept")
        raw_neutral = load_raws_grouped(ldir, "raw_neutral")

        # ---- Mode A: stage-wise concept vectors ----
        # Group POS paragraphs by (stage, concept).
        # Story name is e.g. "POS-01-0-0_P1" -> traj_id 01, paragraph P1.
        # For Mode A, we look up the trajectory's prior/discovery/reaction
        # concept based on the paragraph.
        by_stage_concept: dict[str, list[np.ndarray]] = defaultdict(list)
        for stem, v in raw_concept.items():
            # stem like "POS-01-0-0_P1"
            head, _, tag = stem.rpartition("_")  # "POS-01-0-0", "_", "P1"
            parts = head.split("-")  # ["POS", "01", "0", "0"]
            traj_id = int(parts[1])
            traj = trajectories[traj_id]
            stage_concept_key = {"P1": traj["prior"],
                                 "P2": traj["discovery"],
                                 "P3": traj["reaction"]}[tag]
            by_stage_concept[stage_concept_key].append(v)

        per_concept = {c: np.stack(vs).mean(0) for c, vs in by_stage_concept.items()}

        # ---- Mode B: trajectory vectors ----
        # Average all 3 paragraphs across all stories of one trajectory.
        by_traj: dict[int, list[np.ndarray]] = defaultdict(list)
        for stem, v in raw_concept.items():
            head, _, tag = stem.rpartition("_")
            parts = head.split("-")
            traj_id = int(parts[1])
            by_traj[traj_id].append(v)
        per_traj = {f"traj_{tid:02d}_{trajectories[tid]['name']}": np.stack(vs).mean(0)
                    for tid, vs in by_traj.items()}

        # ---- Mode C: NEG bases per paragraph slot ----
        neg_by_tag: dict[str, list[np.ndarray]] = defaultdict(list)
        for stem, v in raw_neutral.items():
            _, _, tag = stem.rpartition("_")
            neg_by_tag[tag].append(v)

        bases: dict[str, np.ndarray] = {}
        for tag in ("P1", "P2", "P3"):
            arr = np.stack(neg_by_tag.get(tag, []))
            bases[tag] = pca_variance_basis(arr, 0.5) if len(arr) >= 2 else np.zeros((arr.shape[1] if len(arr) else 0, 0))

        # ---- Mean over all concept vectors (centering term) ----
        mean_all = (np.stack(list(per_concept.values())).mean(0)
                    if per_concept else np.zeros(next(iter(raw_concept.values())).shape))

        # ---- Stage-matched basis subtraction for Mode A ----
        # Each concept vector belongs to a stage; subtract that stage's NEG basis.
        concept_to_stage = {}
        for tid, traj in trajectories.items():
            concept_to_stage[traj["prior"]] = "P1"
            concept_to_stage[traj["discovery"]] = "P2"
            concept_to_stage[traj["reaction"]] = "P3"

        final_modeA = {}
        for c, v in per_concept.items():
            stage_tag = concept_to_stage[c]
            centered = v - mean_all
            final_modeA[c] = project_off(centered, bases[stage_tag])

        # ---- For Mode B, average all NEG paragraph bases ----
        all_neg = []
        for vs in neg_by_tag.values():
            all_neg.extend(vs)
        if len(all_neg) >= 2:
            neg_basis_all = pca_variance_basis(np.stack(all_neg), 0.5)
        else:
            neg_basis_all = np.zeros((mean_all.shape[0], 0))

        mean_traj = (np.stack(list(per_traj.values())).mean(0)
                     if per_traj else mean_all.copy())

        final_modeB = {}
        for name, v in per_traj.items():
            centered = v - mean_traj
            final_modeB[name] = project_off(centered, neg_basis_all)

        # ---- Save ----
        np.savez(ldir / "concept_vectors_modeA.npz", **final_modeA)
        np.savez(ldir / "trajectory_vectors_modeB.npz", **final_modeB)
        np.save(ldir / "mean.npy", mean_all)
        for tag in ("P1", "P2", "P3"):
            if bases[tag].shape[1] > 0:
                np.save(ldir / f"neutral_projection_{tag}.npy", bases[tag])
        if neg_basis_all.shape[1] > 0:
            np.save(ldir / "neutral_projection_all.npy", neg_basis_all)

        summary[L] = {
            "n_concept_vectors": len(final_modeA),
            "n_trajectory_vectors": len(final_modeB),
            "neg_basis_k": {tag: bases[tag].shape[1] for tag in ("P1", "P2", "P3")},
            "samples_per_concept": {c: len(vs) for c, vs in by_stage_concept.items()},
            "samples_per_trajectory": {f"traj_{tid:02d}": len(vs) for tid, vs in by_traj.items()},
        }
        n_a = len(final_modeA)
        n_b = len(final_modeB)
        n_neg = sum(len(neg_by_tag.get(t, [])) for t in ("P1", "P2", "P3"))
        print(f"[layer {L}] Mode A: {n_a} concepts, Mode B: {n_b} trajectories, "
              f"NEG paragraphs: {n_neg}")

    (root / "extraction_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Wrote extraction_summary.json")


if __name__ == "__main__":
    main()
