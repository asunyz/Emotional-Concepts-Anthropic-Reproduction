"""
Cognitive v3 — Compare 4 extraction methods on the same stories.

Methods (all run in one pass, sharing forward computations where possible):

  Method A — V2-style whole-story mean
    Forward the full story body (incl. markdown headers), mean across all
    tokens, attribute that single vector to every stage-concept of the
    trajectory. Same vector contributes to v_prior, v_discovery, v_reaction
    of one trajectory (intentional contamination, matches how v2-style
    extraction would behave on a 3-stage story).
    Outputs: concept_vectors_modeA.npz + trajectory_vectors_modeB.npz

  Method B — Paragraph isolation (current v3 baseline)
    Forward each paragraph SEPARATELY, mean across its tokens. Each
    paragraph is processed without seeing the other two stages, so
    cognitive context is stripped. This is the current extract_v3.py.
    Outputs: concept_vectors_modeA.npz only.

  Method C — Paragraph in-context (the fix)
    Forward the whole story ONCE; locate the token ranges of each
    paragraph's content (between markdown headers); mean activations only
    within that range. Model sees complete trajectory, so deep-layer
    cognitive integration is preserved.
    Outputs: concept_vectors_modeA.npz + trajectory_vectors_modeB.npz
    (Method A's trajectory vectors are reused here.)

  Method D — In-context + within-stage contrast
    Same per-paragraph activations as Method C, but aggregate concept
    vectors by subtracting the WITHIN-STAGE mean instead of the global
    mean across all 9 concepts. This cancels "I'm a P1 paragraph" /
    "I'm a P3 paragraph" position artifacts and isolates concept-level
    differences within each stage.
    Outputs: concept_vectors_modeA.npz only.

Output layout:
  runs/<task>/extractions/
    methodA_v2style/        layer_10/concept_vectors_modeA.npz, trajectory_vectors_modeB.npz, ...
    methodB_isolation/      layer_10/concept_vectors_modeA.npz, ...
    methodC_incontext/      layer_10/concept_vectors_modeA.npz, trajectory_vectors_modeB.npz, ...
    methodD_contrast/       layer_10/concept_vectors_modeA.npz, ...
    trajectories.json       (copied for plot_v3.py compatibility)

Usage:
    python scripts/extract_v3_compare.py --run-dir runs/cognitive_v3_sanity --layers 10,20,30,36

Resume note: this script always recomputes everything (it's cheap — ~30s
on top of model load).
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cv_utils import load_model, extract_layer_activations  # noqa: E402
from scripts.v3_validate import parse_blocks, STAGE_HEADER_RE, STAGE_TO_TAG  # noqa: E402


# ---------------------------------------------------------------------------
# Story IO
# ---------------------------------------------------------------------------

def parse_metadata_header(text: str) -> tuple[dict, str]:
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
            print(f"  SKIP {path.name}: failed to parse markdown sections")
            continue
        out.append({"path": path, "meta": meta, "body": body, "blocks": blocks})
    return out


# ---------------------------------------------------------------------------
# Token range location for in-context paragraph slicing
# ---------------------------------------------------------------------------

def find_paragraph_char_ranges(text: str) -> dict[str, tuple[int, int]] | None:
    """Return {P1: (char_start, char_end), P2: ..., P3: ...} for content between
    markdown stage headers. Returns None if headers are missing/misordered."""
    matches = list(STAGE_HEADER_RE.finditer(text))
    if len(matches) != 3:
        return None
    if [m.group(1).lower() for m in matches] != ["prior", "discovery", "reaction"]:
        return None
    ranges = {}
    for i, m in enumerate(matches):
        content_start = m.end()
        content_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        ranges[STAGE_TO_TAG[m.group(1).lower()]] = (content_start, content_end)
    return ranges


def char_to_token_idx(text: str, char_pos: int, tokenizer) -> int:
    """Number of tokens in text[:char_pos]. Used as a token index = first token
    AT or AFTER `char_pos`."""
    if char_pos <= 0:
        return 0
    if char_pos >= len(text):
        return len(tokenizer.encode(text, add_special_tokens=False))
    return len(tokenizer.encode(text[:char_pos], add_special_tokens=False))


def paragraph_token_ranges(body: str, tokenizer) -> dict[str, tuple[int, int]] | None:
    char_ranges = find_paragraph_char_ranges(body)
    if char_ranges is None:
        return None
    return {tag: (char_to_token_idx(body, s, tokenizer),
                  char_to_token_idx(body, e, tokenizer))
            for tag, (s, e) in char_ranges.items()}


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def normalize_npz_dict(d: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Cast all values to float32 so np.savez doesn't choke on mixed dtypes."""
    return {k: np.asarray(v, dtype=np.float32) for k, v in d.items()}


def aggregate_method_A(
    raw_full: dict[str, dict[int, np.ndarray]],   # {story_name: {layer: vec}}
    trajectories: dict[int, dict],
    story_metas: dict[str, dict],
) -> tuple[dict[int, dict[str, np.ndarray]], dict[int, dict[str, np.ndarray]]]:
    """Method A: whole-story mean attributed to all 3 stage-concepts.

    Returns (concept_vectors_modeA[L][concept], trajectory_vectors_modeB[L][traj_name]).
    """
    layers = sorted(next(iter(raw_full.values())).keys())
    concept_acc: dict[int, dict[str, list[np.ndarray]]] = {L: defaultdict(list) for L in layers}
    traj_acc: dict[int, dict[str, list[np.ndarray]]] = {L: defaultdict(list) for L in layers}

    for name, layer_vecs in raw_full.items():
        meta = story_metas.get(name, {})
        if meta.get("type") != "POS":
            continue
        traj_id = int(meta.get("trajectory_id", "-1"))
        if traj_id not in trajectories:
            continue
        traj = trajectories[traj_id]
        traj_name = f"traj_{traj_id:02d}_{traj['name']}"
        for L, v in layer_vecs.items():
            # Attribute the SAME whole-story vec to all three stage-concepts
            concept_acc[L][traj["prior"]].append(v)
            concept_acc[L][traj["discovery"]].append(v)
            concept_acc[L][traj["reaction"]].append(v)
            traj_acc[L][traj_name].append(v)

    concept_vecs = {L: {c: np.stack(vs).mean(0) for c, vs in d.items()}
                    for L, d in concept_acc.items()}
    traj_vecs = {L: {n: np.stack(vs).mean(0) for n, vs in d.items()}
                 for L, d in traj_acc.items()}
    return concept_vecs, traj_vecs


def aggregate_per_paragraph(
    raw_para: dict[str, dict[str, dict[int, np.ndarray]]],   # {story_name: {tag: {layer: vec}}}
    trajectories: dict[int, dict],
    story_metas: dict[str, dict],
) -> dict[int, dict[str, np.ndarray]]:
    """Aggregate per-paragraph activations into 9 stage-concept vectors per
    layer. Used by Methods B and C (they differ only in HOW the paragraph
    activations were computed)."""
    any_layers = next(iter(next(iter(raw_para.values())).values())).keys()
    layers = sorted(any_layers)
    concept_acc: dict[int, dict[str, list[np.ndarray]]] = {L: defaultdict(list) for L in layers}

    for name, by_tag in raw_para.items():
        meta = story_metas.get(name, {})
        if meta.get("type") != "POS":
            continue
        traj_id = int(meta.get("trajectory_id", "-1"))
        if traj_id not in trajectories:
            continue
        traj = trajectories[traj_id]
        stage_concept = {"P1": traj["prior"], "P2": traj["discovery"], "P3": traj["reaction"]}
        for tag, by_layer in by_tag.items():
            for L, v in by_layer.items():
                concept_acc[L][stage_concept[tag]].append(v)

    return {L: {c: np.stack(vs).mean(0) for c, vs in d.items()}
            for L, d in concept_acc.items()}


def aggregate_trajectory_modeB(
    raw_para_or_full: dict[str, dict],    # if dict[L]: full; if dict[tag][L]: para
    trajectories: dict[int, dict],
    story_metas: dict[str, dict],
    is_paragraph: bool,
) -> dict[int, dict[str, np.ndarray]]:
    """One vector per trajectory, averaging either whole-story vectors (A/C) or
    all 3 paragraph vectors of that trajectory (alt for B/D)."""
    layers_set = set()
    if is_paragraph:
        for s in raw_para_or_full.values():
            for d in s.values():
                layers_set.update(d.keys())
    else:
        for d in raw_para_or_full.values():
            layers_set.update(d.keys())
    layers = sorted(layers_set)
    traj_acc: dict[int, dict[str, list[np.ndarray]]] = {L: defaultdict(list) for L in layers}

    for name, payload in raw_para_or_full.items():
        meta = story_metas.get(name, {})
        if meta.get("type") != "POS":
            continue
        traj_id = int(meta.get("trajectory_id", "-1"))
        if traj_id not in trajectories:
            continue
        traj = trajectories[traj_id]
        traj_name = f"traj_{traj_id:02d}_{traj['name']}"
        if is_paragraph:
            for tag, by_layer in payload.items():
                for L, v in by_layer.items():
                    traj_acc[L][traj_name].append(v)
        else:
            for L, v in payload.items():
                traj_acc[L][traj_name].append(v)

    return {L: {n: np.stack(vs).mean(0) for n, vs in d.items()}
            for L, d in traj_acc.items()}


# Stage assignment for Method D's within-stage contrast
STAGE_OF_CONCEPT = {
    "curious": "P1", "uncertain": "P1", "confident": "P1",
    "surprised": "P2", "bored": "P2",
    "stubborn": "P3", "enlightened": "P3", "confused": "P3", "confirmed": "P3",
}


def apply_global_centering(
    concept_vecs: dict[int, dict[str, np.ndarray]]
) -> tuple[dict[int, dict[str, np.ndarray]], dict[int, np.ndarray]]:
    """Subtract the mean across ALL 9 concepts from each (Methods A/B/C).

    Returns (centered_vectors, per_layer_mean). The per-layer mean is what
    the v2 analysis scripts (concept_vs_variable, label_text) need to put
    test-sentence activations in the same coordinate system as the centered
    concept directions.
    """
    out = {}
    means = {}
    for L, by_concept in concept_vecs.items():
        names = list(by_concept.keys())
        if not names:
            out[L] = {}
            means[L] = None
            continue
        m = np.stack([by_concept[c] for c in names]).mean(0)
        out[L] = {c: by_concept[c] - m for c in names}
        means[L] = m
    return out, means


def apply_within_stage_centering(
    concept_vecs: dict[int, dict[str, np.ndarray]]
) -> tuple[dict[int, dict[str, np.ndarray]], dict[int, np.ndarray]]:
    """Subtract the mean across concepts WITHIN THE SAME STAGE (Method D).

    For the saved mean.npy we still report the GLOBAL mean across all 9
    concepts — that's what the test-sentence centering needs to align the
    coordinate system. (The within-stage centering is encoded in the
    concept vectors themselves.)
    """
    out = {}
    means = {}
    for L, by_concept in concept_vecs.items():
        out[L] = {}
        # within-stage centering for the concept vectors
        by_stage: dict[str, list[str]] = defaultdict(list)
        for c in by_concept:
            by_stage[STAGE_OF_CONCEPT[c]].append(c)
        for stage, members in by_stage.items():
            stage_mean = np.stack([by_concept[c] for c in members]).mean(0)
            for c in members:
                out[L][c] = by_concept[c] - stage_mean
        # global mean for the saved mean.npy
        names = list(by_concept.keys())
        if names:
            means[L] = np.stack([by_concept[c] for c in names]).mean(0)
        else:
            means[L] = None
    return out, means


def write_layer_outputs(
    out_root: Path,
    concept_vecs: dict[int, dict[str, np.ndarray]] | None,
    traj_vecs: dict[int, dict[str, np.ndarray]] | None,
    means: dict[int, np.ndarray] | None = None,
) -> None:
    layers = sorted((concept_vecs or traj_vecs).keys())
    for L in layers:
        ldir = out_root / f"layer_{L}"
        ldir.mkdir(parents=True, exist_ok=True)
        # mean.npy — used by v2 analysis scripts to center test-sentence acts
        if means is not None and means.get(L) is not None:
            np.save(ldir / "mean.npy", np.asarray(means[L], dtype=np.float32))
        if concept_vecs is not None:
            np.savez(ldir / "concept_vectors_modeA.npz",
                     **normalize_npz_dict(concept_vecs[L]))
            # alias filename so v2 analysis scripts (concept_similarity.py,
            # concept_vs_variable.py, label_text.py, steer.py) work as-is
            np.savez(ldir / "concept_vectors.npz",
                     **normalize_npz_dict(concept_vecs[L]))
        if traj_vecs is not None:
            np.savez(ldir / "trajectory_vectors_modeB.npz",
                     **normalize_npz_dict(traj_vecs[L]))


# ---------------------------------------------------------------------------
# Main extraction
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--layers", default="10,20,30,36")
    ap.add_argument("--model-path", default=None)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    stories_dir = run_dir / "stories"
    layers = [int(x) for x in args.layers.split(",")]

    traj_cfg = json.loads((run_dir / "trajectories.json").read_text())
    trajectories = {t["id"]: t for t in traj_cfg["trajectories"]}

    print(f"Reading stories from {stories_dir}")
    stories = collect_stories(stories_dir)
    print(f"Loaded {len(stories)} stories")

    # Build a name -> meta map for aggregation
    story_metas = {s["path"].stem: s["meta"] for s in stories}

    print("Loading model...")
    model = load_model(args.model_path)
    print("Model loaded.")
    tokenizer = model.tokenizer

    # =====================================================================
    # Phase 1: forward each story (whole + each paragraph)
    # =====================================================================
    raw_full: dict[str, dict[int, np.ndarray]] = {}        # whole-story mean
    raw_full_seq: dict[str, dict[int, np.ndarray]] = {}    # whole-story full [seq, hid] for slicing
    raw_para_isolation: dict[str, dict[str, dict[int, np.ndarray]]] = {}
    raw_para_incontext: dict[str, dict[str, dict[int, np.ndarray]]] = {}

    pbar = tqdm(stories, desc="extract", unit="story")
    for s in pbar:
        name = s["path"].stem
        body = s["body"]
        pbar.set_postfix_str(name)

        # Whole-story forward — used for Method A and Method C
        full_acts = extract_layer_activations(model, body, layers)
        # Method A: mean over all tokens
        raw_full[name] = {L: h.mean(0).numpy() for L, h in full_acts.items()}

        # Method C: slice by paragraph
        ranges = paragraph_token_ranges(body, tokenizer)
        if ranges is not None:
            raw_para_incontext[name] = {}
            for tag, (ts, te) in ranges.items():
                # Some paragraphs near sequence end may collide with seq_len; clamp
                seq_len = next(iter(full_acts.values())).shape[0]
                te = min(te, seq_len)
                if te <= ts:
                    continue
                raw_para_incontext[name][tag] = {
                    L: h[ts:te].mean(0).numpy() for L, h in full_acts.items()
                }
        else:
            raw_para_incontext[name] = {}

        # Method B: forward each paragraph separately
        raw_para_isolation[name] = {}
        for tag, content in s["blocks"].items():
            iso_acts = extract_layer_activations(model, content, layers)
            raw_para_isolation[name][tag] = {L: h.mean(0).numpy() for L, h in iso_acts.items()}

    # =====================================================================
    # Phase 2: aggregate per method, per layer
    # =====================================================================
    extractions_root = run_dir / "extractions"
    extractions_root.mkdir(parents=True, exist_ok=True)
    # Copy trajectories.json so plot_v3.py can find it under each method dir
    for sub in ("methodA_v2style", "methodB_isolation", "methodC_incontext", "methodD_contrast"):
        d = extractions_root / sub
        d.mkdir(parents=True, exist_ok=True)
        shutil.copy(run_dir / "trajectories.json", d / "trajectories.json")

    summary: dict = {}

    # ---- Method A: whole-story mean ----
    A_concept, A_traj = aggregate_method_A(raw_full, trajectories, story_metas)
    A_concept_centered, A_means = apply_global_centering(A_concept)
    write_layer_outputs(extractions_root / "methodA_v2style",
                        A_concept_centered, A_traj, A_means)
    summary["A"] = {L: {"n_concepts": len(A_concept[L]),
                         "n_trajectories": len(A_traj[L])} for L in layers}

    # ---- Method B: paragraph isolation ----
    B_concept = aggregate_per_paragraph(raw_para_isolation, trajectories, story_metas)
    B_concept_centered, B_means = apply_global_centering(B_concept)
    write_layer_outputs(extractions_root / "methodB_isolation",
                        B_concept_centered, None, B_means)
    summary["B"] = {L: {"n_concepts": len(B_concept[L])} for L in layers}

    # ---- Method C: paragraph in-context ----
    C_concept = aggregate_per_paragraph(raw_para_incontext, trajectories, story_metas)
    C_concept_centered, C_means = apply_global_centering(C_concept)
    # C's trajectory vectors: average the 3 in-context paragraph vecs per trajectory
    C_traj = aggregate_trajectory_modeB(raw_para_incontext, trajectories, story_metas, is_paragraph=True)
    write_layer_outputs(extractions_root / "methodC_incontext",
                        C_concept_centered, C_traj, C_means)
    summary["C"] = {L: {"n_concepts": len(C_concept[L]),
                         "n_trajectories": len(C_traj[L])} for L in layers}

    # ---- Method D: in-context + within-stage contrast ----
    D_concept, D_means = apply_within_stage_centering(C_concept)
    write_layer_outputs(extractions_root / "methodD_contrast",
                        D_concept, None, D_means)
    summary["D"] = {L: {"n_concepts": len(D_concept[L])} for L in layers}

    (extractions_root / "extraction_compare_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    print(f"\nDone. Outputs in {extractions_root}")
    for sub in ("methodA_v2style", "methodB_isolation", "methodC_incontext", "methodD_contrast"):
        print(f"  {extractions_root / sub}")


if __name__ == "__main__":
    main()
