"""
Cognitive v4 — Dialogue probe geometry analysis (Anthropic Table 14 reproduction).

Reads the centered probe vectors from extract_dialogue_probes.py and produces:

1. Cosine matrices (self×self, other×other, self×other) — Anthropic Figure 17 logic:
   - self×self: high cosine for same concept across speaker positions (sanity)
   - other×other: high for same concept across positions (sanity)
   - self×other: should be ~0 if self and other are distinct subspaces

2. Sanity stats: diagonal vs off-diagonal means, present-vs-other correlation.

3. **Table 14 reproduction**: for each concept c, list the top-K self-vectors
   most similar to other(c). Tells us "when I perceive other in c, my self-state
   is closest to what?"

Output (in --output-dir, default = <run-dir>/extractions_dialogue/layer_<L>/analysis):
    cosines_self_self.json
    cosines_other_other.json
    cosines_self_other.json
    sanity_stats.json
    table14.md
    table14.json

Usage:
    python scripts/analyze_dialogue_geometry.py \\
        --vec-dir runs/cognitive_v4_dialogue_sanity/extractions_dialogue/layer_30
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def cosine_matrix(vecs_a: dict[str, np.ndarray],
                   vecs_b: dict[str, np.ndarray]) -> dict:
    """Returns {row_concept: {col_concept: cos}}."""
    out = {}
    for ra, va in vecs_a.items():
        out[ra] = {rb: cos(va, vb) for rb, vb in vecs_b.items()}
    return out


def matrix_diag_offdiag_means(mat: dict, both_keys: list[str]) -> tuple[float, float]:
    """Mean of diagonal entries vs off-diagonal entries for square matrix.

    `both_keys` should be the intersection of mat's row and col keys."""
    diag = []
    off = []
    for k in both_keys:
        for k2 in both_keys:
            v = mat[k][k2]
            (diag if k == k2 else off).append(v)
    return (
        float(np.mean(diag)) if diag else 0.0,
        float(np.mean(off)) if off else 0.0,
    )


def render_table14(self_vecs: dict[str, np.ndarray],
                   other_vecs: dict[str, np.ndarray],
                   top_k: int = 6) -> tuple[dict, str]:
    """For each other_c, find top_k self concepts most similar to other(c).

    Returns (data_dict, markdown_str)."""
    data = {}
    md_lines = ["## Table 14 (cognitive)\n",
                "Closest **present-speaker** vectors to each **other-speaker** vector.\n",
                f"Showing top {top_k} per other-concept.\n"]
    for other_c, other_v in other_vecs.items():
        sims = sorted(
            ((self_c, cos(other_v, self_v)) for self_c, self_v in self_vecs.items()),
            key=lambda x: -x[1]
        )[:top_k]
        data[other_c] = sims
        md_lines.append(f"\n### Other = {other_c}\n")
        for self_c, c in sims:
            md_lines.append(f"- {self_c}: {c:+.3f}")
    return data, "\n".join(md_lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vec-dir", required=True,
                    help="dir containing concept_vectors_self.npz and concept_vectors_other.npz")
    ap.add_argument("--output-dir", default=None,
                    help="default: <vec-dir>/analysis")
    ap.add_argument("--top-k", type=int, default=6,
                    help="how many present concepts to list per other in Table 14")
    args = ap.parse_args()

    vec_dir = Path(args.vec_dir)
    out_dir = Path(args.output_dir) if args.output_dir else vec_dir / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    self_path = vec_dir / "concept_vectors_self.npz"
    other_path = vec_dir / "concept_vectors_other.npz"
    if not self_path.exists() or not other_path.exists():
        raise SystemExit(f"missing vector files in {vec_dir}")

    self_vecs = {k: v for k, v in np.load(self_path).items()}
    other_vecs = {k: v for k, v in np.load(other_path).items()}

    print(f"loaded {len(self_vecs)} self vectors, {len(other_vecs)} other vectors")
    common = sorted(set(self_vecs) & set(other_vecs))
    print(f"  common concepts: {common}")

    # --- 1. Cosine matrices ---
    cos_ss = cosine_matrix(self_vecs, self_vecs)
    cos_oo = cosine_matrix(other_vecs, other_vecs)
    cos_so = cosine_matrix(self_vecs, other_vecs)

    (out_dir / "cosines_self_self.json").write_text(json.dumps(cos_ss, indent=2))
    (out_dir / "cosines_other_other.json").write_text(json.dumps(cos_oo, indent=2))
    (out_dir / "cosines_self_other.json").write_text(json.dumps(cos_so, indent=2))

    # --- 2. Sanity stats ---
    ss_diag, ss_off = matrix_diag_offdiag_means(cos_ss, common)
    oo_diag, oo_off = matrix_diag_offdiag_means(cos_oo, common)
    so_diag, so_off = matrix_diag_offdiag_means(cos_so, common)

    sanity = {
        "self_self": {
            "diag_mean": ss_diag,
            "offdiag_mean": ss_off,
            "interp": "diag should be 1.0 (each self-vec vs itself)",
        },
        "other_other": {
            "diag_mean": oo_diag,
            "offdiag_mean": oo_off,
            "interp": "diag should be 1.0",
        },
        "self_other": {
            "diag_mean": so_diag,
            "offdiag_mean": so_off,
            "interp": "self×other diag = self(c) vs other(c). High → present and other rep are aligned (NOT what Anthropic found). Low/near-0 → distinct present vs other subspaces (Anthropic finding).",
        },
    }
    (out_dir / "sanity_stats.json").write_text(json.dumps(sanity, indent=2))

    # --- 3. Table 14 ---
    table14_data, table14_md = render_table14(self_vecs, other_vecs, args.top_k)
    (out_dir / "table14.json").write_text(json.dumps(table14_data, indent=2))
    (out_dir / "table14.md").write_text(table14_md)

    # --- Stdout summary ---
    print("\n=== sanity stats ===")
    print(f"  self×self    diag={ss_diag:+.3f}  off-diag={ss_off:+.3f}")
    print(f"  other×other  diag={oo_diag:+.3f}  off-diag={oo_off:+.3f}")
    print(f"  self×other   diag={so_diag:+.3f}  off-diag={so_off:+.3f}")
    print(f"  (Anthropic finding: self×other diag ≈ 0 → distinct subspaces)")

    print("\n=== Table 14 (top-3 per row) ===")
    for other_c, sims in table14_data.items():
        top3 = sims[:3]
        s = "  ".join(f"{c}({v:+.2f})" for c, v in top3)
        print(f"  Other={other_c:<11} → {s}")

    print(f"\nFull outputs in {out_dir}")


if __name__ == "__main__":
    main()
