"""
Cognitive v3 — robustness analyses on existing extracted vectors (no GPU needed).

Computes:
  B. Cross-method consistency  — how similar is each concept vector across the
                                  4 extraction methods (A/B/C/D)?
  D. Cross-layer consistency   — how similar is each concept across layers 10/20/30/36
                                  for the chosen method (default Method C)?

Outputs a concise table + a markdown summary.

Usage:
    python scripts/analyze_consistency_v3.py --extractions-dir runs/cognitive_v3_full/extractions
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


CONCEPTS = ["curious", "uncertain", "confident", "surprised", "bored",
            "stubborn", "enlightened", "confused", "confirmed"]
METHODS = ["methodA_v2style", "methodB_isolation", "methodC_incontext", "methodD_contrast"]
METHOD_LABEL = {"methodA_v2style": "A", "methodB_isolation": "B",
                "methodC_incontext": "C", "methodD_contrast": "D"}
LAYERS = [10, 20, 30, 36]


def load_vec(extractions_dir: Path, method: str, layer: int) -> dict:
    p = extractions_dir / method / f"layer_{layer}" / "concept_vectors_modeA.npz"
    return dict(np.load(p))


def cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def cross_method(extractions_dir: Path, layer: int):
    print(f"### B. Cross-method consistency (layer {layer}) ###\n")
    print("Cosine similarity between methods, per concept.")
    print("  ~1.0 = same direction; 0 = independent; <0 = opposite.\n")

    data = {m: load_vec(extractions_dir, m, layer) for m in METHODS}
    pairs = [(METHODS[i], METHODS[j]) for i in range(len(METHODS))
             for j in range(i + 1, len(METHODS))]

    headers = ["concept"] + [f"{METHOD_LABEL[m1]} vs {METHOD_LABEL[m2]}"
                              for m1, m2 in pairs]
    widths = [12] + [9] * len(pairs)
    fmt = "  ".join(f"{{:>{w}}}" for w in widths)
    print(fmt.format(*headers))
    print("-" * (sum(widths) + 2 * len(widths)))

    rows = {}
    for c in CONCEPTS:
        if c not in data[METHODS[0]]:
            continue
        cosines = [cos(data[m1][c], data[m2][c]) for m1, m2 in pairs]
        rows[c] = cosines
        print(fmt.format(c, *[f"{x:+.3f}" for x in cosines]))

    # Summary stats per pair
    print("\nMean across concepts:")
    means = []
    for k, (m1, m2) in enumerate(pairs):
        m = np.mean([rows[c][k] for c in rows])
        means.append(m)
        print(f"  {METHOD_LABEL[m1]} vs {METHOD_LABEL[m2]}:  {m:+.3f}")
    return rows


def cross_layer(extractions_dir: Path, method: str = "methodC_incontext"):
    print(f"\n### D. Cross-layer consistency ({method}) ###\n")
    print("Cosine similarity between layers, per concept.")
    print("  ~1.0 = same direction across depth; ~0 = layer-specific.\n")

    data = {L: load_vec(extractions_dir, method, L) for L in LAYERS}
    pairs = [(LAYERS[i], LAYERS[j]) for i in range(len(LAYERS))
             for j in range(i + 1, len(LAYERS))]

    headers = ["concept"] + [f"L{L1}-L{L2}" for L1, L2 in pairs]
    widths = [12] + [9] * len(pairs)
    fmt = "  ".join(f"{{:>{w}}}" for w in widths)
    print(fmt.format(*headers))
    print("-" * (sum(widths) + 2 * len(widths)))

    rows = {}
    for c in CONCEPTS:
        if c not in data[LAYERS[0]]:
            continue
        cosines = [cos(data[L1][c], data[L2][c]) for L1, L2 in pairs]
        rows[c] = cosines
        print(fmt.format(c, *[f"{x:+.3f}" for x in cosines]))

    print("\nMean across concepts:")
    for k, (L1, L2) in enumerate(pairs):
        m = np.mean([rows[c][k] for c in rows])
        print(f"  L{L1}-L{L2}:  {m:+.3f}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extractions-dir", required=True,
                    help="e.g. runs/cognitive_v3_full/extractions")
    ap.add_argument("--layer", type=int, default=30,
                    help="layer for cross-method analysis")
    ap.add_argument("--method-for-layer-scan", default="methodC_incontext")
    args = ap.parse_args()

    extractions_dir = Path(args.extractions_dir)
    cm = cross_method(extractions_dir, args.layer)
    cl = cross_layer(extractions_dir, args.method_for_layer_scan)

    # Write a markdown report alongside
    report = ["# Consistency analysis report\n"]
    report.append(f"Source: `{extractions_dir}`\n")
    report.append(f"Cross-method layer: {args.layer}\n")
    report.append(f"Cross-layer method: {args.method_for_layer_scan}\n\n")
    report.append("## B. Cross-method consistency\n")
    report.append("Higher cosine = more agreement between methods on what 'curious' (etc.) means.\n\n")
    report.append("Method labels: A = v2-style whole-story, B = paragraph isolation, "
                  "C = paragraph in-context, D = within-stage contrast.\n\n")
    pair_labels_b = [(m1, m2) for i, m1 in enumerate(METHODS) for m2 in METHODS[i+1:]]
    headers = ["concept"] + [f"{METHOD_LABEL[m1]} vs {METHOD_LABEL[m2]}" for m1, m2 in pair_labels_b]
    report.append("| " + " | ".join(headers) + " |")
    report.append("|" + "|".join(["---"] * len(headers)) + "|")
    for c in CONCEPTS:
        if c not in cm:
            continue
        report.append("| " + c + " | " + " | ".join(f"{x:+.3f}" for x in cm[c]) + " |")
    report.append("\n## D. Cross-layer consistency (Method C)\n")
    pair_labels_d = [(L1, L2) for i, L1 in enumerate(LAYERS) for L2 in LAYERS[i+1:]]
    headers = ["concept"] + [f"L{L1}-L{L2}" for L1, L2 in pair_labels_d]
    report.append("| " + " | ".join(headers) + " |")
    report.append("|" + "|".join(["---"] * len(headers)) + "|")
    for c in CONCEPTS:
        if c not in cl:
            continue
        report.append("| " + c + " | " + " | ".join(f"{x:+.3f}" for x in cl[c]) + " |")

    out_path = extractions_dir.parent / "consistency_report.md"
    out_path.write_text("\n".join(report))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
