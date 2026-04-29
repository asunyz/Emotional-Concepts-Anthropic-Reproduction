"""
Relabel cognitive_v2 stories based on metadata headers, re-aggregate vectors.

Many stories were generated with the wrong concept (model deviated from the
target concept and labeled the actual cognitive state in a metadata header).
This script:

1. Scans each story for `**State:** X` or `**Cognitive State:** X` headers
2. Determines the ACTUAL concept the model wrote about
3. Re-groups raw activations by actual concept
4. Recomputes concept vectors and saves to concept_vectors_relabeled.npz

Original files are not modified. New vectors are written alongside originals.

Usage:
    python scripts/relabel_concepts.py --run-dir runs/cognitive_v2

Output (per layer):
    layer_{L}/concept_vectors_relabeled.npz
    relabeling.json (in run dir): full mapping for audit
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


CONCEPTS = ['curious', 'uncertain', 'confident', 'surprised', 'bored',
            'stubborn', 'enlightened', 'confused', 'confirmed']


def detect_actual_concept(text: str, target: str) -> tuple[str, str]:
    """Detect the actual concept from metadata header. Returns (actual, source)."""
    head = '\n'.join(text.splitlines()[:5])

    # Pattern 1: State Pathway with arrows, e.g., "Confident -> Surprised -> Enlightened"
    arrow_match = re.search(
        r'(?:Cognitive State|State|State Pathway|Path)[:\s]+\*?\*?\s*'
        r'([A-Z][a-z]+(?:\s*(?:Prior|Discovery|Reaction))?)'
        r'\s*[-→>]+\s*([A-Z][a-z]+(?:\s*(?:Prior|Discovery|Reaction))?)'
        r'\s*[-→>]+\s*([A-Z][a-z]+(?:\s*(?:Prior|Discovery|Reaction))?)',
        head
    )
    if arrow_match:
        # Use the LAST element (the reaction / terminal state)
        last = arrow_match.group(3).split()[0].lower()
        if last in CONCEPTS:
            return last, 'arrow_path'

    # Pattern 2: Single state with stage label, e.g., "Confident (Prior-stage)"
    single_match = re.search(
        r'(?:Cognitive State|State)[:\s]+\*?\*?\s*([A-Z][a-z]+)\s*\(',
        head
    )
    if single_match:
        actual = single_match.group(1).lower()
        if actual in CONCEPTS:
            return actual, 'single_stage'

    # Pattern 3: bare State: X
    bare_match = re.search(
        r'(?:Cognitive State|State)[:\s]+\*?\*?\s*([A-Z][a-z]+)\b',
        head
    )
    if bare_match:
        actual = bare_match.group(1).lower()
        if actual in CONCEPTS:
            return actual, 'bare'

    # No header — assume target is correct
    return target, 'no_header'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True,
                    help="e.g. runs/cognitive_v2")
    ap.add_argument("--layers", default="10,20,30,36",
                    help="comma-separated layer indices")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    stories_dir = run_dir / "stories"
    layers = [int(x) for x in args.layers.split(",")]

    # ------------------------------------------------------------------
    # Step 1: build relabeling table
    # ------------------------------------------------------------------
    relabeling = {}
    sources = defaultdict(int)
    mismatches = []

    for f in sorted(stories_dir.glob("*.txt")):
        if f.name.startswith("_"):  # skip neutral
            continue
        name = f.stem  # e.g., confirmed-0-0
        target = name.split("-")[0]

        text = f.read_text(errors="replace")
        actual, source = detect_actual_concept(text, target)
        relabeling[name] = {"target": target, "actual": actual, "source": source}
        sources[source] += 1
        if actual != target:
            mismatches.append((name, target, actual))

    print(f"\n=== Relabeling summary ===")
    print(f"  Total stories: {len(relabeling)}")
    print(f"  By detection source:")
    for src, n in sorted(sources.items()):
        print(f"    {src}: {n}")
    print(f"  Mismatched (target ≠ actual): {len(mismatches)}")

    print(f"\n=== Per-target breakdown ===")
    print(f"  {'target':>12s}  {'kept':>5s}  {'lost':>5s}  {'gained':>7s}  {'final n':>7s}")
    for c in CONCEPTS:
        kept = sum(1 for r in relabeling.values()
                   if r['target'] == c and r['actual'] == c)
        lost = sum(1 for r in relabeling.values()
                   if r['target'] == c and r['actual'] != c)
        gained = sum(1 for r in relabeling.values()
                     if r['target'] != c and r['actual'] == c)
        final_n = kept + gained
        print(f"  {c:>12s}  {kept:>5d}  {lost:>5d}  {gained:>7d}  {final_n:>7d}")

    # Save audit JSON
    audit_path = run_dir / "relabeling.json"
    audit_path.write_text(json.dumps({
        "concepts": CONCEPTS,
        "stories": relabeling,
        "stats": dict(sources),
        "n_mismatch": len(mismatches),
    }, indent=2))
    print(f"\n  Audit saved: {audit_path}")

    # ------------------------------------------------------------------
    # Step 2: per layer, recompute concept vectors using actual labels
    # ------------------------------------------------------------------
    for layer in layers:
        raw_dir = run_dir / f"layer_{layer}/raw_concept"
        if not raw_dir.exists():
            print(f"\n  ⚠ layer_{layer}/raw_concept missing — skipping")
            continue

        # Group raw activations by ACTUAL concept
        by_actual = defaultdict(list)
        for f in raw_dir.glob("*.npy"):
            stem = f.stem  # e.g., confirmed-0-0
            if stem not in relabeling:
                continue
            actual = relabeling[stem]["actual"]
            if actual in CONCEPTS:
                by_actual[actual].append(np.load(f))

        # Compute mean per concept (using actual labels)
        raw_means = {}
        for c in CONCEPTS:
            if c in by_actual and len(by_actual[c]) > 0:
                raw_means[c] = np.stack(by_actual[c]).mean(axis=0)
            else:
                raw_means[c] = None

        # Load neutral basis (already computed)
        basis_path = run_dir / f"layer_{layer}/neutral_projection.npy"
        mean_path = run_dir / f"layer_{layer}/mean.npy"

        if not basis_path.exists():
            print(f"\n  ⚠ layer_{layer} neutral_projection.npy missing")
            continue

        basis = np.load(basis_path)        # [d, k]
        mean_neutral = np.load(mean_path)  # [d]

        # Compute final vectors: subtract mean, project off basis
        final = {}
        for c in CONCEPTS:
            if raw_means[c] is None:
                continue
            v = raw_means[c] - mean_neutral
            v = v - basis @ (basis.T @ v)
            final[c] = v.astype(np.float32)

        # Save
        out_path = run_dir / f"layer_{layer}/concept_vectors_relabeled.npz"
        np.savez(out_path, **final)
        print(f"\n  [layer {layer}] saved {out_path.name}: "
              f"{len(final)}/9 concepts (n per concept: "
              f"{', '.join(f'{c}={len(by_actual[c])}' for c in CONCEPTS if c in by_actual)})")

    print("\n✅ Done. Use concept_vectors_relabeled.npz for downstream analysis.")


if __name__ == "__main__":
    main()
