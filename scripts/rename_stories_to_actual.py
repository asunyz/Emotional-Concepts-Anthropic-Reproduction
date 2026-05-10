"""
Rename cognitive_v2 story files based on metadata-detected actual concept.

Effect: stories whose model-acknowledged actual concept differs from the
filename target are RENAMED so the filename reflects the actual concept.
Indices are reassigned to avoid collision with existing files.

This script is idempotent: it reads the rename plan from
`runs/cognitive_v2/rename_plan.json` (created by --plan mode).

Modes:
    --plan       Build the rename plan, save to JSON. Modifies nothing.
    --apply-txt  Apply renames to .txt files in stories/.
    --apply-npy  Apply renames to .npy files in layer_*/raw_concept/.
    --aggregate  Re-aggregate concept_vectors.npz from current raw_concept/.
    --all        Run all four steps in order.

Typical flow:
    Local:   python scripts/rename_stories_to_actual.py --plan --apply-txt
             git add ... && git push
    RunPod:  git pull
             python scripts/rename_stories_to_actual.py --apply-npy --aggregate
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


CONCEPTS = ['curious', 'uncertain', 'confident', 'surprised', 'bored',
            'stubborn', 'enlightened', 'confused', 'confirmed']

RUN_DIR = Path("runs/cognitive_v2")
STORIES_DIR = RUN_DIR / "stories"
PLAN_PATH = RUN_DIR / "rename_plan.json"
LAYERS = [10, 20, 30, 36]


def detect_actual(text: str, fallback: str) -> str:
    head = '\n'.join(text.splitlines()[:5])

    # arrow path: take last (the reaction-stage terminal state)
    am = re.search(
        r'(?:Cognitive State|State|State Pathway|Path)[:\s]+\*?\*?\s*'
        r'[A-Z][a-z]+(?:\s*\([^)]*\))?\s*[-→>]+\s*'
        r'[A-Z][a-z]+(?:\s*\([^)]*\))?\s*[-→>]+\s*'
        r'([A-Z][a-z]+)',
        head
    )
    if am:
        actual = am.group(1).lower()
        if actual in CONCEPTS:
            return actual

    # single state: e.g. "Cognitive State: Curious (Prior-stage)"
    sm = re.search(
        r'(?:Cognitive State|State)[:\s]+\*?\*?\s*([A-Z][a-z]+)\s*\(', head
    )
    if sm:
        actual = sm.group(1).lower()
        if actual in CONCEPTS:
            return actual

    # bare State: X
    bm = re.search(
        r'(?:Cognitive State|State)[:\s]+\*?\*?\s*([A-Z][a-z]+)\b', head
    )
    if bm:
        actual = bm.group(1).lower()
        if actual in CONCEPTS:
            return actual

    return fallback


def build_plan() -> list[dict]:
    """Scan stories, detect mismatches, assign new indices."""
    mismatches = []
    for f in sorted(STORIES_DIR.glob("*.txt")):
        if f.name.startswith("_"):
            continue
        name = f.stem
        parts = name.split("-")
        if len(parts) != 3:
            continue
        target, topic_idx, story_idx = parts[0], parts[1], parts[2]
        text = f.read_text(errors="replace")
        actual = detect_actual(text, target)
        if actual != target:
            mismatches.append({
                "old": name,
                "target": target,
                "actual": actual,
                "topic_idx": topic_idx,
            })

    # find max idx among non-mismatching files (the "stable" set)
    mismatch_set = {m["old"] for m in mismatches}
    existing_max: dict[tuple[str, str], int] = defaultdict(lambda: -1)
    for f in STORIES_DIR.glob("*.txt"):
        if f.name.startswith("_") or f.stem in mismatch_set:
            continue
        parts = f.stem.split("-")
        if len(parts) == 3:
            c, t, i = parts[0], parts[1], int(parts[2])
            existing_max[(c, t)] = max(existing_max[(c, t)], i)

    # assign new indices, grouped by (actual, topic)
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for m in mismatches:
        groups[(m["actual"], m["topic_idx"])].append(m)

    plan = []
    for (actual, topic_idx), group in sorted(groups.items()):
        base = existing_max[(actual, topic_idx)] + 1
        for i, m in enumerate(group):
            m["new"] = f"{actual}-{topic_idx}-{base + i}"
            plan.append(m)
    return plan


def cmd_plan() -> None:
    plan = build_plan()
    PLAN_PATH.write_text(json.dumps(plan, indent=2))
    print(f"Plan: {len(plan)} renames. Saved to {PLAN_PATH}")
    if plan:
        print("First 5 renames:")
        for m in plan[:5]:
            print(f"  {m['old']:>30s} → {m['new']}")


def cmd_apply_txt() -> None:
    if not PLAN_PATH.exists():
        raise SystemExit(f"No plan found at {PLAN_PATH}. Run --plan first.")
    plan = json.loads(PLAN_PATH.read_text())
    n_renamed = 0
    n_skipped = 0
    for m in plan:
        src = STORIES_DIR / f"{m['old']}.txt"
        dst = STORIES_DIR / f"{m['new']}.txt"
        if not src.exists() and dst.exists():
            n_skipped += 1
            continue
        if not src.exists():
            print(f"  ⚠ source missing: {src.name}")
            continue
        src.rename(dst)
        n_renamed += 1
    print(f"Renamed {n_renamed} .txt files (skipped {n_skipped} already-renamed).")


def cmd_apply_npy() -> None:
    if not PLAN_PATH.exists():
        raise SystemExit(f"No plan found at {PLAN_PATH}. Run --plan first.")
    plan = json.loads(PLAN_PATH.read_text())
    total = 0
    for layer in LAYERS:
        raw_dir = RUN_DIR / f"layer_{layer}/raw_concept"
        if not raw_dir.exists():
            print(f"  ⚠ {raw_dir} missing — skipping layer {layer}")
            continue
        n = 0
        for m in plan:
            src = raw_dir / f"{m['old']}.npy"
            dst = raw_dir / f"{m['new']}.npy"
            if not src.exists() and dst.exists():
                continue  # already renamed
            if not src.exists():
                print(f"  ⚠ source missing: layer_{layer}/raw_concept/{src.name}")
                continue
            src.rename(dst)
            n += 1
        print(f"  layer {layer}: renamed {n} .npy files")
        total += n
    print(f"Renamed {total} .npy files total.")


def cmd_aggregate() -> None:
    """Recompute concept_vectors.npz from current raw_concept structure."""
    for layer in LAYERS:
        raw_dir = RUN_DIR / f"layer_{layer}/raw_concept"
        if not raw_dir.exists():
            print(f"  ⚠ {raw_dir} missing — skipping layer {layer}")
            continue

        by_concept: dict[str, list] = defaultdict(list)
        for f in raw_dir.glob("*.npy"):
            concept = f.stem.split("-")[0]
            if concept in CONCEPTS:
                by_concept[concept].append(np.load(f))

        # compute mean per concept
        means = {}
        for c in CONCEPTS:
            if c in by_concept and by_concept[c]:
                means[c] = np.stack(by_concept[c]).mean(axis=0)

        # project off neutral
        basis_path = RUN_DIR / f"layer_{layer}/neutral_projection.npy"
        mean_path = RUN_DIR / f"layer_{layer}/mean.npy"
        if not basis_path.exists():
            print(f"  ⚠ layer_{layer} neutral basis missing")
            continue
        basis = np.load(basis_path)        # [d, k]
        mean_neutral = np.load(mean_path)  # [d]

        final = {}
        for c, raw in means.items():
            v = raw - mean_neutral
            v = v - basis @ (basis.T @ v)
            final[c] = v.astype(np.float32)

        out = RUN_DIR / f"layer_{layer}/concept_vectors.npz"
        np.savez(out, **final)
        n_per = ", ".join(
            f"{c}={len(by_concept[c])}" for c in CONCEPTS if c in by_concept
        )
        print(f"  layer {layer}: saved {out.name} ({n_per})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true",
                    help="Build the rename plan and save to JSON")
    ap.add_argument("--apply-txt", action="store_true",
                    help="Rename .txt files according to plan")
    ap.add_argument("--apply-npy", action="store_true",
                    help="Rename .npy files according to plan")
    ap.add_argument("--aggregate", action="store_true",
                    help="Recompute concept_vectors.npz from current files")
    ap.add_argument("--all", action="store_true",
                    help="Run plan + apply-txt + apply-npy + aggregate")
    args = ap.parse_args()

    if args.all or args.plan:
        cmd_plan()
    if args.all or args.apply_txt:
        cmd_apply_txt()
    if args.all or args.apply_npy:
        cmd_apply_npy()
    if args.all or args.aggregate:
        cmd_aggregate()


if __name__ == "__main__":
    main()
