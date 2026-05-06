"""
Cognitive v3 — Trajectory-pinned story generation with structural validation.

For each (trajectory, topic) pair, generates N stories using the v3 POS prompt,
which fixes the prior / discovery / reaction stages explicitly. After generation,
each story is validated:

  - Has exactly one <P1>...</P1>, <P2>...</P2>, <P3>...</P3> block (in order)
  - Word counts are within tolerance (40-60 / 40-60 / 80-100, ±50%)
  - No banned cognitive-concept words (curious, uncertain, ..., or grammatical variants)
  - No metadata header ("State:", "Trajectory:", "Pathway:", "Cognitive State:", "Stage:")

Stories that fail validation are regenerated, up to --max-retries times. Stories that
still fail are written to `failed/` for inspection.

NEG (neutral) stories are generated similarly with the v3 NEG prompt, which produces
factual third-person paragraphs of the same scenario.

Output layout:
    runs/<task-label>/
        prompts.json                          # full provenance
        trajectories.json                     # copy of input config
        stories/
            POS-<traj_id>-<topic_idx>-<story_idx>.txt
            NEG-<topic_idx>-<story_idx>.txt
        stories/_raw/                         # full model completions (for debug)
        stories/_failed/                      # validation-failed stories
        validation_log.json                   # per-story validation results

Usage (sanity mode — 1 topic, 1 story per trajectory, ~10 stories total):
    python scripts/generate_trajectories_v3.py --sanity

Usage (full run — 5 topics, configurable stories per trajectory):
    python scripts/generate_trajectories_v3.py \\
        --pos-stories-per-traj-topic 5 \\
        --oversample-traj-16 4 \\
        --neg-stories-per-topic 10 \\
        --task-label cognitive_v3_qwen35_nf4
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tqdm.auto import tqdm

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from cv_utils import load_model, generate_story  # noqa: E402
from scripts.v3_validate import validate_story  # noqa: E402


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def build_pos_prompt(template: str, topic: str, traj: dict, stage_defs: dict) -> str:
    return template.format(
        topic=topic,
        prior_concept=traj["prior"],
        prior_def=stage_defs["prior"][traj["prior"]],
        discovery_concept=traj["discovery"],
        discovery_def=stage_defs["discovery"][traj["discovery"]],
        reaction_concept=traj["reaction"],
        reaction_def=stage_defs["reaction"][traj["reaction"]],
    )


def build_neg_prompt(template: str, topic: str) -> str:
    return template.format(topic=topic)


def write_story_file(
    path: Path,
    text: str,
    blocks: dict[str, str] | None,
    metadata: dict,
) -> None:
    header_lines = [f"# {k}: {v}" for k, v in metadata.items()]
    if blocks:
        body = (
            f"<P1>\n{blocks['P1']}\n</P1>\n\n"
            f"<P2>\n{blocks['P2']}\n</P2>\n\n"
            f"<P3>\n{blocks['P3']}\n</P3>\n"
        )
    else:
        body = text
    path.write_text("\n".join(header_lines) + "\n---\n" + body)


def generate_with_validation(
    model,
    prompt: str,
    metadata: dict,
    is_neg: bool,
    target_path: Path,
    failed_dir: Path,
    raw_dir: Path,
    max_new_tokens: int,
    temperature: float,
    max_retries: int,
    log: list[dict],
) -> bool:
    """Generate a story for `prompt`, validate, retry up to max_retries.

    Returns True if a valid story was written to target_path, else False
    (and the last failed attempt is written to failed_dir).
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    last_text, last_reasons, last_blocks = "", ["never_generated"], None
    for attempt in range(1, max_retries + 1):
        text = generate_story(
            model, prompt, max_new_tokens=max_new_tokens, temperature=temperature
        )
        # Save raw completion for every attempt
        raw_path = raw_dir / f"{target_path.stem}_attempt{attempt}.txt"
        raw_path.write_text(text)

        ok, reasons, blocks = validate_story(text, neg=is_neg)
        log.append({
            "story": target_path.stem,
            "attempt": attempt,
            "ok": ok,
            "reasons": reasons,
        })
        last_text, last_reasons, last_blocks = text, reasons, blocks
        if ok:
            write_story_file(target_path, text, blocks, metadata)
            return True
    # All attempts failed — save last attempt to failed dir for inspection
    failed_dir.mkdir(parents=True, exist_ok=True)
    fail_path = failed_dir / target_path.name
    write_story_file(
        fail_path,
        last_text,
        last_blocks,
        {**metadata, "validation_failed_reasons": "; ".join(last_reasons)},
    )
    return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", default="inputs/cognitive_v3")
    ap.add_argument("--output-dir", default="runs")
    ap.add_argument("--task-label", default=None,
                    help="defaults to 'cognitive_v3_sanity' under --sanity, else 'cognitive_v3'")
    ap.add_argument("--sanity", action="store_true",
                    help="sanity mode: 1 story per trajectory on 1 topic, 1 NEG, ~10 stories total")
    ap.add_argument("--pos-stories-per-traj-topic", type=int, default=5,
                    help="(non-sanity) stories per (trajectory, topic) cell")
    ap.add_argument("--oversample-traj-16", type=int, default=4,
                    help="(non-sanity) multiplier for trajectory #16 (uncertain->bored->confirmed)")
    ap.add_argument("--neg-stories-per-topic", type=int, default=10,
                    help="(non-sanity) NEG stories per topic")
    ap.add_argument("--max-new-tokens", type=int, default=600,
                    help="generation budget per story")
    ap.add_argument("--temperature", type=float, default=config.GEN_TEMPERATURE)
    ap.add_argument("--max-retries", type=int, default=3)
    ap.add_argument("--model-path", default=None)
    args = ap.parse_args()

    in_dir = Path(args.input_dir)
    traj_cfg = json.loads((in_dir / "trajectories.json").read_text())
    trajectories = traj_cfg["trajectories"]
    stage_defs = traj_cfg["stage_concepts"]
    topics = [l.strip() for l in (in_dir / "topics.txt").read_text().splitlines() if l.strip()]
    pos_template = (in_dir / "pos_prompt.txt").read_text()
    neg_template = (in_dir / "neg_prompt.txt").read_text()

    # Resolve sanity / full mode
    if args.sanity:
        task_label = args.task_label or "cognitive_v3_sanity"
        topics = topics[:1]
        pos_per_cell = 1
        neg_per_topic = 1
        oversample_16 = 1
        print(f"[SANITY] Running sanity mode: {len(trajectories)} trajectories x "
              f"{len(topics)} topic x {pos_per_cell} story = {len(trajectories) * len(topics) * pos_per_cell} POS + "
              f"{len(topics) * neg_per_topic} NEG.")
    else:
        task_label = args.task_label or "cognitive_v3"
        pos_per_cell = args.pos_stories_per_traj_topic
        neg_per_topic = args.neg_stories_per_topic
        oversample_16 = args.oversample_traj_16

    root = Path(args.output_dir) / task_label
    stories_dir = root / "stories"
    raw_dir = stories_dir / "_raw"
    failed_dir = stories_dir / "_failed"
    stories_dir.mkdir(parents=True, exist_ok=True)

    # Provenance
    (root / "trajectories.json").write_text(json.dumps(traj_cfg, indent=2))
    (root / "prompts.json").write_text(json.dumps({
        "pos_template": pos_template,
        "neg_template": neg_template,
        "topics": topics,
        "pos_stories_per_traj_topic": pos_per_cell,
        "neg_stories_per_topic": neg_per_topic,
        "oversample_traj_16": oversample_16,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "max_retries": args.max_retries,
        "sanity": args.sanity,
    }, indent=2))

    print(f"Loading model...")
    model = load_model(args.model_path)
    print(f"Model loaded.")

    log: list[dict] = []
    n_ok = 0
    n_fail = 0

    # ---------- POS ----------
    pos_jobs = []
    for traj in trajectories:
        n_per_cell = pos_per_cell * (oversample_16 if traj["id"] == 16 else 1)
        for tidx, topic in enumerate(topics):
            for sidx in range(n_per_cell):
                pos_jobs.append((traj, topic, tidx, sidx))

    pbar = tqdm(pos_jobs, desc="POS gen", unit="story")
    for traj, topic, tidx, sidx in pbar:
        prefix = f"POS-{traj['id']:02d}-{tidx}-{sidx}"
        target = stories_dir / f"{prefix}.txt"
        if target.exists():
            continue
        pbar.set_postfix_str(f"{traj['name']} t{tidx}")
        prompt = build_pos_prompt(pos_template, topic, traj, stage_defs)
        metadata = {
            "type": "POS",
            "trajectory_id": traj["id"],
            "trajectory_name": traj["name"],
            "prior": traj["prior"],
            "discovery": traj["discovery"],
            "reaction": traj["reaction"],
            "topic_idx": tidx,
            "topic": topic,
            "story_idx": sidx,
        }
        ok = generate_with_validation(
            model, prompt, metadata, is_neg=False,
            target_path=target, failed_dir=failed_dir, raw_dir=raw_dir,
            max_new_tokens=args.max_new_tokens, temperature=args.temperature,
            max_retries=args.max_retries, log=log,
        )
        n_ok += int(ok)
        n_fail += int(not ok)

    # ---------- NEG ----------
    neg_jobs = [(tidx, topic, sidx)
                for tidx, topic in enumerate(topics)
                for sidx in range(neg_per_topic)]
    pbar = tqdm(neg_jobs, desc="NEG gen", unit="story")
    for tidx, topic, sidx in pbar:
        prefix = f"NEG-{tidx}-{sidx}"
        target = stories_dir / f"{prefix}.txt"
        if target.exists():
            continue
        pbar.set_postfix_str(f"t{tidx}")
        prompt = build_neg_prompt(neg_template, topic)
        metadata = {
            "type": "NEG",
            "topic_idx": tidx,
            "topic": topic,
            "story_idx": sidx,
        }
        ok = generate_with_validation(
            model, prompt, metadata, is_neg=True,
            target_path=target, failed_dir=failed_dir, raw_dir=raw_dir,
            max_new_tokens=args.max_new_tokens, temperature=args.temperature,
            max_retries=args.max_retries, log=log,
        )
        n_ok += int(ok)
        n_fail += int(not ok)

    # Save validation log
    (root / "validation_log.json").write_text(json.dumps(log, indent=2))
    summary = {
        "task_label": task_label,
        "sanity": args.sanity,
        "n_pos_jobs": len(pos_jobs),
        "n_neg_jobs": len(neg_jobs),
        "n_total_jobs": len(pos_jobs) + len(neg_jobs),
        "n_ok": n_ok,
        "n_fail": n_fail,
        "ok_rate": round(n_ok / max(1, len(pos_jobs) + len(neg_jobs)), 3),
        "n_attempts": len(log),
        "avg_attempts_per_story": round(len(log) / max(1, len(pos_jobs) + len(neg_jobs)), 2),
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2))
    print()
    print("=" * 60)
    print(f"Done. ok={n_ok} fail={n_fail} (rate={summary['ok_rate']:.1%}). "
          f"Avg {summary['avg_attempts_per_story']} attempts per story.")
    print(f"Output: {root}")
    print("Next: python scripts/sanity_check_v3.py", root)


if __name__ == "__main__":
    main()
