"""
Cognitive v4 — Two-speaker dialogue generation for present/other speaker analysis.

Reproduces Anthropic emotion paper (Table 14) setup but with cognitive concepts:
    For each (p1_state, p2_state) pair, generate N dialogues where two speakers
    each independently exhibit their assigned cognitive state. The dialogues are
    later parsed turn-by-turn into a 2×2 probe grid (token speaker × emotion label).

Output layout:
    runs/<task-label>/
        prompts.json                            # provenance
        concepts.json                           # copy of input concept config
        dialogues/
            <p1>__<p2>__<topic_idx>__<dlg_idx>.txt
        dialogues/_raw/                         # full model completions
        dialogues/_failed/                      # validation-failed ones
        validation_log.json                     # per-attempt validation results
        summary.json                            # final counts

Usage (sanity, 4 dialogues per pair, 8x8=64 pairs, ~256 dialogues):
    python scripts/generate_dialogues_v4.py \\
        --task-label cognitive_v4_dialogue_sanity \\
        --dialogues-per-pair 4

Usage (mid, 8 per pair, ~512 dialogues):
    python scripts/generate_dialogues_v4.py \\
        --task-label cognitive_v4_dialogue_mid \\
        --dialogues-per-pair 8
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from cv_utils import load_model, generate_story  # noqa: E402
from scripts.v4_dialogue_validate import validate_dialogue  # noqa: E402


def build_prompt(template: str, topic: str, p1: dict, p2: dict,
                 universal_banned: list[str]) -> str:
    """Render the dialogue_prompt.txt template with concrete pair + topic."""
    # banned_stems string: union of p1's, p2's, and universal layer-2 stems
    all_banned = sorted(set(p1["banned_stems"] + p2["banned_stems"] + universal_banned))
    banned_str = ", ".join(all_banned)
    return template.format(
        topic=topic,
        p1_state=p1["name"],
        p1_show=p1["show_through"],
        p2_state=p2["name"],
        p2_show=p2["show_through"],
        banned_stems=banned_str,
    )


def write_dialogue_file(path: Path, text: str, turns: list[dict] | None,
                         metadata: dict) -> None:
    """Write dialogue with metadata header. Body is the parsed-clean turn list
    if available, else the raw text."""
    header_lines = [f"# {k}: {v}" for k, v in metadata.items()]
    if turns:
        body = "\n\n".join(f"{t['speaker'].replace('P1', 'Person 1').replace('P2', 'Person 2')}: {t['text']}"
                           for t in turns)
    else:
        body = text
    path.write_text("\n".join(header_lines) + "\n---\n" + body + "\n")


def generate_with_validation(
    model,
    prompt: str,
    metadata: dict,
    p1_state: str,
    p2_state: str,
    banned_stems: list[str],
    target_path: Path,
    failed_dir: Path,
    raw_dir: Path,
    max_new_tokens: int,
    temperature: float,
    max_retries: int,
    log: list[dict],
) -> bool:
    raw_dir.mkdir(parents=True, exist_ok=True)
    last_text, last_reasons, last_turns = "", ["never_generated"], None
    for attempt in range(1, max_retries + 1):
        text = generate_story(
            model, prompt, max_new_tokens=max_new_tokens, temperature=temperature
        )
        raw_path = raw_dir / f"{target_path.stem}_attempt{attempt}.txt"
        raw_path.write_text(text)

        ok, reasons, turns = validate_dialogue(
            text, p1_state=p1_state, p2_state=p2_state, banned_stems=banned_stems
        )
        log.append({
            "dialogue": target_path.stem,
            "attempt": attempt,
            "ok": ok,
            "reasons": reasons,
            "n_turns": len(turns),
        })
        last_text, last_reasons, last_turns = text, reasons, turns
        if ok:
            write_dialogue_file(target_path, text, turns, metadata)
            return True
    failed_dir.mkdir(parents=True, exist_ok=True)
    fail_path = failed_dir / target_path.name
    write_dialogue_file(
        fail_path, last_text, last_turns,
        {**metadata, "validation_failed_reasons": "; ".join(last_reasons)},
    )
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", default="inputs/cognitive_v4_dialogue")
    ap.add_argument("--output-dir", default="runs")
    ap.add_argument("--task-label", required=True,
                    help="e.g. cognitive_v4_dialogue_sanity")
    ap.add_argument("--dialogues-per-pair", type=int, default=4,
                    help="dialogues to generate for each (p1, p2) pair")
    ap.add_argument("--max-new-tokens", type=int, default=600)
    ap.add_argument("--temperature", type=float, default=config.GEN_TEMPERATURE)
    ap.add_argument("--max-retries", type=int, default=3)
    ap.add_argument("--concepts-subset", nargs="*", default=None,
                    help="(debug) only use this subset of concepts (e.g. curious uncertain)")
    ap.add_argument("--model-path", default=None)
    args = ap.parse_args()

    in_dir = Path(args.input_dir)
    cfg = json.loads((in_dir / "concepts.json").read_text())
    concepts_all = cfg["concepts"]
    universal_banned = cfg.get("universal_banned_stems", [])
    if args.concepts_subset:
        concepts = [c for c in concepts_all if c["name"] in args.concepts_subset]
        print(f"using subset: {[c['name'] for c in concepts]}")
    else:
        concepts = concepts_all
    topics = [l.strip() for l in (in_dir / "topics.txt").read_text().splitlines() if l.strip()]
    template = (in_dir / "dialogue_prompt.txt").read_text()

    out = Path(args.output_dir) / args.task_label
    dlg_dir = out / "dialogues"
    raw_dir = dlg_dir / "_raw"
    failed_dir = dlg_dir / "_failed"
    out.mkdir(parents=True, exist_ok=True)
    dlg_dir.mkdir(parents=True, exist_ok=True)

    # Provenance
    (out / "concepts.json").write_text(json.dumps(cfg, indent=2))
    (out / "prompts.json").write_text(json.dumps({
        "args": vars(args),
        "n_concepts": len(concepts),
        "n_pairs": len(concepts) ** 2,
        "n_topics": len(topics),
        "expected_dialogues": len(concepts) ** 2 * args.dialogues_per_pair,
        "template": template,
    }, indent=2))

    print(f"loading model ...")
    model = load_model(args.model_path)
    print(f"  ok")

    log: list[dict] = []
    pairs = [(p1, p2) for p1 in concepts for p2 in concepts]
    n_total = len(pairs) * args.dialogues_per_pair
    n_ok, n_fail = 0, 0

    pbar = tqdm(total=n_total, desc="dialogues")
    for p1 in concepts:
        for p2 in concepts:
            for d in range(args.dialogues_per_pair):
                topic_idx = d % len(topics)
                topic = topics[topic_idx]
                stem = f"{p1['name']}__{p2['name']}__t{topic_idx}__d{d}"
                target = dlg_dir / f"{stem}.txt"
                if target.exists():
                    pbar.update(1); pbar.set_postfix(skip=True)
                    n_ok += 1
                    continue
                metadata = {
                    "p1_state": p1["name"],
                    "p2_state": p2["name"],
                    "topic_idx": topic_idx,
                    "topic": topic,
                    "dialogue_idx": d,
                }
                pair_banned = sorted(set(
                    p1["banned_stems"] + p2["banned_stems"] + universal_banned
                ))
                prompt = build_prompt(template, topic, p1, p2, universal_banned)
                ok = generate_with_validation(
                    model, prompt, metadata,
                    p1_state=p1["name"], p2_state=p2["name"],
                    banned_stems=pair_banned,
                    target_path=target,
                    failed_dir=failed_dir,
                    raw_dir=raw_dir,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    max_retries=args.max_retries,
                    log=log,
                )
                if ok:
                    n_ok += 1
                else:
                    n_fail += 1
                pbar.update(1)
                pbar.set_postfix(ok=n_ok, fail=n_fail)
                # incremental log save (resumable)
                if (n_ok + n_fail) % 20 == 0:
                    (out / "validation_log.json").write_text(json.dumps(log, indent=2))
    pbar.close()

    (out / "validation_log.json").write_text(json.dumps(log, indent=2))
    (out / "summary.json").write_text(json.dumps({
        "ok": n_ok,
        "failed": n_fail,
        "total": n_total,
        "n_concepts": len(concepts),
        "n_pairs": len(pairs),
        "dialogues_per_pair": args.dialogues_per_pair,
        "pass_rate": round(n_ok / n_total, 3) if n_total else 0,
    }, indent=2))
    print(f"\n=== done ===")
    print(f"  ok:  {n_ok}/{n_total}")
    print(f"  fail: {n_fail}/{n_total}")
    print(f"  pass rate: {n_ok / n_total * 100:.1f}%")
    print(f"  output: {out}")


if __name__ == "__main__":
    main()
