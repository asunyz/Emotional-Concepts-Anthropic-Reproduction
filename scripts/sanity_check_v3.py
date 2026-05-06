"""
Cognitive v3 — Sanity check on generated stories.

After running `generate_trajectories_v3.py --sanity`, this script:
  1. Re-validates every story (structural + banned words)
  2. Groups paragraphs by stage-concept across trajectories
  3. Writes a human-readable report so you can eyeball whether stories sharing
     the same prior/discovery/reaction concept actually feel consistent.

Usage:
    python scripts/sanity_check_v3.py runs/cognitive_v3_sanity
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.v3_validate import (  # noqa: E402
    parse_blocks, validate_story, word_count,
)


def parse_metadata_header(text: str) -> tuple[dict, str]:
    """Files written by generate_trajectories_v3.py start with '# key: value'
    lines, then '---', then the body. Returns (metadata, body)."""
    meta = {}
    body_lines = []
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
        ok, reasons, blocks = validate_story(body)
        out.append({
            "path": path,
            "metadata": meta,
            "body": body,
            "blocks": blocks,
            "ok": ok,
            "reasons": reasons,
        })
    return out


def render_block(text: str, max_chars: int = 600) -> str:
    """Squash whitespace and truncate for a markdown-friendly cell."""
    s = " ".join(text.split())
    if len(s) > max_chars:
        s = s[:max_chars - 1] + "…"
    return s


def report_structural(stories: list[dict]) -> str:
    lines = ["## 1. Structural validation\n"]
    n_total = len(stories)
    n_ok = sum(1 for s in stories if s["ok"])
    lines.append(f"- Total: {n_total}, OK: {n_ok}, Failed: {n_total - n_ok}\n")

    fail = [s for s in stories if not s["ok"]]
    if fail:
        lines.append("### Failures")
        for s in fail:
            lines.append(f"- `{s['path'].name}` — {'; '.join(s['reasons'])}")
        lines.append("")

    if n_ok > 0:
        lines.append("### Word counts (for OK stories)")
        lines.append("| Story | P1 | P2 | P3 |")
        lines.append("|---|---|---|---|")
        for s in stories:
            if not s["ok"]:
                continue
            wc = {tag: word_count(s["blocks"][tag]) for tag in ("P1", "P2", "P3")}
            lines.append(
                f"| `{s['path'].name}` | {wc['P1']} | {wc['P2']} | {wc['P3']} |"
            )
        lines.append("")
    return "\n".join(lines)


def report_consistency(stories: list[dict]) -> str:
    """Group OK stories by stage-concept and dump P1/P2/P3 side-by-side for
    human eyeball verification. The rule we want to check:

      - All P1s with prior=X should feel like the same prior
      - All P2s with discovery=Y should feel like the same discovery
      - All P3s with reaction=Z should feel like the same reaction
    """
    pos_ok = [s for s in stories if s["ok"] and s["metadata"].get("type") == "POS"]
    neg_ok = [s for s in stories if s["ok"] and s["metadata"].get("type") == "NEG"]

    lines = ["\n## 2. Cross-trajectory consistency (POS)\n"]
    lines.append(
        f"_{len(pos_ok)} OK POS stories. For each stage-concept, all paragraphs "
        f"surfacing it are listed below — read them and check whether they feel "
        f"like the same cognitive state._\n"
    )

    # ---- Group P1 by prior ----
    by_prior = defaultdict(list)
    for s in pos_ok:
        by_prior[s["metadata"]["prior"]].append(s)
    lines.append("### 2.1 Prior — P1 grouped by prior concept\n")
    for concept in ["curious", "uncertain", "confident"]:
        group = by_prior.get(concept, [])
        lines.append(f"#### prior = `{concept}` ({len(group)} stories)\n")
        if not group:
            lines.append("_(no stories)_\n")
            continue
        for s in group:
            traj = s["metadata"]["trajectory_name"]
            lines.append(f"- **{traj}**: {render_block(s['blocks']['P1'])}")
        lines.append("")

    # ---- Group P2 by discovery ----
    by_discovery = defaultdict(list)
    for s in pos_ok:
        by_discovery[s["metadata"]["discovery"]].append(s)
    lines.append("### 2.2 Discovery — P2 grouped by discovery concept\n")
    for concept in ["surprised", "bored"]:
        group = by_discovery.get(concept, [])
        lines.append(f"#### discovery = `{concept}` ({len(group)} stories)\n")
        if not group:
            lines.append("_(no stories)_\n")
            continue
        for s in group:
            traj = s["metadata"]["trajectory_name"]
            lines.append(f"- **{traj}**: {render_block(s['blocks']['P2'])}")
        lines.append("")

    # ---- Group P3 by reaction ----
    by_reaction = defaultdict(list)
    for s in pos_ok:
        by_reaction[s["metadata"]["reaction"]].append(s)
    lines.append("### 2.3 Reaction — P3 grouped by reaction concept\n")
    for concept in ["stubborn", "enlightened", "confused", "confirmed"]:
        group = by_reaction.get(concept, [])
        lines.append(f"#### reaction = `{concept}` ({len(group)} stories)\n")
        if not group:
            lines.append("_(no stories)_\n")
            continue
        for s in group:
            traj = s["metadata"]["trajectory_name"]
            lines.append(f"- **{traj}**: {render_block(s['blocks']['P3'])}")
        lines.append("")

    # ---- NEG ----
    lines.append("\n## 3. NEG (factual baseline)\n")
    lines.append(
        f"_{len(neg_ok)} OK NEG stories. These should describe the same scenarios "
        f"in factual third-person register, with no cognitive states or interiority._\n"
    )
    for s in neg_ok:
        topic = s["metadata"].get("topic", "?")
        lines.append(f"### `{s['path'].name}` — topic: {topic}\n")
        for tag in ("P1", "P2", "P3"):
            lines.append(f"- **{tag}**: {render_block(s['blocks'][tag])}")
        lines.append("")

    return "\n".join(lines)


def report_full_stories(stories: list[dict]) -> str:
    """Optional: full text of each OK story, for closer reading."""
    pos_ok = sorted(
        [s for s in stories if s["ok"] and s["metadata"].get("type") == "POS"],
        key=lambda s: int(s["metadata"].get("trajectory_id", "0") or "0"),
    )
    lines = ["\n## 4. Full POS stories (in trajectory order)\n"]
    for s in pos_ok:
        traj = s["metadata"]["trajectory_name"]
        topic = s["metadata"].get("topic", "?")
        lines.append(f"### {traj} — topic: {topic}\n")
        lines.append(f"**P1 ({s['metadata']['prior']}):** {render_block(s['blocks']['P1'], 1000)}\n")
        lines.append(f"**P2 ({s['metadata']['discovery']}):** {render_block(s['blocks']['P2'], 1000)}\n")
        lines.append(f"**P3 ({s['metadata']['reaction']}):** {render_block(s['blocks']['P3'], 1500)}\n")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path,
                    help="e.g. runs/cognitive_v3_sanity")
    args = ap.parse_args()

    stories_dir = args.run_dir / "stories"
    if not stories_dir.exists():
        print(f"ERROR: {stories_dir} does not exist. Did you run "
              f"generate_trajectories_v3.py first?")
        sys.exit(1)

    stories = collect_stories(stories_dir)
    print(f"Found {len(stories)} stories in {stories_dir}")

    parts = [
        f"# Sanity check report — {args.run_dir.name}\n",
        report_structural(stories),
        report_consistency(stories),
        report_full_stories(stories),
    ]
    report = "\n".join(parts)

    out = args.run_dir / "sanity_report.md"
    out.write_text(report)
    print(f"Wrote {out}")
    print()
    # Print just the summary to console
    n_ok = sum(1 for s in stories if s["ok"])
    n_fail = len(stories) - n_ok
    print(f"Summary: {n_ok}/{len(stories)} stories pass structural validation.")
    if n_fail:
        print("Failures:")
        for s in stories:
            if not s["ok"]:
                print(f"  - {s['path'].name}: {'; '.join(s['reasons'])}")


if __name__ == "__main__":
    main()
