# Cognitive v3 — Trajectory-Pinned Inputs

This directory contains the input config for the trajectory-based F2 pipeline.
See `docs/F2_cognitive_v3_design.md` for the full rationale.

## Files

| File | Purpose |
|---|---|
| `trajectories.json` | 9 valid cognitive trajectories + per-stage concept definitions |
| `topics.txt` | 5 information-processing scenarios |
| `pos_prompt.txt` | POS prompt template (trajectory-pinned, 3 explicit stages) |
| `neg_prompt.txt` | NEG prompt template (factual third-person, scenario-matched) |

## Sanity test

```bash
python scripts/generate_trajectories_v3.py --sanity
python scripts/sanity_check_v3.py runs/cognitive_v3_sanity
```

Sanity mode generates 1 story per trajectory + 1 NEG = 10 stories total
(roughly 5 minutes on a single H100). The check script then verifies:

1. Structural conformance — `<P1>/<P2>/<P3>` markers, word counts, no banned words
2. Visual consistency — paragraphs grouped by stage-concept for human eyeball check

## Why this is different from v2

v2 gave the model one concept name and asked for a "3-stage arc" — the model
had to infer prior + discovery + reaction itself. 22.7% of stories drifted
to a different concept (the model honestly logged the actual one in metadata).

v3 pins all three stages in the prompt. The model has no path-choice freedom;
it only chooses how to write the assigned trajectory naturally.
