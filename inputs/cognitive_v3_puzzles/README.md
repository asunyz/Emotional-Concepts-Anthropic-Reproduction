# Cognitive v3 — Puzzle-Context Inputs

Puzzle-context variant of `inputs/cognitive_v3/`. Same trajectories, same prompt
templates — just topics that center on constraint-reasoning / puzzle-solving
scenarios instead of generic information processing.

## Why

The original cognitive_v3 vectors were extracted from stories about generic
information-processing scenarios (scientist examining a result, doctor reviewing
labs, etc.). When we tested those vectors on ZebraLogic puzzle-solving
activations, they did not reliably correspond to their labels (see Notion: F2
Cross-Validation results, v2 n=40, v3 trajectory n=40). The most parsimonious
read: the vectors carry generic-task structure baked into the story-generation
context, not the cognitive states themselves.

Re-extracting on puzzle-context stories should give vectors whose task structure
aligns with the activations we want to project them on (logic-grid puzzle
solving).

## Files

| File | Same as `cognitive_v3/`? | Notes |
|---|---|---|
| `trajectories.json` | ✓ identical | same 9 trajectories, same stage-concept definitions |
| `pos_prompt.txt` | ✓ identical | parameterizes on `{topic}` so no changes needed |
| `neg_prompt.txt` | ✓ identical | parameterizes on `{topic}` so no changes needed |
| `topics.txt` | ✗ rewritten | 8 puzzle/constraint-reasoning scenarios |
| `characters.json` | ✗ rewritten | character pools aligned with the new topics |

## What's still missing (must be added before extraction is meaningful)

**Neutral PCA projection step.** The v3 pipeline (`scripts/extract_v3_compare.py`)
does not run PCA on neutral activations to project off task-generic directions
from the concept vectors. The original emotion pipeline (`extract_concepts.py`)
does. Without this step, the re-extracted puzzle vectors will still carry
task-generic puzzle-solving structure (e.g., "this is a logic-grid prompt")
baked into the concept axis. That defeats most of the point of re-extraction.

See Notion → "Caveat: Missing Neutral Projection Step".

## Run

```bash
# 1. Sanity test (1 story per trajectory, ~5 min on H100)
python scripts/generate_trajectories_v3.py \
  --input-dir inputs/cognitive_v3_puzzles \
  --task-label cognitive_v3_puzzles_sanity \
  --sanity

python scripts/sanity_check_v3.py runs/cognitive_v3_puzzles_sanity

# 2. Full generation
python scripts/generate_trajectories_v3.py \
  --input-dir inputs/cognitive_v3_puzzles \
  --task-label cognitive_v3_puzzles \
  --pos-stories-per-traj-topic 5 \
  --neg-stories-per-topic 10

# 3. Extract concept vectors at layer 30
python scripts/extract_v3_compare.py \
  --run-dir runs/cognitive_v3_puzzles \
  --layers 30
```

Outputs land in `runs/cognitive_v3_puzzles/extractions/methodC_incontext/layer_30/`:
- `concept_vectors_modeA.npz` — 9 concept vectors
- `mean.npy` — global mean for centering
- `trajectories.json` — copied for downstream tooling

Drop the npz + mean.npy into `arcarae-active-selection/data/f2_vectors/` and
re-run validation (`scripts/02_validate_mcq.py` + `scripts/03_validate_f2_zebralogic.py`)
to compare against the story-derived vectors.
