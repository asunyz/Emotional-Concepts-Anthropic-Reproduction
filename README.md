# concept-vector

Toolkit for extracting, visualizing, and steering with **concept vectors** in a
Llama-3.1 model, built on top of [NNSight](https://nnsight.net/) and
[transformers](https://github.com/huggingface/transformers). Inspired by
Representation Engineering (Zou et al. 2023) and the activation-addition
literature.

The core idea: for each concept `c` (e.g. an emotion like `happy`) we produce a
direction in the residual stream at each chosen layer. That direction can be
used two ways:

- **Read** — project a new text's activations onto it to measure how strongly
  each token expresses `c`.
- **Write** — add `α · unit(v_c)` to the residual stream during generation to
  steer the model toward or away from `c`.

## Setup

```bash
bash setup_env.sh                    # creates conda env `nnsight`
conda activate nnsight
export HF_HOME=/mnt/e/models/hf_cache
huggingface-cli login                # Llama-3.1 is gated
python nnsight_intro.py              # downloads + quantizes on first run
```

Model / quantization / cache paths are centralized in [`config.py`](config.py).
Swap `MODEL_ID` or `QUANTIZATION` (`"nf4"` / `"int8"` / `"none"`) and every
script picks up the change. Quantized weights are saved to
`/mnt/e/models/<stem>-<quant>/` after the first download so subsequent runs
skip the Hub round-trip.

## Modules

| File | Role |
|---|---|
| `config.py` | Model ID, quantization, cache/save paths, generation defaults |
| `cv_utils.py` | Shared helpers: `load_model`, `generate_story`, `extract_layer_activations` |
| `nnsight_intro.py` | Guided tour of the NNSight API (tracing, saving, patching, steering) |
| `extract_concepts.py` | End-to-end extraction pipeline (generate → embed → reduce) |
| `label_text.py` | Color-code tokens by signed projection onto concept direction(s) |
| `steer.py` | Generate completions at multiple steering strengths |
| `concept_similarity.py` | Cosine-similarity heatmap between concept vectors |
| `concept_cluster.py` | PCA scatter of concept vectors (2D or 3D) |
| `concept_vs_variable.py` | Sweep a prompt variable, plot per-concept activation |

## Extraction pipeline

`extract_concepts.py` runs in two phases:

1. **Generate** — for each (concept, topic) it prompts the model once and asks
   for N stories in a single completion, split on `[story N]` (or `[dialogue
   N]` for neutral dialogues). Raw completions are archived under
   `stories/_raw/` for debugging parse issues.
2. **Embed** — each sub-story is fed back through the model *in isolation* (no
   prompt, no sibling stories) so its embedding reflects only its own text.
   Activations are averaged from the 50th token onward.

Per-concept final vector: `v_c_final = (I − B Bᵀ) (v_c − mean_c)`, where `B`
is the orthonormal basis of the top neutral-corpus PCs explaining ≥ 50% of
variance. This removes (a) a shared concept-writing bias and (b) task-generic
directions that show up when the model is doing "any dialogue."

Layout under `<output-dir>/<task-label>/`:

```
prompts.json
stories/<concept>-<topic_idx>-<story_idx>.txt
stories/_neutral-<topic_idx>-<story_idx>.txt
stories/_raw/<concept>-<topic_idx>.txt
layer_<L>/
    raw_concept/<concept>-<topic_idx>-<story_idx>.npy
    raw_neutral/<topic_idx>-<story_idx>.npy
    concept_vectors.npz      # one named array per concept
    mean.npy                 # centering vector
    neutral_projection.npy   # [d, k] PCA basis removed from concepts
```

The pipeline is resumable: any file already on disk is reused. Delete files to
force regeneration.

### Example

```bash
python extract_concepts.py \
    --concept-prompt inputs/emotions/concept_prompt.txt \
    --concept-topics inputs/emotions/concept_topics.txt \
    --concepts       inputs/emotions/concepts.csv \
    --neutral-prompt inputs/emotions/neutral_prompt.txt \
    --neutral-topics inputs/emotions/concept_topics.txt \
    --layers 16,24 \
    --task-label emotions_8b_nf4
```

## Using the vectors

```bash
# Color tokens by projection
python label_text.py --text story.txt --layers 16 \
    --concept-dir runs/emotions_8b_nf4 \
    --concepts happy,sad --colors red,blue --output labeled.html

# Steer generation
python steer.py --prompt "Tell me about your day." \
    --concept-dir runs/emotions_8b_nf4 --layer 16 --concept happy \
    --strengths=-6,-3,0,3,6

# Inspect concept space
python concept_similarity.py --concept-dir runs/emotions_8b_nf4 --layers 16,24
python concept_cluster.py    --concept-dir runs/emotions_8b_nf4 --layers 16,24

# Sweep a prompt variable
python concept_vs_variable.py --prompt "I took {x} mg of tylenol." \
    --values 0,250,500,1000,2000 --concept-dir runs/emotions_8b_nf4 \
    --layer 16 --concepts happy,sad --plot line --xlabel "dose (mg)"
```

### Scoring convention

Both `label_text.py` and `concept_vs_variable.py` use the field-standard
signed projection onto the unit concept direction:

```
score = (act − mean) · v_c / ‖v_c‖
```

Only the concept vector is normalized, not the activation — residual-stream
magnitudes carry real signal that full cosine similarity would wash out.
`label_text.py` additionally rescales per-text by `max(|score|)` and clips
negatives to 0 so each concept only tints tokens it points *toward*.

## Notes / gotchas

- **BOS is dropped** before scoring. Its residual norm is an order of
  magnitude larger than real tokens (attention-sink phenomenon) and otherwise
  dominates per-text rescaling.
- **NNSight trace order matters.** Inside a `with model.trace(...)` save
  modules in forward-pass order (`self_attn → mlp → layer_output` within a
  block, ascending layer index across blocks). Otherwise you'll hit
  `OutOfOrderError`.
- **Few-concept degeneracy.** Centering with only 2 concepts forces the two
  final vectors to be exact opposites (`v₁_final = −v₂_final`). The tooling
  still works, but similarity/cluster plots aren't meaningful until you have
  3+ concepts.

## Hardware

Tested on an 11 GB RTX 2080 Ti using 4-bit NF4 quantization. For 8-bit or full
precision on larger GPUs, flip `QUANTIZATION` in `config.py` — local save
directories are keyed by quantization so different builds don't collide.
