# Qwen3.6-35B-A3B MoE Port — 7-Concept Run Plan

**Owner:** Yadong
**Target:** surprise7_qwen3moe — reproduce 6 emotion concepts + add `surprise` as 7th
**Deadline:** week of 2026-04-26

## TL;DR

Minimum-diff port: reuse the existing pipeline, add one concept, swap the model. The 6 known-working emotions act as a control when debugging the MoE run.

## Current state

- `inputs/emotions/concepts.csv` — `happy, sad, afraid, angry, calm, indifferent` (6)
- Target: add `surprise` → 7 concepts
- Model: Llama-3.1-8B/70B (dense) → `Qwen/Qwen3.6-35B-A3B` (MoE, 40 layers, hidden=2048, 256 experts / 9 active)

## Four decisions to make before running

### 1. Layer selection (don't reuse Llama's 16/24)

Llama-3.1-8B has 32 layers; 16/24 ≈ mid/late. Qwen3.6-35B-A3B has **40 layers**; equivalent is ~20/30. For the first MoE run, scan wider:

```
--layers 10,20,30,36
```

Pipeline natively supports multi-layer — marginal cost is more `.npy` files, avoids rerunning if we pick the wrong single layer.

### 2. `n_stories` should probably grow

Angelica observed that 5 stories × 6 concepts felt too small on Llama-70B (transcript 17:03). For first MoE validation run, use:

```
--n-stories 12
```

Qwen3-MoE generation is slower than Llama-8B (MoE router overhead) but resumability makes this low-risk.

### 3. Verify `AVG_FROM_TOKEN = 50` still makes sense

`extract_concepts.py:44` — this magic number skips story setup and averages the body. For `surprise` specifically, the "reveal" moment may land early. If token 50 starts *after* the reveal, the signal gets diluted.

**Action:** before the full run, run the pipeline on a tiny test subset, print token distribution of a few `stories/surprise-0-*.txt`, and either keep 50 or drop to ~20.

### 4. Review the synonym-ban prompt for `surprise`

`inputs/emotions/concept_prompt.txt` bans the concept word + direct synonyms. For `happy`/`sad` that's clean. For `surprise`, synonyms range wider: *shocked, amazed, startled, stunned, taken aback, unexpected* — and the semantic field (cognitive reaction vs. emotional state) is different.

**Risk:** model over-avoids and writes event descriptions with no character-level surprise reaction.

**Action:** hand-check the first `stories/surprise-0-*.txt` batch; if characters don't actually *feel* surprise, tune the prompt before the full run.

## Concrete change list

1. **`config.py`** — `MODEL_ID = "Qwen/Qwen3.6-35B-A3B"`, confirm `QUANTIZATION = "nf4"`, possibly adjust `MODELS_ROOT`.
2. **`inputs/emotions/concepts.csv`** — append `,surprise`.
3. **Command line:**

   ```bash
   python extract_concepts.py \
       --concept-prompt inputs/emotions/concept_prompt.txt \
       --concept-topics inputs/emotions/concept_topics.txt \
       --concepts       inputs/emotions/concepts.csv \
       --neutral-prompt inputs/emotions/neutral_prompt.txt \
       --neutral-topics inputs/emotions/concept_topics.txt \
       --layers 10,20,30,36 \
       --n-stories 12 \
       --task-label surprise7_qwen36moe
   ```

## Pre-flight diagnostic (before the big run)

Write a ~20-line script that:

1. Loads Qwen3.6-35B-A3B via `cv_utils.load_model()`.
2. Runs a dummy trace: `model.model.layers[L].output[0]` for L in [0, 12, 24, 36, 44] — confirm shape `[1, seq, hidden]` or `[seq, hidden]`.
3. On a chat-templated sample, prints per-token residual norms — identify any outlier tokens (Qwen has no BOS, but `<|im_start|>` / `<|im_end|>` could behave like one).
4. Confirms NF4 + MoE + NNSight trace all coexist without error.

If all four green → proceed to full run.

## Post-run validation

After `concept_vectors.npz` exists for all 4 layers:

1. Run `concept_similarity.py` — the 6 known emotions should form sensible cosine structure (happy ⊥ sad negative, etc.) at one of the mid/late layers. If not, the MoE port has an issue.
2. Run `concept_cluster.py` — look for valence × arousal layout, with `surprise` sitting off-axis.
3. Run `label_text.py` on a few sample sentences to see per-token behavior of `v_surprise`.
4. Compare to Angelica's Llama-70B emotion vectors as a sanity check (if results diverge wildly, investigate).

## Open risks

- **Device map with MoE** — `device_map="auto"` may split layers across GPUs; `steer.py`'s "grab first cuda param" pattern may pick wrong device. Not blocking for extraction, but fix before running `steer.py`.
- **Surprise is punctuated, not diffuse** — even with the correct pipeline, `v_surprise` may be noisy because only a few tokens per story carry the signal. If so, upgrade to contrast-pair subtraction (the coin-edge / coin-heads minimal-pair idea from the 2026-04-24 meeting).
- **Qwen chat template leakage** — `dialogue_to_chat_text` uses the tokenizer's chat template for neutral dialogues; Qwen's system prompt wrapper is different from Llama's, so the neutral PCA basis may capture different "task-generic" directions.
