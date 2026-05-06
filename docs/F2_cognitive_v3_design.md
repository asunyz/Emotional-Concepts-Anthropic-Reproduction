# F2 Cognitive v3 — Trajectory-Pinned Pipeline

**Status:** v3 (post-mortem of v2 + redesign)
**Branch:** `f2-cognitive-v3`
**Goal:** Eliminate the trajectory-drift / sample-imbalance / baseline-mismatch problems that made v2's vectors unreliable.

---

## 1. Why v3 — what went wrong in v2

V2 generated stories by giving the model a single concept word (`enlightened`, `confirmed`, ...) and asking it to write a 3-stage narrative arc. The audit ([F2_cognitive_v2_analysis_relabel.md](F2_cognitive_v2_analysis_relabel.md)) found three structural problems:

| Layer | Problem |
|---|---|
| **Prompt** | Concepts aren't independent — `confirmed` only makes sense as the reaction to `uncertain → bored`. Asking the model to write `confirmed` for a topic phrased as "may contradict the hypothesis" forces a contradiction; the model honestly relabels in metadata. **22.7% of stories drifted to a different concept**. |
| **Extraction** | Each story was a full trajectory (prior → discovery → reaction). Whole-story activation mean is a **trajectory mean**, not a concept vector. The "saturated with one state" claim and "3-stage arc" requirement are physically incompatible. |
| **Baseline** | Neutral stories were independent scientific topics ("hydrogen combustion") with no narrative or discovery framing. The PCA basis built from them couldn't subtract off the "discovery scenario" features that pervade concept stories. |

Concrete consequences in the audit data:
- `confirmed` ended up with n=5 stories (target 25); `uncertain` n=7; `curious` swelled to 31
- `curious` dominated 7/8 sentence-level selectivity probes
- `confident + surprised → stubborn` came out **−0.232** (should be positive)

---

## 2. v3 core design

Three changes that address the three problem layers, plus one new requirement (validation).

### 2.1 Pin trajectories — eliminate model's path-choice freedom

V3 enumerates the cognitive space as **3 priors × 2 discoveries × 4 reactions = 24 combinations**, then keeps only **9 valid trajectories** (8 clearly valid + 1 marginal). See [`inputs/cognitive_v3/trajectories.json`](../inputs/cognitive_v3/trajectories.json) for the full list.

Each story prompt fixes ALL THREE stages:

```
Required trajectory (FIXED — do not deviate):
- Paragraph 1 (PRIOR — uncertain): The character holds a tentative belief...
- Paragraph 2 (DISCOVERY — bored): The outcome is plain and unremarkable...
- Paragraph 3 (REACTION — confirmed): The expected match consolidates...
```

The model has zero degrees of freedom in path selection — it only chooses how to write the assigned trajectory naturally.

### 2.2 Stage-localized extraction — paragraph-level vectors

Story output uses mandatory `<P1>...</P1> / <P2>...</P2> / <P3>...</P3>` markers. At extraction time we parse the markers, identify the token range of each paragraph, and **average activations only within the corresponding paragraph**.

Then we aggregate by stage-concept across trajectories:

```
v_uncertain = mean( P1 activations from all stories where prior = uncertain )
            (sourced from trajectories #9, #10, #11, #16)

v_confirmed = mean( P3 activations from all stories where reaction = confirmed )
            (sourced from trajectory #16 only)
```

Each vector is now genuinely "saturated with one cognitive state" — the relevant paragraph's tokens carry only that stage's content. (Modulo the late-layer leakage discussed in §6.)

### 2.3 POS / NEG — same scenario, paired baseline

For each topic, we generate both:

- **POS** — the trajectory story (subjective interiority, cognitive state)
- **NEG** — same scenario, factual third-person, no interiority

The PCA basis is built per-paragraph from NEG stories: `neutral_basis_P1`, `neutral_basis_P2`, `neutral_basis_P3`. When extracting `v_uncertain` (P1 concept), only `neutral_basis_P1` is projected off — true stage-matched baseline.

### 2.4 Generation-time validation — close the drift loop

V2 had no quality gate. V3 enforces:

- **Structural:** exactly one `<P1>/<P2>/<P3>` block, in order
- **Word counts:** P1/P2 in 25–90 words, P3 in 50–150 words (design target ±50%)
- **Banned words:** none of the 9 cognitive concept stems (`curious`, `uncertain`, ..., `confirm`) anywhere in the body
- **No metadata leak:** no `State:`, `Trajectory:`, `Pathway:`, `Cognitive State:`, `Stage:` headers

Stories failing validation are regenerated up to N times. Persistent failures are written to `_failed/` for inspection.

---

## 3. Trajectory enumeration

| # | Prior | Discovery | Reaction | Notes |
|---|---|---|---|---|
| 1 | confident | surprised | stubborn | Strong prior violated; refuses to update |
| 2 | confident | surprised | enlightened | Strong prior violated; framework restructured |
| 3 | confident | surprised | confused | Strong prior collapses; cannot process |
| 9 | uncertain | surprised | stubborn | Weak prior meets counter-evidence; held (marginal) |
| 10 | uncertain | surprised | enlightened | Weak prior opened by counter-evidence |
| 11 | uncertain | surprised | confused | Weak prior disrupted by counter-evidence |
| 16 | uncertain | bored | confirmed | Weak prior reinforced by expected outcome |
| 18 | curious | surprised | enlightened | Exploration yields new understanding |
| 19 | curious | surprised | confused | Exploration yields unprocessable surprise |

Excluded paths (15 of 24) are either contradictions (`confident + surprised + confirmed`), no-update flows (`* + bored + stubborn/enlightened/confused`), or violate the definition of `confirmed` (which requires an `uncertain` prior).

---

## 4. Sample plan

### Sanity (this PR's deliverable)

```
9 trajectories × 1 topic × 1 story  =  9 POS
1 NEG       × 1 topic × 1 story     =  1 NEG
─────────────────────────────────────────────
Total: 10 stories, ~5 min on H100
```

### Full run (after sanity passes)

```
8 normal trajectories × 5 topics × 5 stories       =  200 POS
1 oversampled #16     × 5 topics × 20 stories      =  100 POS
NEG                   × 5 topics × 10 stories      =   50 NEG
─────────────────────────────────────────────────────────────
Total: 350 stories, ~2-3 hours on H100
```

Per-stage-concept counts after stage-localized aggregation:

| Stage-Concept | Sources | Count |
|---|---|---|
| uncertain (P1) | #9,10,11,16 | 25+25+25+100 = **175** |
| confident (P1) | #1,2,3 | 75 |
| curious (P1) | #18,19 | 50 |
| surprised (P2) | #1,2,3,9,10,11,18,19 | 200 |
| **bored (P2)** | #16 | **100** (oversampled) |
| stubborn (P3) | #1,9 | 50 |
| enlightened (P3) | #2,10,18 | 75 |
| confused (P3) | #3,11,19 | 75 |
| **confirmed (P3)** | #16 | **100** (oversampled) |

Min 50, max 200 (4× ratio). Compare v2: 5–33, ratio 6.6×.

---

## 5. What this design does NOT solve (residual risks)

These are honest limitations — the new design moves us from "data not usable" to "data usable, with caveats":

1. **Subtle drift** — the model may write a paragraph that satisfies structure + banned-words check but is a "weak rendition" of the assigned concept. Not detectable without a semantic judge. Mitigation: rely on sample size; the *systematic* expression survives averaging.

2. **Late-layer contextual leakage** — by layer 30/36, P1 token activations have integrated downstream context (the model "knows" what trajectory it's writing). `v_uncertain` in different trajectories carries different anticipations. Mitigation: prefer earlier layers (10/20) for stage-isolated analysis; treat layer 30/36 results as composite-state vectors.

3. **Sample asymmetry by source-trajectory count** — `v_bored` and `v_confirmed` come exclusively from trajectory #16 (only one trajectory has `bored` discovery). They are "context-locked": these vectors represent "bored after uncertain" rather than "bored in general." This is intrinsic to the cognitive ontology.

4. **Within-story correlation** — each story contributes to 3 stage vectors; per-story noise is shared. n_effective < n_stories. Mitigation: for inferential statistics, treat per-story random effects.

5. **Concept word ban is leaky** — close synonyms ("tentative" ≈ uncertain, "perplexed" ≈ confused) carry similar activation signatures and aren't banned. Trying to ban them all produces awkward prose. Accepted limitation.

6. **Ontology assumption** — the prior/discovery/reaction split presumes Qwen's representation actually carves cognitive space this way. If not, no extraction design recovers a structure that isn't there. The pipeline is a falsifiable test of the ontology.

---

## 6. Implementation map

| File | Role |
|---|---|
| [inputs/cognitive_v3/trajectories.json](../inputs/cognitive_v3/trajectories.json) | 9 trajectories + per-stage concept definitions |
| [inputs/cognitive_v3/topics.txt](../inputs/cognitive_v3/topics.txt) | 5 information-processing scenarios |
| [inputs/cognitive_v3/pos_prompt.txt](../inputs/cognitive_v3/pos_prompt.txt) | POS template (3 stages explicitly pinned) |
| [inputs/cognitive_v3/neg_prompt.txt](../inputs/cognitive_v3/neg_prompt.txt) | NEG template (factual third-person) |
| [scripts/v3_validate.py](../scripts/v3_validate.py) | Pure validation primitives (stdlib-only) |
| [scripts/generate_trajectories_v3.py](../scripts/generate_trajectories_v3.py) | Generation + retry on validation failure |
| [scripts/sanity_check_v3.py](../scripts/sanity_check_v3.py) | Post-gen consistency report |

Extraction (stage-localized) and analysis scripts are deferred until the sanity test confirms generation quality. Adding them later requires modifying neither the prompts nor `extract_concepts.py` directly — a separate `extract_v3.py` will read the marker-tagged stories.

---

## 7. Sanity test workflow

```bash
# On the GPU box (RunPod / H100):
git fetch && git checkout f2-cognitive-v3
python scripts/generate_trajectories_v3.py --sanity
python scripts/sanity_check_v3.py runs/cognitive_v3_sanity

# Output:
#   runs/cognitive_v3_sanity/sanity_report.md
#   runs/cognitive_v3_sanity/validation_log.json
#   runs/cognitive_v3_sanity/summary.json
```

The report has three sections to read:

1. **Structural validation** — pass/fail breakdown, word counts per story
2. **Cross-trajectory consistency** — paragraphs grouped by stage-concept (e.g., all P1s with prior=uncertain side-by-side); read to check whether they "feel like the same prior"
3. **Full POS stories** — each trajectory's complete story for closer reading

Pass criterion: ≥ 8/9 POS stories pass structural validation, and the consistency check shows clear within-concept similarity and between-concept distinguishability. If pass, scale up to full run; if fail, iterate on the prompt template or stage definitions.
