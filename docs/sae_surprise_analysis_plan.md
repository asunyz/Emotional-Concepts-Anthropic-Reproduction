# SAE-based Surprise Feature Analysis — Plan

**Branch:** `asuka`
**Owner:** Yadong
**Goal:** Use Qwen-Scope SAE to find whether `surprise` is a single conflated direction (high-arousal + epistemic violation) or two separable feature subsets — and whether an *epistemic-surprise* feature exists that mean-difference could not detect.

---

## TL;DR

Mean-difference vectors gave us `cos(surprise, afraid) = 0.70` at layer 30 (see `docs/qwen3_moe_port_plan.md`). That's high enough to suspect the surprise direction is conflating "high-arousal negative" with "violated expectations." SAE features are sparse and decorrelated, so they can in principle pull these apart. We'll:

1. Run the existing 2880 stories through **Qwen3.5-35B-A3B-Base** (the SAE's native model) and pull per-token residuals at layers 30/31/35.
2. Encode through the Qwen-Scope TopK SAE (k=100, width=131K).
3. Rank features by `mean_act(surprise stories) − mean_act(neutral stories)`, take top-50 → filter to ~10.
4. For each candidate, check (a) is it equally high on `afraid` stories? (conflate test); (b) does it light up on 20 *epistemic violation* prompts vs matched controls? (epistemic test).

---

## Decisions made (frozen)

| # | Decision | Why |
|---|---|---|
| **A** | Use **`Qwen/Qwen3.5-35B-A3B-Base`** for Phase 1, not the project's `Qwen3.6-35B-A3B`. Re-extract residuals against this model. | The SAE was trained on 3.5-Base; using it on 3.6-Instruct mixes two distribution shifts (weight version + base→instruct). Reuse stories as text only (model-agnostic). Phase 2 (later) will repeat on 3.6-Instruct as a transfer-robustness experiment. |
| **B** | **Re-run forward pass** on all 2880 stories. | Existing `raw_concept/*.npy` are mean-pooled `[d]`; SAE encoder is non-linear so per-token activations are required. |
| **C** | **Detect hook point empirically** in Phase 1 smoke test. Try post-block, pre-block, mid-block; pick lowest reconstruction MSE. | Card just says "Residual stream" without specifying. Industry default is post-block (`hook_resid_post`), which matches our existing `model.layers[L].output[0]`. |
| **D** | **Top-K = 50, then filter to ~10**. | Gives slack for the conflate + epistemic filters. |
| **E** | **20 epistemic prompts, 5 sub-categories × 4 each**. | Statistical floor (paired t-test df=19, |t|>2.09 → p<0.05). Sub-categories distinguish *true* domain-general epistemic surprise from narrower "wrong-number" / "wrong-location" features. |

---

## Setup confirmed via web (2026-05-04)

**SAE:** [`Qwen/SAE-Res-Qwen3.5-35B-A3B-Base-W128K-L0_100`](https://huggingface.co/Qwen/SAE-Res-Qwen3.5-35B-A3B-Base-W128K-L0_100)
- TopK SAE, k=100, width 131072, d_model 2048
- One file per layer: `layer{n}.sae.pt` for n in 0..39
- Each file is a Python dict: `W_enc (131072, 2048)`, `W_dec (2048, 131072)`, `b_enc (131072,)`, `b_dec (2048,)`
- Encoding: `acts = topk(residual @ W_enc.T + b_enc, k=100)` (other dims zeroed)

**Base model:** [`Qwen/Qwen3.5-35B-A3B-Base`](https://huggingface.co/Qwen/Qwen3.5-35B-A3B-Base)
- 40 layers, hidden 2048, 256 experts (8 routed + 1 shared) → architecturally identical to Qwen3.6
- Vocab 248320

---

## File layout (additions only — do not touch existing runs)

```
inputs/epistemic/
    prompts.tsv                                # 20 (control, violation) pairs

runs/emotions_qwen35_BASE/                    # NEW — Phase 2 output
    layer_{30,31,35}/
        per_token/<story_stem>.npy             # [seq_len, 2048] fp16 (optional, compressed)
        sae_acts/<story_stem>.npz              # sparse [seq_len, 131072] (indices+values)
        agg_mean.npz                           # {story_stem: [131072]} mean act per story (post AVG_FROM_TOKEN)
        agg_max.npz                            # {story_stem: [131072]} max act per story

runs/sae_analysis/
    epistemic/
        per_token/<pair_id>_{c,v}.npy
        sae_acts/<pair_id>_{c,v}.npz

outputs/sae_surprise/
    candidates_layer_{L}.csv                   # top-50 features + scores
    final_layer_{L}.csv                        # filtered to verdict in {epistemic, conflate, other}
    examples_layer_{L}.html                    # max-activating story snippets
    summary.md
```

Models on disk:
- `/workspace/models/qwen3.6-35b-a3b-nf4/` — existing (kept untouched)
- `/workspace/models/qwen3.5-35b-a3b-base-nf4/` — NEW (Phase 1 downloads it)
- `/workspace/models/sae/qwen3.5-35b-a3b-base-w128k-l0_100/layer{30,31,35}.sae.pt` — NEW

---

## Code additions (no existing-file mutations beyond config + cv_utils opt-ins)

| File | Status | Role |
|---|---|---|
| `config.py` | extend | Add `BASE_MODEL_ID`, `SAE_REPO`, `SAE_LAYERS`, `SAE_LOCAL_DIR`, `SAE_K` |
| `cv_utils.py` | extend | `load_model()` accepts optional `model_id` override; new `extract_per_token_residuals(model, text, layers)` |
| `scripts/download_sae.py` | new | Pull only the 3 needed `layer{N}.sae.pt` files via `snapshot_download(allow_patterns=...)` |
| `scripts/sae_loader.py` | new | `load_sae(layer)` and `encode_topk(x, sae, k)` |
| `scripts/smoke_test_hook.py` | new | Phase 1 hook-point detector — 3 candidate hooks × reconstruction MSE |
| `scripts/extract_sae_acts.py` | new | Phase 2 — story → per-token residuals → SAE → store agg_mean / agg_max |
| `scripts/contrastive_features.py` | new | Phase 3 — surprise vs neutral mean-diff ranking |
| `scripts/epistemic_probe.py` | new | Phase 4 — paired test on epistemic prompts |

Total new code estimate: ~350 LoC.

---

## Phase plan with checkpoints

### Phase 1 — Setup & smoke test (~30–60 min)
1. Add config + write `download_sae.py` + `sae_loader.py`.
2. Download 3 SAE files (~3 GB total).
3. Trigger Qwen3.5-Base download via `cv_utils._materialize_local_copy` (~17 GB NF4).
4. Run `smoke_test_hook.py`: load model + layer-30 SAE, push a neutral sentence, compute normalized reconstruction MSE for `post-block` / `pre-block` / `mid-block`.
5. **🛑 CHECKPOINT**: lowest MSE < 0.20 → use that hook; if all > 0.5 → SAE not usable, halt and report.
6. Append Phase 1 results (chosen hook, all 3 MSEs) to changelog.

### Phase 2 — Extract SAE acts on stories (~2–3 h)
- 2880 stories × 3 layers, fp16. Per-story aggregates (mean + max over tokens ≥ AVG_FROM_TOKEN=50, with the same `len//2` fallback as `extract_concepts.py:188`).
- Sparse storage: TopK=100 means each token has ≤100 non-zero features → store as `(indices, values)` per token, then aggregate.

### Phase 3 — Contrastive ranking (~10 min)
- Per layer, score each feature: `score(f) = mean_surprise(f) − mean_neutral(f)`. Output top-50 to `candidates_layer_{L}.csv`.

### Phase 4 — Conflate + epistemic verification (~30 min)
- For each candidate:
  - **Conflate test**: `specificity = mean_surprise(f) / (mean_surprise(f) + mean_afraid(f))`. >0.7 = surprise-specific, ~0.5 = conflated.
  - **Epistemic test**: paired t on 20 prompts; report `mean_delta`, `t`, `hit_rate_overall`, `hit_rate_per_subcat`, `subcat_coverage`.
- `verdict`:
  - `subcat_coverage ≥ 4` AND `paired_t > 2.5` → **epistemic**
  - `specificity > 0.7` AND not epistemic → **surprise-affective**
  - otherwise → **conflate / other**
- Export top-2 features per verdict with max-activating story snippets (HTML, token-level highlighting).

---

## Epistemic prompts (the 20)

5 sub-categories × 4 pairs. Target token marked with `〈〉`.

### A. Numeric / quantitative (tests "wrong-number" feature)
1. Two plus two equals 〈four〉. / Two plus two equals 〈seventeen〉.
2. A triangle has 〈three〉 sides. / A triangle has 〈eleven〉 sides.
3. Humans have 〈two〉 lungs. / Humans have 〈seven〉 lungs.
4. A week has 〈seven〉 days. / A week has 〈nineteen〉 days.

### B. Geographic / spatial (tests "wrong-location" feature)
5. The capital of France is 〈Paris〉. / The capital of France is 〈Jupiter〉.
6. Tokyo is located in 〈Japan〉. / Tokyo is located in 〈Brazil〉.
7. The Eiffel Tower stands in 〈Paris〉. / The Eiffel Tower stands on 〈Mars〉.
8. The sun rises in the 〈east〉. / The sun rises in the 〈west〉.

### C. Category / taxonomy (tests "wrong-category" feature)
9. A whale is a 〈mammal〉. / A whale is a 〈vegetable〉.
10. A banana is a 〈fruit〉. / A banana is a 〈planet〉.
11. A chair is a piece of 〈furniture〉. / A chair is a piece of 〈weather〉.
12. Mount Everest is a 〈mountain〉. / Mount Everest is a 〈sandwich〉.

### D. Historical / cultural (tests "wrong-date/person" feature)
13. World War Two ended in 〈1945〉. / World War Two ended in 〈2010〉.
14. Albert Einstein was a 〈physicist〉. / Albert Einstein was a 〈hairdresser〉.
15. Shakespeare wrote 〈Hamlet〉. / Shakespeare wrote 〈Python〉.
16. Sherlock Holmes lived in 〈London〉. / Sherlock Holmes lived on 〈Saturn〉.

### E. Physical / sensory law (tests "physical-impossibility" feature)
17. Water boils at one hundred 〈degrees〉 Celsius. / Water boils at zero 〈degrees〉 Celsius.
18. Objects fall 〈down〉 due to gravity. / Objects fall 〈up〉 due to gravity.
19. Snow is 〈white〉. / Snow is 〈purple〉.
20. We see with our 〈eyes〉. / We see with our 〈elbows〉.

A "true epistemic-surprise" feature must light up in **≥ 4/5 sub-categories**, not just one.

---

## Risks & contingencies

| Risk | Mitigation |
|---|---|
| All three hook-point MSEs are huge | SAE may need a different normalization (e.g., RMSNorm pre-multiply). Read SAE repo's example code, retry. If still bad → halt, report, no Phase 2. |
| Per-token storage explodes (288 stories × 200 tokens × 131K features) | Sparse storage: only ≤100 non-zero per token. Aggregates (mean/max per story) are dense `[131072]` fp16 ≈ 256 KB each → a few GB total per layer. |
| Multi-token target words | For each pair, take the **first violation token**. Tokenize control + violation in advance, store target indices in `prompts.tsv`. |
| No feature passes (a) AND (b) | Negative result is fine — write up "epistemic surprise is not localized in this layer/SAE basis." Try concatenating features across layers 30+31+35 as fallback. |

---

## Out of scope for Phase 1

- Steering experiments using the discovered features (separate plan).
- Per-expert routing analysis (`docs/moe_interp_review.md` Adaptation 1; needs `extract_layer_activations_with_routing` rerun).
- Qwen3.6-Instruct transfer experiment (Phase 6).
