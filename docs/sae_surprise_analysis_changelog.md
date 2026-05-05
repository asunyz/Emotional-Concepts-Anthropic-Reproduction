# SAE Surprise Analysis — Changelog

Append-only log of every working session on the SAE surprise-feature analysis.
Most recent entries at the bottom. Each session starts with a `## YYYY-MM-DD HH:MM` header.

Format per entry:
- **Phase**: which phase of the plan
- **What ran**: commands or scripts executed
- **Result**: numbers, file paths, decisions
- **Next**: what to do in the next session

See `sae_surprise_analysis_plan.md` for the master plan.

---

## 2026-05-04 — Session 0 (planning)

- **Phase**: pre-Phase 1 (planning)
- **What ran**: Read code on `asuka` branch; web-verified SAE on HF; designed 20 epistemic prompts.
- **Result**:
  - Confirmed SAE repo: `Qwen/SAE-Res-Qwen3.5-35B-A3B-Base-W128K-L0_100`. TopK k=100, width 131072, d_model 2048, one `.pt` per layer 0-39.
  - Confirmed Qwen3.5-Base architecture matches Qwen3.6: 40 layers, hidden 2048, 256 experts (8+1).
  - Hook point NOT documented on card → must detect empirically.
  - Plan frozen with 5 decisions (A–E in plan §"Decisions made").
- **Next**: Phase 1 — write code, download SAE + Qwen3.5-Base, run hook-point smoke test.

---

## 2026-05-04 — Session 1 (Phase 1, in progress)

- **Phase**: Phase 1 (Setup & smoke test)
- **What ran**:
  - Added `BASE_MODEL_ID`, `SAE_REPO`, `SAE_LAYERS=(30,31,35)`, `SAE_K=100`, `SAE_LOCAL_DIR`, `SAE_HOOK_POINT="auto"` to `config.py`.
  - Made `cv_utils.load_model()` and `_materialize_local_copy()` accept `model_id` override (default still `MODEL_ID`).
  - Added `cv_utils.extract_per_token_residuals(model, text, layers, hook_point)` with `post_block / pre_block / mid_block` support. `mid_block` reads `block.post_attention_layernorm.input[0]` (post-attn-residual, pre-MLP).
  - New scripts: `scripts/download_sae.py`, `scripts/download_base_model.py`, `scripts/sae_loader.py`, `scripts/smoke_test_hook.py`.
  - Wrote `inputs/epistemic/prompts.tsv` (20 pairs, 5 sub-categories × 4).
  - Downloaded SAE weights for layers 30/31/35 (~6.1 GB) — fast, no issues.
- **Disk-quota incident**: First Qwen3.5-Base download attempt failed with `Disk quota exceeded` — `/workspace` mount has a per-tenant quota despite showing 286T cluster-wide. Freed ~74 GB by deleting `/workspace/models/qwen3.6-35b-a3b-nf4` (re-downloadable for Phase 6) and the partial `/workspace/models/hf_cache`. Restarted base-model download.
- **Result so far**:
  - `/workspace/models/sae/qwen3.5-35b-a3b-base-w128k-l0_100/layer{30,31,35}.sae.pt` — 2148 MB each ✓
  - Qwen3.5-Base download in progress (background task `bijggbf7s`)
- **Next**: when base model finishes, run `scripts/smoke_test_hook.py --layers 30` and append the chosen hook + all 3 normalized MSEs here. Then commit Phase 1 work.

### Phase 1 smoke test (completed)

Probe text (49 tokens): one paragraph about the Industrial Revolution.

| hook | normalized_mse |
|---|---|
| **post_block** | **0.2870** |
| mid_block | 0.3376 |
| pre_block | 0.3448 |

- **Decision**: `SAE_HOOK_POINT = "post_block"` (committed to `config.py`). Matches `model.model.layers[L].output[0]` — same tensor the existing `extract_layer_activations` reads, so Phase 2 doesn't need any plumbing changes.
- **Caveat**: 0.287 is higher than the typical ≤0.15 you'd see on a SAE evaluated on its training distribution. Likely contributors: (i) model is loaded NF4-quantized vs SAE trained on bf16; (ii) only 49 probe tokens; (iii) BOS sink not stripped. Still preserves ~71% of activation variance, which is enough for contrastive feature ranking. Will revisit if Phase 3 yields no clear features.
- **Phase 1 done.** All scripts in `scripts/` for Phases 1-4 are written; config locked; SAE + base model on disk (~69 G total).
- **Next session**: Phase 2 — run `extract_sae_acts.py` on the 2880 stories under `runs/emotions_qwen35_nf4/stories/` for layers 30/31/35. Output to `runs/emotions_qwen35_BASE/`. Estimate ~2-3 h on this GPU. Then Phase 3 (contrastive ranking) and Phase 4 (epistemic probe + summarize).

---

## 2026-05-05 05:52 UTC — Session 2 (Phases 2 + 3)

- **Phase**: Phase 2 (extract) + Phase 3 (contrastive ranking)
- **What ran**:
  - `scripts/extract_sae_acts.py` on 2880 stories × layers 30/31/35 (Qwen3.5-Base, NF4, hook=`post_block`).
  - `scripts/contrastive_features.py --concept-a surprise --concept-b _neutral --top-k 50` for each of layers 30/31/35.
- **Result**:
  - Phase 2 finished in **20:42** wall (vs. 2-3 h estimate — much faster than expected; worth noting in case it indicates an issue, but n_skip=0 and outputs look healthy). 6 files written: `runs/emotions_qwen35_BASE/layer_{30,31,35}/agg_{mean,max}.npz`. Each story is a 131072-dim fp16 vector; 10 concepts × 288 stories = 2880 confirmed.
  - Phase 3 outputs: `outputs/sae_surprise/candidates_layer_{30,31,35}_surprise_vs__neutral.csv` (top-50 each).
  - Top features per layer (delta = mean_surprise − mean_neutral; nz = non-zero rate):
    - **Layer 30**: feat 110120 Δ=0.60 (nz 100/100% — ambient); feat 26093 Δ=0.55 (nz 100/9.4% — clean); feat 4689 Δ=0.33 (nz 100/1.0%).
    - **Layer 31**: feat 73152 Δ=0.49 (nz 100/12.5%); feat 85930 Δ=0.44 (nz 100/100%); feat 37706 Δ=0.31 (nz 100/0.3%).
    - **Layer 35**: feat 127526 Δ=0.65 (nz 100/0.7%); feat 23498 Δ=0.65 (nz 100/37%); feat 95838 Δ=0.64 (nz 100/17%); feat 94071 Δ=0.44 (nz 100/100%).
  - Pattern: at every layer the very top spot is shared between (a) "ambient" features that fire on everything but stronger on surprise (nz~100%/100%) and (b) clean discriminators (nz~100% / <15%). The clean discriminators are the candidates for the conflate + epistemic test in Phase 4.
- **Phase 4 (this session, 06:00 UTC)**:
  - `scripts/contrastive_features.py --concept-a afraid --concept-b _neutral --top-k 131072` for each of L30/31/35. Full table needed because `summarize_features` looks up surprise candidates by `feature_id` in the afraid CSV and falls back to `mean_afraid=0.0` on miss — top-50 alone would spuriously mark most surprise candidates as surprise-specific.
  - `scripts/epistemic_probe.py --candidates ..._surprise_vs__neutral.csv --top 50 --layer {30,31,35}` against `inputs/epistemic/prompts.tsv`.
  - `scripts/summarize_features.py --layer {30,31,35}`.
- **Phase 4 result — verdicts**:
  | layer | conflate | surprise_affective | epistemic | other |
  |---|---|---|---|---|
  | 30 | 46 | 2 | 0 | 2 |
  | 31 | 47 | 0 | 0 | 3 |
  | 35 | 47 | 0 | **1** | 2 |
  - **One epistemic feature**: **L35 feat 94071**. paired_t = **6.06** across 20 epistemic prompt pairs, subcat_coverage = **4/5** (≥0.75 hit-rate in numeric / geographic / category / historical), specificity = 0.498 (mean_surprise=1.04, mean_afraid=1.05 → fires equally on afraid stories at the story level — the conflate pattern). Δ_surprise_vs_neutral = 0.435 (mean_neutral=0.61).
  - **Interpretation**: matches the original hypothesis. Mean-diff couldn't isolate epistemic surprise because at the story level it's conflated with fear/negative-arousal (cos(surprise, afraid)=0.70 from earlier work). The SAE feature axis pulls "epistemic violation" apart: feat 94071 lights up cleanly on (control fact → violated fact) prompt pairs *beyond* what mean-diff could detect. Specificity ~0.5 says the same feature *also* fires on affective surprise/fear stories — so it is "shared by both", not "epistemic-only". Still, this is the first separable epistemic-surprise feature in this analysis.
- **Operational notes**:
  - Two concurrent epistemic probe loops ran (`logs/phase4_epistemic.log` and `logs/phase4_probe.log`). They competed for GPU on the 2nd model load. After both wrote `epistemic_layer_30.csv` (identical), one was killed; the other completed L31 + L35 cleanly.
  - `transformers.modeling_utils.caching_allocator_warmup` preallocates a ~63 GB block sized for bf16 weights even when loading NF4 — so two simultaneous Qwen-35B loads on an 80 GB H100 will OOM. Run epistemic probes one model load at a time.
- **Outputs**:
  - `outputs/sae_surprise/candidates_layer_{30,31,35}_surprise_vs__neutral.csv` (top-50 each)
  - `outputs/sae_surprise/candidates_layer_{30,31,35}_afraid_vs__neutral.csv` (full 131 072 rows each)
  - `outputs/sae_surprise/epistemic_layer_{30,31,35}.csv` (paired t per candidate)
  - `outputs/sae_surprise/final_layer_{30,31,35}.csv` (joined with verdict)
- **Next**: run `scripts/max_activating_examples.py --layer 35 --features 94071` (and for the L30 surprise_affective hits) to get HTML token-highlighted story snippets — needed to verify the epistemic interpretation by hand. Then write `outputs/sae_surprise/summary.md` and decide whether to extend (a) beyond top-50, (b) to the L30 surprise_affective features, or (c) to the Qwen3.6-Instruct transfer experiment in the plan.

### 06:15 UTC — manual inspection of feat 94071 (verdict refined)

- **What ran**:
  - `scripts/max_activating_examples.py --layer 35 --features 94071 --top-k 8` → `outputs/sae_surprise/examples_layer_35_feat94071.html` (gitignored).
  - Ad-hoc per-story argmax-token extraction over the top-8 stories.
  - Ad-hoc per-prompt argmax-token extraction over the 20 epistemic prompt pairs (where does the feature actually peak in each prompt?).
- **Story-level peaks** (top-8 max-activating stories, peak token shown in «»):
  | rank | story | peak token | context |
  |---|---|---|---|
  | 1 | surprise-22-1 | «counselor» | "Which counselor was it?" |
  | 2 | surprise-5-3 | «blood» | "the sudden roar of blood rushing" |
  | 3 | sad-14-9 | «click» | "the click of the lock was loud" |
  | 4 | indifferent-1-2 | «mail» | "to check if the mail had arrived" |
  | 5 | afraid-17-7 | «hum» | "the hum of the refrigerator" |
  | 6 | sad-22-3 | «grandmother» | "My grandmother used to sing it" |
  | 7 | inspired-3-1 | «Tears» | "Tears pricked at the corners" |
  | 8 | afraid-1-11 | «click» | "the click of the latch echoing" |
  Top-8 spans 5 different concepts (surprise×2, sad×2, afraid×2, indifferent, inspired). Peaks land on **concrete sensory/relational nouns at narrative inflection points** — not on emotion words. "Click" is the peak in two unrelated stories.
- **Prompt-level falsification** of the "violation noun is the peak" hypothesis:
  - Argmax token == divergent (substituted) target: control 1/20, violation 2/20. So the feature is *not* a discrete "wrong-word flag."
  - But activation **at the target position** is reliably higher in violation than control: 18/20 positive deltas, mean Δ=+0.90 (consistent with paired_t=6.06 from `epistemic_layer_35.csv`).
  - Argmax often lands on the *same* sentence-position token in both control and violation ("The", "Two", "whale", "Snow", "boils") → driven by sentence/template structure, not the specific lexical violation.
  - Biggest target-position deltas correlate with how impossible the substituted token is in context: "Jupiter" as France's capital (+1.50), "Einstein had hair" (+2.13), "weather is a vegetable/sandwich" (+1.28/+1.61).
- **Refined verdict**: feat 94071 is a **broad surprisal-modulated content feature**, not an "epistemic violation" detector. Always-on (control activations 0.5–2.0 are typical), magnitude scales with in-context token surprisal. The epistemic test passes because the substituted target token is high-surprisal — so the feature reads higher *at that position* — but the feature is a continuous "this token is unexpected" signal, not a discrete wrongness flag. Story-level peaks on `click`, `blood`, `Tears`, `hum` fit the same theory: locally high-surprisal narrative beats.
- **Implication for the headline claim**: the previous entry's "matches the hypothesis that mean-diff missed an epistemic-surprise direction the SAE basis can isolate" is *partially* right — the SAE basis does isolate a useful surprisal-related direction that mean-diff misses — but it's not specifically epistemic. The verdict pipeline's `epistemic` label is a measurement-level pass, not a mechanism-level claim. Plan §"verdict" thresholds (cov≥4 ∧ t>2.5) should be treated as a screening filter, not a typing decision.
- **Next**: revisit the L30 `surprise_affective` candidates (feats with specificity > 0.7) and rerun the manual peak-token inspection on those — they're the only candidates that survived the conflate-with-afraid test, and may yield a cleaner mechanism than the surprisal-feature shape we found at L35. Then decide on (a) extending beyond top-50, (b) the Qwen3.6-Instruct transfer experiment, or (c) summarizing and stopping.


