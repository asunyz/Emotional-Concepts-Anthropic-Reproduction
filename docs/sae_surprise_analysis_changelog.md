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


