# Mechanistic Interpretability of MoE LMs — Literature Review and Adaptation Plan

Branch: `asuka` · Target model: Qwen3.6-35B-A3B (256 routed experts top-8 + 1 shared, 40 layers, hidden=2048).
Pipeline scope: `extract_concepts.py`, `steer.py`, `label_text.py`, `concept_similarity.py`, `concept_cluster.py`, `concept_vs_variable.py`.
Generated: 2026-04-30. **Note on sourcing:** live web search was unavailable during compilation; all citations come from training knowledge through Jan 2026. Verify arxiv IDs and venues before formal citation. Mark as TODO any line where we want to confirm a 2025+ paper.

---

## TL;DR — what changes when you go from dense Llama-3.1-8B to Qwen3.6-35B-A3B MoE

The Anthropic emotion-concepts methodology is **residual-stream-additive**: build a mean-difference vector, project off neutral PCs, add it back at a chosen layer to steer. This is architecture-agnostic in *principle*, but MoE introduces three non-obvious confounds:

1. **Routing flips.** Adding `α · v̂` to the residual at layer L perturbs the input to the router at layer L+1. The discrete top-k gate then re-routes some tokens to different experts. Steering effects are no longer "purely additive" — they are partly mediated by routing changes. The pipeline currently has no instrumentation to detect this.
2. **Concept-vector heterogeneity.** Each token's residual at layer L is a sum of contributions from a *different* 8-of-256 expert subset (plus the shared expert). The mean-difference direction is therefore an average over subspaces that may be only partially aligned. Per-prompt variance of the concept vector should be higher than on dense models; bootstrapped CIs on `concept_similarity.py` heatmaps will quantify this.
3. **Shared expert as a load-bearing piece.** DeepSeek-MoE / Qwen3-MoE put one always-on expert per layer. It sees every token and likely carries general affect-modulation features. Standard MoE interp methods (per-expert ablation, gate intervention) treat it as one-of-N; for our pipeline it should be treated as a separate object.

Net assessment: residual-stream mean-diff vectors will probably still work (Mixtral sentiment probes work; OLMoE residual probes work), but reproducibility, stability, and *mechanistic claims* about "what the steering is doing" need MoE-specific controls. The good news is that hidden=2048 is small for a 35B model — the residual stream is a tight bottleneck, which is *favorable* for residual-stream interpretability.

The current pipeline extracts and steers at `model.model.layers[L].output[0]` (post-block, post-residual-add) — see `cv_utils.py:76-87`, `extract_concepts.py:158-169`, `steer.py:86-88`. This is the right default for a black-box MoE-as-MLP-replacement view, but it forecloses the diagnostics in §C and §E below until we add pre-MoE / per-expert / router hooks.

---

## A. Foundational MoE interpretability

### A.1 Routing analysis and expert specialization

- **Zoph et al., "ST-MoE: Designing Stable and Transferable Sparse Expert Models"** (arXiv:2202.08906, 2022). Encoder experts specialize on shallow features (punctuation, conjunctions, numerals, proper nouns); decoder experts specialize less cleanly. Specialization is *position*- and *token*-driven, not topic-driven, in the encoder. Z-loss stabilizes routing.
  *Transfer:* Qwen3-MoE is decoder-only — expect *weaker* lexical specialization than ST-MoE encoders, and topic/affect signal smeared across many experts. Implication: per-expert concept vectors are likely under-determined; aggregated MoE-output extraction is the correct first cut, but **log routing during extraction** (see Adaptation 1).

- **Jiang et al., "Mixtral of Experts"** (arXiv:2401.04088, 2024). Routing § found *little* topical/domain specialization — no "math expert," no "code expert." Routing correlates with syntactic/positional structure (consecutive tokens often share an expert; indented code routes consistently). Deeper layers showed slightly more semantic clustering.
  *Transfer:* don't expect "the sadness expert." The mean-diff vector captures a global *post-mixture* latent, so the residual-stream pipeline should still work, but the concept vector is a superposition over many experts' outputs — weakening per-expert ablation as a control.

- **Muennighoff et al., "OLMoE: Open Mixture-of-Experts Language Models"** (arXiv:2409.02060, 2024). With *fine-grained* experts (64 experts top-8) domain specialization emerges more strongly than in Mixtral. Vocabulary specialization is real for some experts. Routing saturates early in training and is stable after.
  *Transfer:* Qwen3.6-MoE has **256 routed top-8 + 1 shared** — even more fine-grained than OLMoE. Expect *meaningful* per-expert specialization, possibly enough that emotion signal partially localizes to a small expert subset. This is the strongest reason to add per-expert logging.

- **Dai et al., "DeepSeekMoE: Towards Ultimate Expert Specialization"** (arXiv:2401.06066, 2024). Introduces fine-grained + shared expert (the architecture Qwen3-MoE inherits). Argues the shared expert isolates "common knowledge" so routed experts are free to specialize. Removing the shared expert hurts general benchmarks more than removing routed experts.
  *Transfer:* the **shared expert is the most likely home of global affect-modulation features** — it sees every token. (a) The residual delta you add in steering will *always* pass through the shared expert at the next layer; (b) per-expert SAEs/probes should treat it as a separate object; (c) it is plausible that ablating the shared expert kills emotion steering more than ablating any routed subset. (See Adaptation 5.)

### A.2 Expert ablation / patching

- **Lo et al., "A Closer Look into Mixture-of-Experts in Large Language Models"** (arXiv:2406.18219, 2024). Systematic expert ablation on Mixtral, DeepSeekMoE, Qwen-MoE. Most experts are individually ablatable with small perplexity loss; a small subset is critical. Routing decisions are similar across architectures despite differing aux losses; gate weights are partially redundant.
  *Transfer:* directly motivates the **expert-ablation steering control** — for each emotion direction, ablate top-k frequently-routed experts and re-test the steering effect. If steering survives, the concept is genuinely residual-stream-level (good for the Anthropic methodology); if it collapses, the concept lives in a small expert circuit.

- **Lu et al., "Not All Experts are Equal" / Expert Pruning literature** (arXiv:2402.14800, 2024). ~30–50% of experts can be pruned per layer with minor capability loss, but the *identity* of prunable experts is task-dependent.
  *Transfer:* warns that "concept lives across experts" is task-conditional. For emotion stories vs. math, the active expert set differs — stratify analysis by task (which we already do via the `concept_topics.txt`/`neutral_topics.txt` split, but never quantify in expert space).

### A.3 Router as a circuit

- **Chi et al., "On the Representation Collapse of Sparse Mixture of Experts"** (arXiv:2204.09179, NeurIPS 2022). Router projection matrix exhibits low-rank structure; tokens cluster in a low-dimensional subspace of router input.
  *Transfer:* directly testable — compute the cosine of each emotion concept vector with the top right-singular vectors of `model.model.layers[L+1].mlp.gate.weight`. If concept vectors strongly project onto router space, our steering is *partly* a router intervention. (Adaptation 7.)

- **Csordás et al. "MoEUT"** (NeurIPS 2024); **Puigcerver et al. "From Sparse to Soft Mixtures of Experts"** (arXiv:2308.00951, ICLR 2024). Soft MoE is differentiable; useful as a *control* model when you want gradient-based attribution. Not relevant for our Qwen3 target but worth noting if we ever want a small-scale soft-MoE baseline.

### A.4 Shared expert role

- **Qwen2-MoE technical report** (arXiv:2407.10671) and **Qwen3 technical report** (Qwen team, 2025). Shared expert carries general linguistic features; removing it disproportionately hurts low-resource languages and stylistic consistency.
  *Transfer:* for English emotion concepts, the shared expert is likely necessary but not sufficient. Keep its activations as a separate logging stream from the routed experts.

---

## B. Sparse autoencoders / dictionary learning on MoE

This area is genuinely thin as of early 2026.

- **Marks et al., "Sparse Feature Circuits"** (arXiv:2403.19647, 2024). Dense models. Methodology: train SAE on residual stream, identify causal circuits over SAE features.
  *Transfer:* the *methodology* is what to import. Mean-diff vectors are a coarse version of the same idea; eventually replace with SAE features (Adaptation 8).

- **Templeton et al., "Scaling Monosemanticity"** (Anthropic, 2024). Dense Claude 3 Sonnet. SAE features for emotions, sycophancy, deception. Methodologically the closest precedent to the emotion-concepts paper we are reproducing.
  *Transfer:* if we eventually want to compare mean-diff vectors to SAE features, train a residual-stream SAE at the same layer where steering works.

- **Lieberum et al., "Gemma Scope"** (arXiv:2408.05147, 2024). Dense Gemma 2 SAE suite. JumpReLU/top-k recipe is what people now apply to MoE.

- **MoE-SAE work emerging in 2024-2025 (verify with current search):** the empirical pattern is — a single SAE on the **post-MoE residual** works similarly to dense, but features are noisier; **per-expert SAEs** give cleaner per-expert features but need much more data per expert (each routed expert sees only ~3.1% of tokens at top-8/256).
  *Transfer:* a single residual-stream SAE *just after the MoE block* is the practical first step; per-expert SAEs are a research project of their own (and per-expert SAEs on the *shared* expert are easy because it sees every token).

- **Crosscoders** (Anthropic, "Sparse Crosscoders for Cross-Layer Features," 2024). One SAE that reads/writes across multiple sites.
  *Transfer:* an MoE-aware crosscoder reading from `pre-MoE residual + each expert's output + post-MoE residual` would directly expose where each feature is "computed." Most natural MoE-aware extension; flag as a follow-up paper.

- **Transcoders** (Dunefsky et al., arXiv:2406.11944, 2024). Replace an MLP with a wider sparse MLP for analysis.
  *Transfer:* the natural MoE analog is to replace the entire MoE block with a transcoder; preserves the I/O contract but bypasses the gate, giving a differentiable interpretable surrogate.

---

## C. Steering / activation engineering on MoE

### C.1 Methods our pipeline already uses (mostly dense)

- **Turner et al., "Activation Addition / Steering Language Models With Activation Engineering"** (arXiv:2308.10248, 2023). Direct ancestor of `steer.py`.
- **Rimsky et al., "Steering Llama 2 via Contrastive Activation Addition" (CAA)** (arXiv:2312.06681, 2023). Mean-diff over contrastive pairs at one layer — what `extract_concepts.py` does.
- **Zou et al., "Representation Engineering: A Top-Down Approach to AI Transparency" (RepE)** (arXiv:2310.01405, 2023). PCA over contrastive activations; reading + control vectors. Includes an emotion-control demo on Llama-2 — direct prior art.
- **Li et al., "Inference-Time Intervention (ITI)"** (arXiv:2306.03341, NeurIPS 2023). Per-attention-head intervention.
  *Transfer:* attention is dense in Qwen3-MoE, so ITI transfers directly. Useful as an *MoE-architecture-independent control* — if a head-edit-based steering works, the residual-stream-vs-MoE concern is moot for that intervention.

### C.2 MoE-specific steering

The published literature here is sparse and most of it is workshop-track. Concrete claims that have emerged (treat as preliminary, verify):

- **Routing-flip rate under residual steering.** When you add a residual delta of magnitude comparable to natural activation norms, **top-k routing changes for ~5–20% of tokens** at the next MoE layer, depending on layer depth and delta norm. Effects are larger in deeper layers where router projections have higher condition number.
  *Transfer:* this is the central concern. Our `--strengths -6,-3,0,3,6` sweep in `steer.py` is exactly the regime that will trigger routing flips — we just don't measure it. (Adaptation 2.)

- **Gate-logit interventions.** Add a vector directly to router logits to *force* a routing change. Useful as ablation: does steering *need* the routing flip, or is it incidental?
- **Expert-masked steering.** Add the residual delta only to tokens that route to a specific expert subset. Both are research-grade; cite carefully.

### C.3 Concept vector stability on MoE

Preliminary evidence (workshop track NeurIPS 2024 / ICLR 2025) that mean-diff concept vectors on Mixtral exhibit **higher variance across prompt subsets** than on Llama-2 of comparable scale, attributed to discrete routing introducing per-prompt heterogeneity in which expert subspace contributes.
*Transfer:* argues for (i) more contrastive pairs, (ii) bootstrapped CIs on `concept_similarity.py`, (iii) reporting concept-vector stability stratified by dominant routed expert. (Adaptation 3.)

### C.4 Frozen-gate path patching

Patching the residual stream after the MoE block is straightforward; patching *inside* the block requires holding the gate fixed across clean and corrupted runs. The de-facto standard is "frozen-gate patching": (a) cache gate decisions on clean run, (b) replay them on corrupted run while patching the chosen activation. (Adaptation 6.)

---

## D. Probing / linear concept directions on MoE

- **Tigges et al., "Linear Representations of Sentiment in LLMs"** (arXiv:2310.15154, 2023). Sentiment is linearly decodable in dense Llama-2 residual stream and the direction has causal effect under steering. Direct precedent for the emotion-concepts methodology.
  *Transfer:* the *linear decodability* claim is what to test on Qwen3-MoE. Prediction: probe accuracy will be *similar* to dense at the residual-stream level but *substantially worse* on individual expert outputs (each expert sees only its routed token subset).

- **Belrose et al., "Eliciting Latent Predictions from Transformers with the Tuned Lens"** (arXiv:2303.08112, 2023). Layer-wise probing.
  *Transfer:* train a tuned lens for Qwen3.6-MoE and check whether the layer at which emotion becomes linearly decodable matches the layer where steering works. Free sanity check on which `--layers` to hook in `extract_concepts.py`.

- **Park, Choe, Veitch, "The Linear Representation Hypothesis and the Geometry of Large Language Models"** (arXiv:2311.03658, ICML 2024). Geometric framework; introduces the *causal inner product*.
  *Transfer:* if MoE concept vectors look non-orthogonal in raw cosine but orthogonal in causal inner product, that's the canonical explanation. Our `concept_similarity.py` uses raw cosine — a causal-inner-product variant is one short PR away.

- **Anisotropy under MoE.** Workshop observations 2025: residual-stream activations on MoE models show **slightly higher anisotropy** than dense — the post-MoE residual sits in a cone shaped by the union of expert output ranges.
  *Transfer:* `extract_concepts.py:289-298` already centers by `mean_all` and projects off the top-k neutral PCs (variance fraction ≥ 0.5) — this is a partial isotropy correction. Worth checking what fraction of variance the discarded PCs account for *as a function of layer depth*: deeper MoE layers may need a larger `variance_fraction`.

---

## E. Circuit-level work on MoE

- **Conmy et al., "Towards Automated Circuit Discovery (ACDC)"** (arXiv:2304.14997, NeurIPS 2023). Dense baseline.
- **Syed et al., "Attribution Patching Outperforms ACDC" (EAP)** (arXiv:2310.10348, 2023); **Hanna et al., "EAP-IG"** (arXiv:2403.17806, 2024). Gradient-based circuit discovery.
  **MoE problem:** the discrete top-k gate has zero gradient w.r.t. unselected experts; gradient-based attribution under-estimates contributions of experts that *would have been* selected. Workarounds:
  - **Straight-through estimator** on the gate during attribution (treat top-k as identity in backward).
  - **Soft routing for analysis** — replace top-k with softmax during the attribution pass.
  - **Expected attribution** averaged over multiple corrupted runs to marginalize gate stochasticity.
  Active area; treat as research-grade.

- **IOI / induction-head replications on MoE.** Preliminary work on Mixtral and OLMoE for IOI, induction heads, and greater-than circuits in 2025 workshop tracks. Qualitative finding: **attention-head circuits transfer cleanly** (induction heads exist in MoE, IOI heads exist in Mixtral); **MLP-localized circuits transfer poorly** because the "MLP" is now 8 of 256 experts per token.
  *Transfer:* if we ever want circuit-level claims about emotion (beyond residual-stream steering), expect the attention side to look familiar and the MLP side to require frozen-gate patching.

---

## F. Qwen3-MoE-specific facts that matter for interp

- **Architecture:** 256 routed + 1 shared expert per layer, top-8 routing, **normalized router scores** (softmax over selected experts only, not over all 256), grouped expert load balancing. (Qwen3 technical report, 2025.)
  - Normalized scores mean unselected logits are not directly informative — only relative weights among the top-8 are meaningful when reading router output. Be careful when interpreting raw router logits.
  - 256 experts → each routed expert sees ~`8/256 = 3.1%` of tokens; per-expert SAE training needs ~30× more raw tokens than dense.
  - Hidden size 2048 is small for 35B params → residual stream is a tight bottleneck → *favorable for residual-stream interpretability* (concepts have less room to hide).

- **Qwen3 vs DeepSeek-V2/V3:** DeepSeek uses MLA (multi-head latent attention) alongside MoE; Qwen3 uses standard GQA, so attention-side methods (ITI, attention patching) transfer more cleanly to Qwen3 than to DeepSeek.

- **Dedicated Qwen3-MoE interpretability publications** (as of early 2026): I am not aware of any. The closest analogs are OLMoE expert-analysis papers and Mixtral routing analyses. **This project would be one of the early ones — a contribution opportunity.**

---

## G. Emotion / affect representations in LLMs

- **Anthropic, "Emotion Concepts and their Function in a Large Language Model"** (transformer-circuits.pub, 2026). The paper being reproduced. Llama-3.1-8B; mean-diff concept vectors; projection-based token-level analysis (analogous to `label_text.py`); causal validation via steering. Key claims to replicate on Qwen3-MoE: (i) emotion concepts are linearly represented in residual stream; (ii) concepts cluster by valence/arousal; (iii) steering produces coherent affective shifts; (iv) concept directions partially explain refusal behavior.
- **Tigges et al.** (above) — sentiment as the predecessor.
- **RepE emotion vectors (Zou et al.)** — earlier emotion-direction probing.
- **Hendrycks et al., "Discovering Latent Knowledge Without Supervision"** (arXiv:2212.03827, CCS 2022). Probe-stability concerns transfer to MoE.
- **MoE replications of emotion probing:** to my knowledge none published as of early 2026. This is the user's contribution gap.

---

## H. Tooling

- **NNSight** (Fiotto-Kaufman et al., 2024, ndif.us). Already in use. Module access for Qwen3-MoE:
  - `model.model.layers[L].mlp` — the MoE block as a whole.
  - `model.model.layers[L].mlp.gate` — router (verify with `diagnose_qwen.py`).
  - `model.model.layers[L].mlp.experts[i]` — individual experts.
  - **Caveat:** some HF MoE implementations dispatch experts via index-gather kernels that bypass per-expert `forward`. Hooks on `experts[i]` may not fire. **Verify with a tiny test** before relying on per-expert hooks: trace a forward and assert the saved tensor is non-empty.
  - Shared expert is typically `model.model.layers[L].mlp.shared_expert` in Qwen3 conventions — confirm against `diagnose_qwen.py` output.

- **TransformerLens.** MoE support added piecemeal through 2024-2025 (Mixtral, OLMoE). For Qwen3-MoE specifically: check `HookedTransformer.from_pretrained` registry; may need a custom config. TL exposes `blocks.{L}.mlp.hook_post`; for MoE there are `hook_expert_outputs` and `hook_router_logits`.

- **Pyvene** (Wu et al., 2024). Intervention library; generic but you'd write custom intervention objects for router / per-expert outputs.

- **Specialized MoE-interp libs** (`moe-explorer`, `moe-lens`, etc., emerging in 2025): nascent, verify maintenance.

**Concrete next step:** add to `diagnose_qwen.py` a small block that prints whether `with model.trace(text): r = model.model.layers[L].mlp.gate.output.save()` returns a `[batch, seq, 256]` tensor. If yes, every MoE-aware adaptation below is unblocked.

---

## Synthesis: what the literature says about our specific pipeline

1. **Residual-stream mean-diff vectors will probably work** — post-MoE residual is still a sum, and linear concepts survive in MoE residuals (Mixtral sentiment, OLMoE probes).
2. **But concept-vector stability will be lower** than dense Llama-3.1-8B due to per-prompt routing heterogeneity.
3. **Steering will partially flip routing.** Real confound; should be measured, not assumed away.
4. **The shared expert is load-bearing** for any global feature including emotion; analyze separately.
5. **Per-expert specialization is real but mild** in Mixtral, possibly stronger in 256-expert Qwen3-MoE; worth checking whether emotions correlate with a small expert subset.

---

## Prioritized adaptations for our pipeline

Ordered by payoff per cost. File references are to the current `asuka` branch.

### 1. Log routing during extraction *(cheap, foundational)*
**What:** in `extract_concepts.py` and `cv_utils.extract_layer_activations`, alongside residual activations save `gate_output` (`[seq, n_experts]`) and `top_k_indices` (`[seq, k]`) for every MoE layer of interest. One extra `.save()` per layer in the NNSight trace.
**Cost:** hours.
**Unblocks:** Adaptations 2, 3, 7.
**Output:** new `raw_routing/<concept>-<tidx>-<sidx>.npz` next to `raw_concept/`.

### 2. Routing-flip-rate metric in `steer.py` *(cheap, directly addresses the dense-vs-MoE concern)*
**What:** for each steered example, run two forward passes — clean and steered. At every MoE layer below the steering site, record the fraction of tokens whose top-8 set changed and the Jaccard similarity of top-8 sets. Report as a primary table.
**Cost:** ~1 day.
**Why high priority:** this is the single most important MoE-specific control. Without it we can't make the claim "the steering is residual-stream-additive" — the alternative hypothesis "the steering works by reshaping routing at the next layer" is unaddressed.
**Output:** add `--diagnose-routing` flag to `steer.py` that emits a `routing_flips.json` per strength.

### 3. Stratified concept vectors by dominant routed expert *(medium, addresses stability)*
**What:** using routing logs from (1), partition contrastive examples by their dominant expert at the extraction layer. Compute a separate concept vector per stratum. Compare to the global mean-diff vector via cosine and via causal effect under steering. If strata vectors are nearly parallel, the global vector is faithful; if not, you've found expert-subspace heterogeneity.
**Cost:** few days.
**Payoff:** paper-worthy methodology contribution. Plot stratum-vector cosines as a per-layer matrix.

### 4. Pre-MoE vs post-MoE extraction comparison *(cheap, mechanistic)*
**What:** `extract_concepts.py` currently extracts from `model.model.layers[L].output[0]` (post-block, post-residual-add). Add parallel extraction at:
  - **Pre-MoE residual:** input to the MoE block (`mlp.input` or the residual-stream just before the MoE block).
  - **Post-MoE pre-residual:** raw output of the MoE block (`mlp.output`), before the residual add.
Compare the three concept vectors per layer.
**Hypothesis:** post-MoE vector gives the strongest steering; pre-MoE vector gives the cleanest reading direction. If true, steering and reading should hook *different* sites — currently we use the same site for both.
**Cost:** ~1 day refactor.
**Payoff:** locates where the concept is "computed" vs where it's "carried."

### 5. Shared-expert ablation control *(medium, mechanistic)*
**What:** add a steering mode that *zeroes the shared-expert output* during generation while applying concept-vector steering. If steering still works → concept lives in routed experts / residual stream. If it collapses → shared expert is essential. Mirror with "routed-experts ablation" (zero all routed, keep shared).
**Cost:** 2-3 days (need careful NNSight or HF forward-hook plumbing).
**Payoff:** cleanly answers DeepSeek-MoE-inspired "shared vs routed" question for emotion concepts. High publishability.

### 6. Frozen-gate path patching *(medium, foundational for circuits)*
**What:** implement a generation mode that caches gate decisions on a clean run and replays them on a corrupted run. Use this to do clean activation patching across emotion-concept positions.
**Cost:** ~1 week.
**Unblocks:** any future circuit work; required for gradient-free EAP-style analysis on Qwen3-MoE.

### 7. Concept-direction vs router-row-space alignment *(cheap analysis on top of (1))*
**What:** compute SVD of `model.model.layers[L+1].mlp.gate.weight` per layer. Compute cosine of each emotion concept vector with the top-k right-singular vectors. If concept vectors strongly project onto router space → our steering is partly a router intervention, and the routing-flip-rate from (2) should be high in correlated directions.
**Cost:** ~1 day.
**Payoff:** directly diagnoses a hidden mechanism; can be combined with (2) into a single figure.

### 8. Train a residual-stream SAE around one MoE block *(expensive, big payoff)*
**What:** small JumpReLU or top-k SAE on, say, layer 20 residual pre- and post-MoE. Identify SAE features correlating with each emotion concept; check whether the mean-diff vector is well-approximated by a sparse sum of SAE features.
**Cost:** 1–3 weeks compute + engineering.
**Payoff:** bridges to Anthropic's monosemanticity methodology. Makes the reproduction publishable as a *methods* paper rather than a pure replication.
**Optional follow-on:** per-expert SAE on the shared expert and on the top-3 most-active routed experts at the steering layer; compare features to the residual-stream SAE to localize where each emotion sub-feature is computed.

---

## Quick cross-references to existing code

| Concern | Current code | Adaptation |
|---|---|---|
| Post-MoE residual extraction | `cv_utils.py:76-87`, `extract_concepts.py:158-169` | Adaptations 1, 4 |
| Post-block residual steering | `steer.py:86-88` | Adaptations 2, 5, 6 |
| Cosine similarity over concept vectors | `concept_similarity.py:31-33` | Adaptation 7 (add causal-inner-product variant); bootstrap CIs |
| Anisotropy correction | `extract_concepts.py:289-298` (mean center + neutral PCA basis ≥50% variance) | Worth tuning `variance_fraction` per layer; check what fraction is explained by the discarded PCs |
| Token-projection visualization | `label_text.py:45-52` (signed projection, no cosine norm of H) | Add per-token routing-flip overlay if Adaptation 2 lands |
| Layer choice for hooking | `--layers 16,24` examples | Adaptation 4 + tuned-lens probe to confirm those are the right depths on Qwen3-MoE |

---

## Verification TODOs before formal citation

- [ ] Confirm arxiv IDs and venues for every paper named above (web search was unavailable at compile time).
- [ ] Find specific 2025 / 2026 papers backing the "5–20% routing-flip rate under residual steering" claim — currently paraphrased from a literature trend, not a single source.
- [ ] Confirm there is no published Qwen3-MoE-specific interpretability paper (search arxiv + transformer-circuits + ICLR/NeurIPS 2025-2026 proceedings).
- [ ] Check current state of NNSight / TransformerLens MoE support — versions move fast.
- [ ] Check the Anthropic emotion-concepts paper for any MoE-specific commentary added since first release.
