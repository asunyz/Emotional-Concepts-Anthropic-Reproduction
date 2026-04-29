# F2 Cognitive Group (v2)

Input files for extracting a group of cognitive concept vectors using the existing Anthropic-style pipeline. Reuses `extract_concepts.py` with no code changes.

## Concept structure (9 concepts in 3 cognitive stages)

The 9 concepts map to a Bayesian inference flow: **prior → discovery → reaction**.

| Stage | Concepts | Cognitive role |
|-------|----------|----------------|
| Prior | curious, uncertain, confident | What the character believes/expects before evidence arrives |
| Discovery | surprised, bored | The character's reaction at the moment evidence arrives |
| Reaction | stubborn, enlightened, confused, confirmed | How the character processes the evidence afterward |

### Concept definitions

- **curious**: open expectation, no specific prediction
- **uncertain**: tentative belief with low confidence
- **confident**: strong specific prediction, expects a particular outcome
- **surprised**: outcome registers as unexpected and engaging
- **bored**: outcome is uninteresting, expected, or unremarkable
- **stubborn**: outcome contradicted prior, but the character refuses to update
- **enlightened**: outcome contradicted prior, and the character genuinely updates
- **confused**: outcome contradicted prior, but cannot be resolved
- **confirmed**: outcome aligned with a previously uncertain prior, now consolidated

## Files

- `concepts.csv` — 9 concept names, comma-separated (one row)
- `concept_topics.txt` — 5 discovery-focused topics (each topic supports all 9 cognitive variations)
- `concept_prompt.txt` — story prompt template; each story has a 3-stage narrative structure with the given concept as the dominant cognitive state
- `neutral_topics.txt` — 5 scientific topics (chemistry, physics, biology, geology, medicine) for the neutral baseline
- `neutral_prompt.txt` — generates purely objective scientific descriptions; no characters, no subjective cognitive states

## Why scientific objective neutral

The neutral baseline is designed to share the "narrative + discovery scenario" features of the concept stories without containing any subjective cognitive state. By projecting concept vectors off the PCA basis derived from these descriptions, we remove "narrative format" residuals while preserving the cognitive-state signal.

## Why discovery-focused topics

The previous emotion run (F1) used 24 topics that were predominantly negative-valence interpersonal scenarios (plagiarism, betrayal, partner secrecy). This biased the surprise vector toward affective negativity. The new topics are valence-neutral information-processing scenarios where any of the 9 cognitive states can naturally arise.

## Usage

```bash
python extract_concepts.py \
    --concepts inputs/cognitive_v2/concepts.csv \
    --concept-topics inputs/cognitive_v2/concept_topics.txt \
    --concept-prompt inputs/cognitive_v2/concept_prompt.txt \
    --neutral-prompt inputs/cognitive_v2/neutral_prompt.txt \
    --neutral-topics inputs/cognitive_v2/neutral_topics.txt \
    --output runs/cognitive_v2_qwen35_nf4 \
    --layers 10 20 30 36 \
    --n-stories 8 \
    --max-new-tokens 600 \
    --temperature 0.7 \
    --avg-from-token 50
```

Total stories: 9 × 5 × 8 = 360 concept stories + 5 × 8 = 40 neutral stories = 400 stories. Approx 2.5–3 hours on a single RunPod GPU.

## Analysis targets

After extraction, the 9 vectors per layer can support several analyses:

1. **Cosine matrix (9×9)** — pairwise similarities, with stage block structure highlighted
2. **PCA visualization** — color-coded by stage, looking for 3-cluster structure
3. **Layer scan** — geometry across layers 10/20/30/36 for stability check
4. **Vector arithmetic** — test whether `v_curious + v_surprised ≈ v_enlightened` and similar Bayesian-flow relationships
5. **Stage-conditioned steering** (optional, follow-up) — inject vectors at narrative positions to test causal control over cognitive trajectory
