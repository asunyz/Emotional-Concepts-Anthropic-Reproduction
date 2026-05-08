# MCQ Experiment Results Memo (English)

**Experiment**: Multiple-Choice Question Surprise-Signal experiment
**Research question**: which cognitive vector responds **differently** when the model reads a correct vs incorrect answer? That difference is our candidate **surprise as learning signal**.
**Model**: Qwen3.6-35B-A3B (NF4 quantized)
**Probe vectors**: v3 cognitive concept vectors (9 vectors, layer 30, Method C)
**Date**: 2026-05-08
**Author**: meridah7

---

## 0. One-sentence headline

**Found it.**

We fed 40 common-sense questions through the model with each option (correct + 3 wrong) filled in as the answer, and measured 9 cognitive vector activations at the answer-token. The result:

- **Wrong answers** strongly activate `confused` (Cohen's d = **−1.52**) and `stubborn` (d = **−1.34**)
- **Correct answers** strongly activate `bored` (d = +0.66), `confident` (d = +0.62), `confirmed` (d = +0.43)

This is the candidate **surprise-as-learning-signal** the team has been looking for — a **quantifiable, readable** internal signal that distinguishes "model sees familiar information" from "model sees dissonant information".

---

## 1. Experimental Design (40 questions × 4 options = 160 forward passes)

### 1.1 MCQ dataset

40 hand-designed common-sense questions (designed for ~95% expected accuracy):

- 10 geography (capitals, oceans, continents)
- 10 science (chemical symbols, planet count, speed of light)
- 10 math (7+8, 12×12, square roots)
- 10 general (days/week, primary colors)

**Why simple questions**: we need to verify "the model HAS this knowledge", otherwise the model treating a wrong answer as correct (because it doesn't know) confounds the signal. Simple questions ensure "correct is correct, wrong is wrong, and model can tell".

### 1.2 Prompt template

For each question, build 4 prompts (one per option), format:

```
Question: What is the capital of France?
A) London
B) Paris
C) Rome
D) Berlin
Answer: B) Paris        ← this run: correct option
```

or:

```
Answer: A) London       ← this run: wrong option
```

### 1.3 What we measure

After each forward pass, capture **layer 30 hidden state at the last token** (i.e., internal state at the "Paris" / "London" position). Project onto each of the 9 v3 cognitive vectors → 9 cosine similarities.

Per question, we get:
- 1 "9-d projection vector when model read the correct answer"
- 3 "9-d projection vectors when model read a wrong answer"

40 questions × 4 options = **160 data points**.

### 1.4 Analysis

For each concept, split the 160 data points into two groups:
- **correct group** (n=40): projections when reading correct answers
- **incorrect group** (n=120): projections when reading wrong answers

Compute **Cohen's d** (standardized mean difference):

```
d > 0:  vector higher on correct answers
d < 0:  vector higher on incorrect answers
|d| ≥ 0.8: large effect
|d| ≥ 0.5: medium effect
|d| ≥ 0.2: small effect
```

---

## 2. Results

![Cohen's d ranking](../outputs/cognitive_v3_mcq/cohen_d_summary_last.png)

### 2.1 Numerical table

| Concept | Cohen's d | mean(correct) | mean(incorrect) | Interpretation |
|---|---|---|---|---|
| **confused** | **−1.52** 🔴 | −0.098 | −0.062 | Wrong answers **strongly** activate confusion |
| **stubborn** | **−1.34** 🔴 | −0.002 | +0.046 | Wrong answers **strongly** activate refusal stance |
| **bored** | +0.66 🟢 | −0.070 | −0.085 | Correct answers activate "no novelty" feeling |
| **confident** | +0.62 🟢 | +0.160 | +0.134 | Correct answers activate self-confidence |
| **curious** | +0.47 🟢 | +0.042 | +0.026 | Correct answers slightly activate curiosity |
| **confirmed** | +0.43 🟢 | — | — | Correct answers activate "expectation validated" |
| uncertain | +0.26 | — | — | Weak effect |
| surprised | −0.17 | — | — | Near-zero effect |
| enlightened | +0.04 | — | — | No effect |

### 2.2 Distribution scatter

![Strip plots](../outputs/cognitive_v3_mcq/strip_plot_per_concept_last.png)

Each panel is one concept:
- 🟢 Upper green dots = correct answers (n=40)
- 🔴 Lower red dots = incorrect answers (n=120)
- Dashed lines = group means

The `confused` and `stubborn` panels show **clearly separated red vs green clusters** — that's our judgment signal.

---

## 3. Interpretation

### 3.1 The model performs internal judgment

We never asked the model to answer; we only had it **read** the prompt. But the hidden state already shows **internal reaction** to the answer it just read:

| Model sees | Internal activation pattern | Plain-English |
|---|---|---|
| Correct answer | `confirmed` ↑, `bored` ↑, `confident` ↑ | "Yes, that's it; no surprise; I knew this" |
| Wrong answer | `confused` ↑, `stubborn` ↑ | "This doesn't fit; I reject it" |

**Even without generating any tokens, the residual stream already encodes a "cognitive judgment" signal.** This is the core value of the experiment.

### 3.2 Proposed surprise-score formula

Combining the two discovered groups of vectors:

```
surprise_score(t)  =   α · confused(t)
                     + β · stubborn(t)
                     − γ · bored(t)
                     − δ · confident(t)
                     − ε · confirmed(t)
```

- Coefficients can be fit via linear regression (target: "is this answer correct?" label)
- Compute per-token in real time
- High score = "the model thinks something is off here"

This delivers the **first computable real-time detector** for the "let the model actively discover new things" goal.

### 3.3 Why does `surprised` itself have d ≈ 0?

An interesting null finding — the `surprised` vector alone has very small effect size (d=−0.17).

Possible explanations:
- "surprised" in v3's training data corresponds to "unprocessable surprise" (mid-discovery), not "read a fact that contradicts knowledge"
- "Factual contradiction" is closer to cognitive **dissonance** (conflicts with prior belief) — exactly the `confused + stubborn` combination we found
- So surprise has two distinct meanings (cognitive vs informational), and our experiment **revealed the latter**

---

## 4. Limitations + Next Steps

### 4.1 Limitations

- **Only tested wrong-because-incorrect**: all incorrect answers are objectively wrong (London is not the capital of France). **Did NOT test "correct-but-novel"** — facts the model doesn't know but are true
- **Easy questions**: 40 are deliberately easy. **Did not test model on unfamiliar domains** — confused/stubborn responses might differ
- **Class imbalance**: 1:3 (40 vs 120), but Cohen's d normalizes this
- **No accuracy baseline**: we assumed model knows all 40 answers. Next run should first measure baseline accuracy ≥90% before running forward experiment

### 4.2 Next experiments (priority-sorted)

| Priority | Experiment | Expected finding |
|---|---|---|
| ★★★ | **Novel-but-correct test**: use word problems / reasoning questions where model doesn't know but answer is correct — test if `enlightened` activates here | Identify the "real learning opportunity" signal |
| ★★ | Verify model accuracy on 40 questions (baseline) | Rule out "model lacks knowledge" confound |
| ★★ | Fit a linear classifier on `surprise_score` formula, test AUC on holdout | Quantify signal reliability |
| ★ | Cross-layer comparison (layer 10/20/30/36) | Find which depth has strongest signal |
| ★ | Add neutral projection (the v3 gap), retest | See if effects become cleaner |

---

## 5. Resources & Files

| | Path |
|---|---|
| Question bank | `inputs/cognitive_v3_mcq/questions.json` |
| Experiment script | `scripts/run_mcq_experiment.py` |
| Analysis script | `scripts/analyze_mcq.py` |
| Raw projections | `outputs/cognitive_v3_mcq/raw_projections.json` |
| Main figure | `outputs/cognitive_v3_mcq/cohen_d_summary_last.png` |
| Strip plot | `outputs/cognitive_v3_mcq/strip_plot_per_concept_last.png` |

GPU time: ~1 minute (160 forwards on already-loaded model) + 1 min model load = **~2 min total**.
