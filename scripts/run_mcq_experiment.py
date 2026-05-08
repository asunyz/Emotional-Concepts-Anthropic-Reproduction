"""
MCQ surprise-signal experiment (D's suggestion).

For each multiple-choice question with 4 options:
  - Build a prompt with one option filled in as "Answer: X) text"
  - Forward through model, capture hidden state at the answer position
  - Project that hidden state onto each of the 9 v3 concept vectors

Group activations by correct vs incorrect answer. Hypothesis: some concept
vectors will respond differently to correct-answer states vs wrong-answer
states — those are candidate "surprise / learning" signal vectors.

Output:
  outputs/cognitive_v3_mcq/raw_projections.json
  outputs/cognitive_v3_mcq/per_question.json

Usage:
  python scripts/run_mcq_experiment.py \\
      --vec-dir runs/cognitive_v3_full/extractions/methodC_incontext \\
      --layer 30 \\
      --questions inputs/cognitive_v3_mcq/questions.json \\
      --output-dir outputs/cognitive_v3_mcq
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cv_utils import load_model  # noqa: E402


def resolve_hf_model(model):
    for attr in ("_model", "_module", "module"):
        cand = getattr(model, attr, None)
        if cand is not None and hasattr(cand, "generate"):
            return cand
    return model


def build_prompt(question: dict, chosen_letter: str) -> str:
    """Build "Question: ...\\nA)...\\nB)...\\nC)...\\nD)...\\nAnswer: X) text"."""
    lines = [f"Question: {question['question']}"]
    for letter in ("A", "B", "C", "D"):
        lines.append(f"{letter}) {question['options'][letter]}")
    chosen_text = question["options"][chosen_letter]
    lines.append(f"Answer: {chosen_letter}) {chosen_text}")
    return "\n".join(lines)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vec-dir", required=True,
                    help="e.g. runs/cognitive_v3_full/extractions/methodC_incontext")
    ap.add_argument("--layer", type=int, default=30)
    ap.add_argument("--questions", default="inputs/cognitive_v3_mcq/questions.json")
    ap.add_argument("--output-dir", default="outputs/cognitive_v3_mcq")
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--limit", type=int, default=None,
                    help="(debug) only first N questions")
    ap.add_argument("--n-pool-tokens", type=int, default=8,
                    help="how many trailing tokens to pool for the 'answer-pool' "
                         "projection (default 8 ≈ 'X) <answer text>')")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ----- Load v3 concept vectors at target layer -----
    vec_dir = Path(args.vec_dir) / f"layer_{args.layer}"
    vec_path = vec_dir / "concept_vectors_modeA.npz"
    mean_path = vec_dir / "mean.npy"
    vectors = {k: v.astype(np.float32) for k, v in np.load(vec_path).items()}
    mean = (np.load(mean_path).astype(np.float32)
            if mean_path.exists() else None)
    concepts = sorted(vectors.keys())
    print(f"loaded {len(concepts)} concept vectors at layer {args.layer}: {concepts}")

    # ----- Load questions -----
    qcfg = json.loads(Path(args.questions).read_text())
    questions = qcfg["questions"]
    if args.limit:
        questions = questions[:args.limit]
    print(f"loaded {len(questions)} questions")

    # ----- Load model -----
    print(f"loading model ...")
    model = load_model(args.model_path)
    hf_model = resolve_hf_model(model)
    tok = model.tokenizer
    target_layer = model.model.layers[args.layer]
    device = next(model.parameters()).device

    # ----- Forward hook to capture hidden states -----
    captured: list[torch.Tensor] = []

    def hook(_m, _i, output):
        h = output[0] if isinstance(output, tuple) else output
        captured.append(h.detach())  # (B, T, H)

    handle = target_layer.register_forward_hook(hook)

    results = []
    try:
        for q in tqdm(questions, desc="questions"):
            for letter in ("A", "B", "C", "D"):
                prompt = build_prompt(q, letter)
                captured.clear()
                inputs = tok(prompt, return_tensors="pt").to(device)
                with torch.no_grad():
                    hf_model(**inputs)
                h_full = captured[0][0].cpu().float().numpy()  # (T, H)

                # Two projections:
                # (1) last-token: hidden state at very last token of answer
                h_last = h_full[-1]
                # (2) answer-pool: mean over last n_pool_tokens (covers "X) <text>")
                n_pool = min(args.n_pool_tokens, h_full.shape[0])
                h_pool = h_full[-n_pool:].mean(axis=0)

                if mean is not None:
                    h_last = h_last - mean
                    h_pool = h_pool - mean

                proj_last = {c: cosine(h_last, vectors[c]) for c in concepts}
                proj_pool = {c: cosine(h_pool, vectors[c]) for c in concepts}

                results.append({
                    "question_id": q["id"],
                    "category": q["category"],
                    "question": q["question"],
                    "letter": letter,
                    "answer_text": q["options"][letter],
                    "is_correct": (letter == q["correct"]),
                    "n_tokens": int(h_full.shape[0]),
                    "proj_last": proj_last,
                    "proj_pool": proj_pool,
                })
    finally:
        handle.remove()

    # ----- Save -----
    out_file = out_dir / "raw_projections.json"
    out_file.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nwrote {out_file}")
    print(f"  {len(results)} (question, option) pairs")
    print(f"  {sum(1 for r in results if r['is_correct'])} correct, "
          f"{sum(1 for r in results if not r['is_correct'])} incorrect")


if __name__ == "__main__":
    main()
