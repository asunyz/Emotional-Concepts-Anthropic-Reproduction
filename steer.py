"""
Activation steering: add `strength * unit(concept_vec)` to the residual stream
at a chosen layer, on every generated token, and compare completions across
strengths.

Usage:
    python steer.py \
        --prompt "Tell me about your day." \
        --concept-dir runs/emotions_8b_nf4 \
        --layer 16 --concept happy \
        --strengths -6,-3,0,3,6 \
        --output steered_happy.txt
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from cv_utils import load_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--concept-dir", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--concept", required=True)
    ap.add_argument("--strengths", required=True,
                    help="comma-separated floats, e.g. -6,-3,0,3,6")
    ap.add_argument("--max-new-tokens", type=int, default=150)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--no-chat-template", action="store_true",
                    help="skip the tokenizer's chat template (use for base models)")
    ap.add_argument("--output", default="steered.txt")
    args = ap.parse_args()

    strengths = [float(x) for x in args.strengths.split(",")]
    cv = np.load(Path(args.concept_dir) / f"layer_{args.layer}" / "concept_vectors.npz")[args.concept]
    cv_unit = cv / (np.linalg.norm(cv) + 1e-9)

    model = load_model()
    tok = model.tokenizer

    if not args.no_chat_template and getattr(tok, "chat_template", None):
        input_text = tok.apply_chat_template(
            [{"role": "user", "content": args.prompt}],
            tokenize=False, add_generation_prompt=True,
        )
    else:
        input_text = args.prompt
    prompt_len = tok(input_text, return_tensors="pt").input_ids.shape[1]

    # Place the steering vector on the same device/dtype as the model weights.
    target_param = next(p for p in model.parameters() if p.device.type == "cuda")
    cv_unit_t = torch.tensor(cv_unit, dtype=target_param.dtype, device=target_param.device)

    sections = [f"=== PROMPT ===\n{args.prompt}\n",
                f"=== CONCEPT: {args.concept} @ layer {args.layer} ===\n"]

    for s in strengths:
        delta = s * cv_unit_t  # [d]; broadcasts across [batch, seq, d]
        with model.generate(input_text, max_new_tokens=args.max_new_tokens,
                            do_sample=(args.temperature > 0),
                            temperature=args.temperature,
                            pad_token_id=tok.eos_token_id):
            # `.all()` applies the intervention on every forward pass (every
            # new token during generation), not just the prefill.
            model.model.layers[args.layer].all()
            h = model.model.layers[args.layer].output[0]
            model.model.layers[args.layer].output[0][:] = h + delta
            out = model.generator.output.save()

        completion = tok.decode(out[0, prompt_len:].cpu(), skip_special_tokens=True).strip()
        sections.append(f"\n--- strength = {s:+g} ---\n{completion}\n")
        print(f"strength {s:+g}: done")

    Path(args.output).write_text("".join(sections))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
