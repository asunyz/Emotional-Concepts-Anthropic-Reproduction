"""
Pre-flight diagnostic for Qwen3.6-35B-A3B MoE port.

Run this once on the GPU machine BEFORE the smoke test. It verifies:
  1. Model loads (NF4 + MoE + NNSight all coexist)
  2. Tokenizer special tokens (BOS / EOS / im_start / chat template)
  3. Layer structure (model.model.layers exists, count is 48)
  4. NNSight trace returns the expected residual-stream shape
  5. Per-token residual norms — find any outlier tokens that should be
     dropped before averaging (the Llama BOS analog)
  6. MoE structure: gate hookable, shape [seq, n_experts], top-k routing
     histogram (required by extract_concepts.py routing-log pass)

Usage:
    python diagnose_qwen.py

Exit code 0 = all checks pass. Read each section's output by hand to decide
on AVG_FROM_TOKEN and any token-drop logic before running extract_concepts.py.
"""
import sys

import numpy as np

from cv_utils import load_model, extract_layer_activations_with_routing


def section(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def main() -> int:
    # ------------------------------------------------------------------
    section("[1/5] Loading model")
    # ------------------------------------------------------------------
    model = load_model()
    print(f"model class:        {type(model).__name__}")
    print(f"underlying module:  {type(model.model).__name__}")

    # ------------------------------------------------------------------
    section("[2/5] Tokenizer special tokens")
    # ------------------------------------------------------------------
    tok = model.tokenizer
    print(f"bos_token:  {tok.bos_token!r:25s} id={tok.bos_token_id}")
    print(f"eos_token:  {tok.eos_token!r:25s} id={tok.eos_token_id}")
    print(f"pad_token:  {tok.pad_token!r:25s} id={tok.pad_token_id}")
    print(f"unk_token:  {tok.unk_token!r:25s} id={tok.unk_token_id}")
    print(f"all special tokens map: {tok.special_tokens_map}")

    # Show what the chat template injects in front of an empty user message —
    # this is the prefix overhead added on top of any chat-templated text.
    chat_text = tok.apply_chat_template(
        [{"role": "user", "content": "Hello."}],
        tokenize=False, add_generation_prompt=True,
    )
    chat_ids = tok(chat_text, return_tensors="pt").input_ids[0]
    print(f"\nchat template wrap of user 'Hello.':  ({len(chat_ids)} tokens)")
    print(f"raw text: {chat_text!r}")
    for i, tid in enumerate(chat_ids):
        print(f"  [{i:3d}] id={int(tid):>7d}  {tok.decode([int(tid)])!r}")

    # ------------------------------------------------------------------
    section("[3/5] Layer structure")
    # ------------------------------------------------------------------
    n_layers = len(model.model.layers)
    print(f"total layers:       {n_layers}")
    print(f"expected (Qwen3.6-35B-A3B): 40")
    print(f"layer 0 type:       {type(model.model.layers[0]).__name__}")
    print(f"layer 0 submodules:")
    for name, child in model.model.layers[0].named_children():
        print(f"  .{name}  ({type(child).__name__})")

    if n_layers != 40:
        print(f"\nWARN: expected 40 layers, got {n_layers}. "
              f"Either the model ID is wrong or Qwen released a new size.")

    # ------------------------------------------------------------------
    section("[4/5] NNSight trace smoke test")
    # ------------------------------------------------------------------
    test_text = "The cat sat on the mat. It was a bright sunny day."
    test_layers = sorted({0, n_layers // 4, n_layers // 2,
                          (3 * n_layers) // 4, n_layers - 1})

    saved = {}
    with model.trace(test_text):
        for L in test_layers:
            saved[L] = model.model.layers[L].output[0].save()

    for L in test_layers:
        t = saved[L].detach().cpu().float()
        if t.ndim == 3:
            t = t[0]
        nrm = t.norm(dim=-1)
        print(f"layer {L:>2d}  shape={tuple(t.shape)}  "
              f"norm min/median/max = "
              f"{nrm.min().item():7.2f} / {nrm.median().item():7.2f} / "
              f"{nrm.max().item():7.2f}")

    # ------------------------------------------------------------------
    section("[5/5] Per-token residual norms (find outliers)")
    # ------------------------------------------------------------------
    # Two passes: (a) raw text without chat template, (b) chat-templated.
    # The concept-vector pipeline runs raw text; the neutral pipeline runs
    # chat-templated text. Both need to be checked for outliers.
    raw_sample = (
        "Once upon a time, there was a young girl named Lily. She lived in "
        "a small village by a quiet river. Every morning, she would walk to "
        "the market with her mother to buy fresh bread."
    )
    chat_sample = tok.apply_chat_template(
        [{"role": "system", "content": "You are a friendly assistant."},
         {"role": "user", "content": "Tell me about your day."},
         {"role": "assistant", "content": "I had a wonderful day exploring the city."}],
        tokenize=False,
    )

    mid_layer = n_layers // 2
    for label, text in [("RAW story text", raw_sample),
                        ("CHAT-TEMPLATED dialogue", chat_sample)]:
        print(f"\n--- {label}  (layer {mid_layer}) ---")
        ids = tok(text, return_tensors="pt").input_ids[0]
        with model.trace(text):
            h = model.model.layers[mid_layer].output[0].save()
        h = h.detach().cpu().float()
        if h.ndim == 3:
            h = h[0]
        norms = h.norm(dim=-1).numpy()
        median = float(np.median(norms))
        mean = float(norms.mean())
        print(f"tokens: {len(ids)}  hidden_dim: {h.shape[-1]}  "
              f"mean={mean:.2f}  median={median:.2f}")
        print(f"per-token norms (★ flags norm > 3× median):")
        for i, (tid, n) in enumerate(zip(ids, norms)):
            flag = "  ★ OUTLIER" if n > 3 * median else ""
            # Print first 30 + any flagged token + last 5
            if i < 30 or flag or i >= len(ids) - 5:
                print(f"  [{i:3d}] norm={float(n):7.2f}  "
                      f"{tok.decode([int(tid)])!r}{flag}")

    # ------------------------------------------------------------------
    section("[6/6] MoE structure: gate hook + routing histogram")
    # ------------------------------------------------------------------
    # Confirms (a) NNSight can hook `mlp.gate` on a real MoE layer despite
    # the eager-fallback _grouped_mm patch in cv_utils, (b) gate output has
    # the [seq, n_experts] shape that extract_layer_activations_with_routing
    # expects, and (c) top-k routing on a sample text isn't degenerate
    # (same expert every token would suggest a load-balance / config issue).
    moe_text = ("She felt a deep wave of joy rising as the music started, "
                "and her eyes shone with quiet wonder.")
    L = mid_layer

    # Print mlp submodule tree so we can sanity-check `gate`/`experts` paths.
    print(f"layer {L} .mlp submodules:")
    mlp = model.model.layers[L].mlp
    for name, child in mlp.named_children():
        # ModuleList prints its size; everything else prints its class name.
        extra = f" (n={len(child)})" if hasattr(child, "__len__") else ""
        print(f"  .{name}  ({type(child).__name__}){extra}")

    if not hasattr(mlp, "gate"):
        print(f"\nERR: layer {L} mlp has no .gate — routing logging will skip "
              f"this layer. If every layer reports this, the model isn't MoE "
              f"or the attribute path differs from `mlp.gate`.")
    else:
        try:
            res, routing = extract_layer_activations_with_routing(
                model, moe_text, [L])
        except Exception as e:
            print(f"\nERR: gate hook raised {type(e).__name__}: {e}")
            print(f"     If the message mentions a missing attr or a kernel "
                  f"path, check that _can_use_grouped_mm is patched (see "
                  f"cv_utils.py) and that the eager MoE forward calls "
                  f"`self.gate(...)` as a Python module call.")
            return 1

        h = res[L]
        g = routing[L]
        if g is None:
            print(f"\nERR: routing for layer {L} came back None despite "
                  f"`mlp.gate` existing. Inspect cv_utils.extract_layer_"
                  f"activations_with_routing for the dense-layer branch.")
            return 1

        n_experts = g.shape[-1]
        print(f"\nresidual shape:  {tuple(h.shape)}")
        print(f"gate logits:     {tuple(g.shape)}  dtype={g.dtype}  "
              f"(expected [seq, n_experts] with n_experts=256 for Qwen3.6-A3B)")
        if g.shape[0] != h.shape[0]:
            print(f"WARN: gate seq dim {g.shape[0]} != residual seq dim "
                  f"{h.shape[0]} — reshape logic in cv_utils may be off.")

        # Compute top-k from logits. Qwen3-MoE uses k=8.
        k = 8
        gf = g.float()
        topk_vals, topk_idx = gf.topk(k, dim=-1)            # both [seq, k]
        flat = topk_idx.flatten().tolist()
        # Count how many distinct experts saw at least one token in their top-k.
        unique_used = sorted(set(flat))
        # Per-expert frequency over all token-slot pairs (seq * k slots total).
        from collections import Counter
        counts = Counter(flat)
        top_used = counts.most_common(10)
        print(f"\ntop-{k} routing on a {h.shape[0]}-token emotional sample:")
        print(f"  distinct experts seen in top-{k}: "
              f"{len(unique_used)} / {n_experts}")
        print(f"  10 most-frequently-routed experts (out of "
              f"seq*k={h.shape[0]*k} routing slots):")
        for eid, cnt in top_used:
            pct = 100 * cnt / (h.shape[0] * k)
            print(f"    expert {eid:>3d}  count={cnt:>4d}  ({pct:5.1f}% of slots)")
        # Router entropy (over the top-k softmax) — degenerate if ~0.
        # Real Qwen3-MoE applies softmax over selected experts only.
        topk_w = topk_vals.softmax(dim=-1)                  # [seq, k]
        ent = -(topk_w * (topk_w + 1e-9).log()).sum(-1).mean().item()
        max_ent = float(np.log(k))
        print(f"  mean top-{k} entropy: {ent:.3f} nats  "
              f"(uniform = {max_ent:.3f}; near 0 = single expert dominates)")

    print("\n" + "=" * 70)
    print("Diagnostic complete.")
    print("=" * 70)
    print("""
Interpretation guide:
  - Section 2: if a token shows up with norm >> others in section 5, it
    should probably be dropped (analog of Llama's BOS handling in
    label_text.py / concept_vs_variable.py).
  - Section 3: layer count must match what you pass to --layers.
  - Section 4: shape should be (seq, hidden) after the [0] index
    normalization. Norms should look stable across layers (mid-layers
    a bit higher than early/late is normal).
  - Section 5 RAW: this is what extract_concepts.py Phase 2 actually
    feeds the model. The first ~10-20 tokens are usually scene setup;
    decide AVG_FROM_TOKEN by looking for where the story body begins.
  - Section 5 CHAT: this is what the neutral-PCA pipeline feeds. Check
    whether chat template tokens (like <|im_start|>) show outlier norms.
  - Section 6: gate output must be [seq, 256] for Qwen3.6-A3B. If the
    distinct-experts count is unusually small (<30) or one expert
    dominates >40% of slots, routing may be degenerate — investigate
    before trusting downstream stratification analyses.

If everything looks reasonable, proceed to the smoke test:
  python extract_concepts.py \\
      --concept-prompt inputs/emotions/concept_prompt.txt \\
      --concept-topics inputs/emotions/concept_topics.txt \\
      --concepts       inputs/emotions/concepts.csv \\
      --neutral-prompt inputs/emotions/neutral_prompt.txt \\
      --neutral-topics inputs/emotions/concept_topics.txt \\
      --layers 20 --n-stories 3 --task-label smoke_qwen36moe
""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
