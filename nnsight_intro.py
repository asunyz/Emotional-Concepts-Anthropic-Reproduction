"""
A tour of NNSight with Llama-3.1-8B (quantized — see config.py).

NNSight's core idea: you write PyTorch-looking code inside a `with model.trace(...)`
(or `model.generate(...)`) context. That code is *traced*, not immediately executed;
NNSight then runs the model once and splices your interventions in at the right
points. To pull a value out of the trace into Python, call `.save()`.

Run:
    conda activate nnsight
    huggingface-cli login            # Llama 3.1 is gated
    python nnsight_intro.py
"""

import torch

from cv_utils import load_model  # handles config, HF cache, download+save, nnsight wrap

# -- 1. Load the model --------------------------------------------------------
# All model / quantization / cache knobs live in config.py. `load_model`:
#   (a) first run: downloads from the Hub, quantizes per config.QUANTIZATION,
#       saves a pre-quantized copy under /mnt/e/models/<stem>-<quant>/.
#   (b) later runs: loads that folder directly — no Hub round-trip.
# Inference mode + grad-off are set inside.
model = load_model()

# -- 2. Print architecture ----------------------------------------------------
# Just `print(model)` — NNSight forwards __repr__ to the underlying nn.Module.
# This is how you discover module paths like `model.model.layers[15].mlp`.
print("=" * 60, "\nARCHITECTURE\n", "=" * 60, sep="")
print(model)
print("num layers:", len(model.model.layers))

PROMPT = "The capital of China is Beijing. The capital of France is"

# -- 3. Text completion (generation) ------------------------------------------
# `model.generate` works like HF's .generate(), but inside a trace context.
# Grab the produced token ids with `model.generator.output.save()`
# To sample ungreedy,   
# model.generate(PROMPT, max_new_tokens=20, do_sample=True, temperature=1.3, top_p=0.9, top_k=50)
print("\n" + "=" * 60, "\n3. TEXT COMPLETION\n", "=" * 60, sep="")
with model.generate(PROMPT, max_new_tokens=20, do_sample=False):
    out_ids = model.generator.output.save()
print(model.tokenizer.decode(out_ids[0].cpu()))

# -- 4. Read an inner activation ---------------------------------------------
# Every module exposes `.input` and `.output` proxies inside a trace. Saving
# the residual stream after layer 15 is a one-liner.
print("\n" + "=" * 60, "\n4. READ INNER ACTIVATION\n", "=" * 60, sep="")
with model.trace(PROMPT):
    # IMPORTANT: save in forward-pass order (self_attn → mlp → layer output),
    # otherwise nnsight raises OutOfOrderError.
    attn_15 = model.model.layers[15].self_attn.output[0].save()
    mlp_15 = model.model.layers[15].mlp.output.save()
    # layer output is a tuple (hidden_states, ...); take [0]
    resid_15 = model.model.layers[15].output[0].save()
print("resid @ layer 15:", resid_15.shape, resid_15.dtype)
print("mlp   @ layer 15:", mlp_15.shape)
print("attn  @ layer 15:", attn_15.shape)

# -- 5. Write (patch) an inner activation ------------------------------------
# Assigning to `.output` replaces the module's output for the rest of the forward.
# Here we zero-ablate the MLP at layer 15 and compare next-token logits.
print("\n" + "=" * 60, "\n5. SET / PATCH INNER ACTIVATION\n", "=" * 60, sep="")
with model.trace(PROMPT):
    clean_logits = model.lm_head.output[0, -1].save()

with model.trace(PROMPT):
    mlp_out = model.model.layers[15].mlp.output
    model.model.layers[15].mlp.output = torch.zeros_like(mlp_out)
    ablated_logits = model.lm_head.output[0, -1].save()

top_clean = clean_logits.argmax().item()
top_ablated = ablated_logits.argmax().item()
print("clean top token  :", repr(model.tokenizer.decode([top_clean])))
print("ablated top token:", repr(model.tokenizer.decode([top_ablated])))
print("logit delta L1   :", (clean_logits - ablated_logits).abs().mean().item())

# -- 6. Activation patching across prompts (a classic mech-interp move) ------
# Run prompt A, stash layer-15 residual, then run prompt B with A's residual
# spliced in. Requires two nested traces via `model.trace(...) as tracer`.
print("\n" + "=" * 60, "\n6. CROSS-PROMPT ACTIVATION PATCHING\n", "=" * 60, sep="")
src = "The capital of China is Beijing. The capital of France is"
dst = "The capital of China is Beijing. The capital of Japan is"
with model.trace(src):
    donor = model.model.layers[15].output[0].save()

with model.trace(dst):
    # In-place write into the hidden-states tensor — avoids rebuilding the
    # output tuple (which may also carry attn weights / KV cache).
    model.model.layers[15].output[0][:] = donor
    patched_logits = model.lm_head.output[0, -1].save()

print("dst + src-resid top token:",
      repr(model.tokenizer.decode([patched_logits.argmax().item()])))

# -- 7. Gather activations across all layers at once -------------------------
# Any Python control flow (list comprehensions, loops) works inside trace.
print("\n" + "=" * 60, "\n7. ALL-LAYER RESIDUAL STREAM\n", "=" * 60, sep="")
all_resid = []
with model.trace(PROMPT):
    for layer in model.model.layers:
        all_resid.append(layer.output[0].save())
print(f"captured {len(all_resid)} layers, each {all_resid[0].shape}")
# Cosine similarity of final-token residuals between adjacent layers:
# r may come back as [batch, seq, d] or [seq, d] depending on nnsight version —
# strip any leading batch dim, then take the last token.
final = torch.stack([(r[0] if r.ndim == 3 else r)[-1].float() for r in all_resid])
sims = torch.nn.functional.cosine_similarity(final[:-1], final[1:], dim=-1)
print("adjacent-layer cos-sim (first 5):", sims[:5].tolist())

# -- 8. Intervene during generation ------------------------------------------
# Inside `model.generate`, each forward pass is one token. Use `.all()` to apply
# an intervention on every generation step.
print("\n" + "=" * 60, "\n8. INTERVENE DURING GENERATION\n", "=" * 60, sep="")
with model.generate(PROMPT, max_new_tokens=10, do_sample=False):
    # scale up layer-20 residual on every generated token
    model.model.layers[20].all()
    model.model.layers[20].output[0][:] = model.model.layers[20].output[0] * 1.1
    scaled_ids = model.generator.output.save()
print("with layer-20 x1.1 :", model.tokenizer.decode(scaled_ids[0].cpu()))

print("\nDone. Good modules to poke at next:")
print("  model.model.embed_tokens         # token embeddings")
print("  model.model.layers[i].input_layernorm / post_attention_layernorm")
print("  model.model.layers[i].self_attn.{q,k,v,o}_proj")
print("  model.model.layers[i].mlp.{gate,up,down}_proj")
print("  model.model.norm, model.lm_head")
