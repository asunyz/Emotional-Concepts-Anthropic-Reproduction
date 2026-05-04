"""Shared helpers: loading the quantized model and pulling layer activations."""
import os
import shutil
from pathlib import Path

import config  # single source of truth for model / quant / paths

# Point HF at the big disk *before* importing transformers/nnsight.
os.environ.setdefault("HF_HOME", str(config.HF_CACHE))
os.environ.setdefault("TRANSFORMERS_CACHE", str(config.HF_CACHE))
config.HF_CACHE.mkdir(parents=True, exist_ok=True)

import torch                                                # noqa: E402
from transformers import AutoTokenizer, AutoModelForCausalLM  # noqa: E402
from nnsight import LanguageModel                           # noqa: E402

# Blackwell (SM 12.0) workaround: torch 2.8's `_grouped_mm` kernel ships only
# a Hopper (SM 9.0) implementation, but transformers' `_can_use_grouped_mm`
# check passes on any SM ≥ 9.0 and then crashes at kernel call. Force the
# eager fallback so the MoE path works on newer GPUs.
import transformers.integrations.moe as _moe                # noqa: E402
_moe._can_use_grouped_mm = lambda *args, **kwargs: False


def _materialize_local_copy(model_id: str | None = None) -> Path:
    """Download + (re)quantize from the Hub once, save to `config.local_model_dir()`."""
    mid = model_id if model_id is not None else config.MODEL_ID
    target = config.local_model_dir(mid)
    if (target / "config.json").exists():
        return target
    print(f"First run: downloading {mid}, quantizing ({config.QUANTIZATION}), "
          f"saving to {target}")
    kwargs = dict(device_map=config.DEVICE_MAP, torch_dtype=config.COMPUTE_DTYPE,
                  cache_dir=str(config.HF_CACHE))
    qcfg = config.build_quant_config()
    if qcfg is not None:
        kwargs["quantization_config"] = qcfg
    hf_model = AutoModelForCausalLM.from_pretrained(mid, **kwargs)
    tok = AutoTokenizer.from_pretrained(mid, cache_dir=str(config.HF_CACHE))
    # Drop the fp16 cache before writing the quantized copy: keeping both
    # spikes disk to ~88 GB on a 100 GB volume and the save fails midway.
    # Weights are already on GPU, cache files aren't needed anymore.
    shutil.rmtree(config.HF_CACHE, ignore_errors=True)
    target.mkdir(parents=True, exist_ok=True)
    hf_model.save_pretrained(str(target))
    tok.save_pretrained(str(target))
    del hf_model, tok
    torch.cuda.empty_cache()
    return target


def load_model(model_path: str | None = None,
               model_id: str | None = None) -> LanguageModel:
    """Load (or reload) the configured model. Caller reuses the handle.

    `model_path` (explicit local dir) takes priority. Otherwise, materialize
    `model_id` (defaulting to `config.MODEL_ID`) under `MODELS_ROOT`. The
    `model_id` override lets SAE scripts load `BASE_MODEL_ID` without
    mutating global config.
    """
    path = Path(model_path) if model_path else _materialize_local_copy(model_id)
    kwargs = dict(device_map=config.DEVICE_MAP, torch_dtype=config.COMPUTE_DTYPE, dispatch=True)
    qcfg = config.build_quant_config()
    if qcfg is not None:
        kwargs["quantization_config"] = qcfg
    model = LanguageModel(str(path), **kwargs)
    # Qwen3 saves lm_head as fp16 while the rest of the body stays bf16
    # (Qwen's native dtype), causing a dtype-mismatch at the final F.linear.
    # Cast any leftover fp16 non-quantized weights to bf16 to unify.
    for _, mod in model.named_modules():
        if 'Linear4bit' in type(mod).__name__ or 'Params4bit' in type(mod).__name__:
            continue
        for _, p in mod.named_parameters(recurse=False):
            if p.dtype == torch.float16:
                p.data = p.data.to(torch.bfloat16)
    model.eval()
    torch.set_grad_enabled(False)
    return model


def extract_layer_activations(model, text: str, layers: list[int]) -> dict[int, torch.Tensor]:
    """Forward `text` once and return {layer: [seq_len, d] float32 on CPU}."""
    saved: dict[int, "torch.Tensor"] = {}
    with model.trace(text):
        for L in sorted(layers):  # forward-pass order
            saved[L] = model.model.layers[L].output[0].save()
    out = {}
    for L, val in saved.items():
        t = val.detach().cpu().float()
        # nnsight sometimes returns [batch, seq, d], sometimes [seq, d].
        # Normalize to [seq, d] so callers don't have to care.
        if t.ndim == 3:
            t = t[0]
        out[L] = t
    return out


def extract_per_token_residuals(
    model, text: str, layers: list[int],
    hook_point: str = "post_block",
) -> dict[int, torch.Tensor]:
    """Forward `text` once; return per-token residual at each requested layer.

    Unlike `extract_layer_activations` (which is the same), this is named to
    flag intent: callers that need raw [seq, d] activations to feed an SAE
    encoder should use this and explicitly pick a hook point.

    hook_point options:
      "post_block" — model.model.layers[L].output[0]   (after attn + MLP residual adds)
      "pre_block"  — model.model.layers[L].input[0]    (block input residual)
      "mid_block"  — between attn-residual-add and MLP (post-attn, pre-MLP)
    """
    saved: dict[int, "torch.Tensor"] = {}
    with model.trace(text):
        for L in sorted(layers):  # forward-pass order, ascending
            block = model.model.layers[L]
            if hook_point == "post_block":
                saved[L] = block.output[0].save()
            elif hook_point == "pre_block":
                saved[L] = block.input[0].save()
            elif hook_point == "mid_block":
                # Qwen3 block: x = x + attn(norm(x)); x = x + mlp(norm(x))
                # post_attention_layernorm INPUT is the post-attn-residual.
                saved[L] = block.post_attention_layernorm.input[0].save()
            else:
                raise ValueError(f"Unknown hook_point={hook_point!r}")
    out: dict[int, torch.Tensor] = {}
    for L, val in saved.items():
        t = val.detach().cpu().float()
        if t.ndim == 3:
            t = t[0]
        out[L] = t
    return out


def extract_layer_activations_with_routing(
    model, text: str, layers: list[int],
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor | None]]:
    """Forward `text` once; return per-layer residual AND router gate logits.

    residuals: {layer: [seq, d] float32 on CPU} (same as extract_layer_activations).
    routing:   {layer: [seq, n_experts] float16 on CPU, or None for dense layers}.

    The router output is what the model's gate Linear produced *before* top-k
    selection / normalization, so downstream code can reproduce top-k itself
    or analyze full-distribution properties (entropy, expert load, etc.).
    For Qwen3-MoE the gate is `model.model.layers[L].mlp.gate` and outputs
    [bsz*seq or bsz, seq, n_experts]; we normalize to [seq, n_experts].

    Layers that lack `mlp.gate` (dense MLP) get None — callers must handle.
    """
    saved_res: dict[int, "torch.Tensor"] = {}
    saved_gate: dict[int, "torch.Tensor"] = {}
    layer_has_gate: dict[int, bool] = {}
    with model.trace(text):
        for L in sorted(layers):
            saved_res[L] = model.model.layers[L].output[0].save()
            mlp = model.model.layers[L].mlp
            if hasattr(mlp, "gate"):
                saved_gate[L] = mlp.gate.output.save()
                layer_has_gate[L] = True
            else:
                layer_has_gate[L] = False

    res_out: dict[int, torch.Tensor] = {}
    route_out: dict[int, torch.Tensor | None] = {}
    for L, val in saved_res.items():
        t = val.detach().cpu().float()
        if t.ndim == 3:
            t = t[0]
        res_out[L] = t

        if not layer_has_gate[L]:
            route_out[L] = None
            continue

        g = saved_gate[L].detach().cpu().float()
        seq = t.shape[0]
        if g.ndim == 3:
            g = g[0]                       # [bsz, seq, E] -> [seq, E]
        elif g.ndim == 2 and g.shape[0] != seq:
            # [bsz*seq, E] with bsz>1 — we use bsz=1, so this would mean a
            # reshape happened upstream; recover seq dim from residual.
            g = g.view(seq, -1)
        # else g is already [seq, E]
        route_out[L] = g.to(torch.float16)
    return res_out, route_out


def generate_story(model, prompt: str,
                   max_new_tokens: int = config.GEN_MAX_NEW_TOKENS,
                   temperature: float = config.GEN_TEMPERATURE,
                   top_p: float = config.GEN_TOP_P,
                   top_k: int = config.GEN_TOP_K,
                   repetition_penalty: float = config.GEN_REPETITION_PENALTY,
                   apply_chat_template: bool = True) -> str:
    """Generate from `prompt` and return ONLY the newly generated text.

    If the tokenizer defines a chat template (e.g. Llama-3.1-Instruct), wrap
    `prompt` as a user turn so the model follows instructions properly.
    """
    tok = model.tokenizer
    if apply_chat_template and getattr(tok, "chat_template", None):
        # `enable_thinking=False` suppresses Qwen3's <think> reasoning block;
        # tokenizers without that kwarg (e.g. Llama-3.1) silently ignore it.
        input_text = tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
    else:
        input_text = prompt
    prompt_len = tok(input_text, return_tensors="pt").input_ids.shape[1]
    with model.generate(input_text, max_new_tokens=max_new_tokens,
                        do_sample=True, temperature=temperature, top_p=top_p,
                        top_k=top_k, repetition_penalty=repetition_penalty,
                        pad_token_id=tok.eos_token_id):
        out = model.generator.output.save()
    completion = out[0, prompt_len:].cpu()
    return tok.decode(completion, skip_special_tokens=True).strip()
