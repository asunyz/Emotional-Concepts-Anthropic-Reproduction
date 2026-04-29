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


def _materialize_local_copy() -> Path:
    """Download + (re)quantize from the Hub once, save to `config.local_model_dir()`."""
    target = config.local_model_dir()
    if (target / "config.json").exists():
        return target
    print(f"First run: downloading {config.MODEL_ID}, quantizing ({config.QUANTIZATION}), "
          f"saving to {target}")
    kwargs = dict(device_map=config.DEVICE_MAP, torch_dtype=config.COMPUTE_DTYPE,
                  cache_dir=str(config.HF_CACHE))
    qcfg = config.build_quant_config()
    if qcfg is not None:
        kwargs["quantization_config"] = qcfg
    hf_model = AutoModelForCausalLM.from_pretrained(config.MODEL_ID, **kwargs)
    tok = AutoTokenizer.from_pretrained(config.MODEL_ID, cache_dir=str(config.HF_CACHE))
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


def load_model(model_path: str | None = None) -> LanguageModel:
    """Load (or reload) the configured model. Caller reuses the handle."""
    # If the pre-quantized model is already on disk at the canonical path,
    # use it directly. Avoids both the re-quantize round trip and a
    # symlink-induced tokenizer load failure observed on RunPod.
    canonical = Path("/workspace/models/qwen3.6-35b-a3b-nf4")
    if model_path:
        path = Path(model_path)
    elif canonical.exists() and (canonical / "config.json").exists():
        path = canonical
    else:
        path = _materialize_local_copy()
    kwargs = dict(device_map=config.DEVICE_MAP, torch_dtype=config.COMPUTE_DTYPE, dispatch=True)
    qcfg = config.build_quant_config()
    if qcfg is not None:
        kwargs["quantization_config"] = qcfg
    # Qwen2's tokenizer.json sometimes fails to convert through the fast
    # tokenizers backend on certain transformers/tokenizers version combos.
    # Pre-load the slow tokenizer and pass it explicitly so nnsight skips
    # its own auto-load (which would retry the failing fast path).
    tok = AutoTokenizer.from_pretrained(str(path), use_fast=False)
    model = LanguageModel(str(path), tokenizer=tok, **kwargs)
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
