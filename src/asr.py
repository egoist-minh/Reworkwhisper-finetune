"""Batched transcription for eval. PROJECT_CORE.md §2.1 (metrics.py's baseline
and gate calls), §6 Stage 1/4.

Eval needs no gradients, so weights load fp16 (measured: 3.1 GiB on
PhoWhisper-large) instead of the fp32/autocast setup training uses -- eval is
the dominant cost (~10,700 decodes for the full gate), so this is the
highest-value speedup available before the deadline.
"""

from pathlib import Path

import numpy as np

# Whisper's own anti-loop guard, off by default in transformers. A hypothesis
# that compresses better than this ratio is the signature of a repeated phrase,
# and generation is retried at the next temperature up. 1 of v5's 299
# cross-domain segments decodes as such a loop and that one segment alone adds
# 0.82 CER points (docs/v6-ondomain-plan.md §1b). These are Whisper's published
# defaults, not tuned here.
#
# NOT `no_repeat_ngram_size`: banning repeated n-grams costs ~3 CER points on
# Vietnamese, because it forces tone swerves on syllables that legitimately repeat.
FALLBACK_TEMPERATURES = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
COMPRESSION_RATIO_THRESHOLD = 1.35


def pick_dtype(device_index: int = 0):
    """fp16 vs bf16 by compute capability, never by `torch.cuda.is_bf16_supported()`:
    that call returns True on T4 (sm_75), which has no native bf16 -- it lies.
    Capability < 8.0 (Ampere) -> fp16."""
    import torch

    if not torch.cuda.is_available():
        return torch.float32
    major, _ = torch.cuda.get_device_capability(device_index)
    return torch.float16 if major < 8 else torch.bfloat16


def load_for_eval(base_model: str, adapter_dir: str | Path | None = None):
    """Load a Whisper(-family) checkpoint, optionally with a LoRA adapter, in
    eval dtype with grad disabled. `base_model` and `adapter_dir` are always
    config-driven -- never hardcode a checkpoint id here."""
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    dtype = pick_dtype()
    # Default load first. `use_safetensors=False` used to be passed unconditionally
    # to dodge transformers' safetensors auto-conversion probe (403 on repos with
    # discussions disabled, e.g. PhoWhisper-*), but it does not stop that probe --
    # TRANSFORMERS_AUTO_CONVERSION=0 plus compat.silence_hf_discussions_403_noise()
    # do, and callers must run compat.apply() before this (src/compat.py,
    # src/lora.py). What the flag DID do is hide safetensors from transformers, so a
    # safetensors-only repo raised `does not appear to have a file named
    # pytorch_model.bin or model.safetensors` -- how winhsss/Reworkwhisper-large-v5
    # failed on 2026-09-09. Kept only as a fallback for a .bin-only repo that the
    # default path somehow refuses.
    try:
        model = WhisperForConditionalGeneration.from_pretrained(base_model, torch_dtype=dtype)
    except OSError:
        model = WhisperForConditionalGeneration.from_pretrained(
            base_model, torch_dtype=dtype, use_safetensors=False)
    processor = WhisperProcessor.from_pretrained(base_model)

    if adapter_dir is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, processor


def fallback_kwargs(model, temperature, compression_ratio_threshold) -> dict:
    """The anti-loop generate() kwargs, but only on a transformers build whose
    Whisper `generate` names them. `generate` also takes `**kwargs`, so a build
    that predates temperature fallback would swallow them there and then raise
    from `_validate_model_kwargs` -- the named parameters are the only honest
    feature test."""
    import inspect

    params = inspect.signature(model.generate).parameters
    if "compression_ratio_threshold" not in params or "temperature" not in params:
        return {}
    return {"temperature": tuple(temperature),
            "compression_ratio_threshold": compression_ratio_threshold}


def transcribe_batch(model, processor, audios: list[np.ndarray], language: str = "vi",
                      num_beams: int = 1, batch_size: int = 8, desc: str | None = None,
                      temperature=FALLBACK_TEMPERATURES,
                      compression_ratio_threshold: float = COMPRESSION_RATIO_THRESHOLD) -> list[str]:
    """Transcribe `audios` (16 kHz mono float32 arrays) in chunks of `batch_size`.
    Returns hypotheses in the same order as `audios`. `desc` labels the progress
    bar (which split/lambda/tier is decoding) -- eval has no other feedback for
    what the module docstring calls ~10,700 decodes across a full gate run."""
    import torch
    from tqdm.auto import tqdm

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    extra = fallback_kwargs(model, temperature, compression_ratio_threshold)

    hyps: list[str] = []
    starts = range(0, len(audios), batch_size)
    for start in tqdm(starts, desc=desc or "transcribe", unit="batch", leave=False):
        chunk = audios[start:start + batch_size]
        inputs = processor(chunk, sampling_rate=16000, return_tensors="pt")
        features = inputs.input_features.to(device=device, dtype=dtype)
        with torch.no_grad():
            # language/task, not forced_decoder_ids: the latter is gone from both
            # generate()'s signature and GenerationConfig as of transformers 5.x,
            # and _validate_model_kwargs raises ValueError on kwargs the model
            # doesn't consume -- so passing it fails loud on any 5.x runtime.
            # language/task have been the supported path since 4.2x, so this
            # works on the older Kaggle images too.
            ids = model.generate(input_features=features, language=language,
                                  task="transcribe", num_beams=num_beams, **extra)
        hyps.extend(processor.batch_decode(ids, skip_special_tokens=True))
    return hyps
