"""Batched transcription for eval. PROJECT_CORE.md §2.1 (metrics.py's baseline
and gate calls), §6 Stage 1/4.

Eval needs no gradients, so weights load fp16 (measured: 3.1 GiB on
PhoWhisper-large) instead of the fp32/autocast setup training uses -- eval is
the dominant cost (~10,700 decodes for the full gate), so this is the
highest-value speedup available before the deadline.
"""

from pathlib import Path

import numpy as np


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
    # use_safetensors=False doesn't stop transformers' safetensors auto-conversion
    # probe thread from firing (403 on repos with discussions disabled, e.g.
    # PhoWhisper-*) -- the traceback it used to dump is silenced by
    # compat.silence_hf_discussions_403_noise(). Callers must run compat.apply()
    # before this (see src/compat.py, src/lora.py).
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


def transcribe_batch(model, processor, audios: list[np.ndarray], language: str = "vi",
                      num_beams: int = 1, batch_size: int = 8) -> list[str]:
    """Transcribe `audios` (16 kHz mono float32 arrays) in chunks of `batch_size`.
    Returns hypotheses in the same order as `audios`."""
    import torch

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    forced_ids = processor.get_decoder_prompt_ids(language=language, task="transcribe")

    hyps: list[str] = []
    for start in range(0, len(audios), batch_size):
        chunk = audios[start:start + batch_size]
        inputs = processor(chunk, sampling_rate=16000, return_tensors="pt")
        features = inputs.input_features.to(device=device, dtype=dtype)
        with torch.no_grad():
            ids = model.generate(input_features=features, forced_decoder_ids=forced_ids, num_beams=num_beams)
        hyps.extend(processor.batch_decode(ids, skip_special_tokens=True))
    return hyps
