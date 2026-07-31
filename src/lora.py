"""LoRA build / lambda scaling / save. PROJECT_CORE.md §6 Stage 3 (Path A).

Call `src.compat.apply()` before importing peft anywhere upstream of this
module -- the Kaggle torchao probe bug aborts injection for every layer
otherwise (see compat.py).

Lambda shipping contract (decided 2026-07-31, verified by live run -- see
handoff): a LoRA delta is Δ = (alpha / r) * B @ A, so scaling by λ is exactly
`alpha <- λ * alpha` with `B` left bit-identical to training. No SVD, no
reconstruction error. Three routes (in-memory `LoraLayer.scaling`,
multiplying `lora_B`, halving `lora_alpha`) were measured to agree to
0.000e+00 after save/reload; baking into `adapter_config.json`'s `lora_alpha`
is the one that needs no runtime hook to reproduce, so that is what
`save_with_lambda` writes.
"""

import json
from pathlib import Path


def build_lora_config(cfg):
    """cfg: src.config.Config. Never hardcode target_modules or rank -- both
    come from cfg.lora."""
    from peft import LoraConfig, TaskType

    return LoraConfig(
        r=cfg.lora.rank,
        lora_alpha=cfg.lora.alpha,
        target_modules=list(cfg.lora.target_modules),
        lora_dropout=cfg.lora.dropout,
        use_rslora=cfg.lora.use_rslora,
        task_type=TaskType.SEQ_2_SEQ_LM,
    )


def _lora_layers(model):
    """Every injected LoRA layer. MUST filter by type, not `hasattr(m, "scaling")`:
    `WhisperAttention` also has a `.scaling` attribute (`head_dim**-0.5`), and a
    hasattr check corrupts it -- this was caught live (192 layers on
    PhoWhisper-small, not the 228 a hasattr scan finds)."""
    from peft.tuners.lora import LoraLayer

    return [m for m in model.modules() if isinstance(m, LoraLayer)]


def set_lambda(model, lam: float) -> int:
    """Scale all LoRA layers' effective contribution to λ in-memory, via
    `layer.scaling`, not by touching weights. Returns the number of layers
    scaled, for the caller to assert against the expected count."""
    layers = _lora_layers(model)
    for layer in layers:
        for key in layer.scaling:
            base = layer.lora_alpha[key] / layer.r[key]
            if layer.use_rslora:
                import math
                base = layer.lora_alpha[key] / math.sqrt(layer.r[key])
            layer.scaling[key] = lam * base
    return len(layers)


def save_with_lambda(model, lam: float, out_dir: str | Path, tol: float = 1e-5) -> Path:
    """Bake λ into `adapter_config.json`'s `lora_alpha`, leave `lora_B` bit-identical
    to training, then reload and assert logits match the in-memory λ within `tol`
    (not `== 0`: λ values that are not exact binary fractions drift ~1e-7 after
    the save/reload round trip)."""
    import torch

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    n_scaled = set_lambda(model, lam)
    model.save_pretrained(out)

    cfg_path = out / "adapter_config.json"
    adapter_cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    adapter_cfg["lora_alpha"] = lam * adapter_cfg["lora_alpha"]
    cfg_path.write_text(json.dumps(adapter_cfg, indent=2), encoding="utf-8")

    _verify_saved_adapter(model, out, tol)
    return out


def _verify_saved_adapter(model, saved_dir: Path, tol: float) -> None:
    """Reload the saved adapter fresh and compare logits against the live
    in-memory model on a fixed dummy input. Any mismatch beyond `tol` means the
    save path silently changed behavior -- fail loud, do not ship."""
    import torch
    from peft import PeftModel

    device = next(model.parameters()).device
    dummy = torch.zeros(1, model.config.num_mel_bins, 3000, device=device)
    decoder_ids = torch.tensor([[model.config.decoder_start_token_id]], device=device)

    model.eval()
    with torch.no_grad():
        live_logits = model(input_features=dummy, decoder_input_ids=decoder_ids).logits

    base = model.get_base_model() if hasattr(model, "get_base_model") else model.base_model
    reloaded = PeftModel.from_pretrained(base, saved_dir)
    reloaded.eval()
    with torch.no_grad():
        reloaded_logits = reloaded(input_features=dummy, decoder_input_ids=decoder_ids).logits

    max_diff = (live_logits - reloaded_logits).abs().max().item()
    if max_diff > tol:
        raise RuntimeError(
            f"saved adapter diverges from in-memory model: max logit diff "
            f"{max_diff:.2e} > tol {tol:.0e}"
        )
