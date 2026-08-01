"""SFT training loop. PROJECT_CORE.md §6 Stage 2.

UNTESTED end-to-end (no torch/transformers on this dev machine -- see
handoff). Smoke-test on Kaggle before trusting this against the deadline;
the transformers 5.0.0 `Trainer(eval_dataset=dict)` multi-eval-set API this
relies on has not been exercised in this repo yet.

Checkpoint strategy: save to `checkpoints/best/` whenever `val_cer` improves.
Early stopping: 3 eval-round patience. OOD eval runs every eval round (not
just at the end) so forgetting is visible during training, not only after
(§0 problem 2).
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.metrics import score
from src.normalize import Normalizer


@dataclass
class WhisperCollator:
    """Pads input_features and labels separately -- they have unrelated
    padding conventions (mel frames vs token ids, -100 ignore-index)."""
    processor: object

    def __call__(self, batch: list[dict]) -> dict:
        import torch

        feats = self.processor.feature_extractor(
            [b["audio"] for b in batch], sampling_rate=16000, return_tensors="pt"
        )
        labels = self.processor.tokenizer(
            [b["text"] for b in batch], return_tensors="pt", padding=True
        )
        label_ids = labels.input_ids.masked_fill(labels.attention_mask.ne(1), -100)
        return {"input_features": feats.input_features, "labels": label_ids}


def _make_compute_metrics(processor, normalizer: Normalizer):
    def compute_metrics(pred) -> dict:
        pred_ids = pred.predictions
        label_ids = pred.label_ids.copy()
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        hyps = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        refs = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        hyps = [normalizer(h) for h in hyps]
        refs = [normalizer(r) for r in refs]
        result = score(refs, hyps)
        return {"cer": result["cer"], "wer": result["wer"]}

    return compute_metrics


def train(cfg, base_model, train_ds, val_ds, ood_ds, out_dir: str | Path):
    """cfg: src.config.Config. Returns the path to checkpoints/best/."""
    import torch
    from transformers import (Seq2SeqTrainer, Seq2SeqTrainingArguments,
                               WhisperProcessor, EarlyStoppingCallback)
    from peft import get_peft_model

    from src.lora import build_lora_config

    out = Path(out_dir)
    processor = WhisperProcessor.from_pretrained(cfg.base_model)
    model = get_peft_model(base_model, build_lora_config(cfg))
    model.print_trainable_parameters()
    if cfg.training.gradient_checkpointing:
        # Base model is entirely frozen except the adapter, so the graph's input
        # tensor has requires_grad=False -- gradient checkpointing then breaks
        # backprop ("element 0 of tensors does not require grad and does not
        # have a grad_fn") unless this is called. Standard PEFT + Trainer gotcha.
        model.enable_input_require_grads()

    normalizer = Normalizer(
        strip_punctuation=cfg.normalization.strip_punctuation,
        lowercase=cfg.normalization.lowercase,
        number_convention=cfg.normalization.number_convention,
        filler_tokens=cfg.normalization.filler_tokens,
    )

    args = Seq2SeqTrainingArguments(
        output_dir=str(out / "checkpoints"),
        per_device_train_batch_size=cfg.training.batch_size,
        per_device_eval_batch_size=cfg.eval.batch_size,
        gradient_accumulation_steps=cfg.training.grad_accum_steps,
        learning_rate=cfg.training.learning_rate,
        warmup_ratio=cfg.training.warmup_ratio,
        num_train_epochs=cfg.training.epochs,
        gradient_checkpointing=cfg.training.gradient_checkpointing,
        fp16=torch.cuda.is_available(),
        predict_with_generate=True,
        generation_num_beams=cfg.eval.num_beams,
        eval_strategy="epoch",
        save_strategy="epoch",
        metric_for_best_model="val_cer",
        greater_is_better=False,
        load_best_model_at_end=True,
        report_to=[],
        # Trainer auto-disables tqdm when the transformers logger's effective level is
        # above WARNING (see pipeline.py's _quiet_known_noise, which raises it to ERROR
        # to cut unrelated warning spam) -- force it back on so progress is still visible;
        # a silently "hung" vs. slow-but-progressing run is not something to guess at.
        disable_tqdm=False,
        # ManifestDataset is a plain Dataset, not datasets.Dataset -- Trainer's default
        # column-removal wraps the *collator* in that case and strips any key not in
        # WhisperForConditionalGeneration.forward's signature (audio, text, segment_id,
        # meeting_id all get dropped) before WhisperCollator ever sees the batch. Caught
        # live on Kaggle as `KeyError: 'audio'` inside WhisperCollator.
        remove_unused_columns=False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset={"val": val_ds, "ood": ood_ds},
        data_collator=WhisperCollator(processor),
        compute_metrics=_make_compute_metrics(processor, normalizer),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )
    trainer.train()

    best_dir = out / "checkpoints" / "best"
    trainer.save_model(str(best_dir))

    _write_training_csv(trainer.state.log_history, out / "metrics" / "training.csv")
    return best_dir


def _write_training_csv(log_history: list[dict], out_path: Path) -> Path:
    import csv

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["step", "loss", "learning_rate", "eval_val_cer", "eval_ood_cer"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in log_history:
            writer.writerow({k: row.get(k, "") for k in fields})
    return out_path
