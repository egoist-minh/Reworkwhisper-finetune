"""SFT training loop. PROJECT_CORE.md §6 Stage 2.

UNTESTED end-to-end (no torch/transformers on this dev machine -- see
handoff). Smoke-test on Kaggle before trusting this against the deadline;
the transformers 5.0.0 `Trainer(eval_dataset=dict)` multi-eval-set API this
relies on has not been exercised in this repo yet.

Checkpoint strategy: save to `checkpoints/best/` whenever `val_cer` improves.
Early stopping: 3 eval-round patience. OOD eval runs every eval round (not
just at the end) so forgetting is visible during training, not only after
(§0 problem 2).

Early stopping AND best-checkpoint selection both go through one custom
callback (`_EarlyStoppingState` + `RobustEvalTrackingCallback`, built in
`train()`), not transformers' built-in `EarlyStoppingCallback` /
`load_best_model_at_end`+`metric_for_best_model`. Fixed 2026-08-02: with a
dict-valued `eval_dataset` ({"val": ..., "ood": ...}), the built-in
`EarlyStoppingCallback` looked up `eval_val_cer` in a `metrics` dict that
didn't reliably carry it on every `on_evaluate` call, logged "did not find
eval_val_cer", and permanently disabled itself for the rest of training on
the first miss -- harmless in the first Kaggle run only because
`patience == epochs`. `load_best_model_at_end`/`metric_for_best_model` do the
exact same `f"eval_{metric_for_best_model}"` lookup internally (unverified
whether they hit the same miss) to decide which checkpoint to restore at the
end of `.train()` -- silently shipping the last epoch under the name "best"
if they do. `RobustEvalTrackingCallback` replaces both: it skips any
`on_evaluate` call whose `metrics` lacks `eval_val_cer` instead of guessing at
HF's internal key shape, and saves `checkpoints/best/` itself the moment
`eval_val_cer` improves, so nothing downstream depends on Trainer's own
best-model bookkeeping.
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


class _EarlyStoppingState:
    """Pure best-value + patience tracking -- no transformers/torch dependency,
    testable on its own. Callers only call `.update()` when the metric is
    actually present in that eval round, so this never has to guess at HF's
    internal metric-key naming."""

    def __init__(self, patience: int, greater_is_better: bool = False):
        self.patience = patience
        self.greater_is_better = greater_is_better
        self.best: float | None = None
        self.rounds_without_improvement = 0

    def update(self, value: float) -> bool:
        """Record one eval round's metric value. Returns True if training should stop."""
        improved = self.best is None or (
            value > self.best if self.greater_is_better else value < self.best
        )
        if improved:
            self.best = value
            self.rounds_without_improvement = 0
        else:
            self.rounds_without_improvement += 1
        return self.rounds_without_improvement >= self.patience


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
                               WhisperProcessor, TrainerCallback)
    from transformers.trainer_callback import PrinterCallback
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
        # Default is 500 -- TrainingDisplayCallback's TrainLoss column would stay
        # "-" for the whole run on anything under that many steps.
        logging_steps=1,
        report_to=[],
        # HF's own train/eval bars interleave into unreadable noise with a dict-valued
        # eval_dataset (one bar per split, redrawn on top of each other). Killed in favor
        # of TrainingDisplayCallback's single bar + table below.
        disable_tqdm=True,
        # ManifestDataset is a plain Dataset, not datasets.Dataset -- Trainer's default
        # column-removal wraps the *collator* in that case and strips any key not in
        # WhisperForConditionalGeneration.forward's signature (audio, text, segment_id,
        # meeting_id all get dropped) before WhisperCollator ever sees the batch. Caught
        # live on Kaggle as `KeyError: 'audio'` inside WhisperCollator.
        remove_unused_columns=False,
    )

    stopping = _EarlyStoppingState(patience=3, greater_is_better=False)
    best_dir = out / "checkpoints" / "best"

    class RobustEvalTrackingCallback(TrainerCallback):
        """See the module docstring for why this replaces both the built-in
        EarlyStoppingCallback and load_best_model_at_end/metric_for_best_model:
        a `metrics` dict missing `eval_val_cer` on a given `on_evaluate` call is
        skipped outright, never treated as a reason to disable early stopping
        or to fall back to "whatever checkpoint is currently loaded" for best-
        model selection. Saves `checkpoints/best/` itself the instant
        `eval_val_cer` improves, so this is the sole authority on which
        checkpoint is "best" -- Trainer's own bookkeeping is not consulted."""

        def on_evaluate(self, args, state, control, metrics=None, model=None, **kwargs):
            if metrics is None or "eval_val_cer" not in metrics:
                return control
            should_stop = stopping.update(metrics["eval_val_cer"])
            if stopping.rounds_without_improvement == 0:
                model.save_pretrained(str(best_dir))
            if should_stop:
                control.should_training_stop = True
            return control

    class TrainingDisplayCallback(TrainerCallback):
        """One tqdm bar for the whole run + one table row per eval round (val+ood
        merged), replacing HF's default per-split bars that redraw on top of each
        other and the raw metrics-dict prints. `eval_dataset` is {"val":.., "ood":..}
        (fixed order), so a row is flushed once an `eval_ood_*` key shows up --
        that's always the second/last on_evaluate call of the round."""

        _ROW_FMT = "{:>7}{:>8}{:>12}{:>10}{:>9}{:>9}{:>10}"

        def __init__(self):
            self.bar = None
            self.eval_bar = None
            self.last_train_loss = None
            self._pending = {}

        def on_train_begin(self, args, state, control, **kwargs):
            from tqdm.auto import tqdm

            self.bar = tqdm(total=state.max_steps, desc="train", unit="step")
            tqdm.write(self._ROW_FMT.format(
                "Epoch", "Step", "TrainLoss", "ValLoss", "ValCER", "ValWER", "OOD_CER"
            ))

        def on_prediction_step(self, args, state, control, **kwargs):
            # eval_dataset is {"val":.., "ood":..} -- "ood" keys only land in
            # _pending once val's on_evaluate has already fired, so an empty
            # _pending means the val split is the one currently decoding.
            if self.eval_bar is None:
                from tqdm.auto import tqdm

                split = "ood" if self._pending else "val"
                self.eval_bar = tqdm(desc=f"eval:{split}", unit="batch",
                                      position=1, leave=False)
            self.eval_bar.update(1)

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs and "loss" in logs:
                self.last_train_loss = logs["loss"]
                if self.bar is not None:
                    self.bar.set_postfix(loss=f"{logs['loss']:.3f}")

        def on_step_end(self, args, state, control, **kwargs):
            if self.bar is not None:
                self.bar.update(1)

        def on_evaluate(self, args, state, control, metrics=None, model=None, **kwargs):
            if self.eval_bar is not None:
                self.eval_bar.close()
                self.eval_bar = None
            if not metrics:
                return control
            self._pending.update(metrics)
            if not any(k.startswith("eval_ood") for k in metrics):
                return control
            row = self._ROW_FMT.format(
                f"{state.epoch:.1f}",
                state.global_step,
                f"{self.last_train_loss:.3f}" if self.last_train_loss is not None else "-",
                f"{self._pending.get('eval_val_loss', float('nan')):.3f}",
                f"{self._pending.get('eval_val_cer', float('nan')):.4f}",
                f"{self._pending.get('eval_val_wer', float('nan')):.4f}",
                f"{self._pending.get('eval_ood_cer', float('nan')):.4f}",
            )
            from tqdm.auto import tqdm
            tqdm.write(row)
            self._pending = {}
            return control

        def on_train_end(self, args, state, control, **kwargs):
            if self.bar is not None:
                self.bar.close()

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset={"val": val_ds, "ood": ood_ds},
        data_collator=WhisperCollator(processor),
        compute_metrics=_make_compute_metrics(processor, normalizer),
        callbacks=[RobustEvalTrackingCallback(), TrainingDisplayCallback()],
    )
    # disable_tqdm=True makes Trainer default-add PrinterCallback, which dumps every
    # log event as a raw dict -- that's the noise TrainingDisplayCallback replaces.
    trainer.remove_callback(PrinterCallback)
    trainer.train()

    if stopping.best is None:
        raise RuntimeError(
            "eval_val_cer was never observed in any on_evaluate call -- "
            "checkpoints/best/ was never written. compute_metrics or the "
            "'val' eval_dataset key isn't producing the expected metric; "
            "fix that before trusting any downstream stage."
        )

    _write_training_csv(trainer.state.log_history, out / "metrics" / "training.csv")
    return best_dir


def _write_training_csv(log_history: list[dict], out_path: Path) -> Path:
    import csv

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["step", "loss", "learning_rate", "eval_val_loss", "eval_val_cer",
              "eval_val_wer", "eval_ood_cer"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in log_history:
            writer.writerow({k: row.get(k, "") for k in fields})
    return out_path
