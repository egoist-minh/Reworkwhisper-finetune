"""SFT training loop. PROJECT_CORE.md §6 Stage 2.

UNTESTED end-to-end (no torch/transformers on this dev machine -- see
handoff). Smoke-test on Kaggle before trusting this against the deadline;
the transformers 5.0.0 `Trainer(eval_dataset=dict)` multi-eval-set API this
relies on has not been exercised in this repo yet.

Checkpoint strategy: save to `checkpoints/best/` whenever `val_cer` improves.
An eval round is once per epoch by default, or every `training.eval_steps`
steps when that is set -- see `_schedule_kwargs`. Early stopping: 3 eval-round
patience, so the interval sets how much training a stop costs. OOD eval runs
every eval round (not just at the end) so forgetting is visible during
training, not only after (§0 problem 2).

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

import inspect
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.asr import pick_dtype
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

    def replay(self, values: list[float]) -> None:
        """Rebuild state from a previous run's eval rounds, for a resume.

        Trainer restores state.log_history from trainer_state.json, so the rounds
        that already ran are known exactly and both the best value and the
        patience counter come back. Without this the first eval round after a
        resume always counts as an improvement, and RobustEvalTrackingCallback
        overwrites checkpoints/best/ with a worse adapter -- the silent
        best-model failure this class exists to prevent, reintroduced through
        the back door.
        """
        for value in values:
            self.update(value)

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


def _latest_checkpoint(checkpoints_dir: Path | str) -> Path | None:
    """Highest-numbered resumable `checkpoint-<step>` under `checkpoints_dir`.

    Trainer's own save directories only. `checkpoints/best/` holds an adapter and
    nothing else -- RobustEvalTrackingCallback writes it with save_pretrained --
    so it can restore weights but not the optimizer state, the LR schedule or the
    data order, and resuming from it would restart step 1 with a warm adapter and
    a cold everything else. A directory with no trainer_state.json is a save that
    a dying box interrupted; it is skipped rather than resumed from.
    """
    d = Path(checkpoints_dir)
    if not d.is_dir():
        return None
    found = [
        (int(p.name.split("-", 1)[1]), p)
        for p in d.glob("checkpoint-*")
        if p.is_dir() and p.name.split("-", 1)[1].isdigit()
        and (p / "trainer_state.json").is_file()
    ]
    return max(found, key=lambda pair: pair[0])[1] if found else None


def _eval_val_cer_history(log_history: list[dict]) -> list[float]:
    """Each eval round's val CER in order, from a (possibly restored) log_history.

    log_history carries one row per logging event, and only eval rounds carry
    eval_val_cer -- the per-step loss rows do not.
    """
    return [row["eval_val_cer"] for row in log_history if "eval_val_cer" in row]


def _schedule_kwargs(eval_steps: int | None, total_steps: int) -> dict:
    """Eval/save cadence for Seq2SeqTrainingArguments.

    "epoch" was the only cadence until now, and on the 1-epoch runs the larger
    corpora call for it collapses to a single eval after the last step: no CER
    or OOD row while training is still running, and no checkpoint on disk to
    resume from if the box dies at 80% -- the whole run is lost. eval_steps
    turns both into a fixed step interval instead. save and eval share the
    interval so every saved checkpoint has a val_cer measured at that step.

    Kept as a separate function because train() cannot be imported without
    torch, and this is the part worth a test.
    """
    if eval_steps is None:
        return {"eval_strategy": "epoch", "save_strategy": "epoch"}
    # An interval past the end of the run fires no eval round at all: nothing is
    # scored, checkpoints/best/ is never written, and train() raises -- after the
    # entire run has been paid for. Refuse before the GPU is touched.
    if eval_steps > total_steps:
        raise ValueError(
            f"training.eval_steps={eval_steps} exceeds the {total_steps} steps this "
            "run has, so no eval round would ever fire and no best checkpoint would "
            f"be saved -- use at most {total_steps // 2} to get more than one round")
    return {
        "eval_strategy": "steps",
        "save_strategy": "steps",
        "eval_steps": eval_steps,
        "save_steps": eval_steps,
        # Adapter + optimizer state per checkpoint, once per interval instead of
        # once per epoch -- unbounded on a long run. checkpoints/best/ is written
        # separately by RobustEvalTrackingCallback and is not one of these, so
        # rotation here cannot delete the best model.
        "save_total_limit": 2,
    }


def train(cfg, base_model, train_ds, val_ds, ood_ds, out_dir: str | Path,
          resume: bool = False):
    """cfg: src.config.Config. Returns the path to checkpoints/best/.

    resume=True picks up from the newest `checkpoints/checkpoint-<step>` instead
    of step 1, restoring optimizer state, LR schedule, data order and the eval
    history behind early stopping. No checkpoint on disk is not an error -- an
    interrupted run that never reached its first save has nothing to resume, and
    starting over is the only option.
    """
    import torch
    from transformers import (Seq2SeqTrainer, Seq2SeqTrainingArguments,
                               WhisperProcessor, TrainerCallback)
    from transformers.trainer_callback import PrinterCallback
    from peft import get_peft_model

    from src.lora import build_lora_config

    out = Path(out_dir)
    train_dtype = pick_dtype() if torch.cuda.is_available() else torch.float32
    # transformers 5.17 dropped warmup_ratio and kept only warmup_steps. Feature-detect
    # rather than branch on a version string (src/compat.py), so an older transformers
    # keeps taking the ratio it understands.
    steps_per_epoch = math.ceil(
        len(train_ds) / (cfg.training.batch_size * cfg.training.grad_accum_steps))
    total_steps = math.ceil(steps_per_epoch * cfg.training.epochs)
    if "warmup_ratio" in inspect.signature(Seq2SeqTrainingArguments.__init__).parameters:
        warmup_kwargs = {"warmup_ratio": cfg.training.warmup_ratio}
    else:
        warmup_kwargs = {"warmup_steps": round(cfg.training.warmup_ratio * total_steps)}
    schedule_kwargs = _schedule_kwargs(cfg.training.eval_steps, total_steps)
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
        **warmup_kwargs,
        num_train_epochs=cfg.training.epochs,
        gradient_checkpointing=cfg.training.gradient_checkpointing,
        # Same capability rule as src/asr.py:pick_dtype, so training and eval run the
        # same numeric type on the same GPU. Hardcoding fp16 here put a Hopper/Ampere
        # run in fp16 while load_for_eval loaded bf16.
        fp16=train_dtype is torch.float16,
        bf16=train_dtype is torch.bfloat16,
        # ManifestDataset.__getitem__ decodes a wav and resamples 24k -> 16k per item
        # (src/data.py), and the collator extracts mel on top of that. HF's default of
        # 0 workers runs all of it in the training process, which starves any GPU
        # faster than the T4 this was first measured on.
        dataloader_num_workers=min(8, os.cpu_count() or 1),
        predict_with_generate=True,
        generation_num_beams=cfg.eval.num_beams,
        **schedule_kwargs,
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

    class StepTimingCallback(TrainerCallback):
        """Per-step wall clock, split into compute and the gap before the next step.

        The gap is where a slow dataloader shows up. Trainer pulls the batch off
        the iterator before on_step_begin fires, so whatever elapses between one
        on_step_end and the next on_step_begin is mostly waiting on audio to be
        decoded and resampled. That wait is the part that grows with the corpus
        -- 100 h is ~14 GB of 24 kHz wavs read once per epoch, against the ~100 MB
        that stays in page cache on a short probe -- and train_runtime cannot
        separate it from compute.
        """

        def __init__(self):
            self.begins: list[float] = []
            self.ends: list[float] = []

        def on_step_begin(self, args, state, control, **kwargs):
            self.begins.append(time.perf_counter())
            return control

        def on_step_end(self, args, state, control, **kwargs):
            self.ends.append(time.perf_counter())
            return control

    step_timing = StepTimingCallback()

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

        def on_train_begin(self, args, state, control, **kwargs):
            # Empty on a fresh run; on a resume it is the pre-crash eval rounds,
            # restored from trainer_state.json before this fires.
            history = _eval_val_cer_history(state.log_history)
            if history:
                stopping.replay(history)
                print(f"resume: {len(history)} eval rounds replayed, "
                      f"best val_cer so far {stopping.best:.6f}")
            return control

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
            self._progress_every = 1
            self._t0 = None
            self._step0 = 0

        def on_train_begin(self, args, state, control, **kwargs):
            from tqdm.auto import tqdm

            self.bar = tqdm(total=state.max_steps, initial=state.global_step,
                             desc="train", unit="step")
            self._t0 = time.perf_counter()
            # Nonzero on a resume: seconds-per-step and the ETA have to divide by
            # the steps THIS process ran, not by an absolute step number that
            # counts steps some earlier, already-paid-for process did.
            self._step0 = state.global_step
            # The bar redraws in place with a carriage return, which is unreadable
            # in the run.log a detached run leaves behind. These lines are the
            # progress trace that survives a plain tail: ~50 of them however
            # many steps the run has.
            self._progress_every = max(1, state.max_steps // 50)
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
            done = state.global_step - self._step0
            if not done or state.global_step % self._progress_every:
                return control
            from tqdm.auto import tqdm

            elapsed = time.perf_counter() - self._t0
            per_step = elapsed / done
            remaining = per_step * (state.max_steps - state.global_step)
            loss = f"{self.last_train_loss:.3f}" if self.last_train_loss is not None else "-"
            tqdm.write(
                f"progress step {state.global_step}/{state.max_steps} "
                f"loss={loss} sec_per_step={per_step:.3f} "
                f"elapsed={elapsed / 60:.1f}m eta={remaining / 60:.1f}m"
            )
            return control

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
        callbacks=[RobustEvalTrackingCallback(), TrainingDisplayCallback(),
                   step_timing],
    )
    # disable_tqdm=True makes Trainer default-add PrinterCallback, which dumps every
    # log event as a raw dict -- that's the noise TrainingDisplayCallback replaces.
    trainer.remove_callback(PrinterCallback)
    checkpoint = _latest_checkpoint(out / "checkpoints") if resume else None
    if resume:
        # Trainer replays the consumed batches to restore the exact data order
        # (ignore_data_skip stays False). That costs a dataloader pass over what
        # was already seen, which the probe measured at 83.9x the rate the GPU
        # consumes it -- seconds, not minutes, and worth it for a data order that
        # matches what the checkpoint was trained on.
        print(f"resume: {checkpoint}" if checkpoint is not None else
              "resume: no resumable checkpoint found, starting from step 1")
    result = trainer.train(resume_from_checkpoint=str(checkpoint) if checkpoint else None)

    if stopping.best is None:
        raise RuntimeError(
            "eval_val_cer was never observed in any on_evaluate call -- "
            "checkpoints/best/ was never written. compute_metrics or the "
            "'val' eval_dataset key isn't producing the expected metric; "
            "fix that before trusting any downstream stage."
        )

    _write_training_csv(trainer.state.log_history, out / "metrics" / "training.csv")
    _write_timing_json(
        result.metrics,
        trainer.state.log_history,
        trainer.state.global_step,
        step_timing.begins,
        step_timing.ends,
        out / "metrics" / "timing.json",
    )
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


def _summarize_step_times(begins: list[float], ends: list[float], warmup: int = 10) -> dict | None:
    """Steady-state per-step cost, warmup excluded.

    The opening steps of a run pay for cuDNN autotune, allocator growth and the
    first touch of every weight, so a mean taken over all of them inflates any
    projection built on it. Returns None when the run is too short to have a
    steady state at all -- better than a number nobody can tell is warmup.

    Three series, because they answer different questions: `compute` is
    on_step_begin to on_step_end, `dataloader_gap` is the wait before the next
    step begins, and `cycle` is begin-to-begin, which is the one to multiply by
    a step count when projecting a longer run.
    """
    n = min(len(begins), len(ends))
    if n - warmup < 3:
        return None

    def stats(xs: list[float]) -> dict:
        xs = sorted(xs)
        return {
            "median": xs[len(xs) // 2],
            "p95": xs[min(len(xs) - 1, int(0.95 * len(xs)))],
            "max": xs[-1],
        }

    return {
        "warmup_steps_excluded": warmup,
        "steps_measured": n - warmup,
        "first_step_seconds": ends[0] - begins[0],
        # First on_step_begin to last on_step_end. train_only_seconds minus this
        # is what an epoch costs outside its steps -- spawning dataloader
        # workers, prefetching the first batch, tearing the loop down. One-off,
        # so it must not be folded into a per-step figure and multiplied.
        "steps_span_seconds": ends[n - 1] - begins[0],
        "compute_seconds": stats([ends[i] - begins[i] for i in range(warmup, n)]),
        "dataloader_gap_seconds": stats([begins[i + 1] - ends[i] for i in range(warmup, n - 1)]),
        "cycle_seconds": stats([begins[i + 1] - begins[i] for i in range(warmup, n - 1)]),
    }


def _write_timing_json(
    train_metrics: dict,
    log_history: list[dict],
    global_step: int,
    begins: list[float],
    ends: list[float],
    out_path: Path,
) -> Path:
    """Record what the run cost in wall-clock seconds.

    training.csv keeps loss and CER per step and drops every runtime field the
    Trainer reports, so the only way to answer "how long would N hours of audio
    take on this box" was to time the process by hand from outside. Eval is
    subtracted out because it scales with the eval set and the eval interval,
    not with the size of the training corpus -- extrapolating a total that has
    eval baked into it overstates a longer run.

    stage_seconds and model_load_seconds are filled in later by
    merge_stage_timing: they happen outside trainer.train() and are not visible
    from here.
    """
    import json
    import torch

    eval_seconds = sum(
        v
        for row in log_history
        for k, v in row.items()
        if k.endswith("_runtime") and not k.startswith("train")
    )
    total = train_metrics.get("train_runtime")
    train_only = total - eval_seconds if total is not None else None
    steady = _summarize_step_times(begins, ends)
    timing = {
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "peak_vram_allocated_gb": (
            torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else None
        ),
        # Reserved is what nvidia-smi shows: the allocator holds on to freed
        # blocks, so allocated alone understates what the card needs to have free.
        "peak_vram_reserved_gb": (
            torch.cuda.max_memory_reserved() / 1024**3 if torch.cuda.is_available() else None
        ),
        "train_runtime_seconds": total,
        "eval_seconds": eval_seconds,
        "train_only_seconds": train_only,
        "steps": global_step,
        "mean_seconds_per_step": train_only / global_step if train_only and global_step else None,
        "samples_per_second": train_metrics.get("train_samples_per_second"),
        "steady_state": steady,
        # Inside train_runtime, but outside both eval and the steps themselves.
        # Named because it is invisible otherwise: on the whisper-tiny run that
        # first exercised this it was 51 of 55 seconds.
        "train_overhead_seconds": (
            train_only - steady["steps_span_seconds"]
            if train_only is not None and steady is not None
            else None
        ),
        "stage_seconds": None,
        "model_load_seconds": None,
        "dead_seconds": None,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(timing, indent=2), encoding="utf-8")
    return out_path


def merge_stage_timing(
    out_path: Path, *, stage_seconds: float, model_load_seconds: float
) -> Path:
    """Fold the costs that live outside trainer.train() into timing.json.

    Pulling a 6.2 GB checkpoint off disk and building the datasets both happen
    before the first step and never reach train_runtime, so a projection built
    from train_runtime alone silently drops them. dead_seconds is what is left
    of the stage once training and eval are accounted for.
    """
    import json

    if not out_path.exists():
        return out_path
    timing = json.loads(out_path.read_text(encoding="utf-8"))
    train_runtime = timing.get("train_runtime_seconds")
    timing["stage_seconds"] = stage_seconds
    timing["model_load_seconds"] = model_load_seconds
    timing["dead_seconds"] = (
        stage_seconds - train_runtime if train_runtime is not None else None
    )
    out_path.write_text(json.dumps(timing, indent=2), encoding="utf-8")
    return out_path
