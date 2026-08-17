"""Orchestrator. PROJECT_CORE.md §2.1, §5.

Every stage boundary is a file under outputs/{run_id}/ -- nothing is passed
in-memory across stages, so any stage can be re-run standalone and the
pipeline can resume from `.pipeline_state.json` without recomputing earlier
stages (§2.1 invariant 1).

Stages, run in order (`--stage` picks the entry point, not an isolated run --
each stage assumes prior stages' artifacts already exist on disk):
    smoke       -- CPU-only sanity pass: config load, tiny data slice, compat
                   patch, one forward pass. No GPU, no download. Run this
                   before ANY Kaggle GPU session (see CLAUDE.md memory:
                   Kaggle code never works first try).
    baseline    -- Stage 1: validate manifest, eval base model, write
                   metrics/baseline.json.
    train       -- Stage 2: SFT training -> checkpoints/best/.
    sweep-gate  -- Stage 3 + 4: lambda sweep (hard fail if no lambda fits the
                   OOD budget) -> gate (tiers 1, 2, 4a only, see gate.py) ->
                   push to HF iff gate passes and hub.push is true.
"""

import argparse
import json
import math
import sys
from pathlib import Path

from src.config import load, freeze
from src.normalize import Normalizer


def select_lambda(sweep_rows: list[dict], baseline_ood_cer: float,
                   ood_cer_budget: float, elbow_ratio_threshold: float) -> float:
    """Cost/benefit lambda selection (PROJECT_CORE.md §6 Stage 3, SESSIONS.md E2).

    Replaces "largest lambda within the OOD budget": that rule picked
    lambda=1.0 on v3-r16's own sweep even though 1.0 sits well past the point
    where diminishing val-CER gains start costing disproportionate OOD
    regression (catastrophic-forgetting territory, see docs/finetune-results-report-v3.md).

    Walks `sweep_rows` in ascending lambda order. A lambda still must keep
    ood_cer within `ood_cer_budget` of `baseline_ood_cer` (hard constraint --
    no soft fallback, CLAUDE.md). Among budget-safe candidates, stop
    advancing once a step's marginal cost/benefit ratio
    (delta_ood_cer / delta_val_cer vs. the previous grid point) exceeds
    `elbow_ratio_threshold` times the previous step's own ratio -- that step
    and every larger lambda are rejected, and the last accepted lambda wins.

    Only a step with a POSITIVE ratio becomes the baseline for the next step's
    comparison. A step that bought val CER while OOD CER held or improved has a
    ratio <= 0, and `threshold * (a non-positive number)` is a non-positive bar
    that every real cost clears -- which stopped v4-mixed-r16's walk dead at
    lambda=0.25, shipping a weaker adapter than its own sweep supported
    (SESSIONS.md, the v5 production regression). A free step carries no
    cost/benefit signal, so it defines no elbow.

    Raises if no lambda in the grid is budget-safe (mirrors the previous
    hard-fail contract).
    """
    rows = sorted((r for r in sweep_rows if r["ood_cer"] is not None),
                  key=lambda r: r["lambda"])

    best = None
    prev_ratio = None
    prev = None  # (val_cer, ood_cer) of the last ACCEPTED row
    for row in rows:
        lam, val_cer, ood_cer = row["lambda"], row["val_cer"], row["ood_cer"]
        if ood_cer > baseline_ood_cer + ood_cer_budget:
            break  # out of budget -- stop, this and every larger lambda are worse

        if prev is not None:
            delta_val = prev[0] - val_cer
            delta_ood = ood_cer - prev[1]
            ratio = delta_ood / delta_val if delta_val > 0 else math.inf
            if prev_ratio is not None and ratio > prev_ratio * elbow_ratio_threshold:
                break  # elbow: marginal cost/benefit blew past the previous step
            if ratio > 0:
                prev_ratio = ratio

        best = lam
        prev = (val_cer, ood_cer)

    if best is None:
        raise RuntimeError(
            "select_lambda: no lambda in cfg.sweep.lambdas keeps OOD CER within "
            f"ood_cer_budget={ood_cer_budget} of baseline ({baseline_ood_cer}) -- "
            "HARD FAIL, no adapter selected, no push."
        )
    return best


def _write_state(out_dir: Path, stage: str) -> None:
    state_path = out_dir / ".pipeline_state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    state["last_completed_stage"] = stage
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def stage_smoke(cfg) -> None:
    """No GPU, no network. Proves config + data + normalization + compat
    patch all work before spending Kaggle GPU time on anything else."""
    from src import compat
    from src.data import load_manifests, resolve_splits, split_stats

    versions = compat.apply()
    print("compat:", versions["neutralized"])

    records = load_manifests(cfg.data.dataset_path)
    resolved = resolve_splits(records, cfg.data.val_meetings)
    stats = split_stats(resolved)
    print("split_stats:", stats)

    normalizer = Normalizer(
        strip_punctuation=cfg.normalization.strip_punctuation,
        lowercase=cfg.normalization.lowercase,
        number_convention=cfg.normalization.number_convention,
        filler_tokens=cfg.normalization.filler_tokens,
    )
    sample_text = resolved[0]["text"]
    print("normalize sample:", sample_text, "->", normalizer(sample_text))
    print("SMOKE OK")


def stage_baseline(cfg) -> Path:
    from src.data import (load_manifests, resolve_splits, split_stats,
                           write_validated_manifest, ManifestDataset)
    from src.asr import load_for_eval
    from src.gate import _eval_split, write_predictions

    out = cfg.out_dir
    freeze(cfg, out / "config.json")

    records = load_manifests(cfg.data.dataset_path)
    resolved = resolve_splits(records, cfg.data.val_meetings)
    write_validated_manifest(resolved, out / "validated_manifest.jsonl")
    stats = split_stats(resolved)
    (out / "metrics").mkdir(parents=True, exist_ok=True)
    (out / "metrics" / "split_stats.json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8")

    normalizer = Normalizer(
        strip_punctuation=cfg.normalization.strip_punctuation,
        lowercase=cfg.normalization.lowercase,
        number_convention=cfg.normalization.number_convention,
        filler_tokens=cfg.normalization.filler_tokens,
    )
    ds = ManifestDataset(records=resolved, audio_root=Path(cfg.data.dataset_path) / "audio")
    model, processor = load_for_eval(cfg.base_model)

    def _eval_and_record(name: str, dataset) -> float:
        result = _eval_split(model, processor, dataset, normalizer, cfg.eval, desc=f"baseline:{name}")
        predictions = result.pop("_predictions")
        write_predictions(predictions, out / "audit" / f"predictions_baseline_{name}.csv")
        if name == "real":
            from src.gate import score_real
            return score_real(predictions)["cer"]
        return result["cer"]

    baseline = {"cer_test": _eval_and_record("test", ds.filter_split("test"))}
    if cfg.data.ood_eval_path:
        from src.data import load_manifests as load_ood
        ood_records = load_ood(cfg.data.ood_eval_path)
        ood_ds = ManifestDataset(records=ood_records,
                                  audio_root=Path(cfg.data.ood_eval_path) / "audio")
        baseline["cer_ood"] = _eval_and_record("ood", ood_ds)
    if cfg.data.real_bench_path:
        from src.data import load_manifests as load_real
        real_records = load_real(cfg.data.real_bench_path)
        real_ds = ManifestDataset(records=real_records,
                                   audio_root=Path(cfg.data.real_bench_path) / "audio")
        baseline["cer_real"] = _eval_and_record("real", real_ds)

    baseline_path = out / "metrics" / "baseline.json"
    baseline_path.write_text(json.dumps(baseline, indent=2), encoding="utf-8")
    _write_state(out, "baseline")
    return baseline_path


def stage_train(cfg) -> Path:
    from src.data import ManifestDataset
    from src.train import train as run_train
    from transformers import WhisperForConditionalGeneration

    out = cfg.out_dir
    manifest_path = out / "validated_manifest.jsonl"
    if not manifest_path.exists():
        raise FileNotFoundError(f"{manifest_path} missing -- run stage baseline first")

    records = [json.loads(l) for l in manifest_path.read_text(encoding="utf-8").splitlines()]
    audio_root = Path(cfg.data.dataset_path) / "audio"
    train_ds = ManifestDataset(records=records, audio_root=audio_root).filter_split("train")
    val_ds = ManifestDataset(records=records, audio_root=audio_root).filter_split("val")

    ood_ds = None
    if cfg.data.ood_eval_path:
        from src.data import load_manifests
        ood_records = load_manifests(cfg.data.ood_eval_path)
        ood_ds = ManifestDataset(records=ood_records,
                                  audio_root=Path(cfg.data.ood_eval_path) / "audio")

    if cfg.training.limit:
        # Quick end-to-end dry run: exercise the real Trainer/collator/eval code
        # paths on a handful of segments instead of the full split, before
        # spending GPU time on the full run.
        train_ds = train_ds.limit(cfg.training.limit)
        val_ds = val_ds.limit(cfg.training.limit)
        if ood_ds is not None:
            ood_ds = ood_ds.limit(cfg.training.limit)

    # use_safetensors=False doesn't stop transformers' safetensors auto-conversion
    # probe thread from firing (403 on repos with discussions disabled, e.g.
    # PhoWhisper-*) -- the traceback it used to dump is silenced by
    # compat.silence_hf_discussions_403_noise(), applied via compat.apply() in main().
    base_model = WhisperForConditionalGeneration.from_pretrained(cfg.base_model, use_safetensors=False)
    best_dir = run_train(cfg, base_model, train_ds, val_ds, ood_ds, out)
    _write_state(out, "train")
    return best_dir


def stage_sweep_gate(cfg) -> Path:
    from src.data import ManifestDataset, load_manifests
    from src.asr import load_for_eval
    from src.gate import run_gate, write_gate_results, _eval_split, _meeting_to_source
    from src.lora import set_lambda, save_with_lambda

    out = cfg.out_dir
    baseline = json.loads((out / "metrics" / "baseline.json").read_text(encoding="utf-8"))
    checkpoint_dir = out / "checkpoints" / "best"
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"{checkpoint_dir} missing -- run stage train first")

    normalizer = Normalizer(
        strip_punctuation=cfg.normalization.strip_punctuation,
        lowercase=cfg.normalization.lowercase,
        number_convention=cfg.normalization.number_convention,
        filler_tokens=cfg.normalization.filler_tokens,
    )

    val_records = [r for r in load_manifests_from_validated(out)]
    val_ds = ManifestDataset(records=val_records, audio_root=Path(cfg.data.dataset_path) / "audio")
    ood_records = load_manifests(cfg.data.ood_eval_path) if cfg.data.ood_eval_path else []
    ood_ds = ManifestDataset(records=ood_records,
                              audio_root=Path(cfg.data.ood_eval_path) / "audio") if ood_records else None

    model, processor = load_for_eval(cfg.base_model, checkpoint_dir)

    val_meeting_to_source = _meeting_to_source(val_ds.records)
    sweep_rows = []
    for lam in cfg.sweep.lambdas:
        set_lambda(model, lam)
        val_metrics = _eval_split(model, processor, val_ds, normalizer, cfg.eval,
                                   desc=f"sweep:lambda={lam}:val")
        val_cer_by_source = _cer_by_source(val_metrics["_predictions"], val_meeting_to_source)
        ood_cer = (_eval_split(model, processor, ood_ds, normalizer, cfg.eval,
                                desc=f"sweep:lambda={lam}:ood")["cer"]
                   if ood_ds is not None else None)
        sweep_rows.append({
            "lambda": lam, "val_cer": val_metrics["cer"], "ood_cer": ood_cer,
            "val_cer_synthetic": val_cer_by_source.get("synthetic"),
            "val_cer_youtube": val_cer_by_source.get("youtube"),
        })

    _write_sweep_csv(sweep_rows, out / "metrics" / "lambda_sweep.csv")

    best_lambda = select_lambda(sweep_rows, baseline.get("cer_ood", float("inf")),
                                 cfg.sweep.ood_cer_budget, cfg.sweep.elbow_ratio_threshold)

    adapter_dir = save_with_lambda(model, best_lambda, out / "adapter")

    validated = [json.loads(l) for l in
                 (out / "validated_manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    test_ds = ManifestDataset(records=[r for r in validated if r["split"] == "test"],
                              audio_root=Path(cfg.data.dataset_path) / "audio")

    real_ds = None
    if cfg.data.real_bench_path:
        real_records = load_manifests(cfg.data.real_bench_path)
        real_ds = ManifestDataset(records=real_records,
                                   audio_root=Path(cfg.data.real_bench_path) / "audio")

    baseline_real_csv = out / "audit" / "predictions_baseline_real.csv"
    baseline_test_csv = out / "audit" / "predictions_baseline_test.csv"
    results = run_gate(cfg, model, processor, normalizer, test_ds, ood_ds, real_ds, baseline,
                        baseline_real_csv=baseline_real_csv, baseline_test_csv=baseline_test_csv)
    gate_path = write_gate_results(results, out)

    if results["overall_pass"] and cfg.hub.push:
        from src.hub import push_adapter
        url = push_adapter(adapter_dir, cfg.hub.repo_id, private=cfg.hub.private,
                            gate_results=json.loads(gate_path.read_text(encoding="utf-8")))
        print(f"pushed adapter to {url}")

    _write_state(out, "sweep-gate")
    return gate_path


def load_manifests_from_validated(out_dir: Path) -> list[dict]:
    path = out_dir / "validated_manifest.jsonl"
    records = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()]
    return [r for r in records if r["split"] == "val"]


def _cer_by_source(predictions: list[dict], meeting_to_source: dict[str, str]) -> dict[str, float]:
    """val CER per `source` slice (synthetic/youtube) -- for `lambda_sweep.csv`'s
    per-slice columns, so it's visible which slice drove a given lambda's selection
    (PROJECT_CORE.md, mixed-noisy-v1 plan). Reuses the same `_predictions` the sweep
    loop already decoded -- no extra GPU cost."""
    from src.metrics import score

    by_source: dict[str, list[dict]] = {}
    for row in predictions:
        by_source.setdefault(meeting_to_source[row["meeting_id"]], []).append(row)
    return {src: score([r["ref"] for r in rows], [r["hyp"] for r in rows])["cer"]
            for src, rows in by_source.items()}


def _write_sweep_csv(rows: list[dict], out_path: Path) -> Path:
    import csv

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["lambda", "val_cer", "ood_cer", "val_cer_synthetic", "val_cer_youtube"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, restval="")
        writer.writeheader()
        writer.writerows(rows)
    return out_path


STAGES = {
    "smoke": stage_smoke,
    "baseline": stage_baseline,
    "train": stage_train,
    "sweep-gate": stage_sweep_gate,
}


def _quiet_known_noise() -> None:
    """Silence transformers/torch warnings observed to be harmless and
    repetitive across every stage (deprecation notices, tied-weights/attention-
    mask/logits-processor advisories) -- cosmetic only, never a correctness
    signal we rely on. Does not touch exceptions or the compat patch."""
    import warnings

    from transformers.utils import logging as hf_logging

    hf_logging.set_verbosity_error()
    warnings.filterwarnings("ignore", message=".*warmup_ratio is deprecated.*")
    warnings.filterwarnings("ignore", message=".*gather along dimension.*")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--stage", choices=list(STAGES), required=True)
    ap.add_argument("--override", action="append", default=[])
    args = ap.parse_args()

    cfg = load(args.config, overrides=args.override)
    from src import compat
    compat.apply()  # must run before any peft import -- see compat.py
    _quiet_known_noise()

    try:
        STAGES[args.stage](cfg)
    except Exception:
        from src.compat import version_table
        print("PIPELINE FAILED. Environment:", version_table(), file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
