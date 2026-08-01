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
import sys
from pathlib import Path

from src.config import load, freeze
from src.normalize import Normalizer


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
        result = _eval_split(model, processor, dataset, normalizer, cfg.eval)
        write_predictions(result.pop("_predictions"), out / "audit" / f"predictions_baseline_{name}.csv")
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

    # use_safetensors=False skips transformers' auto-conversion probe, which
    # otherwise spawns a background thread that hits HF's discussions API to
    # check for an existing conversion PR -- 403s (PhoWhisper-small has
    # discussions disabled) and dumps a harmless but noisy traceback every run.
    base_model = WhisperForConditionalGeneration.from_pretrained(cfg.base_model, use_safetensors=False)
    best_dir = run_train(cfg, base_model, train_ds, val_ds, ood_ds, out)
    _write_state(out, "train")
    return best_dir


def stage_sweep_gate(cfg) -> Path:
    from src.data import ManifestDataset, load_manifests
    from src.asr import load_for_eval
    from src.gate import run_gate, write_gate_results, _eval_split
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

    sweep_rows = []
    best_lambda = None
    for lam in cfg.sweep.lambdas:
        set_lambda(model, lam)
        val_cer = _eval_split(model, processor, val_ds, normalizer, cfg.eval)["cer"]
        ood_cer = (_eval_split(model, processor, ood_ds, normalizer, cfg.eval)["cer"]
                   if ood_ds is not None else None)
        row = {"lambda": lam, "val_cer": val_cer, "ood_cer": ood_cer}
        sweep_rows.append(row)
        if ood_cer is not None and ood_cer <= baseline.get("cer_ood", float("inf")) + cfg.sweep.ood_cer_budget:
            if best_lambda is None or lam > best_lambda:
                best_lambda = lam

    _write_sweep_csv(sweep_rows, out / "metrics" / "lambda_sweep.csv")

    if best_lambda is None:
        raise RuntimeError(
            "sweep-gate: no lambda in cfg.sweep.lambdas keeps OOD CER within "
            f"cfg.sweep.ood_cer_budget={cfg.sweep.ood_cer_budget} of baseline "
            f"({baseline.get('cer_ood')}) -- HARD FAIL, no adapter selected, no push."
        )

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

    results = run_gate(cfg, model, processor, normalizer, test_ds, ood_ds, real_ds, baseline)
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


def _write_sweep_csv(rows: list[dict], out_path: Path) -> Path:
    import csv

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["lambda", "val_cer", "ood_cer"])
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
