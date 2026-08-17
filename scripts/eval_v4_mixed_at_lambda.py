"""Re-score v4-mixed-r16 (HF `winhsss/Reworkwhisper-large-v5`) at a lambda the
sweep never got to select, without retraining anything.

Why this exists (SESSIONS.md, "Hoi quy Reworkwhisper-large-v5 trong
production"): `select_lambda` walked v4-mixed-r16's grid and stopped at
lambda=0.25 because the 0.0 -> 0.25 step *lowered* OOD CER, giving that step a
negative cost/benefit ratio; `threshold * (a negative number)` is a negative
bar that the next step's positive ratio always clears, so the walk broke before
ever accepting lambda=0.5. Fixed in src/pipeline.py (a step with a
non-positive ratio no longer defines the elbow) and pinned by
tests/test_pipeline.py::test_select_lambda_a_free_step_does_not_reject_every_later_lambda,
which re-runs the real sweep rows and now selects 0.5.

What the fix does NOT do is tell us whether 0.5 is actually better on the axis
production regressed on. The sweep only ever recorded val CER and OOD CER --
both of which already favour 0.5 (val 0.0364 vs 0.0548, OOD 0.0262 against a
0.0428 bound) -- and CER is exactly the metric that could not see the loanword
loss this investigation is about. English-token retention rises with adapter
strength on every set measured so far (base lambda=0: 0.5574 synthetic /
0.3158 real; v4-mixed-r16 at lambda=0.25: 0.7599 / 0.4545), so 0.5 is expected
to retain more, but expected is not measured. This script measures it.

Reads `checkpoints/best` -- the raw pre-lambda-bake adapter, the same source
`stage_sweep_gate` scales for every lambda in a sweep -- NOT `adapter/`, which
is already baked to the selected lambda and cannot be rescaled up.

Scoring reuses the gate's own `_score_by_source` (so `retention`,
`retention_baseline`, `pass` and `retention_pass` mean exactly what they mean
in a real gate run) and `rejoin_real_chunks`/`score_real` for tier 4a. Real-bench
retention is reported on raw ingest sub-chunks, matching the 0.4083 / 0.4545
numbers already in SESSIONS.md rather than the rejoined parents.

Must run on GPU (Kaggle T4) with `mixed-noisy-v1` attached -- UNTESTED on this
machine (no torch/peft/GPU here; CLAUDE.md memory "Kaggle code never works
first try" applies, expect at least one iteration). Inference only, no training.

    python -m scripts.eval_v4_mixed_at_lambda \\
        --audio-root /kaggle/input/datasets/<user>/mixed-noisy-v1/mixed-noisy-v1/audio \\
        --real-bench-path /kaggle/input/datasets/<user>/real-meetings-bench/real-meetings-bench \\
        --lam 0.5
"""

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace

# v4-mixed-r16's shipped lambda=0.25 numbers, for the side-by-side print. Sources:
# Outputs/v4-mixed-r16/metrics/gate_results.json (CER) and SESSIONS.md H3/H6 (retention).
SHIPPED = {
    "youtube": {"cer": 0.07643118148599269, "retention": 0.7094},
    "synthetic": {"cer": 0.025828203779888027, "retention": 0.7599},
    "real": {"cer": 0.2890171436377297, "retention": 0.4545},
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", default="Outputs/v4-mixed-r16",
                    help="run dir: config.json, checkpoints/best, validated_manifest.jsonl, "
                    "audit/predictions_baseline_*.csv")
    ap.add_argument("--audio-root", required=True,
                    help="mixed-noisy-v1's audio/ dir as mounted on this machine")
    ap.add_argument("--real-bench-path", default=None,
                    help="real-meetings-bench root as mounted here; tier 4a is skipped without it")
    ap.add_argument("--lam", type=float, default=0.5)
    ap.add_argument("--out-dir", default=None,
                    help="where to write predictions/summary (default: <run-dir>/lambda<lam>)")
    args = ap.parse_args()

    from src import compat
    compat.apply()

    from src.asr import load_for_eval
    from src.config import Gates
    from src.data import ManifestDataset, load_manifests
    from src.gate import (_eval_split, _meeting_to_source, _score_by_source,
                          rejoin_real_chunks, score_real, write_predictions)
    from src.lora import set_lambda
    from src.metrics import english_token_retention
    from src.normalize import Normalizer

    run_dir = Path(args.run_dir)
    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    checkpoint_dir = run_dir / "checkpoints" / "best"
    if not checkpoint_dir.exists():
        raise FileNotFoundError(
            f"{checkpoint_dir} missing -- the raw pre-lambda-bake adapter is required; "
            f"{run_dir / 'adapter'} is baked to the selected lambda and cannot be rescaled up")

    out_dir = Path(args.out_dir) if args.out_dir else run_dir / f"lambda{args.lam}"
    out_dir.mkdir(parents=True, exist_ok=True)

    validated = [json.loads(l) for l in
                 (run_dir / "validated_manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    test_records = [r for r in validated if r["split"] == "test"]
    if not test_records:
        raise RuntimeError(f"no split=test records in {run_dir / 'validated_manifest.jsonl'}")
    test_ds = ManifestDataset(records=test_records, audio_root=Path(args.audio_root))
    print(f"{len(test_records)} test segments selected from {run_dir / 'validated_manifest.jsonl'}")

    model, processor = load_for_eval(cfg["base_model"], checkpoint_dir)
    n_scaled = set_lambda(model, args.lam)
    print(f"set_lambda: {n_scaled} LoRA layers scaled to lambda={args.lam}")

    norm_cfg = cfg["normalization"]
    normalizer = Normalizer(
        strip_punctuation=norm_cfg["strip_punctuation"],
        lowercase=norm_cfg["lowercase"],
        number_convention=norm_cfg["number_convention"],
        filler_tokens=norm_cfg["filler_tokens"],
    )
    eval_cfg = SimpleNamespace(**cfg["eval"])

    summary = {"lambda": args.lam, "run_dir": str(run_dir)}

    test_metrics = _eval_split(model, processor, test_ds, normalizer, eval_cfg,
                               desc=f"tier1_in_domain:lambda={args.lam}")
    test_predictions = test_metrics.pop("_predictions")
    write_predictions(test_predictions, out_dir / "predictions_tier1_in_domain.csv")

    baseline_test_csv = run_dir / "audit" / "predictions_baseline_test.csv"
    baseline_test_rows = None
    if baseline_test_csv.exists():
        with open(baseline_test_csv, encoding="utf-8") as f:
            baseline_test_rows = list(csv.DictReader(f))
    else:
        print(f"WARNING: {baseline_test_csv} missing -- no baseline, no pass/retention_pass verdict")

    summary["tier1_cer"] = test_metrics["cer"]
    summary["tier1_by_source"] = _score_by_source(
        test_predictions, _meeting_to_source(test_records), baseline_test_rows,
        cfg["gates"]["min_improvement_pct"], Gates().max_retention_regression_pp,
    )

    if args.real_bench_path:
        real_records = load_manifests(args.real_bench_path)
        real_ds = ManifestDataset(records=real_records,
                                  audio_root=Path(args.real_bench_path) / "audio")
        real_metrics = _eval_split(model, processor, real_ds, normalizer, eval_cfg,
                                   desc=f"tier4a_real:lambda={args.lam}")
        real_predictions = real_metrics.pop("_predictions")
        write_predictions(real_predictions, out_dir / "predictions_tier4a_real.csv")
        summary["tier4a_cer"] = score_real(rejoin_real_chunks(real_predictions))["cer"]
        summary["tier4a_retention"] = english_token_retention(
            [p["ref"] for p in real_predictions], [p["hyp"] for p in real_predictions])["retention"]
    else:
        print("tier 4a skipped -- pass --real-bench-path to include it")

    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n{'slice':12s} {'CER@' + str(args.lam):>12s} {'CER@0.25':>12s} "
          f"{'ret@' + str(args.lam):>12s} {'ret@0.25':>12s}")
    for source, entry in summary["tier1_by_source"].items():
        shipped = SHIPPED.get(source, {})
        print(f"{source:12s} {entry['cer']:12.4f} {shipped.get('cer', float('nan')):12.4f} "
              f"{entry['retention']:12.4f} {shipped.get('retention', float('nan')):12.4f}")
    if "tier4a_cer" in summary:
        print(f"{'real':12s} {summary['tier4a_cer']:12.4f} {SHIPPED['real']['cer']:12.4f} "
              f"{summary['tier4a_retention']:12.4f} {SHIPPED['real']['retention']:12.4f}")

    print(f"\nlambda={args.lam} beats the shipped lambda=0.25 only if BOTH CER holds and "
          "retention rises -- CER alone is what let the regression ship (SESSIONS.md H4).")
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
