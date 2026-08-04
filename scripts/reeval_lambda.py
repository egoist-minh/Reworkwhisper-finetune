"""Re-evaluate tier1_in_domain and tier4a_real at an alternate lambda for an
existing run, without re-running the val/ood lambda sweep (those numbers are
already known for any lambda in cfg.sweep.lambdas -- see metrics/lambda_sweep.csv).

Why this exists (2026-08-04 grilling session, v3-r16): `src.pipeline.stage_sweep_gate`
picks the LARGEST lambda that keeps OOD CER within budget, not the cost/benefit
elbow. On v3-r16's own sweep, cost-per-benefit (delta_ood_cer / delta_val_cer)
jumps ~12x between lambda=0.5 and 0.75, then ~31x by 1.0 -- the elbow is ~0.5,
matching the predecessor repo's own finding (docs/finetune-results-report-v2.md):
lambda=1.0 caused real truncation/hallucination on real audio that a bare CER
number doesn't fully surface. tier1/tier4a were only ever evaluated once, at
whatever lambda `stage_sweep_gate` picked (1.0 for this run) -- this script gets
those same two tiers' numbers at a different lambda so the choice can be
verified on the one benchmark that matters (real audio) before merging/pushing.

Usage: python -m scripts.reeval_lambda <run_dir> <lambda>
Writes metrics/gate_results_lambda{L}.json and
audit/predictions_{tier1_in_domain,tier4a_real}_lambda{L}.csv -- does NOT
touch the run's official metrics/gate_results.json (that stays the
already-gated record for the lambda `stage_sweep_gate` picked).
"""

import argparse
import json
from pathlib import Path

from src.config import Config, _build
from src.data import ManifestDataset, load_manifests
from src.gate import (_eval_split, _score_by_meeting, score_real, write_predictions)
from src.lora import set_lambda
from src.metrics import bootstrap_ci, bootstrap_delta_ci, verdict
from src.normalize import Normalizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("lam", type=float)
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    lam = args.lam

    from src import compat
    compat.apply()  # must run before any peft import -- see src/compat.py
    from src.asr import load_for_eval

    raw_cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    cfg = _build(Config, raw_cfg)
    baseline = json.loads((run_dir / "metrics" / "baseline.json").read_text(encoding="utf-8"))

    normalizer = Normalizer(
        strip_punctuation=cfg.normalization.strip_punctuation,
        lowercase=cfg.normalization.lowercase,
        number_convention=cfg.normalization.number_convention,
        filler_tokens=cfg.normalization.filler_tokens,
    )

    checkpoint_dir = run_dir / "checkpoints" / "best"
    model, processor = load_for_eval(cfg.base_model, checkpoint_dir)
    set_lambda(model, lam)

    validated = [json.loads(l) for l in
                 (run_dir / "validated_manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    test_ds = ManifestDataset(records=[r for r in validated if r["split"] == "test"],
                               audio_root=Path(cfg.data.dataset_path) / "audio")
    real_ds = None
    if cfg.data.real_bench_path:
        real_records = load_manifests(cfg.data.real_bench_path)
        real_ds = ManifestDataset(records=real_records,
                                   audio_root=Path(cfg.data.real_bench_path) / "audio")

    tag = f"lambda{lam}"
    results = {"lambda": lam}

    test_metrics = _eval_split(model, processor, test_ds, normalizer, cfg.eval,
                                desc=f"reeval:{tag}:tier1_in_domain")
    test_predictions = test_metrics.pop("_predictions")
    tier1_bound = (1 - cfg.gates.min_improvement_pct / 100) * baseline["cer_test"]
    results["tier1_in_domain"] = {
        "cer": test_metrics["cer"], "bound": tier1_bound,
        "pass": test_metrics["cer"] <= tier1_bound,
    }
    write_predictions(test_predictions, run_dir / "audit" / f"predictions_tier1_in_domain_{tag}.csv")

    if cfg.normalization.audit_conversions:
        # Digit-normalization attribution (report §, "how much of the tier1 gain
        # is learning to write numbers vs actual acoustic learning") -- bundled
        # into this same GPU pass rather than a separate re-run.
        alt_convention = ("as_written" if cfg.normalization.number_convention == "word_to_digit"
                           else "word_to_digit")
        alt_normalizer = Normalizer(
            strip_punctuation=cfg.normalization.strip_punctuation,
            lowercase=cfg.normalization.lowercase,
            number_convention=alt_convention,
            filler_tokens=cfg.normalization.filler_tokens,
        )
        alt_test_metrics = _eval_split(model, processor, test_ds, alt_normalizer, cfg.eval,
                                        desc=f"reeval:{tag}:tier1_in_domain_normcheck")
        alt_test_metrics.pop("_predictions")
        results["tier1_in_domain"]["normalization_check"] = {
            cfg.normalization.number_convention: test_metrics["cer"],
            alt_convention: alt_test_metrics["cer"],
            "delta_pp": round(abs(test_metrics["cer"] - alt_test_metrics["cer"]) * 100, 3),
        }

    if real_ds is not None:
        real_metrics = _eval_split(model, processor, real_ds, normalizer, cfg.eval,
                                    desc=f"reeval:{tag}:tier4a_real")
        real_predictions = real_metrics.pop("_predictions")
        write_predictions(real_predictions, run_dir / "audit" / f"predictions_tier4a_real_{tag}.csv")

        real_scored = score_real(real_predictions)
        lo, hi = bootstrap_ci(real_scored["_char_counts"])
        tier4a_bound = baseline["cer_real"] + cfg.gates.real_cer_regression_pp
        results["tier4a_real"] = {
            "cer": real_scored["cer"], "ci": [lo, hi], "bound": tier4a_bound,
            "pass": real_scored["cer"] <= tier4a_bound,
            "by_meeting": _score_by_meeting(real_scored["_rejoined"]),
        }

        baseline_real_csv = run_dir / "audit" / "predictions_baseline_real.csv"
        if baseline_real_csv.exists():
            import csv as _csv_module
            with open(baseline_real_csv, encoding="utf-8") as f:
                base_rows = list(_csv_module.DictReader(f))
            base_counts = score_real(base_rows)["_char_counts"]
            cand_counts = real_scored["_char_counts"]
            if len(base_counts) == len(cand_counts):
                d_lo, d_hi = bootstrap_delta_ci(base_counts, cand_counts)
                results["tier4a_real"]["delta_ci"] = [d_lo, d_hi]
                results["tier4a_real"]["verdict"] = verdict(d_lo, d_hi)

        # length-ratio signal (generated chars / ref chars) -- cheap truncation/
        # hallucination check the predecessor repo used (docs/finetune-results-
        # report-v2.md SS2.2): CER alone can hide this because a model that cuts a
        # clause and one that mishears it the same length score differently even
        # at similar CER.
        rejoined = real_scored["_rejoined"]
        ref_chars = sum(len(r["ref"]) for r in rejoined)
        hyp_chars = sum(len(r["hyp"]) for r in rejoined)
        results["tier4a_real"]["hyp_ref_char_ratio"] = round(hyp_chars / ref_chars, 3) if ref_chars else None

    from src.lora import save_with_lambda
    adapter_dir = save_with_lambda(model, lam, run_dir / f"adapter_{tag}")

    out_path = run_dir / "metrics" / f"gate_results_{tag}.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"lambda={lam}")
    print(f"tier1_in_domain cer: {results['tier1_in_domain']['cer']:.4f}  "
          f"bound: {results['tier1_in_domain']['bound']:.4f}  pass: {results['tier1_in_domain']['pass']}")
    if "normalization_check" in results["tier1_in_domain"]:
        print(f"tier1_in_domain normalization_check: {results['tier1_in_domain']['normalization_check']}")
    if "tier4a_real" in results:
        t4 = results["tier4a_real"]
        print(f"tier4a_real cer:     {t4['cer']:.4f}  bound: {t4['bound']:.4f}  pass: {t4['pass']}")
        print(f"delta_ci: {t4.get('delta_ci')}  verdict: {t4.get('verdict')}")
        print(f"hyp/ref char ratio: {t4.get('hyp_ref_char_ratio')}  "
              "(compare against the lambda=1.0 run's own ratio -- lower means more truncation)")
    print(f"written: {out_path}")
    print(f"adapter (lambda={lam} baked in): {adapter_dir}")


if __name__ == "__main__":
    main()
