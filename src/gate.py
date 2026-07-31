"""Eval gate. PROJECT_CORE.md §6 Stage 4.

Scope cut for the 2026-08-02 deadline (see handoff): only tiers 1 (in-domain),
2 (OOD), and 4a (real audio, segmented) run. Tier 3 (RTF) is dropped and
tier 4b (long-form) is deferred -- do not re-expand scope here without
updating configs/experiment.yaml + PROJECT_CORE.md first.

Any tier FAIL halts the pipeline: no adapter is marked pass, nothing pushed
to HF (§0 problem 4, "Fail Loudly").
"""

from pathlib import Path

from src.asr import transcribe_batch
from src.metrics import score, bootstrap_ci


def _eval_split(model, processor, dataset, normalizer, eval_cfg) -> dict:
    """Returns the usual score() dict plus `_predictions`: one row per segment
    (segment_id, meeting_id, ref, hyp) so a run's evidence includes what the
    model actually said, not just the aggregate CER (§0 "Evidence-Based")."""
    n = min(len(dataset), eval_cfg.limit) if eval_cfg.limit else len(dataset)
    items = [dataset[i] for i in range(n)]
    audios = [it["audio"] for it in items]
    refs = [normalizer(it["text"]) for it in items]
    hyps_raw = transcribe_batch(model, processor, audios, language=eval_cfg.language,
                                 num_beams=eval_cfg.num_beams, batch_size=eval_cfg.batch_size)
    hyps = [normalizer(h) for h in hyps_raw]
    result = score(refs, hyps)
    result["_predictions"] = [
        {"segment_id": it.get("segment_id"), "meeting_id": it.get("meeting_id"),
         "ref": r, "hyp": h}
        for it, r, h in zip(items, refs, hyps)
    ]
    return result


def run_gate(cfg, model, processor, normalizer, test_ds, ood_ds, real_ds, baseline: dict) -> dict:
    """cfg: src.config.Config. `baseline` is metrics/baseline.json's parsed dict
    (cer_test, cer_ood, cer_real -- all computed once at Stage 1, never
    recomputed here -- PROJECT_CORE.md §2.1 invariant 3). `results["_predictions"]`
    holds per-tier prediction rows for `write_predictions` -- pop it before
    treating `results` as pure tier/pass data (e.g. `overall_pass`)."""
    results = {}
    predictions = {}

    test_metrics = _eval_split(model, processor, test_ds, normalizer, cfg.eval)
    predictions["tier1_in_domain"] = test_metrics.pop("_predictions")
    tier1_bound = (1 - cfg.gates.min_improvement_pct / 100) * baseline["cer_test"]
    results["tier1_in_domain"] = {
        "cer": test_metrics["cer"], "bound": tier1_bound,
        "pass": test_metrics["cer"] <= tier1_bound,
    }

    ood_metrics = _eval_split(model, processor, ood_ds, normalizer, cfg.eval)
    predictions["tier2_ood"] = ood_metrics.pop("_predictions")
    tier2_bound = baseline["cer_ood"] + cfg.sweep.ood_cer_budget
    results["tier2_ood"] = {
        "cer": ood_metrics["cer"], "bound": tier2_bound,
        "pass": ood_metrics["cer"] <= tier2_bound,
    }

    if real_ds is not None:
        real_metrics = _eval_split(model, processor, real_ds, normalizer, cfg.eval)
        predictions["tier4a_real"] = real_metrics.pop("_predictions")
        lo, hi = bootstrap_ci(real_metrics["_char_counts"])
        tier4a_bound = baseline["cer_real"] + cfg.gates.real_cer_regression_pp
        results["tier4a_real"] = {
            "cer": real_metrics["cer"], "ci": [lo, hi], "bound": tier4a_bound,
            "pass": real_metrics["cer"] <= tier4a_bound,
        }
    else:
        results["tier4a_real"] = {"pass": None, "note": "real_bench_path not configured"}

    results["overall_pass"] = all(
        t.get("pass") is not False for t in results.values() if isinstance(t, dict)
    )
    results["_predictions"] = predictions
    return results


def write_gate_results(results: dict, out_dir: str | Path) -> Path:
    import json

    out_dir = Path(out_dir)
    predictions = results.pop("_predictions", {})

    out = out_dir / "metrics" / "gate_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    for tier, rows in predictions.items():
        write_predictions(rows, out_dir / "audit" / f"predictions_{tier}.csv")
    return out


def write_predictions(rows: list[dict], out_path: str | Path) -> Path:
    import csv

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["segment_id", "meeting_id", "ref", "hyp"])
        writer.writeheader()
        writer.writerows(rows)
    return out_path
