"""Recompute an existing run's tier4a_real gate metrics with the rejoin fix
(src.gate.score_real, added 2026-08-04 -- see gate.py module docstring
"Rejoin-before-scoring"), from that run's own audit CSVs. No model
re-inference needed: predictions_baseline_real.csv / predictions_tier4a_real.csv
already hold every ref/hyp pair the fix needs to rejoin and rescore.

Does not touch tier4a_real.normalization_check -- that field needs the
alt-number-convention model pass, which wasn't persisted to audit/, so it's
left as originally gated (diagnostic only, doesn't affect pass/overall_pass).

Usage: python -m scripts.recompute_tier4a <run_dir>
Backs up the pre-fix metrics/baseline.json and metrics/gate_results.json
alongside the originals (*.pre_rejoin_fix.json) before overwriting.
"""

import argparse
import csv
import json
from pathlib import Path

from src.gate import score_real, _score_by_meeting
from src.metrics import bootstrap_ci, bootstrap_delta_ci, verdict


def _load_rows(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    args = ap.parse_args()
    run_dir = Path(args.run_dir)

    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    regression_pp = cfg["gates"]["real_cer_regression_pp"]

    baseline_rows = _load_rows(run_dir / "audit" / "predictions_baseline_real.csv")
    candidate_rows = _load_rows(run_dir / "audit" / "predictions_tier4a_real.csv")
    baseline_scored = score_real(baseline_rows)
    candidate_scored = score_real(candidate_rows)

    baseline_path = run_dir / "metrics" / "baseline.json"
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    baseline_path.with_suffix(".pre_rejoin_fix.json").write_text(
        json.dumps(baseline, indent=2, ensure_ascii=False), encoding="utf-8")
    baseline["cer_real"] = baseline_scored["cer"]
    baseline_path.write_text(json.dumps(baseline, indent=2, ensure_ascii=False), encoding="utf-8")

    gate_path = run_dir / "metrics" / "gate_results.json"
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    gate_path.with_suffix(".pre_rejoin_fix.json").write_text(
        json.dumps(gate, indent=2, ensure_ascii=False), encoding="utf-8")

    lo, hi = bootstrap_ci(candidate_scored["_char_counts"])
    bound = baseline["cer_real"] + regression_pp
    tier4a = gate["tier4a_real"]
    tier4a["cer"] = candidate_scored["cer"]
    tier4a["ci"] = [lo, hi]
    tier4a["bound"] = bound
    tier4a["pass"] = candidate_scored["cer"] <= bound
    tier4a["by_meeting"] = _score_by_meeting(candidate_scored["_rejoined"])

    if len(baseline_scored["_char_counts"]) == len(candidate_scored["_char_counts"]):
        d_lo, d_hi = bootstrap_delta_ci(baseline_scored["_char_counts"], candidate_scored["_char_counts"])
        tier4a["delta_ci"] = [d_lo, d_hi]
        tier4a["verdict"] = verdict(d_lo, d_hi)

    tier4a["_note"] = ("rescored at parent-segment level, rejoin fix 2026-08-04 -- "
                        "pre-fix chunk-level numbers in gate_results.pre_rejoin_fix.json; "
                        "normalization_check below predates the fix, left as-is")
    gate["tier4a_real"] = tier4a
    gate["overall_pass"] = all(t.get("pass") is not False for t in gate.values() if isinstance(t, dict))

    gate_path.write_text(json.dumps(gate, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"baseline cer_real: {baseline['cer_real']:.4f}")
    print(f"tier4a_real cer:   {tier4a['cer']:.4f}  bound: {bound:.4f}  pass: {tier4a['pass']}")
    print(f"delta_ci: {tier4a.get('delta_ci')}  verdict: {tier4a.get('verdict')}")
    print(f"overall_pass: {gate['overall_pass']}")


if __name__ == "__main__":
    main()
