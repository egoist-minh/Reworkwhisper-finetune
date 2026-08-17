"""Tests for the pure-logic pieces of src/pipeline.py. No model/GPU needed.
SESSIONS.md E2: cost/benefit lambda selection replacing "largest lambda
within OOD budget"."""

import pytest

from src.pipeline import select_lambda, _write_sweep_csv, _cer_by_source

# v3-r16's actual metrics/lambda_sweep.csv (Outputs/v3-r16/metrics/lambda_sweep.csv).
V3_R16_SWEEP_ROWS = [
    {"lambda": 0.0, "val_cer": 0.05829596412556054, "ood_cer": 0.022832927484899276},
    {"lambda": 0.25, "val_cer": 0.025579485883913344, "ood_cer": 0.023402169444356597},
    {"lambda": 0.5, "val_cer": 0.01597928377439525, "ood_cer": 0.024888523449606275},
    {"lambda": 0.75, "val_cer": 0.01200025263689762, "ood_cer": 0.0326365390088865},
    {"lambda": 1.0, "val_cer": 0.010231794353565337, "ood_cer": 0.041428164827171814},
]
V3_R16_BASELINE_OOD_CER = 0.022832927484899276  # baseline.json's cer_ood
V3_R16_OOD_CER_BUDGET = 0.02

# v4-mixed-r16's actual metrics/lambda_sweep.csv. Unlike v3-r16's, the first
# step (0.0 -> 0.25) *improves* OOD CER, so its cost/benefit ratio is negative.
V4_MIXED_R16_SWEEP_ROWS = [
    {"lambda": 0.0, "val_cer": 0.10337330648114244, "ood_cer": 0.022832927484899276},
    {"lambda": 0.25, "val_cer": 0.054833394361039914, "ood_cer": 0.022611555611776985},
    {"lambda": 0.5, "val_cer": 0.036364884657634565, "ood_cer": 0.026185130135036844},
    {"lambda": 0.75, "val_cer": 0.03316093006224826, "ood_cer": 0.033427152841466114},
    {"lambda": 1.0, "val_cer": 0.03256590992310509, "ood_cer": 0.04256664874608646},
]


def test_select_lambda_picks_elbow_not_largest_budget_safe_lambda():
    # All five lambdas are within the 0.02 OOD budget (even 1.0 regresses only
    # 0.0186), so "largest within budget" picks 1.0. The elbow rule must pick
    # 0.5 instead -- the step to 0.75 costs ~12.6x the previous step's OOD
    # regression per unit of val-CER gain, well past the 10x threshold.
    best = select_lambda(V3_R16_SWEEP_ROWS, V3_R16_BASELINE_OOD_CER,
                          V3_R16_OOD_CER_BUDGET, elbow_ratio_threshold=10.0)
    assert best == 0.5


def test_select_lambda_respects_budget_as_a_hard_constraint():
    # A tight budget that only lambda=0.0 satisfies -- elbow logic never runs.
    best = select_lambda(V3_R16_SWEEP_ROWS, V3_R16_BASELINE_OOD_CER,
                          ood_cer_budget=0.0002, elbow_ratio_threshold=10.0)
    assert best == 0.0


def test_select_lambda_raises_when_no_lambda_fits_budget():
    rows = [{"lambda": 0.0, "val_cer": 0.1, "ood_cer": 0.05}]
    with pytest.raises(RuntimeError, match="HARD FAIL"):
        select_lambda(rows, baseline_ood_cer=0.0, ood_cer_budget=0.01,
                      elbow_ratio_threshold=10.0)


def test_select_lambda_ignores_rows_with_no_ood_eval():
    rows = [
        {"lambda": 0.0, "val_cer": 0.1, "ood_cer": None},
        {"lambda": 0.5, "val_cer": 0.05, "ood_cer": 0.01},
    ]
    best = select_lambda(rows, baseline_ood_cer=0.01, ood_cer_budget=0.02,
                          elbow_ratio_threshold=10.0)
    assert best == 0.5


def test_select_lambda_a_free_step_does_not_reject_every_later_lambda():
    # v4-mixed-r16's real sweep. The 0.0 -> 0.25 step lowers OOD CER
    # (0.022833 -> 0.022612), so its ratio is negative -- it bought val CER for
    # free. A negative ratio must not become the elbow baseline: threshold * a
    # negative number is a negative bar, which every positive ratio clears, so
    # the walk used to stop dead at 0.25. 0.5 is the real elbow here -- val CER
    # 0.0364 vs 0.0548 and OOD 0.0262 against a 0.0428 bound -- and the step to
    # 0.75 is what the 10x rule should reject (11.7x the 0.5 step's ratio).
    best = select_lambda(V4_MIXED_R16_SWEEP_ROWS, V3_R16_BASELINE_OOD_CER,
                          V3_R16_OOD_CER_BUDGET, elbow_ratio_threshold=10.0)
    assert best == 0.5


def test_select_lambda_higher_threshold_lets_the_elbow_through():
    # A permissive threshold accepts the same 12.6x jump that the default
    # (10.0) rejects, walking all the way to lambda=1.0.
    best = select_lambda(V3_R16_SWEEP_ROWS, V3_R16_BASELINE_OOD_CER,
                          V3_R16_OOD_CER_BUDGET, elbow_ratio_threshold=20.0)
    assert best == 1.0


# ------------------------------------------------- lambda_sweep.csv per-slice cols

def test_cer_by_source_splits_predictions_by_meeting_source():
    predictions = [
        {"meeting_id": "m1", "ref": "aaaa", "hyp": "aaaa"},
        {"meeting_id": "m2", "ref": "bbbb", "hyp": "bbba"},
    ]
    meeting_to_source = {"m1": "synthetic", "m2": "youtube"}
    result = _cer_by_source(predictions, meeting_to_source)
    assert result["synthetic"] == 0.0
    assert result["youtube"] > 0.0


def test_write_sweep_csv_includes_per_source_columns(tmp_path):
    rows = [{"lambda": 0.0, "val_cer": 0.05, "ood_cer": 0.02,
             "val_cer_synthetic": 0.04, "val_cer_youtube": 0.09}]
    out = _write_sweep_csv(rows, tmp_path / "lambda_sweep.csv")
    header = out.read_text(encoding="utf-8").splitlines()[0]
    assert "val_cer_synthetic" in header
    assert "val_cer_youtube" in header


def test_write_sweep_csv_backward_compatible_without_per_source_columns(tmp_path):
    # Old callers (or a synthetic-only run) may not have per-source CER --
    # DictWriter must not choke on rows missing those keys.
    rows = [{"lambda": 0.0, "val_cer": 0.05, "ood_cer": 0.02}]
    out = _write_sweep_csv(rows, tmp_path / "lambda_sweep.csv")
    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
