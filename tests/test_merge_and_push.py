"""Tests for scripts/merge_and_push.py:check_no_regression_vs_production
(SESSIONS.md H4a) -- the check the gate itself never runs, since every tier
compares a candidate only against its own base-model baseline, never against
whatever run is currently serving production. No torch/peft needed: this
function only reads two predictions CSVs and calls src.metrics.score.

The real-CSV case reproduces H3's finding (v4-mixed-r16 regresses vs v3-r16 by
51% relative on 426 shared synthetic segments) against Outputs/ local run
artifacts, skipped when absent."""

import csv
from pathlib import Path

import pytest

from scripts.merge_and_push import check_no_regression_vs_production

V3_CSV = Path("Outputs/v3-r16/audit/predictions_tier1_in_domain.csv")
V4MIX_CSV = Path("Outputs/v4-mixed-r16/audit/predictions_tier1_in_domain.csv")
pytestmark_real = pytest.mark.skipif(
    not (V3_CSV.exists() and V4MIX_CSV.exists()),
    reason="v3-r16 / v4-mixed-r16 local run artifacts not present on this machine",
)


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["segment_id", "meeting_id", "ref", "hyp"])
        writer.writeheader()
        writer.writerows(rows)


def test_raises_when_candidate_cer_is_worse_than_production_on_shared_segments(tmp_path):
    shared = [
        {"segment_id": "seg_0000", "meeting_id": "m1", "ref": "xin chào các bạn", "hyp": "xin chào các bạn"},
        {"segment_id": "seg_0001", "meeting_id": "m1", "ref": "hôm nay họp lúc chín giờ", "hyp": "hôm nay họp lúc chín giờ"},
    ]
    production = shared
    candidate = [
        {**shared[0], "hyp": "xin chào các bạn"},
        {**shared[1], "hyp": "hôm nay hộp lúc chín giờ"},  # one char wrong -> worse CER
    ]
    prod_csv, cand_csv = tmp_path / "prod.csv", tmp_path / "cand.csv"
    _write_csv(prod_csv, production)
    _write_csv(cand_csv, candidate)

    with pytest.raises(RuntimeError, match="regresses vs production"):
        check_no_regression_vs_production(cand_csv, prod_csv)


def test_does_not_raise_when_candidate_matches_or_improves_on_production(tmp_path):
    rows = [
        {"segment_id": "seg_0000", "meeting_id": "m1", "ref": "xin chào các bạn", "hyp": "xin chào các bạn"},
    ]
    prod_csv, cand_csv = tmp_path / "prod.csv", tmp_path / "cand.csv"
    _write_csv(prod_csv, rows)
    _write_csv(cand_csv, rows)

    cand_cer, prod_cer, n = check_no_regression_vs_production(cand_csv, prod_csv)
    assert n == 1
    assert cand_cer == prod_cer == 0.0


def test_raises_when_shared_key_has_different_ref_text(tmp_path):
    prod_csv, cand_csv = tmp_path / "prod.csv", tmp_path / "cand.csv"
    _write_csv(prod_csv, [{"segment_id": "seg_0000", "meeting_id": "m1", "ref": "a", "hyp": "a"}])
    _write_csv(cand_csv, [{"segment_id": "seg_0000", "meeting_id": "m1", "ref": "b", "hyp": "b"}])

    with pytest.raises(RuntimeError, match="different `ref` text"):
        check_no_regression_vs_production(cand_csv, prod_csv)


def test_raises_when_no_keys_are_shared(tmp_path):
    prod_csv, cand_csv = tmp_path / "prod.csv", tmp_path / "cand.csv"
    _write_csv(prod_csv, [{"segment_id": "seg_0000", "meeting_id": "m1", "ref": "a", "hyp": "a"}])
    _write_csv(cand_csv, [{"segment_id": "seg_0001", "meeting_id": "m2", "ref": "a", "hyp": "a"}])

    with pytest.raises(RuntimeError, match="share no"):
        check_no_regression_vs_production(cand_csv, prod_csv)


@pytestmark_real
def test_v4_mixed_r16_regresses_vs_v3_r16_on_shared_synthetic_segments():
    # SESSIONS.md H3: 426 shared segments, 0.0258 (v4-mixed-r16) vs 0.0171
    # (v3-r16) -- 51% relative regression. This is the exact case H4a exists
    # to catch; not raising here means the check does not work.
    with pytest.raises(RuntimeError, match="regresses vs production"):
        check_no_regression_vs_production(V4MIX_CSV, V3_CSV)
