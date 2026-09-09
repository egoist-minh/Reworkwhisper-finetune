"""Tests for scripts/merge_and_push.py:check_no_regression_vs_production
(SESSIONS.md H4a) -- the check the gate itself never runs, since every tier
compares a candidate only against its own base-model baseline, never against
whatever run is currently serving production. No torch/peft needed: this
function only reads two predictions CSVs and calls src.metrics.score.

The real-CSV case reproduces H3's finding (v4-mixed-r16 regresses vs v3-r16 by
51% relative on 426 shared synthetic segments) against Outputs/ local run
artifacts, skipped when absent.

Also covers `check_retention_vs_production` and `check_module4_evidence` (added
2026-08-19), the pair that gates a push on module-4 predictions -- same joins, no
torch, and the CSVs they read are `scripts/eval_module4.py` output rather than the
gate's, so they are just as testable from a tmp_path fixture."""

import csv
from pathlib import Path

import pytest

from scripts.merge_and_push import (check_module4_evidence,
                                     check_no_regression_vs_production,
                                     check_retention_vs_production)

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


# --- module-4 evidence gate (2026-08-19) ---------------------------------------
# check_no_regression_vs_production is unchanged and covered above; these cover
# the retention condition added beside it and the wrapper that runs both on both
# splits using scripts/eval_module4.py output.

def _module4_dir(root: Path, real: list[dict], test: list[dict]) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    _write_csv(root / "predictions_tier4a_real.csv", real)
    _write_csv(root / "predictions_tier1_in_domain.csv", test)
    return root


REF_WITH_LOANWORDS = "ờ team mình sẽ build cái pipeline này trong sprint tới"


def test_retention_raises_when_candidate_substitutes_loanwords(tmp_path):
    # `team`->`tim`, `build`->`bill`: 4 edit characters over a 52-character
    # reference, which CER cannot separate from noise -- retention sees it.
    prod = [{"segment_id": "seg_0000", "meeting_id": "m1", "ref": REF_WITH_LOANWORDS,
             "hyp": REF_WITH_LOANWORDS}]
    cand = [{**prod[0], "hyp": REF_WITH_LOANWORDS.replace("team", "tim").replace("build", "bill")}]
    prod_csv, cand_csv = tmp_path / "prod.csv", tmp_path / "cand.csv"
    _write_csv(prod_csv, prod)
    _write_csv(cand_csv, cand)

    with pytest.raises(RuntimeError, match="loses English tokens vs production"):
        check_retention_vs_production(cand_csv, prod_csv, max_regression_pp=0.0)


def test_retention_allows_a_drop_inside_the_configured_tolerance(tmp_path):
    prod = [{"segment_id": "seg_0000", "meeting_id": "m1", "ref": REF_WITH_LOANWORDS,
             "hyp": REF_WITH_LOANWORDS}]
    cand = [{**prod[0], "hyp": REF_WITH_LOANWORDS.replace("team", "tim")}]
    prod_csv, cand_csv = tmp_path / "prod.csv", tmp_path / "cand.csv"
    _write_csv(prod_csv, prod)
    _write_csv(cand_csv, cand)

    cand_r, prod_r, n = check_retention_vs_production(cand_csv, prod_csv,
                                                      max_regression_pp=50.0)
    assert prod_r == 1.0 and cand_r < prod_r and n == 1


def test_retention_skips_when_there_are_no_english_shaped_tokens(tmp_path):
    rows = [{"segment_id": "seg_0000", "meeting_id": "m1", "ref": "chào các bạn",
             "hyp": "chào các bạn"}]
    prod_csv, cand_csv = tmp_path / "prod.csv", tmp_path / "cand.csv"
    _write_csv(prod_csv, rows)
    _write_csv(cand_csv, rows)

    cand_r, prod_r, n = check_retention_vs_production(cand_csv, prod_csv,
                                                      max_regression_pp=0.0)
    assert cand_r is None and prod_r is None and n == 1


def test_module4_evidence_passes_when_both_splits_match_production(tmp_path):
    real = [{"segment_id": "seg_0000", "meeting_id": "real_0001",
             "ref": REF_WITH_LOANWORDS, "hyp": REF_WITH_LOANWORDS}]
    test = [{"segment_id": "seg_0000", "meeting_id": "m1",
             "ref": REF_WITH_LOANWORDS, "hyp": REF_WITH_LOANWORDS}]
    cand = _module4_dir(tmp_path / "cand", real, test)
    prod = _module4_dir(tmp_path / "prod", real, test)

    check_module4_evidence(cand, prod, max_retention_regression_pp=0.0)


def test_module4_evidence_raises_when_only_the_real_split_regresses(tmp_path):
    # The test split alone would clear the push. tier4a is where the reported v5
    # production regression lives, so a wrapper that checked one split would be
    # blind to exactly the case it was written for.
    good = [{"segment_id": "seg_0000", "meeting_id": "m1",
             "ref": "hôm nay họp lúc chín giờ", "hyp": "hôm nay họp lúc chín giờ"}]
    worse = [{**good[0], "meeting_id": "real_0001", "hyp": "hôm nay hộp lúc chín giờ"}]
    prod_real = [{**good[0], "meeting_id": "real_0001"}]
    cand = _module4_dir(tmp_path / "cand", worse, good)
    prod = _module4_dir(tmp_path / "prod", prod_real, good)

    with pytest.raises(RuntimeError, match="regresses vs production"):
        check_module4_evidence(cand, prod, max_retention_regression_pp=0.0)


def test_module4_evidence_raises_when_a_split_was_never_measured(tmp_path):
    rows = [{"segment_id": "seg_0000", "meeting_id": "m1", "ref": "a", "hyp": "a"}]
    cand = _module4_dir(tmp_path / "cand", rows, rows)
    prod = _module4_dir(tmp_path / "prod", rows, rows)
    (cand / "predictions_tier4a_real.csv").unlink()

    with pytest.raises(RuntimeError, match="scripts.eval_module4 --split real"):
        check_module4_evidence(cand, prod, max_retention_regression_pp=0.0)
