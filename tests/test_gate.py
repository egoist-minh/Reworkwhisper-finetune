"""Tests for the pure-logic pieces of src/gate.py added 2026-08-02 for tier
4a rigor: per-meeting breakdown and reconstructing Counts from a predictions
CSV for the paired baseline-vs-candidate comparison. No model/GPU needed."""

import csv
from pathlib import Path

import pytest

from src.gate import (_score_by_meeting, _load_char_counts_from_predictions,
                       rejoin_real_chunks, score_real, score_cross_domain,
                       _meeting_to_source, _score_by_source)


def test_score_by_meeting_splits_by_meeting_id():
    predictions = [
        {"segment_id": "seg_0000", "meeting_id": "real_0001", "ref": "hello", "hyp": "hello"},
        {"segment_id": "seg_0001", "meeting_id": "real_0001", "ref": "world", "hyp": "world"},
        {"segment_id": "seg_0000", "meeting_id": "real_0002", "ref": "abc", "hyp": "xyz"},
    ]
    by_meeting = _score_by_meeting(predictions)
    assert set(by_meeting) == {"real_0001", "real_0002"}
    assert by_meeting["real_0001"]["cer"] == 0.0
    assert by_meeting["real_0001"]["n_segments"] == 2
    assert by_meeting["real_0002"]["n_segments"] == 1
    assert by_meeting["real_0002"]["cer"] > 0


def test_score_by_meeting_one_bad_recording_does_not_hide_in_pooled_average():
    # real_0001 perfect, real_0002 all-wrong -- a pooled CER would look "only
    # somewhat bad"; the per-meeting breakdown must show real_0002 as 100% wrong.
    predictions = [
        {"segment_id": "s0", "meeting_id": "real_0001", "ref": "aaaa", "hyp": "aaaa"},
        {"segment_id": "s0", "meeting_id": "real_0002", "ref": "bbbb", "hyp": "zzzz"},
    ]
    by_meeting = _score_by_meeting(predictions)
    assert by_meeting["real_0001"]["cer"] == 0.0
    assert by_meeting["real_0002"]["cer"] == 1.0


def test_load_char_counts_from_predictions_matches_manual_computation(tmp_path):
    csv_path = tmp_path / "predictions_baseline_real.csv"
    rows = [
        {"segment_id": "seg_0000", "meeting_id": "real_0001", "ref": "hello", "hyp": "hallo"},
        {"segment_id": "seg_0001", "meeting_id": "real_0001", "ref": "world", "hyp": "world"},
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["segment_id", "meeting_id", "ref", "hyp"])
        writer.writeheader()
        writer.writerows(rows)

    counts = _load_char_counts_from_predictions(csv_path)
    assert len(counts) == 2
    assert counts[0].edits == 1  # hello -> hallo, 1 substitution
    assert counts[0].ref_len == 5
    assert counts[1].edits == 0


def test_rejoin_real_chunks_reproduces_boundary_shifted_text():
    # scripts/ingest_real_bench.py splits text proportionally by chunk duration,
    # not real alignment -- a fast talker near a chunk boundary can push a
    # trailing word into the next chunk's hyp even though the model heard it
    # correctly. Rejoining must recover the correct comparison regardless.
    predictions = [
        {"segment_id": "seg_0000_0", "meeting_id": "real_0001",
         "ref": "one two", "hyp": "one"},
        {"segment_id": "seg_0000_1", "meeting_id": "real_0001",
         "ref": "three four", "hyp": "two three four"},
    ]
    rejoined = rejoin_real_chunks(predictions)
    assert len(rejoined) == 1
    assert rejoined[0]["segment_id"] == "seg_0000"
    assert rejoined[0]["ref"] == "one two three four"
    assert rejoined[0]["hyp"] == "one two three four"


def test_rejoin_real_chunks_orders_by_numeric_suffix_not_string_sort():
    predictions = [
        {"segment_id": f"seg_0000_{i}", "meeting_id": "real_0001", "ref": w, "hyp": w}
        for i, w in enumerate(["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k"])
    ]
    rejoined = rejoin_real_chunks(predictions)
    assert rejoined[0]["ref"] == "a b c d e f g h i j k"


def test_rejoin_real_chunks_leaves_single_chunk_segments_unmerged():
    predictions = [
        {"segment_id": "seg_0000", "meeting_id": "real_0001", "ref": "aaaa", "hyp": "aaaa"},
        {"segment_id": "seg_0001", "meeting_id": "real_0001", "ref": "bbbb", "hyp": "bbbb"},
    ]
    rejoined = rejoin_real_chunks(predictions)
    assert {r["segment_id"] for r in rejoined} == {"seg_0000", "seg_0001"}


def test_score_real_boundary_shift_inflates_chunk_level_but_not_rejoined():
    # Same underlying transcript as the boundary-shift test above: scoring the
    # raw chunks directly overstates CER because "two" lands in the wrong
    # chunk's hyp; rejoined scoring should show 0 CER since the model actually
    # got every word right.
    predictions = [
        {"segment_id": "seg_0000_0", "meeting_id": "real_0001",
         "ref": "one two", "hyp": "one"},
        {"segment_id": "seg_0000_1", "meeting_id": "real_0001",
         "ref": "three four", "hyp": "two three four"},
    ]
    chunk_level_edits = sum(
        len(p["ref"]) != len(p["hyp"]) or p["ref"] != p["hyp"] for p in predictions
    )
    assert chunk_level_edits > 0  # raw chunk comparison sees mismatches
    assert score_real(predictions)["cer"] == 0.0  # rejoined sees the true (perfect) transcript


# --------------------------------------------------- tier1_in_domain by_source

def test_meeting_to_source_maps_from_records():
    records = [{"meeting_id": "m1", "source": "synthetic"}, {"meeting_id": "m2", "source": "youtube"}]
    assert _meeting_to_source(records) == {"m1": "synthetic", "m2": "youtube"}


def test_score_by_source_splits_and_reports_ci_without_baseline():
    predictions = [
        {"segment_id": "s0", "meeting_id": "m1", "ref": "hello world", "hyp": "hello world"},
        {"segment_id": "s1", "meeting_id": "m2", "ref": "chao ban", "hyp": "chao bat"},
    ]
    meeting_to_source = {"m1": "synthetic", "m2": "youtube"}
    result = _score_by_source(predictions, meeting_to_source, baseline_rows=None)
    assert set(result) == {"synthetic", "youtube"}
    assert result["synthetic"]["cer"] == 0.0
    assert result["synthetic"]["n_segments"] == 1
    assert "ci" in result["synthetic"]
    assert "cer_baseline" not in result["synthetic"]
    assert result["synthetic"]["verdict"] == "SKIPPED (predictions_baseline_test.csv not found)"


def test_score_by_source_computes_delta_ci_and_verdict_when_paired():
    predictions = [
        {"segment_id": "s0", "meeting_id": "m1", "ref": "aaaa", "hyp": "aaaa"},
        {"segment_id": "s1", "meeting_id": "m2", "ref": "bbbb", "hyp": "bbbb"},
    ]
    baseline_rows = [
        {"segment_id": "s0", "meeting_id": "m1", "ref": "aaaa", "hyp": "aaab"},
        {"segment_id": "s1", "meeting_id": "m2", "ref": "bbbb", "hyp": "bbba"},
    ]
    meeting_to_source = {"m1": "synthetic", "m2": "youtube"}
    result = _score_by_source(predictions, meeting_to_source, baseline_rows)
    assert result["synthetic"]["cer"] == 0.0
    assert result["synthetic"]["cer_baseline"] > 0
    assert "delta_ci" in result["synthetic"]
    assert result["synthetic"]["verdict"] in {"IMPROVED", "REGRESSED", "INCONCLUSIVE"}


def test_score_by_source_skips_verdict_when_segment_counts_mismatch_but_keeps_cer_baseline():
    predictions = [
        {"segment_id": "s0", "meeting_id": "m1", "ref": "aaaa", "hyp": "aaaa"},
        {"segment_id": "s1", "meeting_id": "m1", "ref": "bbbb", "hyp": "bbbb"},
    ]
    baseline_rows = [
        {"segment_id": "s0", "meeting_id": "m1", "ref": "aaaa", "hyp": "aaab"},
    ]
    meeting_to_source = {"m1": "synthetic"}
    result = _score_by_source(predictions, meeting_to_source, baseline_rows)
    assert "SKIPPED" in result["synthetic"]["verdict"]
    assert "cer_baseline" in result["synthetic"]
    assert "delta_ci" not in result["synthetic"]


def test_score_by_source_fails_a_regressing_slice_even_when_the_other_slice_improves():
    # The case the pooled bound cannot catch: youtube (long refs, most of the
    # pooled character denominator) improves a lot, synthetic regresses. Each
    # slice is judged against its own baseline by tier 1's own rule.
    predictions = [
        {"segment_id": "s0", "meeting_id": "syn", "ref": "aaaa", "hyp": "aaab"},   # 25% CER
        {"segment_id": "s0", "meeting_id": "yt", "ref": "b" * 100, "hyp": "b" * 100},
    ]
    baseline_rows = [
        {"segment_id": "s0", "meeting_id": "syn", "ref": "aaaa", "hyp": "aaaa"},   # 0% CER
        {"segment_id": "s0", "meeting_id": "yt", "ref": "b" * 100, "hyp": "c" * 100},
    ]
    meeting_to_source = {"syn": "synthetic", "yt": "youtube"}
    result = _score_by_source(predictions, meeting_to_source, baseline_rows,
                              min_improvement_pct=10.0)
    assert result["youtube"]["pass"] is True
    assert result["synthetic"]["pass"] is False


def test_score_by_source_omits_pass_when_min_improvement_pct_not_given():
    predictions = [{"segment_id": "s0", "meeting_id": "m1", "ref": "aaaa", "hyp": "aaaa"}]
    baseline_rows = [{"segment_id": "s0", "meeting_id": "m1", "ref": "aaaa", "hyp": "aaab"}]
    result = _score_by_source(predictions, {"m1": "synthetic"}, baseline_rows)
    assert "pass" not in result["synthetic"]
    assert "bound" not in result["synthetic"]


def test_score_by_source_skips_source_missing_from_baseline_entirely():
    predictions = [{"segment_id": "s0", "meeting_id": "m2", "ref": "aaaa", "hyp": "aaaa"}]
    baseline_rows = [{"segment_id": "s0", "meeting_id": "m1", "ref": "aaaa", "hyp": "aaaa"}]
    meeting_to_source = {"m1": "synthetic", "m2": "youtube"}
    result = _score_by_source(predictions, meeting_to_source, baseline_rows)
    assert "SKIPPED" in result["youtube"]["verdict"]
    assert "cer_baseline" not in result["youtube"]


# --------------------------------------- tier1_in_domain retention (H4b, H6)

def test_score_by_source_reports_retention_regardless_of_baseline():
    # SESSIONS.md H6: CER alone missed the v4-mixed-r16 loanword-drop symptom
    # -- retention has to show up even when there's no baseline to gate on.
    predictions = [
        {"segment_id": "s0", "meeting_id": "m1", "ref": "chao team hello build", "hyp": "chao tim hello bill"},
    ]
    result = _score_by_source(predictions, {"m1": "synthetic"}, baseline_rows=None)
    assert result["synthetic"]["retention"] == 1 / 3  # only "hello" survives; "team"/"build" lost


def test_score_by_source_retention_pass_fails_when_candidate_drops_more_than_budget():
    # Candidate loses "team" (1 of 2 English tokens), baseline keeps both --
    # a 50pp drop should fail a 5pp budget.
    predictions = [{"segment_id": "s0", "meeting_id": "m1", "ref": "chao team build", "hyp": "chao tim build"}]
    baseline_rows = [{"segment_id": "s0", "meeting_id": "m1", "ref": "chao team build", "hyp": "chao team build"}]
    result = _score_by_source(predictions, {"m1": "synthetic"}, baseline_rows,
                              max_retention_regression_pp=0.05)
    assert result["synthetic"]["retention"] == 0.5
    assert result["synthetic"]["retention_baseline"] == 1.0
    assert result["synthetic"]["retention_pass"] is False


def test_score_by_source_retention_pass_true_when_within_budget():
    predictions = [{"segment_id": "s0", "meeting_id": "m1", "ref": "chao team build", "hyp": "chao team build"}]
    baseline_rows = [{"segment_id": "s0", "meeting_id": "m1", "ref": "chao team build", "hyp": "chao team build"}]
    result = _score_by_source(predictions, {"m1": "synthetic"}, baseline_rows,
                              max_retention_regression_pp=0.05)
    assert result["synthetic"]["retention_pass"] is True


def test_score_by_source_omits_retention_pass_when_slice_has_no_english_tokens():
    # No English-shaped reference tokens on either side -> retention is None
    # on both sides -- absence of loanwords is not a loanword loss.
    predictions = [{"segment_id": "s0", "meeting_id": "m1", "ref": "chao ban", "hyp": "chao ban"}]
    baseline_rows = [{"segment_id": "s0", "meeting_id": "m1", "ref": "chao ban", "hyp": "chao bat"}]
    result = _score_by_source(predictions, {"m1": "synthetic"}, baseline_rows,
                              max_retention_regression_pp=0.05)
    assert result["synthetic"]["retention"] is None
    assert result["synthetic"]["retention_baseline"] is None
    assert "retention_pass" not in result["synthetic"]


def test_score_by_source_omits_retention_pass_when_threshold_not_given():
    predictions = [{"segment_id": "s0", "meeting_id": "m1", "ref": "chao team", "hyp": "chao tim"}]
    baseline_rows = [{"segment_id": "s0", "meeting_id": "m1", "ref": "chao team", "hyp": "chao team"}]
    result = _score_by_source(predictions, {"m1": "synthetic"}, baseline_rows)
    assert "retention" in result["synthetic"]
    assert "retention_pass" not in result["synthetic"]


# ------------------------------------------- cross_domain_bench (v7 plan §2.3)

# Two segments carrying loanwords, one without -- the three numbers the check
# reports are computed over different slices of the same predictions.
CROSS_DOMAIN_PREDICTIONS = [
    {"segment_id": "s0", "meeting_id": "m1", "ref": "chao team", "hyp": "chao tim"},
    {"segment_id": "s1", "meeting_id": "m1", "ref": "mot build", "hyp": "mot build"},
    {"segment_id": "s2", "meeting_id": "m1", "ref": "chao ban", "hyp": "chao ban"},
]


def test_score_cross_domain_splits_segments_by_whether_the_reference_has_a_loanword():
    result = score_cross_domain(CROSS_DOMAIN_PREDICTIONS)
    assert result["n_segments"] == 3
    assert result["n_segments_with_loanword"] == 2
    assert result["n_segments_no_loanword"] == 1
    assert result["cer_no_loanword"] == 0.0
    assert result["retention"] == 0.5      # "build" kept, "team" lost


def test_score_cross_domain_with_no_thresholds_reports_without_gating():
    result = score_cross_domain(CROSS_DOMAIN_PREDICTIONS)
    assert result["pass"] is None
    assert "cer_pass" not in result


def test_score_cross_domain_fails_on_cer_and_on_retention_independently():
    lenient_cer = score_cross_domain(CROSS_DOMAIN_PREDICTIONS, cer_max=1.0,
                                      retention_min=0.9)
    assert lenient_cer["cer_pass"] is True
    assert lenient_cer["retention_pass"] is False
    assert lenient_cer["pass"] is False

    lenient_retention = score_cross_domain(CROSS_DOMAIN_PREDICTIONS, cer_max=0.0,
                                            retention_min=0.4)
    assert lenient_retention["cer_pass"] is False
    assert lenient_retention["retention_pass"] is True
    assert lenient_retention["pass"] is False


def test_score_cross_domain_no_loanword_cer_catches_the_opposite_failure():
    # A candidate that keeps every loanword but wrecks the plain Vietnamese
    # segments passes CER-with-loanwords and retention, and must still fail.
    predictions = [
        {"segment_id": "s0", "meeting_id": "m1", "ref": "chao team", "hyp": "chao team"},
        {"segment_id": "s1", "meeting_id": "m1", "ref": "chao ban", "hyp": "xxxx xxx"},
    ]
    result = score_cross_domain(predictions, cer_max=1.0, retention_min=0.5,
                                 no_loanword_cer_max=0.0586)
    assert result["retention_pass"] is True
    assert result["no_loanword_cer_pass"] is False
    assert result["pass"] is False


def test_score_cross_domain_passes_when_every_set_threshold_is_met():
    result = score_cross_domain(CROSS_DOMAIN_PREDICTIONS, cer_max=1.0, retention_min=0.5,
                                 no_loanword_cer_max=0.1)
    assert result["pass"] is True


def test_score_cross_domain_refuses_a_retention_floor_it_cannot_measure():
    # No non-Vietnamese-shaped reference token anywhere: the threshold has
    # nothing to answer, which is a wrong benchmark rather than a pass.
    predictions = [{"segment_id": "s0", "meeting_id": "m1", "ref": "chao ban", "hyp": "chao ban"}]
    with pytest.raises(ValueError, match="cross_domain_retention_min"):
        score_cross_domain(predictions, retention_min=0.666)


def test_score_cross_domain_refuses_a_no_loanword_bound_it_cannot_measure():
    predictions = [{"segment_id": "s0", "meeting_id": "m1", "ref": "chao team", "hyp": "chao team"}]
    with pytest.raises(ValueError, match="cross_domain_no_loanword_cer_max"):
        score_cross_domain(predictions, no_loanword_cer_max=0.0586)


# The acceptance criterion docs/v6-curriculum-plan.md §2.3 states for this check:
# it must FAIL on the adapter that already exists. Scored from the v6 and v5
# hypotheses recorded by the cross-domain benchmark run, so it needs no GPU --
# but those artifacts are untracked (Outputs/ is gitignored), hence the skip.
_BENCH = Path("Outputs/bench-cross-v6")
_V5_THRESHOLDS = {"cer_max": 0.0819, "retention_min": 0.666}


def _bench_predictions(name: str) -> list[dict]:
    import json

    from src.normalize import Normalizer

    normalizer = Normalizer(strip_punctuation=True, lowercase=True,
                             number_convention="word_to_digit",
                             filler_tokens=["ừm", "ờm", "ehm", "uhm", "hmm"])
    path = _BENCH / f"{name}.cross-domain.persegment.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    return [{"segment_id": r["segment_id"], "meeting_id": r["group_id"],
             "ref": normalizer(r["ref"]), "hyp": normalizer(r["hyp"])} for r in rows]


@pytest.mark.skipif(not _BENCH.is_dir(), reason="Outputs/bench-cross-v6 not on this machine")
def test_cross_domain_check_fails_the_v6_adapter_it_was_written_for():
    result = score_cross_domain(_bench_predictions("rework_whisper_v6"), **_V5_THRESHOLDS)
    assert result["n_segments"] == 299
    assert result["cer_pass"] is False        # 10.85% against an 8.19% bound
    assert result["retention_pass"] is False  # 48.6% against a 66.6% floor
    assert result["pass"] is False


@pytest.mark.skipif(not _BENCH.is_dir(), reason="Outputs/bench-cross-v6 not on this machine")
def test_cross_domain_check_passes_the_v5_adapter_the_thresholds_came_from():
    # Same thresholds, same audio: the adapter in production must clear its own
    # numbers, or the bounds are transcribed wrong.
    result = score_cross_domain(_bench_predictions("winhsss_reworkwhisper_large_v5"),
                                 **_V5_THRESHOLDS)
    assert result["cer_pass"] is True
    assert result["retention_pass"] is True
