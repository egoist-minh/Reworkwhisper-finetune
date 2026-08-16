"""Tests for the pure-logic pieces of src/gate.py added 2026-08-02 for tier
4a rigor: per-meeting breakdown and reconstructing Counts from a predictions
CSV for the paired baseline-vs-candidate comparison. No model/GPU needed."""

import csv
from pathlib import Path

from src.gate import (_score_by_meeting, _load_char_counts_from_predictions,
                       rejoin_real_chunks, score_real,
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
