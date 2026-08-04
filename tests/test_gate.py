"""Tests for the pure-logic pieces of src/gate.py added 2026-08-02 for tier
4a rigor: per-meeting breakdown and reconstructing Counts from a predictions
CSV for the paired baseline-vs-candidate comparison. No model/GPU needed."""

import csv
from pathlib import Path

from src.gate import (_score_by_meeting, _load_char_counts_from_predictions,
                       rejoin_real_chunks, score_real)


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
