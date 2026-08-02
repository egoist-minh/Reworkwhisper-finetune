"""Tests for the pure-logic pieces of src/gate.py added 2026-08-02 for tier
4a rigor: per-meeting breakdown and reconstructing Counts from a predictions
CSV for the paired baseline-vs-candidate comparison. No model/GPU needed."""

import csv
from pathlib import Path

from src.gate import _score_by_meeting, _load_char_counts_from_predictions


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
