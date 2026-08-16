"""Tests for scripts/inspect_errors.py. No model/GPU needed.

SESSIONS.md E5: consolidates ad hoc scratch analyses (edit-char ranking,
loanword-drop detection, digit-mismatch detection) into a reusable tool.
Synthetic cases pin exact behavior; the real-CSV cases run against
v3-r16's actual audit output (Outputs/, gitignored local artifact) when
present, skipped otherwise.

Honest gap: the real-v3-r16 numbers this reproduces are NOT bit-identical
to the ad hoc figures noted in SESSIONS.md E5 (23% / 83 of 361), because
that analysis's exact original code is gone (summarized-away session).
digit_mismatch_segments does reproduce its figure (42/183) exactly against
predictions_tier2_ood.csv. The other two are close in rate but not count --
see SESSIONS.md E5 for the numbers this actually produces and why."""

from pathlib import Path

import pytest

from scripts.inspect_errors import (
    is_vietnamese_shaped, strip_tone, rank_by_edit_chars,
    top_segment_edit_share, loanword_dropped_segments,
    digit_mismatch_segments, load_predictions,
)

AUDIT_DIR = Path("Outputs/v3-r16/audit")
pytestmark_real = pytest.mark.skipif(
    not AUDIT_DIR.exists(), reason="v3-r16 local run artifacts not present on this machine"
)


def test_is_vietnamese_shaped_accepts_common_words_with_digraphs_and_diphthongs():
    # PROJECT_CORE.md §4's literal one-consonant/one-vowel pattern rejects all
    # of these (digraph onsets kh/ph/tr/nh/th, diphthong/triphthong nuclei).
    for word in ["không", "cũng", "được", "nhưng", "trước", "điều", "phương", "hoặc", "tuần"]:
        assert is_vietnamese_shaped(word), word


def test_is_vietnamese_shaped_rejects_english_loanwords():
    for word in ["report", "model", "dependency", "mapping", "rollback"]:
        assert not is_vietnamese_shaped(word), word


def test_strip_tone_keeps_special_vowel_letters():
    assert strip_tone("được") == "đươc"
    assert strip_tone("không") == "không"  # no tone mark to begin with


def test_rank_by_edit_chars_orders_by_absolute_count_not_rate():
    rows = [
        {"segment_id": "s0", "meeting_id": "m", "ref": "a" * 3, "hyp": "b" * 3},   # 3 edits, 100% rate
        {"segment_id": "s1", "meeting_id": "m", "ref": "a" * 100, "hyp": "b" * 20 + "a" * 80},  # 20 edits, 20% rate
    ]
    ranked = rank_by_edit_chars(rows)
    assert [r["segment_id"] for r in ranked] == ["s1", "s0"]


def test_top_segment_edit_share():
    rows = [
        {"segment_id": "s0", "meeting_id": "m", "ref": "aaaa", "hyp": "bbbb"},  # 4 edits
        {"segment_id": "s1", "meeting_id": "m", "ref": "aa", "hyp": "aa"},      # 0 edits
    ]
    result = top_segment_edit_share(rows)
    assert result["segment_id"] == "s0"
    assert result["share"] == 1.0


def test_loanword_dropped_segments_flags_missing_candidate():
    rows = [
        {"segment_id": "s0", "meeting_id": "m", "ref": "chúng ta cần xem report này", "hyp": "chúng ta cần xem này"},
        {"segment_id": "s1", "meeting_id": "m", "ref": "chúng ta cần xem report này", "hyp": "chúng ta cần xem report này"},
        {"segment_id": "s2", "meeting_id": "m", "ref": "không có từ nào lạ ở đây cả", "hyp": "không có từ nào lạ ở đây cả"},
    ]
    result = loanword_dropped_segments(rows)
    assert result["n_candidate_segments"] == 2  # s0, s1 contain "report"; s2 has none
    assert result["dropped_segment_ids"] == ["s0"]


def test_digit_mismatch_segments_flags_differing_digit_sequence():
    rows = [
        {"segment_id": "s0", "meeting_id": "m", "ref": "có 4 chiếc trống", "hyp": "có 4 chiếc trống"},
        {"segment_id": "s1", "meeting_id": "m", "ref": "có 4 chiếc trống", "hyp": "có 5 chiếc trống"},
        {"segment_id": "s2", "meeting_id": "m", "ref": "không có số nào", "hyp": "không có số nào"},
    ]
    result = digit_mismatch_segments(rows)
    assert result["n_segments_with_digits"] == 2  # s0, s1
    assert result["mismatch_segment_ids"] == ["s1"]


@pytestmark_real
def test_digit_mismatch_reproduces_vivos_figure_on_tier2_ood():
    rows = load_predictions(AUDIT_DIR / "predictions_tier2_ood.csv")
    result = digit_mismatch_segments(rows)
    assert (result["n_mismatch_segments"], result["n_segments_with_digits"]) == (42, 183)


@pytestmark_real
def test_top_segment_edit_share_on_real_bench_is_the_known_bad_segment():
    # real_0002/seg_0074: 4.2s audio holding ~2000 chars of text (see
    # tests/test_ingest_real_bench.py's E1 speech-rate check on the same
    # segment) -- consistently the single largest edit-char contributor.
    rows = load_predictions(AUDIT_DIR / "predictions_tier4a_real.csv")
    result = top_segment_edit_share(rows)
    assert result["meeting_id"] == "real_0002"
    assert result["segment_id"] == "seg_0074"
    assert result["share"] > 0.15


@pytestmark_real
def test_loanword_dropped_segments_smoke_on_test_set():
    rows = load_predictions(AUDIT_DIR / "predictions_tier1_in_domain.csv")
    result = loanword_dropped_segments(rows)
    assert result["n_candidate_segments"] > 200
    assert result["n_dropped_segments"] > 40
