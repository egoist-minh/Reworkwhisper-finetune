"""Tests for scripts/ingest_youtube.py (SESSIONS.md F3). Unit tests are fully
offline/synthetic; the corpus-level tests run against F2's real, already
downloaded dataset/youtube-meetings/raw/ and are skipped if that is absent."""

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from scripts.draft_sources import Word, load as load_draft, transcript
from scripts.ingest_real_bench import MAX_SEGMENT_SEC
from scripts.ingest_youtube import (
    cut_segments, drop_bracket_words, DEFAULT_TEST_MEETINGS, gap_candidates,
    ingest_meeting, is_bracket_label, middle_window, window_words, RAW_ROOT,
    TRIM_TARGET_SEC,
)
from src.data import ManifestDataset, resolve_splits, split_stats

pytestmark_real = pytest.mark.skipif(
    not RAW_ROOT.exists(), reason="F2 has not fetched dataset/youtube-meetings/raw/ on this machine"
)


def _w(text: str, start: float, end: float) -> Word:
    return Word(text, start, end, None)


def _write_raw_meeting(tmp_path: Path, meeting_id: str, doc: dict, duration_sec: float) -> Path:
    """Build a synthetic dataset/youtube-meetings/raw/<meeting_id>/ directory
    matching F2's output shape, so ingest_meeting can run against it."""
    raw_dir = tmp_path / meeting_id
    raw_dir.mkdir(parents=True)
    sr = 16000
    sf.write(raw_dir / "audio.wav", np.zeros(int(duration_sec * sr), dtype="float32"), sr)
    (raw_dir / "captions.json3").write_text(json.dumps(doc), encoding="utf-8")
    (raw_dir / "provenance.json").write_text(json.dumps({
        "video_id": meeting_id, "video_url": f"https://youtu.be/{meeting_id}",
        "yt_start": 0.0, "yt_end": duration_sec, "download_date": "2026-08-12",
        "sha256": "deadbeef", "source": "youtube", "label_source": "google_asr",
        "asr_draft_model": "vi-orig", "reviewed_by": "", "review_date": "",
    }), encoding="utf-8")
    return raw_dir


# ------------------------------------------------------------- bracket labels

def test_is_bracket_label_matches_exact_strings_f1_measured():
    assert is_bracket_label("[Âm nhạc]")
    assert is_bracket_label("[Hắng giọng]")
    assert not is_bracket_label("chào")
    assert not is_bracket_label("[không đóng]extra")


def test_drop_bracket_words_removes_only_bracket_entries():
    words = [_w("chào", 0.0, 0.5), _w("[Âm nhạc]", 0.5, 1.0), _w("anh", 1.0, 1.5)]
    kept = drop_bracket_words(words)
    assert [w.text for w in kept] == ["chào", "anh"]


# ------------------------------------------------------------ trim window

def test_middle_window_leaves_a_short_meeting_untouched():
    assert middle_window(600.0, target=TRIM_TARGET_SEC) == (0.0, 600.0)


def test_middle_window_centers_a_long_meeting():
    start, end = middle_window(5400.0, target=1800.0)   # 90 min -> middle 30 min
    assert (start, end) == (1800.0, 3600.0)
    assert end - start == 1800.0


def test_window_words_keeps_only_words_inside_and_rebases_to_zero():
    words = [_w("before", 5.0, 5.5), _w("in1", 10.0, 10.5), _w("in2", 15.0, 15.5),
             _w("after", 25.0, 25.5)]
    kept = window_words(words, window_start=10.0, window_end=20.0)
    assert [w.text for w in kept] == ["in1", "in2"]
    assert kept[0].start == 0.0 and kept[1].start == 5.0


# ---------------------------------------------------------------- gap timing

def test_gap_candidates_only_flags_gaps_at_least_min_gap():
    words = [_w("a", 0.0, 0.1), _w("b", 0.1, 0.2), _w("c", 5.0, 5.1)]
    cands = gap_candidates(words, min_gap=0.3)
    assert cands == [(0.1 + 5.0) / 2]   # a->b gap is 0.1s, too small; b->c gap is 4.9s


def test_cut_segments_respects_max_segment_sec_and_preserves_all_words():
    # Words every 1s for 40s -- every consecutive gap clears MIN_GAP_SEC, so
    # _choose_splits has a candidate available near every 15s-since-last-cut
    # mark and cuts there, well before its forced-overflow fallback would fire.
    words = [_w(f"w{i}", float(i), float(i) + 0.5) for i in range(40)]
    spans = cut_segments(words, duration=40.0)

    assert sum(s[1] - s[0] for s in spans) == pytest.approx(40.0)
    for seg_start, seg_end, _ in spans:
        assert seg_end - seg_start <= MAX_SEGMENT_SEC + 1e-6
    total_words = sum(len(seg_words) for _, _, seg_words in spans)
    assert total_words == len(words)


def test_cut_segments_cuts_when_a_candidate_clears_the_since_last_cut_mark():
    words = [_w("a", 0.0, 0.5), _w("b", 1.0, 1.5), _w("c", 20.0, 20.5), _w("d", 21.0, 21.5)]
    spans = cut_segments(words, duration=25.0)
    assert len(spans) == 2
    total_words = sum(len(seg_words) for _, _, seg_words in spans)
    assert total_words == len(words)
    for seg_start, seg_end, _ in spans:
        assert seg_end - seg_start <= MAX_SEGMENT_SEC + 1e-6


# --------------------------------------------------------------- ingest_meeting

JSON3_TWO_SEGMENTS = {
    "events": [
        {"tStartMs": 0, "dDurationMs": 2000, "segs": [
            {"utf8": "chào", "acAsrConf": 0},
            {"utf8": " anh", "tOffsetMs": 500, "acAsrConf": 0},
        ]},
        {"tStartMs": 4000, "dDurationMs": 800, "segs": [{"utf8": "[Âm nhạc]", "acAsrConf": 0}]},
        {"tStartMs": 20000, "dDurationMs": 2000, "segs": [
            {"utf8": "vâng", "acAsrConf": 0},
            {"utf8": " ạ", "tOffsetMs": 500, "acAsrConf": 0},
        ]},
    ],
}

SCRIBE_TWO_SEGMENTS = {
    "words": [
        {"text": "chào", "start": 0.0, "end": 0.5, "type": "word"},
        {"text": "anh", "start": 0.5, "end": 1.0, "type": "word"},
        {"text": "[Âm nhạc]", "start": 4.0, "end": 4.8, "type": "word"},
        {"text": "vâng", "start": 20.0, "end": 20.5, "type": "word"},
        {"text": "ạ", "start": 20.5, "end": 21.0, "type": "word"},
    ],
}


def test_ingest_meeting_drops_bracket_word_and_writes_expected_records(tmp_path):
    raw_root = tmp_path / "raw"
    _write_raw_meeting(raw_root, "m1", JSON3_TWO_SEGMENTS, duration_sec=25.0)
    out_root = tmp_path / "out"

    records = ingest_meeting("m1", "json3", raw_root, out_root)

    assert len(records) == 2
    assert records[0]["text"] == "chào anh vâng"
    assert records[1]["text"] == "ạ"
    assert not any("nhạc" in r["text"] for r in records)   # bracket word never survives
    for r in records:
        assert (out_root / "audio" / r["audio_filepath"]).exists()
        assert r["meeting_id"] == "m1"
        assert r["split"] == "demo"
        assert r["verified"] is False
        assert r["video_id"] == "m1"
        assert r["sha256"] == "deadbeef"


def test_ingest_meeting_clears_stale_wavs_from_a_prior_run_with_more_segments(tmp_path):
    raw_root = tmp_path / "raw"
    _write_raw_meeting(raw_root, "m1", JSON3_TWO_SEGMENTS, duration_sec=25.0)
    out_root = tmp_path / "out"

    audio_dir = out_root / "audio" / "m1"
    audio_dir.mkdir(parents=True)
    for i in range(10):   # simulate a prior run that wrote more segments
        (audio_dir / f"seg_{i:04d}.wav").write_bytes(b"stale")

    records = ingest_meeting("m1", "json3", raw_root, out_root)

    on_disk = sorted(p.name for p in audio_dir.glob("*.wav"))
    expected = sorted(Path(r["audio_filepath"]).name for r in records)
    assert on_disk == expected   # no seg_0002..seg_0009 leftover from the "prior run"


def test_ingest_meeting_is_test_writes_split_test(tmp_path):
    raw_root = tmp_path / "raw"
    _write_raw_meeting(raw_root, "m1", JSON3_TWO_SEGMENTS, duration_sec=25.0)
    records = ingest_meeting("m1", "json3", raw_root, tmp_path / "out", is_test=True)
    assert records and all(r["split"] == "test" for r in records)


def test_ingest_meeting_reproduces_draft_text_modulo_whitespace(tmp_path):
    raw_root = tmp_path / "raw"
    _write_raw_meeting(raw_root, "m1", JSON3_TWO_SEGMENTS, duration_sec=25.0)
    out_root = tmp_path / "out"

    records = ingest_meeting("m1", "json3", raw_root, out_root)
    concatenated = " ".join(r["text"] for r in records)

    reference_words = drop_bracket_words(load_draft(raw_root / "m1" / "captions.json3", "json3"))
    assert concatenated.split() == transcript(reference_words).split()


def test_ingest_meeting_raises_on_synthetic_over_rate_segment(tmp_path):
    # 500 chars packed into a single tOffsetMs-timed word inside a ~0.1s span
    # -- an implausible speech rate that must raise, not silently ingest.
    doc = {"events": [{"tStartMs": 0, "dDurationMs": 100, "segs": [
        {"utf8": "a" * 500, "acAsrConf": 0},
        {"utf8": " b", "tOffsetMs": 50, "acAsrConf": 0},
    ]}]}
    raw_root = tmp_path / "raw"
    _write_raw_meeting(raw_root, "m1", doc, duration_sec=0.2)
    with pytest.raises(ValueError, match="chars/sec"):
        ingest_meeting("m1", "json3", raw_root, tmp_path / "out")


def test_ingest_meeting_swap_source_yields_same_segment_count(tmp_path):
    json3_root = tmp_path / "raw_json3"
    _write_raw_meeting(json3_root, "m1", JSON3_TWO_SEGMENTS, duration_sec=25.0)
    json3_records = ingest_meeting("m1", "json3", json3_root, tmp_path / "out_json3")

    scribe_root = tmp_path / "raw_scribe"
    _write_raw_meeting(scribe_root, "m1", SCRIBE_TWO_SEGMENTS, duration_sec=25.0)
    scribe_records = ingest_meeting("m1", "scribe", scribe_root, tmp_path / "out_scribe")

    assert len(json3_records) == len(scribe_records) == 2


def test_ingest_meeting_scribe_missing_timing_raises_instead_of_falling_back(tmp_path):
    doc = {"words": [{"text": "chào", "type": "word"}]}   # no start/end
    raw_root = tmp_path / "raw"
    _write_raw_meeting(raw_root, "m1", doc, duration_sec=5.0)
    with pytest.raises(ValueError, match="missing start/end"):
        ingest_meeting("m1", "scribe", raw_root, tmp_path / "out")


# ------------------------------------------------------- corpus-level checks

@pytestmark_real
def test_real_corpus_segments_and_splits_and_dataset(tmp_path):
    out_root = tmp_path / "out"
    meeting_ids = sorted(p.name for p in RAW_ROOT.iterdir() if p.is_dir())
    assert meeting_ids

    all_records = []
    for meeting_id in meeting_ids:
        is_test = meeting_id in DEFAULT_TEST_MEETINGS
        records = ingest_meeting(meeting_id, "json3", RAW_ROOT, out_root, is_test=is_test)
        assert records   # every real meeting yields at least one usable segment
        assert all(r["duration"] <= MAX_SEGMENT_SEC + 1e-6 for r in records)
        assert all(r["split"] == ("test" if is_test else "demo") for r in records)

        full_duration = sf.info(str(RAW_ROOT / meeting_id / "audio.wav")).frames / 16000
        window_start, window_end = middle_window(full_duration)
        reference_words = window_words(
            drop_bracket_words(load_draft(RAW_ROOT / meeting_id / "captions.json3", "json3")),
            window_start, window_end)
        concatenated = " ".join(r["text"] for r in records)
        assert concatenated.split() == transcript(reference_words).split()

        all_records.extend(records)

    resolved = resolve_splits(all_records, val_meetings=[])
    stats = split_stats(resolved)
    n_test = sum(1 for r in all_records if r["meeting_id"] in DEFAULT_TEST_MEETINGS)
    assert stats["test"] == n_test
    assert stats["train"] == len(all_records) - n_test   # no val_meetings set -- rest resolve to train
    assert stats["val"] == 0

    dataset = ManifestDataset(records=resolved, audio_root=out_root / "audio")
    sample = dataset[len(dataset) // 2]
    assert sample["audio"].ndim == 1
    assert sample["sampling_rate"] == 16000
