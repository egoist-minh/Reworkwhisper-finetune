"""Tests for scripts/ingest_real_bench.py against the real done/ source
(read-only, predecessor repo). PROJECT_CORE.md §4, §6 Stage 4 tier 4a."""

from pathlib import Path

import numpy as np
import pytest

from scripts.ingest_real_bench import (
    MAX_SEGMENT_SEC, MAX_CHARS_PER_SEC, _split_text_proportionally,
    _choose_splits, resegment, _check_speech_rate, ingest_recording,
)

DONE_DIR = Path("D:/phowhisper-finetune-exp/dataset/done")
pytestmark = pytest.mark.skipif(
    not DONE_DIR.exists(), reason="predecessor repo not present on this machine"
)


def test_split_text_proportionally_preserves_all_words():
    text = "một hai ba bốn năm sáu bảy tám chín mười"
    pieces = _split_text_proportionally(text, chunk_durations=[1.0, 2.0, 1.0])
    assert " ".join(pieces).split() == text.split()
    assert len(pieces) == 3


def test_split_text_proportionally_empty_text():
    pieces = _split_text_proportionally("", chunk_durations=[1.0, 1.0])
    assert pieces == ["", ""]


def test_choose_splits_forces_a_cut_when_no_silence_found():
    # No candidates at all -> must still force cuts to respect MAX_SEGMENT_SEC.
    cuts = _choose_splits(duration=95.0, candidates=[])
    assert cuts == [30.0, 60.0, 90.0]
    for i, c in enumerate([0.0] + cuts + [95.0]):
        pass
    chunk_durs = [b - a for a, b in zip([0.0] + cuts, cuts + [95.0])]
    assert all(d <= MAX_SEGMENT_SEC for d in chunk_durs)


def test_resegment_real_audio_no_chunk_exceeds_limit_and_text_preserved():
    import soundfile as sf

    audio, sr = sf.read(str(DONE_DIR / "real_0001_16k.wav"), dtype="float32")
    # A known >30s segment from real_0001.draft.json (seg 0: 0.0-212.0s).
    seg = audio[int(0.0 * sr):int(212.0 * sr)]
    import json
    draft = json.loads((DONE_DIR / "real_0001.draft.json").read_text(encoding="utf-8"))
    text = draft["segments"][0]["text"]

    chunks = resegment(seg, sr, text)
    assert len(chunks) > 1
    for chunk_audio, chunk_text in chunks:
        assert len(chunk_audio) / sr <= MAX_SEGMENT_SEC + 1e-6
    reconstructed = " ".join(t for _, t in chunks)
    assert reconstructed.split() == text.split()


def test_check_speech_rate_raises_on_synthetic_over_rate_segment():
    text = "a" * 200  # 200 chars over 1s -> 200 chars/sec, way past MAX_CHARS_PER_SEC
    with pytest.raises(ValueError, match=r"200\.0 chars/sec"):
        _check_speech_rate("meeting_x", "seg_0001", text, duration=1.0)


def test_check_speech_rate_allows_normal_rate():
    _check_speech_rate("meeting_x", "seg_0001", "một hai ba bốn năm", duration=2.0)


def test_ingest_recording_raises_on_real_0002_seg_0074(tmp_path):
    # real_0002/seg_0074: a proportional-split rejoin artifact measuring
    # 475.7 chars/sec (1998 chars over a 4.2s chunk) vs. the corpus' next-worst
    # 39.5 chars/sec -- must raise loud, not silently ingest.
    with pytest.raises(ValueError, match=r"real_0002/seg_0074.*475\.7 chars/sec"):
        ingest_recording(
            DONE_DIR / "real_0002.draft.json",
            DONE_DIR / "real_0002_16k.wav",
            "real_0002",
            tmp_path,
        )


def test_ingested_manifest_has_no_long_segments():
    manifest = Path("dataset/real-meetings-bench/manifest.real-meetings-bench.jsonl")
    if not manifest.exists():
        pytest.skip("run scripts/ingest_real_bench.py first")
    import json

    records = [json.loads(l) for l in manifest.read_text(encoding="utf-8").splitlines()]
    assert len(records) > 0
    assert all(r["duration"] <= MAX_SEGMENT_SEC + 1e-6 for r in records)
