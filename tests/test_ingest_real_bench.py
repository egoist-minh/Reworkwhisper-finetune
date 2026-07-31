"""Tests for scripts/ingest_real_bench.py against the real done/ source
(read-only, predecessor repo). PROJECT_CORE.md §4, §6 Stage 4 tier 4a."""

from pathlib import Path

import numpy as np
import pytest

from scripts.ingest_real_bench import (
    MAX_SEGMENT_SEC, _split_text_proportionally, _choose_splits, resegment,
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


def test_ingested_manifest_has_no_long_segments():
    manifest = Path("dataset/real-meetings-bench/manifest.real-meetings-bench.jsonl")
    if not manifest.exists():
        pytest.skip("run scripts/ingest_real_bench.py first")
    import json

    records = [json.loads(l) for l in manifest.read_text(encoding="utf-8").splitlines()]
    assert len(records) > 0
    assert all(r["duration"] <= MAX_SEGMENT_SEC + 1e-6 for r in records)
