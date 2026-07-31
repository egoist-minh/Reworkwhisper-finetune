"""Tests for src/data.py. Reads the predecessor repo's paid-dataset read-only
(CLAUDE.md: never edit it) to check split resolution against the numbers
verified in the handoff: 1316 train / 250 val / 236 test, 1802 total."""

from pathlib import Path

import pytest

from src.data import load_manifests, resolve_splits, split_stats, load_audio_16k

PAID_DATASET = Path("D:/phowhisper-finetune-exp/dataset/paid-dataset")
VAL_MEETINGS = ["paid_meeting_0001", "paid_meeting_0002", "paid_meeting_0011"]

pytestmark = pytest.mark.skipif(
    not PAID_DATASET.exists(), reason="predecessor repo not present on this machine"
)


def test_split_counts_match_handoff():
    records = load_manifests(PAID_DATASET)
    assert len(records) == 1802
    resolved = resolve_splits(records, VAL_MEETINGS)
    stats = split_stats(resolved)
    assert stats == {"train": 1316, "val": 250, "test": 236}


def test_val_meetings_all_resolve_to_val():
    records = load_manifests(PAID_DATASET)
    resolved = resolve_splits(records, VAL_MEETINGS)
    for r in resolved:
        if r["meeting_id"] in VAL_MEETINGS:
            assert r["split"] == "val"


def test_resolve_splits_rejects_meeting_split_conflict():
    records = [
        {"meeting_id": "m1", "split": "demo"},
        {"meeting_id": "m1", "split": "test"},
    ]
    with pytest.raises(ValueError):
        resolve_splits(records, val_meetings=[])


def test_load_audio_16k_resamples_from_24k():
    records = load_manifests(PAID_DATASET)
    sample = records[0]
    wav = PAID_DATASET / "audio" / sample["audio_filepath"]
    audio = load_audio_16k(wav)
    assert audio.ndim == 1
    assert audio.dtype.name == "float32"
