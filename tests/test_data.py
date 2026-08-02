"""Tests for src/data.py. Reads the migrated paid-dataset (dataset/paid-dataset,
verified identical to the predecessor repo's copy via dataset/CHECKSUMS.txt) to
check split resolution against the numbers verified in the handoff: 1316
train / 250 val / 236 test, 1802 total."""

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from src.data import load_manifests, resolve_splits, split_stats, load_audio_16k, ManifestDataset

PAID_DATASET = Path("dataset/paid-dataset")
VAL_MEETINGS = ["paid_meeting_0001", "paid_meeting_0002", "paid_meeting_0011"]

pytestmark = pytest.mark.skipif(
    not PAID_DATASET.exists(), reason="dataset/paid-dataset not present on this machine"
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


def test_manifest_dataset_limit_caps_record_count(tmp_path):
    ds = ManifestDataset(records=[{"audio_filepath": f"{i}.wav", "text": str(i)} for i in range(10)],
                          audio_root=tmp_path)
    limited = ds.limit(3)
    assert len(limited) == 3
    assert len(ds) == 10  # original untouched


def test_manifest_dataset_limit_none_or_zero_is_a_no_op(tmp_path):
    ds = ManifestDataset(records=[{"audio_filepath": "a.wav", "text": "hi"}], audio_root=tmp_path)
    assert len(ds.limit(None)) == 1
    assert len(ds.limit(0)) == 1


def test_manifest_dataset_getitem_without_meeting_id(tmp_path):
    """Regression: crashed on VIVOS records (fetch_vivos.py has no meeting_id/
    segment_id fields, unlike paid-dataset) with KeyError: 'meeting_id'."""
    sf.write(tmp_path / "a.wav", np.zeros(1600, dtype="float32"), 16000)
    ds = ManifestDataset(records=[{"audio_filepath": "a.wav", "text": "hi"}],
                          audio_root=tmp_path)
    item = ds[0]
    assert item["meeting_id"] is None
    assert item["segment_id"] is None
    assert item["text"] == "hi"
