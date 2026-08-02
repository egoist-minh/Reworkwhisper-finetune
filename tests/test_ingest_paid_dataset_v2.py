"""Tests for scripts/ingest_paid_dataset_v2.py. Synthetic tmp_path trees for the
copy/remap logic (no dependency on the real datasets); a couple of checks
against the real drop are skipped if it isn't present on this machine."""

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from scripts.ingest_paid_dataset_v2 import (
    remap_dot2_split, rename_legacy_meeting, ingest_dot2, ingest_legacy_test_as_train,
)

DOT2_SRC = Path("dataset/paid-tts-dataset-dot2/dataset")
LEGACY_SRC = Path("dataset/paid-dataset")


def test_remap_dot2_split():
    assert remap_dot2_split("train") == "demo"
    assert remap_dot2_split("demo") == "demo"
    assert remap_dot2_split("test") == "test"


def test_remap_dot2_split_rejects_unknown():
    with pytest.raises(ValueError):
        remap_dot2_split("val")


def test_rename_legacy_meeting():
    assert rename_legacy_meeting("paid_meeting_test_0001") == "paid_meeting_legacy_0001"


def test_rename_legacy_meeting_rejects_non_test_id():
    with pytest.raises(ValueError):
        rename_legacy_meeting("paid_meeting_0001")


def _write_wav(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.zeros(1600, dtype="float32"), 16000)


def _write_manifest(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def test_ingest_dot2_remaps_split_and_copies_audio(tmp_path):
    dot2_root = tmp_path / "dot2"
    out_root = tmp_path / "out"
    _write_wav(dot2_root / "audio" / "paid_meeting_0001" / "raw_turns" / "seg_0000.wav")
    _write_manifest(dot2_root / "manifests" / "paid_meeting_0001.jsonl", [
        {"audio_filepath": "paid_meeting_0001/raw_turns/seg_0000.wav",
         "meeting_id": "paid_meeting_0001", "segment_id": "seg_0000", "text": "hi",
         "split": "train"},
    ])

    counts = ingest_dot2(dot2_root, out_root)

    assert counts == {"paid_meeting_0001": 1}
    assert (out_root / "audio" / "paid_meeting_0001" / "raw_turns" / "seg_0000.wav").exists()
    written = [json.loads(l) for l in (out_root / "manifest.paid_meeting_0001.jsonl")
               .read_text(encoding="utf-8").splitlines()]
    assert written[0]["split"] == "demo"


def test_ingest_legacy_test_renames_and_folds_into_train(tmp_path):
    legacy_root = tmp_path / "legacy"
    out_root = tmp_path / "out"
    _write_wav(legacy_root / "audio" / "paid_meeting_test_0001" / "raw_turns" / "seg_0000.wav")
    _write_manifest(legacy_root / "manifest.paid_meeting_test_0001.jsonl", [
        {"audio_filepath": "paid_meeting_test_0001/raw_turns/seg_0000.wav",
         "meeting_id": "paid_meeting_test_0001", "segment_id": "seg_0000", "text": "hi",
         "split": "test"},
    ])

    counts = ingest_legacy_test_as_train(legacy_root, out_root)

    assert counts == {"paid_meeting_legacy_0001": 1}
    assert (out_root / "audio" / "paid_meeting_legacy_0001" / "raw_turns" / "seg_0000.wav").exists()
    written = [json.loads(l) for l in (out_root / "manifest.paid_meeting_legacy_0001.jsonl")
               .read_text(encoding="utf-8").splitlines()]
    assert written[0]["split"] == "demo"
    assert written[0]["meeting_id"] == "paid_meeting_legacy_0001"
    assert written[0]["audio_filepath"] == "paid_meeting_legacy_0001/raw_turns/seg_0000.wav"


@pytest.mark.skipif(not DOT2_SRC.exists() or not LEGACY_SRC.exists(),
                     reason="dot2 drop or legacy paid-dataset not present on this machine")
def test_real_ingest_has_no_meeting_id_collision(tmp_path):
    dot2_counts = ingest_dot2(DOT2_SRC, tmp_path)
    legacy_counts = ingest_legacy_test_as_train(LEGACY_SRC, tmp_path)
    assert not (set(dot2_counts) & set(legacy_counts))
    assert len(legacy_counts) == 6
    # dot2 test meetings still carry the paid_meeting_test_ prefix; legacy ones
    # were renamed to paid_meeting_legacy_ specifically so this never collides.
    assert all(m.startswith("paid_meeting_legacy_") for m in legacy_counts)
