"""Tests for scripts/build_v6_corpus_addon.py -- packing delivered TTS batches so
they extract over dataset/v6-corpus without replacing anything in it."""

import json

import pytest

import scripts.build_v6_corpus_addon as addon


def _batch(root, meetings=("paid_meeting_0297",), split="train"):
    (root / "manifests").mkdir(parents=True, exist_ok=True)
    for m in meetings:
        rec = {"audio_filepath": f"{m}/raw_turns/seg_0000.wav", "meeting_id": m,
               "segment_id": "seg_0000", "duration": 4.0, "text": "chạy thử rollout",
               "source": "synthetic", "split": split}
        (root / "manifests" / f"{m}.jsonl").write_text(
            json.dumps(rec, ensure_ascii=False) + "\n", encoding="utf-8")
        wav = root / "audio" / m / "raw_turns" / "seg_0000.wav"
        wav.parent.mkdir(parents=True, exist_ok=True)
        wav.write_bytes(b"RIFF")
    return root


def _corpus(root, meetings=("paid_meeting_0001",)):
    root.mkdir(parents=True, exist_ok=True)
    for m in meetings:
        (root / f"manifest.{m}.jsonl").write_text(
            json.dumps({"meeting_id": m, "segment_id": "seg_0000", "duration": 1.0,
                        "text": "x", "audio_filepath": f"{m}/raw_turns/seg_0000.wav",
                        "split": "demo"}, ensure_ascii=False) + "\n", encoding="utf-8")
    return root


def test_split_train_is_remapped_because_resolve_splits_rejects_it(tmp_path, monkeypatch):
    monkeypatch.setattr(addon, "CORPUS", _corpus(tmp_path / "corpus"))
    addon.build(tmp_path / "out", [_batch(tmp_path / "b") ])
    rec = json.loads((tmp_path / "out" / "manifest.paid_meeting_0297.jsonl").read_text())
    assert rec["split"] == "demo"


def test_a_meeting_already_in_the_corpus_is_refused(tmp_path, monkeypatch):
    # Extracting such an add-on would silently replace a corpus manifest.
    monkeypatch.setattr(addon, "CORPUS", _corpus(tmp_path / "corpus", ("paid_meeting_0297",)))
    with pytest.raises(ValueError, match="already in"):
        addon.build(tmp_path / "out", [_batch(tmp_path / "b")])


def test_two_batches_sharing_a_meeting_id_are_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(addon, "CORPUS", _corpus(tmp_path / "corpus"))
    with pytest.raises(ValueError, match="collision"):
        addon.build(tmp_path / "out", [_batch(tmp_path / "b1"), _batch(tmp_path / "b2")])


def test_an_unexpected_split_value_stops_the_build(tmp_path, monkeypatch):
    monkeypatch.setattr(addon, "CORPUS", _corpus(tmp_path / "corpus"))
    with pytest.raises(ValueError, match="expected split='train'"):
        addon.build(tmp_path / "out", [_batch(tmp_path / "b", split="test")])


def test_building_twice_into_the_same_directory_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(addon, "CORPUS", _corpus(tmp_path / "corpus"))
    addon.build(tmp_path / "out", [_batch(tmp_path / "b")])
    with pytest.raises(FileExistsError):
        addon.build(tmp_path / "out", [_batch(tmp_path / "b")])


def test_audio_lands_at_the_path_the_manifest_points_at(tmp_path, monkeypatch):
    monkeypatch.setattr(addon, "CORPUS", _corpus(tmp_path / "corpus"))
    addon.build(tmp_path / "out", [_batch(tmp_path / "b")])
    assert (tmp_path / "out" / "audio" / "paid_meeting_0297" / "raw_turns"
            / "seg_0000.wav").read_bytes() == b"RIFF"
