"""Tests for scripts/fetch_youtube.py. Offline: `download_and_draft` (the only
function that touches the network) is monkeypatched everywhere, so these
exercise the idempotency guard and the provenance.json shape without yt-dlp."""

import json

import pytest

from scripts.checksum_dataset import _hash_file
from scripts.fetch_youtube import fetch_meeting, read_sources, require_ffmpeg

REQUIRED_PROVENANCE_FIELDS = [
    "video_id", "video_url", "yt_start", "yt_end", "download_date", "sha256",
    "source", "label_source", "asr_draft_model", "reviewed_by", "review_date",
]


def test_read_sources_parses_jsonl(tmp_path):
    path = tmp_path / "sources.jsonl"
    path.write_text(
        '{"meeting_id": "aaa", "video_url": "https://youtu.be/aaa"}\n'
        '\n'
        '{"meeting_id": "bbb", "video_url": "https://youtu.be/bbb"}\n',
        encoding="utf-8",
    )
    rows = read_sources(path)
    assert [r["meeting_id"] for r in rows] == ["aaa", "bbb"]


def test_require_ffmpeg_raises_by_name_when_missing(monkeypatch):
    monkeypatch.setattr("scripts.fetch_youtube.shutil.which", lambda name: None)
    with pytest.raises(RuntimeError, match="ffmpeg"):
        require_ffmpeg()


def test_require_ffmpeg_passes_when_present(monkeypatch):
    monkeypatch.setattr("scripts.fetch_youtube.shutil.which", lambda name: r"C:\ffmpeg.exe")
    require_ffmpeg()   # must not raise


def _fake_download_and_draft(url: str, out_dir: dict) -> dict:
    (out_dir / "audio.wav").write_bytes(b"fake wav bytes")
    return {
        "info": {"id": "vid123", "duration": 123.4},
        "doc": {"events": []},
        "track_key": "vi-orig",
    }


def test_fetch_meeting_writes_expected_files_and_provenance(tmp_path, monkeypatch):
    monkeypatch.setattr("scripts.fetch_youtube.download_and_draft", _fake_download_and_draft)
    out_root = tmp_path / "raw"
    fetch_meeting("m1", "https://youtu.be/m1", out_root)

    out_dir = out_root / "m1"
    assert (out_dir / "audio.wav").read_bytes() == b"fake wav bytes"
    assert json.loads((out_dir / "captions.json3").read_text(encoding="utf-8")) == {"events": []}

    prov = json.loads((out_dir / "provenance.json").read_text(encoding="utf-8"))
    for field in REQUIRED_PROVENANCE_FIELDS:
        assert field in prov, f"missing field {field}"
        assert prov[field] is not None, f"{field} is null"
    assert prov["video_id"] == "vid123"
    assert prov["video_url"] == "https://youtu.be/m1"
    assert prov["yt_start"] == 0.0
    assert prov["yt_end"] == 123.4
    assert prov["source"] == "youtube"
    assert prov["label_source"] == "google_asr"
    assert prov["asr_draft_model"] == "vi-orig"
    assert prov["sha256"] == _hash_file(out_dir / "audio.wav")


def test_fetch_meeting_skips_a_second_run_without_calling_the_network(tmp_path, monkeypatch):
    def _fail_if_called(url, out_dir):
        raise AssertionError("download_and_draft must not be called on a checksum-clean re-run")

    out_root = tmp_path / "raw"
    monkeypatch.setattr("scripts.fetch_youtube.download_and_draft", _fake_download_and_draft)
    fetch_meeting("m1", "https://youtu.be/m1", out_root)

    monkeypatch.setattr("scripts.fetch_youtube.download_and_draft", _fail_if_called)
    fetch_meeting("m1", "https://youtu.be/m1", out_root)   # must not raise, must not download


def test_fetch_meeting_raises_on_checksum_mismatch(tmp_path, monkeypatch):
    monkeypatch.setattr("scripts.fetch_youtube.download_and_draft", _fake_download_and_draft)
    out_root = tmp_path / "raw"
    fetch_meeting("m1", "https://youtu.be/m1", out_root)

    (out_root / "m1" / "audio.wav").write_bytes(b"mutated bytes -- on-disk file changed")
    with pytest.raises(RuntimeError, match="mismatch"):
        fetch_meeting("m1", "https://youtu.be/m1", out_root)
