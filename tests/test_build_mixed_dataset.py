"""Tests for scripts/build_mixed_dataset.py. All fixtures are synthetic, written
into tmp_path -- never touches the real dataset/ directories."""

import json
from pathlib import Path

import pytest
import yaml

from scripts.build_mixed_dataset import build_mixed_dataset
from src.data import load_manifests, resolve_splits, split_stats


def _record(meeting_id: str, segment_id: str, split: str, source: str,
            duration: float = 1.0, verified: bool | None = None,
            audio_filepath: str | None = None) -> dict:
    r = {
        "audio_filepath": audio_filepath or f"{meeting_id}/{segment_id}.wav",
        "meeting_id": meeting_id, "segment_id": segment_id, "split": split,
        "duration": duration, "text": "xin chào", "source": source,
    }
    if verified is not None:
        r["verified"] = verified
    return r


def _write_source(root: Path, records: list[dict]) -> None:
    by_meeting: dict[str, list[dict]] = {}
    for r in records:
        by_meeting.setdefault(r["meeting_id"], []).append(r)
    for mid, recs in by_meeting.items():
        root.mkdir(parents=True, exist_ok=True)
        with open(root / f"manifest.{mid}.jsonl", "w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    for r in records:
        wav = root / "audio" / r["audio_filepath"]
        wav.parent.mkdir(parents=True, exist_ok=True)
        wav.write_bytes(b"fake wav bytes")


def _make_synthetic_source(tmp_path: Path, name: str = "synthetic-src") -> Path:
    src = tmp_path / name
    _write_source(src, [
        _record("paid_0001", "seg_0000", "demo", "synthetic"),
        _record("paid_0001", "seg_0001", "demo", "synthetic"),
        _record("paid_0002", "seg_0000", "test", "synthetic"),
    ])
    return src


def _make_youtube_source(tmp_path: Path, name: str = "youtube-src", verified: bool = True) -> Path:
    src = tmp_path / name
    _write_source(src, [
        _record("videoA", "seg_0000", "demo", "youtube", verified=verified),
        _record("videoB", "seg_0000", "test", "youtube", verified=verified),
    ])
    return src


def _write_config(tmp_path: Path, val_meetings: list[str]) -> Path:
    cfg = {
        "run_id": "test-run",
        "base_model": "vinai/PhoWhisper-small",
        # ood_eval_path is unused by this script but src/config.py:validate
        # requires it -- a merge is only ever done ahead of a real run.
        "data": {"dataset_path": str(tmp_path / "unused"), "val_meetings": val_meetings,
                  "ood_eval_path": str(tmp_path / "unused-ood")},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


# --------------------------------------------------------------------- gates

def test_raises_on_meeting_id_collision(tmp_path):
    src_a = _make_synthetic_source(tmp_path, "a")
    src_b = tmp_path / "b"
    _write_source(src_b, [_record("paid_0001", "seg_0000", "demo", "youtube", verified=True)])
    cfg = _write_config(tmp_path, [])

    with pytest.raises(ValueError, match="paid_0001"):
        build_mixed_dataset([src_a, src_b], tmp_path / "out", str(cfg), dry_run=True)


def test_raises_on_audio_top_level_dir_collision(tmp_path):
    src_a = tmp_path / "a"
    _write_source(src_a, [_record("m1", "seg_0000", "demo", "synthetic",
                                   audio_filepath="shared/seg_0000.wav")])
    src_b = tmp_path / "b"
    _write_source(src_b, [_record("m2", "seg_0000", "demo", "youtube", verified=True,
                                   audio_filepath="shared/seg_0000.wav")])
    cfg = _write_config(tmp_path, [])

    with pytest.raises(ValueError, match="shared"):
        build_mixed_dataset([src_a, src_b], tmp_path / "out", str(cfg), dry_run=True)


def test_raises_on_unverified_segment(tmp_path):
    src_a = _make_synthetic_source(tmp_path, "a")
    src_b = _make_youtube_source(tmp_path, "b", verified=False)
    cfg = _write_config(tmp_path, [])

    with pytest.raises(RuntimeError, match="videoA/seg_0000"):
        build_mixed_dataset([src_a, src_b], tmp_path / "out", str(cfg), dry_run=True)


def test_synthetic_source_without_verified_field_is_not_gated(tmp_path):
    # paid-dataset-v2 records have no "verified" key at all -- must not raise.
    src_a = _make_synthetic_source(tmp_path, "a")
    src_b = _make_youtube_source(tmp_path, "b", verified=True)
    cfg = _write_config(tmp_path, [])

    build_mixed_dataset([src_a, src_b], tmp_path / "out", str(cfg), dry_run=True)  # must not raise


def test_raises_when_dest_exists_and_nonempty(tmp_path):
    src_a = _make_synthetic_source(tmp_path, "a")
    src_b = _make_youtube_source(tmp_path, "b", verified=True)
    out = tmp_path / "out"
    out.mkdir()
    (out / "junk.txt").write_text("x")
    cfg = _write_config(tmp_path, [])

    with pytest.raises(FileExistsError):
        build_mixed_dataset([src_a, src_b], out, str(cfg), dry_run=False)


# ------------------------------------------------------------------ dry-run

def test_dry_run_does_not_create_dest(tmp_path, capsys):
    src_a = _make_synthetic_source(tmp_path, "a")
    src_b = _make_youtube_source(tmp_path, "b", verified=True)
    out = tmp_path / "out"
    cfg = _write_config(tmp_path, ["videoA"])

    build_mixed_dataset([src_a, src_b], out, str(cfg), dry_run=True)

    assert not out.exists()
    assert "split_stats" in capsys.readouterr().out


# ------------------------------------------------------------------- merge

def test_build_merges_manifests_and_audio_matching_split_stats(tmp_path):
    src_a = _make_synthetic_source(tmp_path, "a")
    src_b = _make_youtube_source(tmp_path, "b", verified=True)
    out = tmp_path / "out"
    cfg = _write_config(tmp_path, ["videoA"])

    build_mixed_dataset([src_a, src_b], out, str(cfg), dry_run=False)

    records = load_manifests(out)
    resolved = resolve_splits(records, ["videoA"])
    stats = split_stats(resolved)
    assert stats == {"train": 2, "val": 1, "test": 2}

    for r in records:
        assert (out / "audio" / r["audio_filepath"]).exists()

    val_meeting_ids = {r["meeting_id"] for r in resolved if r["split"] == "val"}
    assert val_meeting_ids == {"videoA"}


def test_build_writes_provenance_with_both_sources(tmp_path):
    src_a = _make_synthetic_source(tmp_path, "a")
    src_b = _make_youtube_source(tmp_path, "b", verified=True)
    out = tmp_path / "out"
    cfg = _write_config(tmp_path, [])

    build_mixed_dataset([src_a, src_b], out, str(cfg), dry_run=False)

    provenance = (out / "provenance.md").read_text(encoding="utf-8")
    assert str(src_a) in provenance
    assert str(src_b) in provenance


def test_build_does_not_copy_raw_directory(tmp_path):
    src_a = _make_synthetic_source(tmp_path, "a")
    (src_a / "raw").mkdir()
    (src_a / "raw" / "junk.wav").write_bytes(b"not audio we care about")
    src_b = _make_youtube_source(tmp_path, "b", verified=True)
    out = tmp_path / "out"
    cfg = _write_config(tmp_path, [])

    build_mixed_dataset([src_a, src_b], out, str(cfg), dry_run=False)

    assert not (out / "raw").exists()
