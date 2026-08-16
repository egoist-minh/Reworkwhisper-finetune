"""Tests for scripts/review_youtube.py (SESSIONS.md F4). Unit/round-trip tests
are fully offline/synthetic; the corpus-level test reads F3's real manifests
but only ever writes into a scratch tmp_path copy, never the real files."""

import json
from pathlib import Path

import pytest

from scripts.review_youtube import (
    apply_corrections, build_html, check_verified, emit_worksheet,
    load_records, MANIFEST_DIR, normalize_for_review, write_records,
)

pytestmark_real = pytest.mark.skipif(
    not MANIFEST_DIR.exists(), reason="F3 has not written dataset/youtube-meetings/ on this machine"
)

RECORD = {
    "text": "chào Anh ạ", "audio_filepath": "m1/seg_0000.wav", "split": "demo",
    "meeting_id": "m1", "segment_id": "seg_0000", "duration": 2.0,
    "video_id": "m1", "video_url": "https://youtu.be/m1", "yt_start": 0.0,
    "yt_end": 2.0, "download_date": "2026-08-12", "sha256": "deadbeef",
    "source": "youtube", "label_source": "google_asr", "asr_draft_model": "vi-orig",
    "reviewed_by": "", "review_date": "", "verified": False,
}


def _write_manifest(manifest_dir: Path, meeting_id: str, records: list[dict]) -> None:
    manifest_dir.mkdir(parents=True, exist_ok=True)
    write_records(meeting_id, records, manifest_dir)


def _write_wav(audio_root: Path, filepath: str) -> None:
    p = audio_root / filepath
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"fake wav bytes")   # never opened by these tests -- audio src is just a URI


# ------------------------------------------------------------------- casing

def test_normalize_for_review_lowercases_and_strips_brackets_and_collapses_space():
    assert normalize_for_review("Là sao Bạn  [Âm nhạc]  hỏi") == "là sao bạn hỏi"


def test_normalize_for_review_idempotent_on_already_lowercase():
    assert normalize_for_review("chào anh ạ") == "chào anh ạ"


# --------------------------------------------------------------------- emit

def test_build_html_unverified_shows_normalized_draft():
    html = build_html("m1", [RECORD], audio_root=Path("."))
    assert 'data-segment-id="seg_0000"' in html
    assert ">chào anh ạ</textarea>" in html   # lowercased, no residual "Anh"
    assert "<audio controls" in html


def test_build_html_verified_shows_stored_text_unchanged():
    verified = {**RECORD, "text": "Chào Anh ạ", "verified": True}
    html = build_html("m1", [verified], audio_root=Path("."))
    assert ">Chào Anh ạ</textarea>" in html   # reviewer's own casing choice, not re-lowercased


def test_emit_worksheet_writes_one_row_per_record(tmp_path):
    manifest_dir = tmp_path / "manifests"
    audio_root = tmp_path / "audio"
    _write_manifest(manifest_dir, "m1", [RECORD, {**RECORD, "segment_id": "seg_0001"}])
    _write_wav(audio_root, "m1/seg_0000.wav")
    _write_wav(audio_root, "m1/seg_0001.wav")

    out_path = emit_worksheet("m1", manifest_dir, audio_root, tmp_path / "review")
    html = out_path.read_text(encoding="utf-8")
    assert html.count('class="seg"') == 2
    assert "exportCorrections" in html


# -------------------------------------------------------------------- apply

def test_apply_corrections_round_trip_preserves_count_and_audio_filepath(tmp_path):
    manifest_dir = tmp_path / "manifests"
    records = [RECORD, {**RECORD, "segment_id": "seg_0001", "audio_filepath": "m1/seg_0001.wav"}]
    _write_manifest(manifest_dir, "m1", records)

    corrections_path = tmp_path / "corrections.m1.json"
    corrections_path.write_text(json.dumps({"seg_0000": "chào anh ạ", "seg_0001": "vâng ạ"}),
                                 encoding="utf-8")

    n = apply_corrections("m1", corrections_path, "minh", manifest_dir, review_date="2026-08-12")
    assert n == 2

    updated = load_records("m1", manifest_dir)
    assert len(updated) == len(records)
    assert [r["audio_filepath"] for r in updated] == [r["audio_filepath"] for r in records]
    assert updated[0]["text"] == "chào anh ạ"
    assert updated[1]["text"] == "vâng ạ"
    assert all(r["verified"] is True for r in updated)
    assert all(r["reviewed_by"] == "minh" for r in updated)
    assert all(r["review_date"] == "2026-08-12" for r in updated)


def test_apply_corrections_sets_label_source_google_asr_plus_human(tmp_path):
    # youtube-data-pilot/README.md step 5: label_source must record that the draft
    # (google_asr) was then human-reviewed, per-record -- not left as the pre-review
    # value nor overwritten with something that drops the "google_asr" provenance.
    manifest_dir = tmp_path / "manifests"
    _write_manifest(manifest_dir, "m1", [RECORD])
    corrections_path = tmp_path / "corrections.m1.json"
    corrections_path.write_text(json.dumps({"seg_0000": "chào anh ạ"}), encoding="utf-8")

    apply_corrections("m1", corrections_path, "minh", manifest_dir)

    updated = load_records("m1", manifest_dir)
    assert updated[0]["label_source"] == "google_asr+human"


def test_apply_corrections_raises_naming_segment_when_missing(tmp_path):
    manifest_dir = tmp_path / "manifests"
    _write_manifest(manifest_dir, "m1", [RECORD])
    corrections_path = tmp_path / "corrections.m1.json"
    corrections_path.write_text(json.dumps({}), encoding="utf-8")   # no entry for seg_0000

    with pytest.raises(ValueError, match="seg_0000"):
        apply_corrections("m1", corrections_path, "minh", manifest_dir)


def test_apply_corrections_raises_on_blank_box_never_keeps_draft(tmp_path):
    manifest_dir = tmp_path / "manifests"
    _write_manifest(manifest_dir, "m1", [RECORD])
    corrections_path = tmp_path / "corrections.m1.json"
    corrections_path.write_text(json.dumps({"seg_0000": "   "}), encoding="utf-8")

    with pytest.raises(ValueError, match="blank"):
        apply_corrections("m1", corrections_path, "minh", manifest_dir)

    # the manifest must be untouched -- apply raised before writing back
    assert load_records("m1", manifest_dir)[0]["verified"] is False


# -------------------------------------------------------------------- check

def test_check_verified_raises_listing_unverified_segments(tmp_path):
    manifest_dir = tmp_path / "manifests"
    _write_manifest(manifest_dir, "m1", [RECORD])
    with pytest.raises(RuntimeError, match="m1/seg_0000"):
        check_verified(manifest_dir)


def test_check_verified_passes_when_all_verified(tmp_path):
    manifest_dir = tmp_path / "manifests"
    _write_manifest(manifest_dir, "m1", [{**RECORD, "verified": True}])
    check_verified(manifest_dir)   # must not raise


def test_check_verified_meeting_id_filter_ignores_other_meetings(tmp_path):
    manifest_dir = tmp_path / "manifests"
    _write_manifest(manifest_dir, "m1", [RECORD])   # unverified
    _write_manifest(manifest_dir, "m2", [{**RECORD, "meeting_id": "m2", "verified": True}])
    check_verified(manifest_dir, meeting_ids=["m2"])   # must not raise -- m1 excluded
    with pytest.raises(RuntimeError):
        check_verified(manifest_dir)   # default (no filter) still sees m1


def test_full_round_trip_emit_apply_check(tmp_path):
    manifest_dir = tmp_path / "manifests"
    audio_root = tmp_path / "audio"
    records = [RECORD, {**RECORD, "segment_id": "seg_0001", "audio_filepath": "m1/seg_0001.wav"}]
    _write_manifest(manifest_dir, "m1", records)
    _write_wav(audio_root, "m1/seg_0000.wav")
    _write_wav(audio_root, "m1/seg_0001.wav")

    emit_worksheet("m1", manifest_dir, audio_root, tmp_path / "review")

    # Simulate a reviewer confirming the normalized draft for every segment
    # (no browser involved -- this is the JSON exportCorrections() would produce).
    before = load_records("m1", manifest_dir)
    corrections = {r["segment_id"]: normalize_for_review(r["text"]) for r in before}
    corrections_path = tmp_path / "corrections.m1.json"
    corrections_path.write_text(json.dumps(corrections), encoding="utf-8")

    apply_corrections("m1", corrections_path, "minh", manifest_dir)

    after = load_records("m1", manifest_dir)
    assert [r["audio_filepath"] for r in after] == [r["audio_filepath"] for r in before]
    assert len(after) == len(before)
    check_verified(manifest_dir)   # must not raise now


# ------------------------------------------------------- corpus-level check

@pytestmark_real
def test_real_corpus_emit_matches_segment_count_and_round_trip_in_scratch(tmp_path):
    from scripts.review_youtube import AUDIO_ROOT

    manifest_files = sorted(MANIFEST_DIR.glob("manifest.*.jsonl"))
    assert manifest_files
    meeting_id = manifest_files[0].stem.split(".", 1)[1]
    real_records = load_records(meeting_id, MANIFEST_DIR)

    out_path = emit_worksheet(meeting_id, MANIFEST_DIR, AUDIO_ROOT, tmp_path / "review")
    html = out_path.read_text(encoding="utf-8")
    assert html.count('class="seg"') == len(real_records)
    for r in real_records:
        assert (AUDIO_ROOT / r["audio_filepath"]).resolve().as_uri() in html

    # Round trip against a SCRATCH copy of the real manifest -- never touch
    # the real dataset/youtube-meetings/ files.
    scratch_dir = tmp_path / "manifests"
    write_records(meeting_id, real_records, scratch_dir)
    corrections = {r["segment_id"]: normalize_for_review(r["text"]) for r in real_records}
    corrections_path = tmp_path / "corrections.json"
    corrections_path.write_text(json.dumps(corrections), encoding="utf-8")

    apply_corrections(meeting_id, corrections_path, "minh", scratch_dir)
    check_verified(scratch_dir)   # must not raise
    # Real on-disk records untouched -- compared field-by-field rather than via
    # `verified`, which apply_corrections is not the only thing that can set.
    assert load_records(meeting_id, MANIFEST_DIR) == real_records
