"""Regression test for the manifest filename bug that crashed stage_baseline's
load_ood() on the first real Kaggle run: fetch_vivos.py wrote "manifest.jsonl",
which does not match src/data.py:load_manifests's glob "manifest.*.jsonl" (that
pattern requires a non-empty middle component)."""

import json

from scripts.fetch_vivos import MANIFEST_FILENAME
from src.data import load_manifests


def test_manifest_filename_matches_load_manifests_glob(tmp_path):
    (tmp_path / MANIFEST_FILENAME).write_text(
        json.dumps({"text": "hi", "audio_filepath": "a.wav"}) + "\n", encoding="utf-8"
    )
    records = load_manifests(tmp_path)
    assert len(records) == 1


def test_bare_manifest_jsonl_would_not_have_matched():
    """Documents the bug: a bare "manifest.jsonl" does NOT match "manifest.*.jsonl"."""
    import fnmatch

    assert not fnmatch.fnmatch("manifest.jsonl", "manifest.*.jsonl")
    assert fnmatch.fnmatch(MANIFEST_FILENAME, "manifest.*.jsonl")
