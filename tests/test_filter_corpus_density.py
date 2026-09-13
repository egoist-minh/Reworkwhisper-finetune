"""Tests for scripts/filter_corpus_density.py -- the train-only loanword-density
filter behind curriculum phase 2 (docs/v6-curriculum-plan.md §2.4). Pure record
filtering, no audio touched, no GPU."""

import json

import pytest

from scripts.filter_corpus_density import filter_dense, _write
from src.normalize import Normalizer

NORMALIZER = Normalizer(strip_punctuation=True, lowercase=True,
                         number_convention="word_to_digit", filler_tokens=[])


def _rec(meeting_id, segment_id, text, split="demo", duration=10.0):
    return {"meeting_id": meeting_id, "segment_id": segment_id, "text": text,
            "split": split, "duration": duration,
            "audio_filepath": f"{meeting_id}/{segment_id}.wav"}


RECORDS = [
    _rec("train_a", "seg_0000", "chao cac ban team"),      # train, has a loanword
    _rec("train_a", "seg_0001", "chao cac ban"),           # train, none -- dropped
    _rec("val_a", "seg_0000", "chao cac ban"),             # val, none -- kept anyway
    _rec("test_a", "seg_0000", "chao cac ban", split="test"),
]
VAL_MEETINGS = ["val_a"]


def test_drops_only_train_segments_without_a_loanword():
    kept, _ = filter_dense(RECORDS, VAL_MEETINGS, NORMALIZER, min_foreign=1)
    assert [(r["meeting_id"], r["segment_id"]) for r in kept] == [
        ("train_a", "seg_0000"), ("val_a", "seg_0000"), ("test_a", "seg_0000")]


def test_val_and_test_pass_through_untouched_so_the_gate_stays_comparable():
    # Filtering these would change what the run's CER means, not just the model.
    _, stats = filter_dense(RECORDS, VAL_MEETINGS, NORMALIZER, min_foreign=1)
    assert stats["val"]["after"]["n_segments"] == stats["val"]["before"]["n_segments"] == 1
    assert stats["test"]["after"]["n_segments"] == stats["test"]["before"]["n_segments"] == 1


def test_reports_density_rising_on_the_train_split():
    _, stats = filter_dense(RECORDS, VAL_MEETINGS, NORMALIZER, min_foreign=1)
    train = stats["train"]
    assert train["before"]["n_segments"] == 2
    assert train["after"]["n_segments"] == 1
    assert train["before"]["density"] == 1 / 7   # one loanword in 7 train tokens
    assert train["after"]["density"] == 1 / 4    # ... in 4 after the filter
    assert train["after"]["hours"] == 10.0 / 3600


def test_min_foreign_counts_instances_not_types():
    records = [_rec("train_a", "seg_0000", "chao team team"),
               _rec("train_a", "seg_0001", "chao team")]
    kept, _ = filter_dense(records, [], NORMALIZER, min_foreign=2)
    assert [r["segment_id"] for r in kept] == ["seg_0000"]


def test_kept_records_keep_their_raw_split_so_resolve_splits_reproduces_it():
    from src.data import resolve_splits

    kept, _ = filter_dense(RECORDS, VAL_MEETINGS, NORMALIZER, min_foreign=1)
    assert {r["split"] for r in kept} == {"demo", "test"}
    assert [r["split"] for r in resolve_splits(kept, VAL_MEETINGS)] == ["train", "val", "test"]


def test_write_groups_manifests_per_meeting_and_keeps_every_field(tmp_path):
    kept, _ = filter_dense(RECORDS, VAL_MEETINGS, NORMALIZER, min_foreign=1)
    src, out = tmp_path / "src", tmp_path / "out"
    (src / "audio").mkdir(parents=True)
    try:
        _write(kept, src, out)
    except OSError as exc:                       # Windows without Developer Mode
        pytest.skip(f"symlinks unavailable here: {exc}")
    rows = [json.loads(line) for line in
            (out / "manifest.train_a.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rows == [r for r in kept if r["meeting_id"] == "train_a"]
    assert sorted(p.name for p in out.glob("manifest.*.jsonl")) == [
        "manifest.test_a.jsonl", "manifest.train_a.jsonl", "manifest.val_a.jsonl"]
    assert (out / "audio").is_symlink()


def test_write_refuses_a_destination_that_already_has_something_in_it(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "manifest.train_a.jsonl").write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError):
        _write([], tmp_path / "src", out)
