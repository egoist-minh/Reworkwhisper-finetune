"""Tests for scripts/filter_corpus_density.py -- the train-only loanword-density
filter behind curriculum phase 2 (docs/v6-curriculum-plan.md §2.4). Pure record
filtering, no audio touched, no GPU."""

import json

import pytest

from scripts.filter_corpus_density import (filter_dense, filter_meeting_quota,
                                           filter_min_meeting_density, _write)
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


# --- whole-meeting modes (docs/v6-ondomain-plan.md §1a) ------------------------

def test_min_meeting_density_keeps_or_drops_a_train_meeting_whole():
    records = [
        _rec("dense", "seg_0000", "chao cac ban team"),      # 1/4 -- meeting kept whole,
        _rec("dense", "seg_0001", "chao cac ban"),           # including this bare segment
        _rec("sparse", "seg_0000", "chao cac ban team"),     # 1/16 overall -- dropped whole,
        _rec("sparse", "seg_0001", "chao cac ban chao cac ban chao cac ban chao cac ban"),
        _rec("val_a", "seg_0000", "chao cac ban"),
    ]
    kept, stats = filter_min_meeting_density(records, VAL_MEETINGS, NORMALIZER, min_density=0.1)
    assert [(r["meeting_id"], r["segment_id"]) for r in kept] == [
        ("dense", "seg_0000"), ("dense", "seg_0001"), ("val_a", "seg_0000")]
    assert stats["train"]["after"]["n_meetings"] == 1
    assert stats["val"]["after"]["n_segments"] == 1


def test_quota_adds_loanword_free_segments_longest_first():
    records = [
        _rec("m", "seg_0000", "chao team"),                  # carrier: 1 foreign / 2 tokens
        _rec("m", "seg_0001", "chao cac ban", duration=30.0),   # 3 tokens -> 1/5 = 20%
        _rec("m", "seg_0002", "chao cac ban", duration=10.0),   # would drop to 1/8 = 12.5%
    ]
    kept, _ = filter_meeting_quota(records, [], NORMALIZER, quota=0.2,
                                    min_meeting_foreign_density=0.0)
    assert [r["segment_id"] for r in kept] == ["seg_0000", "seg_0001"]


def test_quota_skips_a_segment_that_breaches_but_keeps_walking():
    # The long one would breach the quota; the short one after it still fits, so
    # the walk must not stop at the first breach.
    records = [
        _rec("m", "seg_0000", "chao team"),                              # 1/2
        _rec("m", "seg_0001", "chao cac ban chao cac ban", duration=30.0),  # -> 1/8, breaches
        _rec("m", "seg_0002", "chao cac", duration=10.0),                # -> 1/4, fits
    ]
    kept, _ = filter_meeting_quota(records, [], NORMALIZER, quota=0.25,
                                    min_meeting_foreign_density=0.0)
    assert [r["segment_id"] for r in kept] == ["seg_0000", "seg_0002"]


def test_quota_drops_a_meeting_whose_carriers_alone_are_below_theta():
    # Its densest possible subset is still under theta, so no cut can reach the quota.
    records = [_rec("m", "seg_0000", "chao cac ban cac ban team"),   # carriers: 1/6
               _rec("m", "seg_0001", "chao cac ban")]
    kept, _ = filter_meeting_quota(records, [], NORMALIZER, quota=0.2,
                                    min_meeting_foreign_density=0.2)
    assert kept == []


def test_quota_ties_break_on_segment_id_so_the_corpus_rebuilds_identically():
    records = [_rec("m", "seg_0000", "chao team"),
               _rec("m", "seg_0002", "chao cac ban", duration=10.0),
               _rec("m", "seg_0001", "chao cac ban", duration=10.0)]
    kept, _ = filter_meeting_quota(records, [], NORMALIZER, quota=0.2,
                                    min_meeting_foreign_density=0.0)
    assert [r["segment_id"] for r in kept] == ["seg_0000", "seg_0001"]


def test_val_and_test_untouched_by_both_meeting_modes():
    for kept, _ in (filter_min_meeting_density(RECORDS, VAL_MEETINGS, NORMALIZER, 1.0),
                    filter_meeting_quota(RECORDS, VAL_MEETINGS, NORMALIZER, 1.0, 1.0)):
        assert [(r["meeting_id"], r["segment_id"]) for r in kept] == [
            ("val_a", "seg_0000"), ("test_a", "seg_0000")]
