"""Tests for scripts/push_run_adapters.py -- resolving a run's three keeper
adapters. Nothing here touches the network; the upload itself is src/hub.py."""

import pytest

from scripts.push_run_adapters import last_step_dir, push_run_adapters


def _adapter(d):
    d.mkdir(parents=True, exist_ok=True)
    (d / "adapter_config.json").write_text("{}", encoding="utf-8")
    return d


def _run(tmp_path, steps=(400, 800, 6400, 6555)):
    ck = tmp_path / "checkpoints"
    _adapter(ck / "best")
    for s in steps:
        _adapter(ck / f"step-{s}")
    return tmp_path


def test_last_step_is_the_highest_number_not_the_last_listed(tmp_path):
    # 800 sorts after 6400 as a string; the run's last checkpoint is 6555.
    run = _run(tmp_path)
    assert last_step_dir(run / "checkpoints").name == "step-6555"


def test_trainer_s_own_checkpoint_dirs_are_not_adapters(tmp_path):
    run = _run(tmp_path, steps=(400,))
    _adapter(run / "checkpoints" / "checkpoint-9999")     # optimizer state, not an adapter
    assert last_step_dir(run / "checkpoints").name == "step-400"


def test_raises_when_no_step_dir_exists(tmp_path):
    ck = tmp_path / "checkpoints"
    _adapter(ck / "best")
    with pytest.raises(FileNotFoundError, match="no step-<N> directory"):
        last_step_dir(ck)


def test_dry_run_resolves_all_three_without_uploading(tmp_path):
    run = _run(tmp_path)
    urls = push_run_adapters(run, "org/v6-100h-steps", "step-800", dry_run=True)
    assert urls == {
        "best-valcer": "https://huggingface.co/org/v6-100h-steps-best-valcer",
        "best-retention": "https://huggingface.co/org/v6-100h-steps-best-retention",
        "last": "https://huggingface.co/org/v6-100h-steps-last",
    }


def test_a_named_retention_step_that_does_not_exist_fails_before_any_upload(tmp_path):
    run = _run(tmp_path)
    with pytest.raises(FileNotFoundError, match="adapter_config.json"):
        push_run_adapters(run, "org/x", "step-1234", dry_run=True)
