"""Tests for scripts/push_checkpoints_live.py -- picking which checkpoints are
ready to upload. Nothing here touches the network; the upload is src/hub.py."""

import json
import os
import time

from scripts.push_checkpoints_live import is_settled, pending, watch


def _adapter(d, age=3600.0):
    d.mkdir(parents=True, exist_ok=True)
    (d / "adapter_config.json").write_text("{}", encoding="utf-8")
    (d / "adapter_model.safetensors").write_bytes(b"w")
    old = time.time() - age
    for f in d.iterdir():
        os.utime(f, (old, old))
    return d


def test_a_directory_still_being_written_is_not_uploaded(tmp_path):
    # PEFT has written the config but the weights are not there yet.
    d = tmp_path / "step-400"
    d.mkdir()
    (d / "adapter_config.json").write_text("{}", encoding="utf-8")
    assert not is_settled(d, time.time(), settle=60.0)


def test_a_complete_but_freshly_written_directory_waits_for_settle(tmp_path):
    d = _adapter(tmp_path / "step-400", age=0.0)
    assert not is_settled(d, time.time(), settle=60.0)
    assert is_settled(d, time.time() + 120, settle=60.0)


def test_pending_is_ordered_by_step_number_not_by_name(tmp_path):
    for s in (6400, 800, 400):
        _adapter(tmp_path / f"step-{s}")
    names = [p.name for p in pending(tmp_path, set(), time.time(), 60.0)]
    assert names == ["step-400", "step-800", "step-6400"]


def test_already_pushed_and_non_step_dirs_are_skipped(tmp_path):
    _adapter(tmp_path / "step-400")
    _adapter(tmp_path / "step-800")
    _adapter(tmp_path / "best")                 # same weights as some step-<N>
    _adapter(tmp_path / "checkpoint-800")       # Trainer optimizer state
    names = [p.name for p in pending(tmp_path, {"step-400"}, time.time(), 60.0)]
    assert names == ["step-800"]


def test_a_failed_upload_is_retried_and_not_recorded(tmp_path, monkeypatch):
    run = tmp_path / "run"
    _adapter(run / "checkpoints" / "step-400")
    calls = []

    def flaky(d, repo_id, private=True, path_in_repo=None):
        calls.append(path_in_repo)
        if len(calls) == 1:
            raise OSError("connection reset")
        return f"https://huggingface.co/{repo_id}"

    monkeypatch.setattr("scripts.push_checkpoints_live.push_adapter", flaky)
    watch(run, "org/ck", once=True)
    assert not (run / ".pushed_checkpoints.json").exists()
    watch(run, "org/ck", once=True)
    assert calls == ["step-400", "step-400"]
    assert json.loads((run / ".pushed_checkpoints.json").read_text()) == ["step-400"]


def test_the_log_is_re_uploaded_every_sweep_so_the_eval_curve_survives(tmp_path, monkeypatch):
    run = tmp_path / "run"
    _adapter(run / "checkpoints" / "step-400")
    (run / "run.log").write_text("ValCER 0.03", encoding="utf-8")
    seen = []

    def fake_file(path, repo_id, path_in_repo, private=True):
        seen.append(path_in_repo)

    monkeypatch.setattr("scripts.push_checkpoints_live.upload_file", fake_file)
    monkeypatch.setattr("scripts.push_checkpoints_live.push_adapter",
                        lambda *a, **k: "")
    watch(run, "org/ck", once=True, log=run / "run.log")
    watch(run, "org/ck", once=True, log=run / "run.log")
    # Uploaded on both sweeps, unlike a checkpoint, because it keeps growing.
    assert seen == ["run.log", "run.log"]


def test_a_restart_does_not_re_upload_what_is_already_on_the_hub(tmp_path):
    run = tmp_path / "run"
    _adapter(run / "checkpoints" / "step-400")
    (run / ".pushed_checkpoints.json").write_text('["step-400"]', encoding="utf-8")
    assert pending(run / "checkpoints", set(json.loads(
        (run / ".pushed_checkpoints.json").read_text())), time.time(), 60.0) == []
