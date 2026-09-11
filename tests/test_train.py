"""Tests for src/train.py:_EarlyStoppingState -- the pure best-value/patience
tracker behind RobustEarlyStoppingCallback. No transformers/torch dependency,
so these run without a GPU/Kaggle. See src/train.py's module docstring for
why this replaced transformers' built-in EarlyStoppingCallback."""

import pytest

from src.train import (
    _EarlyStoppingState,
    _summarize_step_times,
    _write_training_csv,
)


def test_first_update_always_improves():
    s = _EarlyStoppingState(patience=3)
    assert s.update(0.5) is False
    assert s.best == 0.5
    assert s.rounds_without_improvement == 0


def test_lower_is_better_by_default():
    s = _EarlyStoppingState(patience=2)
    s.update(0.5)
    assert s.update(0.4) is False  # improved
    assert s.best == 0.4
    assert s.rounds_without_improvement == 0


def test_stops_after_patience_rounds_without_improvement():
    s = _EarlyStoppingState(patience=2)
    s.update(0.5)
    assert s.update(0.6) is False  # 1 round without improvement
    assert s.update(0.6) is True   # 2 rounds without improvement -- stop


def test_greater_is_better_mode():
    s = _EarlyStoppingState(patience=1, greater_is_better=True)
    s.update(0.5)
    assert s.update(0.4) is True  # worse under greater_is_better -- stop after patience=1


def test_equal_value_does_not_count_as_improvement():
    s = _EarlyStoppingState(patience=1)
    s.update(0.5)
    assert s.update(0.5) is True  # no change -- not an improvement, patience exhausted


# --------------------------------------------------- training.csv val loss/wer

def test_write_training_csv_includes_eval_val_loss_and_wer(tmp_path):
    # trainer.state.log_history already carries these two (Trainer's own eval loss,
    # compute_metrics' "wer") -- they were just never written to the CSV before.
    log_history = [
        {"step": 10, "loss": 1.23, "learning_rate": 0.0001},
        {"step": 20, "eval_val_loss": 0.5, "eval_val_cer": 0.1, "eval_val_wer": 0.2},
        {"step": 20, "eval_ood_cer": 0.15},
    ]
    out = _write_training_csv(log_history, tmp_path / "training.csv")
    rows = out.read_text(encoding="utf-8").splitlines()
    assert "eval_val_loss" in rows[0]
    assert "eval_val_wer" in rows[0]
    assert "0.5" in rows[2]
    assert "0.2" in rows[2]


# --- _summarize_step_times: the steady-state step cost a 100 h projection is
# built on. Warmup steps carry cuDNN autotune and allocator growth, so leaving
# them in inflates the projection.


def _series(step_costs, gap=0.0, start=100.0):
    """begins/ends for steps of the given compute costs, separated by `gap`."""
    begins, ends, t = [], [], start
    for c in step_costs:
        begins.append(t)
        ends.append(t + c)
        t += c + gap
    return begins, ends


def test_returns_none_when_there_is_no_steady_state_left_after_warmup():
    begins, ends = _series([1.0] * 11)
    assert _summarize_step_times(begins, ends, warmup=10) is None


def test_warmup_steps_are_excluded_from_the_median():
    # 10 slow warmup steps then 10 steady ones: the median must be the steady cost.
    begins, ends = _series([9.0] * 10 + [1.0] * 10)
    out = _summarize_step_times(begins, ends, warmup=10)
    assert out["steps_measured"] == 10
    assert out["compute_seconds"]["median"] == pytest.approx(1.0)
    assert out["first_step_seconds"] == pytest.approx(9.0)


def test_dataloader_gap_is_reported_separately_from_compute():
    begins, ends = _series([1.0] * 20, gap=0.5)
    out = _summarize_step_times(begins, ends, warmup=10)
    assert out["compute_seconds"]["median"] == pytest.approx(1.0)
    assert out["dataloader_gap_seconds"]["median"] == pytest.approx(0.5)
    # cycle is what a step count gets multiplied by, so it has to carry both.
    assert out["cycle_seconds"]["median"] == pytest.approx(1.5)


def test_a_single_stall_moves_p95_but_not_the_median():
    costs = [1.0] * 29 + [20.0]
    begins, ends = _series(costs)
    out = _summarize_step_times(begins, ends, warmup=10)
    assert out["compute_seconds"]["median"] == pytest.approx(1.0)
    assert out["compute_seconds"]["max"] == pytest.approx(20.0)
