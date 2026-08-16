"""Tests for src/train.py:_EarlyStoppingState -- the pure best-value/patience
tracker behind RobustEarlyStoppingCallback. No transformers/torch dependency,
so these run without a GPU/Kaggle. See src/train.py's module docstring for
why this replaced transformers' built-in EarlyStoppingCallback."""

from src.train import _EarlyStoppingState, _write_training_csv


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
