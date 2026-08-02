"""Tests for src/train.py:_EarlyStoppingState -- the pure best-value/patience
tracker behind RobustEarlyStoppingCallback. No transformers/torch dependency,
so these run without a GPU/Kaggle. See src/train.py's module docstring for
why this replaced transformers' built-in EarlyStoppingCallback."""

from src.train import _EarlyStoppingState


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
