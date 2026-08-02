"""Tests for src/config.py:validate -- currently only the rules touched by
this session's changes (training.limit, added 2026-08-02 for a quick
end-to-end dry run). Broader config validation coverage is still a todo
(see SESSIONS.md)."""

from dataclasses import replace

import pytest

from src.config import Config, Data, Training, validate


def _cfg(**training_overrides) -> Config:
    return Config(run_id="t", base_model="m", data=Data(dataset_path="d"),
                  training=replace(Training(), **training_overrides))


def test_training_limit_null_is_valid():
    validate(_cfg(limit=None))  # should not raise


def test_training_limit_positive_is_valid():
    validate(_cfg(limit=20))  # should not raise


def test_training_limit_zero_or_negative_rejected():
    with pytest.raises(ValueError):
        validate(_cfg(limit=0))
    with pytest.raises(ValueError):
        validate(_cfg(limit=-5))
