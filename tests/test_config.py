"""Tests for src/config.py:validate -- currently only the rules touched by
this session's changes (training.limit, added 2026-08-02 for a quick
end-to-end dry run). Broader config validation coverage is still a todo
(see SESSIONS.md)."""

from dataclasses import replace

import pytest

from src.config import Config, Data, Gates, Sweep, Training, validate


def _cfg(**training_overrides) -> Config:
    return Config(run_id="t", base_model="m",
                  data=Data(dataset_path="d", ood_eval_path="ood"),
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


def test_null_ood_eval_path_rejected_before_any_gpu_time():
    # Without it, select_lambda only discovers there is nothing to select AFTER
    # training has finished -- see src/config.py:validate.
    cfg = Config(run_id="t", base_model="m", data=Data(dataset_path="d"))
    with pytest.raises(ValueError, match="ood_eval_path"):
        validate(cfg)


# --------------------------------------------- training.init_adapter (curriculum phase 2)


def test_init_adapter_pointing_at_a_real_adapter_dir_is_valid(tmp_path):
    (tmp_path / "adapter_config.json").write_text("{}", encoding="utf-8")
    validate(_cfg(init_adapter=str(tmp_path)))  # should not raise


def test_init_adapter_with_no_adapter_config_rejected_before_any_model_download(tmp_path):
    with pytest.raises(ValueError, match="init_adapter"):
        validate(_cfg(init_adapter=str(tmp_path / "typo")))


# ------------------------------------------------------------ sweep.retention_floor


def _cfg_sweep(**sweep_overrides) -> Config:
    return Config(run_id="t", base_model="m",
                  data=Data(dataset_path="d", ood_eval_path="ood"),
                  sweep=replace(Sweep(), **sweep_overrides))


def test_retention_floor_null_or_fraction_is_valid():
    validate(_cfg_sweep(retention_floor=None))
    validate(_cfg_sweep(retention_floor=0.666))


def test_retention_floor_outside_zero_to_one_rejected():
    # 66.6 instead of 0.666 -- percent vs fraction is the mistake to catch, and
    # it would otherwise reject every lambda and hard-fail after training.
    with pytest.raises(ValueError, match="retention_floor"):
        validate(_cfg_sweep(retention_floor=66.6))


# ------------------------------------------------------------- cross-domain gates


def _cfg_cross(cross_domain_path=None, **gate_overrides) -> Config:
    return Config(run_id="t", base_model="m",
                  data=Data(dataset_path="d", ood_eval_path="ood",
                            cross_domain_path=cross_domain_path),
                  gates=replace(Gates(), **gate_overrides))


def test_cross_domain_gates_without_a_path_rejected():
    # The check would never run, so the threshold would read as passing.
    with pytest.raises(ValueError, match="cross_domain_path"):
        validate(_cfg_cross(cross_domain_cer_max=0.0819))


def test_cross_domain_gates_with_a_path_are_valid():
    validate(_cfg_cross(cross_domain_path="dataset/cross-domain-bench",
                        cross_domain_cer_max=0.0819, cross_domain_retention_min=0.666))


def test_cross_domain_path_inside_dataset_path_rejected_as_a_leak():
    cfg = Config(run_id="t", base_model="m",
                 data=Data(dataset_path="dataset/v6-corpus", ood_eval_path="ood",
                           cross_domain_path="dataset/v6-corpus/cross"))
    with pytest.raises(ValueError, match="cross_domain_path"):
        validate(cfg)
