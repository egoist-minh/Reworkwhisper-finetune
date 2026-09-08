"""Config loading and validation. PROJECT_CORE.md §3.

Dataclasses, not pydantic -- one fewer dependency whose major version can shift under us.
Validation is fail-loud: a bad config raises before any GPU time is spent.

Downstream stages read outputs/{run_id}/config.json, never configs/experiment.yaml.
"""

import json
from dataclasses import dataclass, field, asdict, fields, is_dataclass
from pathlib import Path

import yaml

# Vietnamese lexical particles that carry meaning. Deleting these from a reference
# transcript destroys real content, so they can never be treated as fillers.
LEXICAL_PARTICLES = {"ạ", "à", "ừ", "ơ", "dạ", "vâng", "nhé", "nhỉ"}


@dataclass
class Data:
    dataset_path: str
    ood_eval_path: str | None = None
    real_bench_path: str | None = None
    real_clip_path: str | None = None
    val_meetings: list[str] = field(default_factory=list)


@dataclass
class Normalization:
    strip_punctuation: bool = True
    lowercase: bool = True
    number_convention: str = "word_to_digit"
    audit_conversions: bool = True
    filler_tokens: list[str] = field(default_factory=list)


@dataclass
class Eval:
    split: str = "test"
    limit: int | None = None
    batch_size: int = 8
    num_beams: int = 1
    language: str = "vi"


@dataclass
class Lora:
    rank: int = 16
    alpha: int = 32
    target_modules: list[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"])
    dropout: float = 0.05
    use_rslora: bool = True
    min_retained_energy: float = 0.99   # Path B (full-FT SVD) only


@dataclass
class Training:
    learning_rate: float = 2.0e-4
    epochs: int = 3
    batch_size: int = 8
    grad_accum_steps: int = 2
    warmup_ratio: float = 0.1
    full_finetune: bool = False
    gradient_checkpointing: bool = True
    limit: int | None = None   # cap train/val (+ood, if present) segment count for a
                                # quick end-to-end dry run; null = full split. Separate
                                # from eval.limit, which only affects baseline/gate evals.


@dataclass
class Sweep:
    lambdas: list[float] = field(default_factory=lambda: [0.0, 0.25, 0.5, 0.75, 1.0])
    ood_cer_budget: float = 0.02   # max absolute OOD CER regression vs baseline
    # Elbow rule for cost/benefit lambda selection (SESSIONS.md E2): stop
    # advancing through the grid once a step's (delta_ood_cer / delta_val_cer)
    # exceeds this multiple of the previous step's ratio. 10.0 separates the
    # ~8.9x step-to-step ratio jump seen at lambda=0.5 (accepted) from the
    # ~12.6x jump at lambda=0.75 (rejected) on v3-r16's own sweep data.
    elbow_ratio_threshold: float = 10.0


@dataclass
class Gates:
    """Eval gate thresholds (PROJECT_CORE.md §6 Stage 4). Only tiers 1, 2, 4a
    are in scope for the first run -- tier 3 (RTF) is dropped and tier 4b
    (long-form) deferred (see handoff, deadline scope cut)."""
    min_improvement_pct: float = 10.0    # tier 1: CER_test <= (1 - pct/100) * CER_base
    real_cer_regression_pp: float = 0.0  # tier 4a: CER_real must not exceed CER_real(base) + this
    max_retention_regression_pp: float = 0.0  # tier1 by_source: english_token_retention must
                                               # not drop more than this (absolute) vs baseline
                                               # on the same source slice (H4b, SESSIONS.md H6)


@dataclass
class Hub:
    push: bool = False
    repo_id: str | None = None
    private: bool = True


@dataclass
class Config:
    run_id: str
    base_model: str
    data: Data
    seed: int = 42
    normalization: Normalization = field(default_factory=Normalization)
    eval: Eval = field(default_factory=Eval)
    lora: Lora = field(default_factory=Lora)
    training: Training = field(default_factory=Training)
    sweep: Sweep = field(default_factory=Sweep)
    gates: Gates = field(default_factory=Gates)
    hub: Hub = field(default_factory=Hub)

    @property
    def out_dir(self) -> Path:
        return Path("outputs") / self.run_id


def _build(cls, raw: dict):
    """Nested dict -> dataclass, rejecting unknown keys so a typo is not silently ignored."""
    known = {f.name: f for f in fields(cls)}
    unknown = set(raw) - set(known)
    if unknown:
        raise ValueError(f"{cls.__name__}: unknown config keys {sorted(unknown)}")
    kwargs = {}
    for name, f in known.items():
        if name not in raw:
            continue
        sub = f.type if is_dataclass(f.type) else None
        kwargs[name] = _build(sub, raw[name] or {}) if sub else raw[name]
    return cls(**kwargs)


def apply_override(raw: dict, expr: str) -> None:
    """`--override lora.rank=32`. The value goes through the YAML parser so ints, floats,
    bools, null and lists all coerce the same way they would in the file."""
    key, _, val = expr.partition("=")
    if not _:
        raise ValueError(f"override must be key=value, got: {expr!r}")
    node, *rest = key.split(".")
    cur = raw
    path = [node] + rest
    for part in path[:-1]:
        if part not in cur or not isinstance(cur[part], dict):
            raise ValueError(f"override path not in config: {key}")
        cur = cur[part]
    if path[-1] not in cur:
        raise ValueError(f"override path not in config: {key}")
    cur[path[-1]] = yaml.safe_load(val)


def validate(cfg: Config) -> None:
    bad = set(t.lower() for t in cfg.normalization.filler_tokens) & LEXICAL_PARTICLES
    if bad:
        raise ValueError(f"filler_tokens contains lexical particles {sorted(bad)} -- "
                         "removing these deletes real content from the reference")
    if cfg.lora.rank <= 0 or cfg.lora.alpha <= 0:
        raise ValueError("lora.rank and lora.alpha must be positive")
    if cfg.eval.limit is not None and cfg.eval.limit <= 0:
        raise ValueError("eval.limit must be a positive int or null")
    if cfg.training.limit is not None and cfg.training.limit <= 0:
        raise ValueError("training.limit must be a positive int or null")
    # PyYAML is YAML 1.1: a float literal needs a decimal point, so
    # `--override training.learning_rate=5e-05` resolves to the *string* "5e-05"
    # and nothing catches it until AdamW compares it to a float, ~200 frames into
    # trainer.train() with the model already on the GPU. Caught live on Kaggle.
    if not isinstance(cfg.training.learning_rate, (int, float)):
        raise ValueError(
            f"training.learning_rate is {cfg.training.learning_rate!r}, a "
            f"{type(cfg.training.learning_rate).__name__} -- YAML needs a decimal "
            "point to read a float, so write 5.0e-05 or 0.00005, not 5e-05")
    if not (0.0 < cfg.lora.min_retained_energy <= 1.0):
        raise ValueError("lora.min_retained_energy must be in (0, 1]")
    if not cfg.sweep.lambdas:
        raise ValueError("sweep.lambdas is empty -- there is nothing to select from")

    # Without an OOD set there is no forgetting measurement at all (tier 2 is the
    # only one) and every sweep row's ood_cer is None, so select_lambda finds no
    # budget-safe lambda and hard-fails -- but only AFTER training and five val
    # decodes have already been paid for. Fail here instead.
    if not cfg.data.ood_eval_path:
        raise ValueError(
            "data.ood_eval_path is unset -- tier 2 (OOD) is the only forgetting "
            "measurement in the gate, and sweep-gate would hard-fail in "
            "select_lambda after training. Run scripts/fetch_vivos.py first."
        )

    # Tier-4 leak guard: the real benchmark must never be reachable as training data.
    ds = Path(cfg.data.dataset_path).resolve()
    for name in ("real_bench_path", "real_clip_path"):
        p = getattr(cfg.data, name)
        if not p:
            continue
        rp = Path(p).resolve()
        if rp == ds or ds in rp.parents or rp in ds.parents:
            raise ValueError(f"data.{name} ({rp}) overlaps data.dataset_path ({ds}) -- "
                             "the real benchmark would leak into training")
    if cfg.hub.push and not cfg.hub.repo_id:
        raise ValueError("hub.push is true but hub.repo_id is unset")


def load(path: str | Path, overrides: list[str] | None = None) -> Config:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    for expr in overrides or []:
        apply_override(raw, expr)
    cfg = _build(Config, raw)
    validate(cfg)
    return cfg


def freeze(cfg: Config, dest: Path | None = None) -> Path:
    """Write the resolved config to outputs/{run_id}/config.json. This file -- not the
    YAML -- is what every downstream stage reads, so a mid-run YAML edit cannot
    retroactively change what a completed stage claims it ran with."""
    out = dest or (cfg.out_dir / "config.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(asdict(cfg), indent=2, ensure_ascii=False), encoding="utf-8")
    return out
