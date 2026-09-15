"""Push a run's three keeper adapters to the Hub instead of copying them home.

A v6-100h-steps run leaves 18 adapters in `checkpoints/` (one per eval round,
plus the last step, plus `best/`) at 111 MB each. Three are worth keeping, and they
are not the same checkpoint:

  * `best-valcer`   -- `checkpoints/best/`, the one `eval_val_cer` picked.
  * `best-retention` -- whichever `checkpoints/step-<N>/` scored highest on
    cross-domain loanword retention (`scripts/score_cross_domain_model.py`).
    val CER cannot see retention, so this has to be named on the command line
    after that screening pass; there is no way to derive it from the run dir.
  * `last`          -- the highest `step-<N>` on disk. The last eval round lands
    on the last multiple of `eval_steps`, so this is a later, separate
    checkpoint whenever the step count is not a multiple of `eval_steps`.

    python -m scripts.push_run_adapters --run-dir outputs/v6-100h-steps \
        --repo-prefix rework-whisper-v6-org/v6-100h-steps \
        --best-retention step-3200

Adapters go up UNSCALED -- the lambda a deployment wants is baked in later by
`src.lora.save_with_lambda`, and `training.init_adapter` refuses a scaled one.
Repos are created private. The token comes from HF_TOKEN in the environment and
is never written into an artifact (src/hub.py).
"""

import argparse
import re
from pathlib import Path

from src.hub import push_adapter

STEP_DIR = re.compile(r"^step-(\d+)$")


def last_step_dir(checkpoints: Path) -> Path:
    """The highest-numbered `step-<N>` under `checkpoints/`. Trainer's own
    `checkpoint-<N>` directories are optimizer state, not adapters, and do not
    match -- the same distinction `archive_run`'s zip filter draws."""
    steps = [(int(m.group(1)), p) for p in checkpoints.iterdir()
             if p.is_dir() and (m := STEP_DIR.match(p.name))]
    if not steps:
        raise FileNotFoundError(
            f"no step-<N> directory under {checkpoints} -- this run predates the "
            "per-round checkpoint save, or trained with an older src/train.py")
    return max(steps)[1]


def _check(d: Path) -> Path:
    if not (d / "adapter_config.json").exists():
        raise FileNotFoundError(f"{d} has no adapter_config.json -- not a PEFT adapter")
    return d


def push_run_adapters(run_dir: Path, repo_prefix: str, best_retention: str,
                       private: bool = True, dry_run: bool = False) -> dict[str, str]:
    checkpoints = run_dir / "checkpoints"
    targets = {
        "best-valcer": _check(checkpoints / "best"),
        "best-retention": _check(checkpoints / best_retention),
        "last": _check(last_step_dir(checkpoints)),
    }
    urls = {}
    for name, d in targets.items():
        repo_id = f"{repo_prefix}-{name}"
        print(f"{name}: {d} -> {repo_id}")
        urls[name] = (f"https://huggingface.co/{repo_id}" if dry_run
                      else push_adapter(d, repo_id, private=private))
    if dry_run:
        print("\ndry run -- nothing uploaded")
    return urls


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--repo-prefix", required=True,
                    help="each adapter is pushed to <prefix>-best-valcer, "
                         "<prefix>-best-retention, <prefix>-last")
    ap.add_argument("--best-retention", required=True,
                    help="the step-<N> directory that won the cross-domain retention "
                         "screen, e.g. step-3200")
    ap.add_argument("--public", action="store_true",
                    help="create the repos public (default: private)")
    ap.add_argument("--dry-run", action="store_true",
                    help="resolve and check the three directories, upload nothing")
    args = ap.parse_args()
    for name, url in push_run_adapters(args.run_dir, args.repo_prefix,
                                        args.best_retention, not args.public,
                                        args.dry_run).items():
        print(f"{name}: {url}")


if __name__ == "__main__":
    main()
