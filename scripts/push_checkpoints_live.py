"""Upload each `checkpoints/step-<N>` to the Hub while the run is still training.

A rented box can die mid-run -- credit limit, preemption, a dropped SSH session --
and everything under `outputs/` dies with it. `scripts/push_run_adapters.py` only
runs after training finishes, so a run that stops at step 3,000 leaves nothing.
This watches `checkpoints/` and uploads every new `step-<N>` as it appears, each
into its own subfolder of one repo:

    python -m scripts.push_checkpoints_live \
        --run-dir outputs/v6-100h-steps \
        --repo-id rework-whisper-v6-org/v6-100h-steps-checkpoints

Run it in a second tmux window next to training. It never touches the training
process, so a network failure here cannot kill the run -- a failed upload is
retried on the next sweep.

`checkpoints/best/` is not uploaded and does not need to be: `on_evaluate` writes
it and `step-<N>` from the same model at the same moment (src/train.py), so every
`best/` there has ever been is byte-identical to some `step-<N>` already up.

Which checkpoints survive is then a question of what finished uploading, not of
what the box still has. The token comes from HF_TOKEN in the environment.
"""

import argparse
import json
import re
import time
from pathlib import Path

from src.hub import push_adapter

STEP_DIR = re.compile(r"^step-(\d+)$")
WEIGHTS = ("adapter_model.safetensors", "adapter_model.bin")


def is_settled(d: Path, now: float, settle: float) -> bool:
    """True when `d` looks like a finished adapter rather than one PEFT is still
    writing: config plus weights present, and nothing in it touched for `settle`
    seconds. Uploading a half-written directory would put a truncated adapter on
    the Hub under a name that reads as complete."""
    if not (d / "adapter_config.json").exists():
        return False
    if not any((d / w).exists() for w in WEIGHTS):
        return False
    newest = max(f.stat().st_mtime for f in d.iterdir() if f.is_file())
    return now - newest >= settle


def pending(checkpoints: Path, pushed: set[str], now: float,
            settle: float) -> list[Path]:
    """Settled `step-<N>` directories not yet uploaded, oldest step first --
    order matters on a box that may die mid-sweep: the early checkpoints are the
    ones §3a of the plan bets on."""
    out = []
    for p in checkpoints.iterdir():
        m = STEP_DIR.match(p.name)
        if m and p.name not in pushed and p.is_dir() and is_settled(p, now, settle):
            out.append((int(m.group(1)), p))
    return [p for _, p in sorted(out)]


def _load(state: Path) -> set[str]:
    return set(json.loads(state.read_text(encoding="utf-8"))) if state.exists() else set()


def _save(state: Path, pushed: set[str]) -> None:
    state.write_text(json.dumps(sorted(pushed), indent=2), encoding="utf-8")


def watch(run_dir: Path, repo_id: str, interval: float = 120.0,
          settle: float = 60.0, private: bool = True, once: bool = False) -> None:
    checkpoints = run_dir / "checkpoints"
    state = run_dir / ".pushed_checkpoints.json"
    pushed = _load(state)
    if pushed:
        print(f"resume: {len(pushed)} already uploaded")
    while True:
        if checkpoints.is_dir():
            for d in pending(checkpoints, pushed, time.time(), settle):
                try:
                    push_adapter(d, repo_id, private=private, path_in_repo=d.name)
                except Exception as exc:                  # noqa: BLE001
                    # Retried on the next sweep. Never re-raise: this process
                    # dying silently is the one failure the run cannot see.
                    print(f"{d.name}: upload failed, will retry -- {exc}")
                    continue
                pushed.add(d.name)
                _save(state, pushed)
                print(f"{d.name}: https://huggingface.co/{repo_id}/tree/main/{d.name}")
        if once:
            return
        time.sleep(interval)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--repo-id", required=True,
                    help="one repo holds every checkpoint, each in a step-<N>/ subfolder")
    ap.add_argument("--interval", type=float, default=120.0,
                    help="seconds between sweeps (default 120)")
    ap.add_argument("--settle", type=float, default=60.0,
                    help="seconds a directory must sit untouched before it is "
                         "considered fully written (default 60)")
    ap.add_argument("--public", action="store_true",
                    help="create the repo public (default: private)")
    ap.add_argument("--once", action="store_true",
                    help="one sweep and exit, for a run that has already finished")
    args = ap.parse_args()
    watch(args.run_dir, args.repo_id, args.interval, args.settle,
          not args.public, args.once)


if __name__ == "__main__":
    main()
