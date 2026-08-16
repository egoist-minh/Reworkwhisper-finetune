"""Short training runs at several learning rates, overlaid on one loss curve.

There is no LR finder in HuggingFace Trainer (Lightning's `Tuner.lr_find()` and
fastai's `learn.lr_find()` have no equivalent here, and
`Trainer.hyperparameter_search` is full HPO, not a range test). This script is
the manual substitute: it shells out to `src.pipeline --stage train` once per
candidate LR with `training.limit` capping the split, then plots the resulting
`metrics/training.csv` loss columns on top of each other so the three shapes
(too low / right / too high) are readable side by side.

    python scripts/lr_probe.py --source-run outputs/v4-mixed-r16 \
        --lrs 2e-4,8e-4,2.4e-3 --limit 1600

Reads nothing the pipeline doesn't already write, and adds no config field --
every knob goes through `--override`, which `src/config.py:apply_override`
already supports.

Two costs worth knowing before starting:

  * `training.limit` caps train, val AND ood at once (src/pipeline.py:192), so
    lowering it to buy fewer train steps does not buy a cheaper end-of-epoch
    eval once it drops under the val/OOD split sizes -- that decode is close to
    a fixed tax per probe run. It cannot be skipped: `src/train.py` hard-fails
    if `eval_val_cer` is never observed.
  * Each probe writes a full `outputs/{run_id}-lr{lr}/` tree, checkpoint
    included. A run whose training.csv already exists is skipped, not repeated
    -- same resume contract as `.pipeline_state.json` (PROJECT_CORE.md §2.1).
"""
import argparse
import csv
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Same palette as scripts/plot_training_curve.py, so probe plots and run plots
# read as one set.
COLORS = ["#7d8b99", "#3f8f5f", "#c9435b", "#e8a33d", "#8e44ad"]


def probe_run_id(base_run_id: str, lr: float) -> str:
    return f"{base_run_id}-lr{lr:g}"


def yaml_float(x: float) -> str:
    """Decimal form, because `--override` values go through `yaml.safe_load` and
    PyYAML follows YAML 1.1: a float literal needs a decimal point, so `5e-05`
    resolves to the *string* "5e-05". Nothing downstream type-checks it and the
    failure lands 200 frames later inside AdamW as
    `'<=' not supported between instances of 'float' and 'str'`."""
    s = f"{x:.12f}".rstrip("0")
    s = s + "0" if s.endswith(".") else s
    if float(s) != x:
        raise ValueError(f"{x!r} does not survive 12-decimal formatting (got {s}) -- "
                         "too small for this grid")
    return s


def run_one(config: Path, run_id: str, lr: float, limit: int, epochs: int,
            source_run: Path, extra_overrides: list[str], extra_args: list[str]) -> Path:
    """Train once at `lr`. Returns the path to that run's training.csv."""
    out_dir = Path("outputs") / run_id
    csv_path = out_dir / "metrics" / "training.csv"
    if csv_path.exists():
        print(f"[skip] {run_id}: {csv_path} already exists")
        return csv_path

    # stage_train reads validated_manifest.jsonl and refuses to run without it,
    # but re-running stage baseline per probe would cost a full test+OOD+real
    # decode each time. The manifest is a pure function of the dataset and
    # val_meetings, so the source run's copy is the same file this probe would
    # have produced.
    manifest = source_run / "validated_manifest.jsonl"
    if not manifest.exists():
        raise FileNotFoundError(
            f"{manifest} missing -- run `--stage baseline` on {source_run.name} first, "
            "or point --source-run at a run that has completed it")
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(manifest, out_dir / "validated_manifest.jsonl")

    cmd = [sys.executable, "-m", "src.pipeline", "--stage", "train", "--config", str(config)]
    cmd += extra_args
    for o in extra_overrides:
        cmd += ["--override", o]
    # apply_override walks the list in order and the last write wins, so the four
    # values this script exists to control go last -- otherwise a caller passing
    # its own `training.learning_rate` through --extra-args (the Kaggle notebook's
    # OVERRIDES string does exactly that) would silently pin every probe to the
    # same LR and the whole sweep would read as noise.
    for o in (f"run_id={run_id}",
              f"training.learning_rate={yaml_float(lr)}",
              f"training.limit={limit}",
              f"training.epochs={epochs}"):
        cmd += ["--override", o]

    print(f"[run ] lr={lr:g} -> {run_id}")
    subprocess.run(cmd, check=True)
    if not csv_path.exists():
        raise FileNotFoundError(f"{csv_path} not written -- training reported success but "
                                "left no metrics; treat this run as failed")
    return csv_path


def read_loss(csv_path: Path) -> tuple[list[int], list[float], int]:
    """(steps, loss, warmup_end_step). The warmup end is read off the
    learning_rate column rather than recomputed from warmup_ratio: the schedule
    rises to the configured LR and decays after, so its argmax IS the boundary,
    whatever rounding Trainer did to the step count."""
    steps, loss, lrs = [], [], []
    with open(csv_path, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if not row["loss"]:
                continue
            steps.append(int(row["step"]))
            loss.append(float(row["loss"]))
            lrs.append(float(row["learning_rate"]) if row["learning_rate"] else 0.0)
    if not steps:
        raise ValueError(f"{csv_path} has no loss rows")
    return steps, loss, steps[lrs.index(max(lrs))]


def summarize(steps: list[int], loss: list[float], warmup_end: int, blocks: int = 10) -> dict:
    """Post-warmup shape of one curve.

    `rise_frac` is computed on block means, not raw steps: `logging_steps=1`
    makes per-step loss noisy enough that a healthy run still steps upward about
    half the time (measured 0.506 on Outputs/v3-r16), so raw step-to-step rises
    carry no signal at all. Averaged into ~`blocks` chunks the same run reads
    monotone, and a bouncing LR still shows its rises."""
    post = [(s, l) for s, l in zip(steps, loss) if s > warmup_end] or list(zip(steps, loss))
    vals = [l for _, l in post]
    tail = vals[-10:]
    width = max(1, len(vals) // blocks)
    means = [sum(vals[i:i + width]) / len(vals[i:i + width])
             for i in range(0, len(vals), width)]
    rises = sum(1 for a, b in zip(means, means[1:]) if b > a)
    diverged = any(l != l or l == float("inf") for l in vals) or (
        sum(tail) / len(tail) > vals[0])
    return {
        "steps": len(steps),
        "warmup_end_step": warmup_end,
        "first_post_warmup_loss": round(vals[0], 4),
        "min_loss": round(min(vals), 4),
        "final_loss": round(sum(tail) / len(tail), 4),
        "rise_frac": round(rises / max(1, len(means) - 1), 3),
        "diverged": diverged,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lrs", required=True,
                    help="comma-separated candidates, e.g. 2e-4,8e-4,2.4e-3")
    ap.add_argument("--source-run", type=Path, required=True,
                    help="run dir holding validated_manifest.jsonl (a completed --stage baseline)")
    ap.add_argument("--config", type=Path, default=Path("configs/experiment.yaml"))
    ap.add_argument("--limit", type=int, default=1600,
                    help="training.limit per probe; 1600 segments at batch 8 x grad_accum 2 "
                         "is ~100 optimizer steps")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--out", type=Path, default=Path("outputs/lr-probe"))
    ap.add_argument("--override", action="append", default=[],
                    help="extra `key=value` passed through to every probe run, e.g. lora.alpha=128")
    ap.add_argument("--extra-args", default="",
                    help="raw argument string forwarded verbatim to every `src.pipeline` call, "
                         "for callers that already hold one (the Kaggle notebook's OVERRIDES). "
                         "run_id / learning_rate / limit / epochs in here are overridden by "
                         "this script's own values.")
    args = ap.parse_args()
    extra_args = shlex.split(args.extra_args)

    lrs = [float(x) for x in args.lrs.split(",")]
    # Named after the source run, not configs/experiment.yaml's run_id: the notebook
    # sets RUN_ID itself and passes it via --override, so the YAML value can differ
    # from the run this probe is actually attached to.
    base_run_id = args.source_run.name

    curves = []
    for lr in lrs:
        run_id = probe_run_id(base_run_id, lr)
        csv_path = run_one(args.config, run_id, lr, args.limit, args.epochs,
                           args.source_run, args.override, extra_args)
        steps, loss, warmup_end = read_loss(csv_path)
        curves.append((lr, steps, loss, warmup_end, summarize(steps, loss, warmup_end)))

    args.out.mkdir(parents=True, exist_ok=True)
    summary_path = args.out / "summary.csv"
    fields = ["learning_rate", "steps", "warmup_end_step", "first_post_warmup_loss",
              "min_loss", "final_loss", "rise_frac", "diverged"]
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for lr, _, _, _, s in curves:
            w.writerow({"learning_rate": f"{lr:g}", **s})

    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=110)
    # log scale for the same reason as the run curves: step 1 loss is an order of
    # magnitude above everything after it and would flatten the rest.
    ax.set_yscale("log")
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("train loss (log)")
    ax.grid(alpha=0.3)
    for i, (lr, steps, loss, warmup_end, s) in enumerate(curves):
        ax.plot(steps, loss, color=COLORS[i % len(COLORS)], linewidth=1.4,
                label=f"lr={lr:g}  final {s['final_loss']:.3f}  rise {s['rise_frac']:.0%}"
                      + ("  DIVERGED" if s["diverged"] else ""))
    warmup_end = curves[0][3]
    ax.axvline(warmup_end, color="#999999", linestyle=":", linewidth=1.2)
    ax.annotate("warmup ends", (warmup_end, ax.get_ylim()[1]), fontsize=9,
                color="#666666", ha="left", va="top", xytext=(4, -4),
                textcoords="offset points")
    ax.set_title(f"{base_run_id} · LR probe ({args.limit} segments, {args.epochs} epoch)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    plot_path = args.out / "loss_curves.png"
    fig.savefig(plot_path)

    print(f"\nwrote {plot_path}\nwrote {summary_path}\n")
    print(f"{'lr':>10}{'final_loss':>12}{'min_loss':>10}{'rise_frac':>11}{'diverged':>10}")
    for lr, _, _, _, s in curves:
        print(f"{lr:>10g}{s['final_loss']:>12.4f}{s['min_loss']:>10.4f}"
              f"{s['rise_frac']:>11.3f}{str(s['diverged']):>10}")
    print("\nPick the lowest final_loss whose rise_frac is not elevated vs. its neighbours "
          "(Outputs/v3-r16's full 3-epoch run scores 0.2 for reference). "
          "If the winner is at either end of the grid, extend the grid that way and re-run.")


if __name__ == "__main__":
    main()
