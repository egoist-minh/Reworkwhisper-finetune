"""Plot a run's training curve: train loss + val/OOD CER per eval point.

Two sources are needed because the CSV and the trainer state each hold half the
picture: metrics/training.csv has step-level loss and eval_val_cer, while the
OOD CER only lands in checkpoints/*/trainer_state.json (log_history).

    python scripts/plot_training_curve.py Outputs/v3-r16 --out docs/training-curves/v3_v3-r16_training-curve.png
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_training_csv(path):
    steps, loss, val, val_loss = [], [], [], []
    with open(path, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            step = int(row["step"])
            if row["loss"]:
                steps.append(step)
                loss.append(float(row["loss"]))
            if row["eval_val_cer"]:
                val.append((step, float(row["eval_val_cer"]) * 100))
            if row.get("eval_val_loss"):
                val_loss.append((step, float(row["eval_val_loss"])))
    return steps, loss, val, val_loss


def read_ood(run_dir):
    """Latest checkpoint's log_history holds every eval_ood_cer of the run."""
    ckpts = sorted(run_dir.glob("checkpoints/checkpoint-*/trainer_state.json"),
                   key=lambda p: int(p.parent.name.split("-")[1]))
    if not ckpts:
        return []
    state = json.loads(ckpts[-1].read_text(encoding="utf-8"))
    return [(e["step"], e["eval_ood_cer"] * 100)
            for e in state["log_history"] if "eval_ood_cer" in e]


def bucket_mean(steps, loss, every):
    """Mean loss per block of `every` steps -- this run logged every step, v1/v2
    logged every 25, so decimating here makes the three curves comparable."""
    out = []
    for i in range(0, len(steps), every):
        block = loss[i:i + every]
        out.append((steps[i + len(block) - 1], sum(block) / len(block)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--title", default=None)
    ap.add_argument("--every", type=int, default=25,
                    help="average train loss over blocks of this many steps")
    ap.add_argument("--no-ood", action="store_true",
                    help="drop the OOD CER line (it is the unscaled adapter, so it reads "
                         "as forgetting -- the tradeoff curve tells that part properly)")
    args = ap.parse_args()

    steps, loss, val, val_loss = read_training_csv(args.run_dir / "metrics" / "training.csv")
    ood = [] if args.no_ood else read_ood(args.run_dir)
    steps_per_epoch = val[0][0] if val else max(steps)

    def ep(step):
        return step / steps_per_epoch

    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=110)
    ax.set_xlabel("epoch")
    ax.set_ylabel("error %")
    ax.grid(alpha=0.3)

    ax.plot([ep(s) for s, _ in val], [c for _, c in val],
            "o-", color="#c9435b", label="val CER%")
    if ood:
        ax.plot([ep(s) for s, _ in ood], [c for _, c in ood],
                "s--", color="#e8a33d", label="OOD (VIVOS) CER%")

    ax2 = ax.twinx()
    # log scale: loss spans ~31 -> ~0.06 here, linear would flatten all but step 1
    ax2.set_yscale("log")
    ax2.set_ylabel("train loss (log)", color="#7d8b99")
    blocks = bucket_mean(steps, loss, args.every)
    ax2.plot([ep(s) for s, _ in blocks], [l for _, l in blocks],
             color="#7d8b99", linewidth=1.4, marker=".", markersize=4,
             label=f"train loss (per {args.every} steps)")
    if val_loss:
        # Same axis as train loss (both are loss, unlike the error-% lines on ax) --
        # this is the line that shows overfitting: train loss keeps dropping while
        # val loss turns back up (see docs/training-curves/README.md).
        ax2.plot([ep(s) for s, _ in val_loss], [l for _, l in val_loss],
                 "d--", color="#8e44ad", linewidth=1.4, markersize=5, label="val loss")
    ax2.tick_params(axis="y", colors="#7d8b99")

    best_step, best_cer = min(val, key=lambda t: t[1])
    ax.axvline(ep(best_step), color="#3f8f5f", linestyle=":", linewidth=1.2)
    ax.annotate(f"best val CER {best_cer:.2f}%", (ep(best_step), best_cer),
                color="#3f8f5f", fontsize=9, ha="right", va="bottom",
                xytext=(-6, 6), textcoords="offset points")

    ax.set_title(args.title or f"{args.run_dir.name} · PhoWhisper-large LoRA — training curve")
    ax.legend(loc="center right", fontsize=9)
    ax2.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    print(f"wrote {args.out}  ({len(steps)} loss pts, {len(val)} val, {len(ood)} ood, "
          f"{len(val_loss)} val_loss)")


if __name__ == "__main__":
    main()
