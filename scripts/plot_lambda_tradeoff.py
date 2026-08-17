"""Plot the lambda tradeoff: in-domain gain (val CER) against OOD cost (VIVOS CER).

Companion to plot_training_curve.py. The training curve shows the adapter learning
at its unscaled amplitude; this one shows what choosing lambda costs. Lower-left is
better on both axes.

    python scripts/plot_lambda_tradeoff.py Outputs/v3-r16 --chosen 0.5 \
        --out docs/training-curves/v3_v3-r16_lambda-tradeoff.png

`--style tradeoff` (default): the scatter above. When `lambda_sweep.csv` has the
per-slice columns `val_cer_synthetic`/`val_cer_youtube` (mixed-noisy-v1 run), two
more lines are overlaid using each slice's own val CER as x against the same OOD
CER -- missing columns fall back to the single pooled line, unchanged from before.

`--style by-lambda`: lambda on the x-axis instead, one line each for pooled val,
synthetic val, YouTube val (when present) and OOD, plus the budget line and chosen
lambda -- answers "which slice actually drove lambda* being picked", which the
scatter doesn't show directly.

`--chosen` is not recomputed here -- src.pipeline.select_lambda owns that rule.
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_sweep(path):
    rows = []
    with open(path, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            row = {
                "lambda": float(r["lambda"]),
                "val_cer": float(r["val_cer"]) * 100,
                "ood_cer": float(r["ood_cer"]) * 100,
            }
            if r.get("val_cer_synthetic"):
                row["val_cer_synthetic"] = float(r["val_cer_synthetic"]) * 100
            if r.get("val_cer_youtube"):
                row["val_cer_youtube"] = float(r["val_cer_youtube"]) * 100
            rows.append(row)
    return rows


def _has_slices(rows) -> bool:
    return all("val_cer_synthetic" in r and "val_cer_youtube" in r for r in rows)


def plot_tradeoff(rows, baseline_ood, args):
    fig, ax = plt.subplots(figsize=(9, 5.6), dpi=110)
    ax.plot([r["val_cer"] for r in rows], [r["ood_cer"] for r in rows],
            "o-", color="#e8a33d", markersize=7,
            label="λ sweep (val gộp)" if _has_slices(rows) else "λ sweep")
    if _has_slices(rows):
        ax.plot([r["val_cer_synthetic"] for r in rows], [r["ood_cer"] for r in rows],
                "^--", color="#3f8f5f", markersize=6, label="λ sweep (val synthetic)")
        ax.plot([r["val_cer_youtube"] for r in rows], [r["ood_cer"] for r in rows],
                "v--", color="#c9435b", markersize=6, label="λ sweep (val youtube)")

    # purple, not the #c9435b red of the youtube slice line -- same colour AND same
    # dash style made the budget read as a fourth sweep curve on the mixed run
    ax.axhline(baseline_ood + args.budget_pp, color="#8e44ad", linestyle="--", linewidth=1.2,
               label=f"ngân sách OOD (+{args.budget_pp:g}pp)")
    ax.axhline(baseline_ood, color="#7d8b99", linestyle=":", linewidth=1,
               label=f"OOD của base ({baseline_ood:.2f}%)")

    for r in rows:
        lam, val, ood = r["lambda"], r["val_cer"], r["ood_cer"]
        if lam == 0.0:
            ax.plot(val, ood, "*", color="#1f2328", markersize=16, zorder=5, label="base (λ=0)")
        if args.chosen is not None and lam == args.chosen:
            ax.plot(val, ood, "s", color="#3f8f5f", markersize=11, zorder=4,
                    label=f"λ={lam:g} · chọn dùng")
        ax.annotate(f"λ={lam:g}", (val, ood), fontsize=9,
                    xytext=(7, 7), textcoords="offset points")

    ax.set_xlabel("CER val % (trong miền)  ←  thấp hơn tốt hơn")
    ax.set_ylabel("CER VIVOS % (ngoài miền)  ←  thấp hơn tốt hơn")
    ax.set_title(args.title or
                 f"{args.run_dir.name} · đánh đổi theo λ — góc dưới-trái là lý tưởng")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig


def plot_by_lambda(rows, baseline_ood, args):
    fig, ax = plt.subplots(figsize=(9, 5.6), dpi=110)
    lambdas = [r["lambda"] for r in rows]

    ax.plot(lambdas, [r["val_cer"] for r in rows], "o-", color="#e8a33d",
            label="val CER% (gộp)")
    if _has_slices(rows):
        ax.plot(lambdas, [r["val_cer_synthetic"] for r in rows], "^--", color="#3f8f5f",
                label="val CER% (synthetic)")
        ax.plot(lambdas, [r["val_cer_youtube"] for r in rows], "v--", color="#c9435b",
                label="val CER% (youtube)")
    ax.plot(lambdas, [r["ood_cer"] for r in rows], "s-", color="#1f77b4", label="OOD CER%")

    ax.axhline(baseline_ood + args.budget_pp, color="#8e44ad", linestyle="--", linewidth=1.2,
               label=f"ngân sách OOD (+{args.budget_pp:g}pp)")
    if args.chosen is not None:
        ax.axvline(args.chosen, color="#1f2328", linestyle=":", linewidth=1.2,
                    label=f"λ={args.chosen:g} · chọn dùng")

    ax.set_xlabel("λ")
    ax.set_ylabel("CER %")
    ax.set_title(args.title or f"{args.run_dir.name} · CER theo λ, từng lát")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--chosen", type=float, default=None, help="lambda that was shipped")
    ap.add_argument("--budget-pp", type=float, default=2.0,
                    help="OOD CER budget above baseline, in percentage points")
    ap.add_argument("--title", default=None)
    ap.add_argument("--style", choices=["tradeoff", "by-lambda"], default="tradeoff")
    args = ap.parse_args()

    rows = sorted(read_sweep(args.run_dir / "metrics" / "lambda_sweep.csv"),
                  key=lambda r: r["lambda"])
    baseline_ood = json.loads((args.run_dir / "metrics" / "baseline.json")
                              .read_text(encoding="utf-8"))["cer_ood"] * 100

    fig = (plot_tradeoff if args.style == "tradeoff" else plot_by_lambda)(rows, baseline_ood, args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    print(f"wrote {args.out}  ({len(rows)} lambda points, baseline OOD {baseline_ood:.3f}%, "
          f"style={args.style}, per-slice cols={'yes' if _has_slices(rows) else 'no'})")


if __name__ == "__main__":
    main()
