"""Plot the lambda tradeoff: in-domain gain (val CER) against OOD cost (VIVOS CER).

Companion to plot_training_curve.py. The training curve shows the adapter learning
at its unscaled amplitude; this one shows what choosing lambda costs. Lower-left is
better on both axes.

    python scripts/plot_lambda_tradeoff.py Outputs/v3-r16 --chosen 0.5 \
        --out docs/training-curves/v3_v3-r16_lambda-tradeoff.png

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
    with open(path, encoding="utf-8") as fh:
        return [(float(r["lambda"]), float(r["val_cer"]) * 100, float(r["ood_cer"]) * 100)
                for r in csv.DictReader(fh)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--chosen", type=float, default=None, help="lambda that was shipped")
    ap.add_argument("--budget-pp", type=float, default=2.0,
                    help="OOD CER budget above baseline, in percentage points")
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    rows = sorted(read_sweep(args.run_dir / "metrics" / "lambda_sweep.csv"))
    baseline_ood = json.loads((args.run_dir / "metrics" / "baseline.json")
                              .read_text(encoding="utf-8"))["cer_ood"] * 100

    fig, ax = plt.subplots(figsize=(9, 5.6), dpi=110)
    ax.plot([v for _, v, _ in rows], [o for _, _, o in rows],
            "o-", color="#e8a33d", markersize=7, label="λ sweep")
    ax.axhline(baseline_ood + args.budget_pp, color="#c9435b", linestyle="--", linewidth=1.2,
               label=f"ngân sách OOD (+{args.budget_pp:g}pp)")
    ax.axhline(baseline_ood, color="#7d8b99", linestyle=":", linewidth=1,
               label=f"OOD của base ({baseline_ood:.2f}%)")

    for lam, val, ood in rows:
        marker_at_base = lam == 0.0
        if marker_at_base:
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
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    print(f"wrote {args.out}  ({len(rows)} lambda points, baseline OOD {baseline_ood:.3f}%)")


if __name__ == "__main__":
    main()
