"""Figure 1 for the paper: what lambda buys in-domain and what it costs on VIVOS.

Companion to plot_lambda_tradeoff.py, which plots val CER against OOD CER. This one
puts lambda on the x-axis and overlays the *test* slices on the *val* slices, because
the two disagree past 0.75 -- val keeps improving, test was never measured at 1.0.
Showing both is the point of the figure; a val-only or test-only plot hides that.

    python scripts/plot_lambda_curve_paper.py --out docs/training-curves/paper-fig1-lambda.png

Test numbers live in four different run dirs (each lambda was gated separately), so
they are read from those artifacts rather than retyped. lambda=1.0 has no test
artifact on disk -- the test lines stop at 0.75 and the gap is left visible.
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SWEEP = "Outputs/v4-mixed-r16/metrics/lambda_sweep.csv"
GATE_025 = "Outputs/v4-mixed-r16/metrics/gate_results.json"
SUMMARY_050 = "Outputs/v4-mixed-r16-lambda0.5/summary.json"
GATE_075 = "Outputs/lambda075-metrics/v4-mixed-r16-lambda0.75/metrics/gate_results.json"


def read_val(path):
    rows = {}
    with open(path, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            rows[float(r["lambda"])] = {
                "youtube": float(r["val_cer_youtube"]) * 100,
                "synthetic": float(r["val_cer_synthetic"]) * 100,
                "ood": float(r["ood_cer"]) * 100,
            }
    return rows


def read_test():
    """lambda -> {youtube, synthetic}, in percent. lambda=0 is the baseline the
    lambda=0.25 gate already recorded, so it needs no separate file."""
    g025 = json.loads(Path(GATE_025).read_text(encoding="utf-8"))["tier1_in_domain"]["by_source"]
    g075 = json.loads(Path(GATE_075).read_text(encoding="utf-8"))["tier1_in_domain"]["by_source"]
    s050 = json.loads(Path(SUMMARY_050).read_text(encoding="utf-8"))["tier1_by_source"]
    pick = lambda d, k: {s: d[s][k] * 100 for s in ("youtube", "synthetic")}
    return {
        0.0: pick(g025, "cer_baseline"),
        0.25: pick(g025, "cer"),
        0.5: pick(s050, "cer"),
        0.75: pick(g075, "cer"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/training-curves/paper-fig1-lambda.png")
    ap.add_argument("--chosen", type=float, default=0.75)
    args = ap.parse_args()

    val, test = read_val(SWEEP), read_test()
    vl, tl = sorted(val), sorted(test)

    fig, ax = plt.subplots(figsize=(8.2, 5.2), dpi=150)
    ax.plot(tl, [test[x]["youtube"] for x in tl], "o-", color="#c0392b",
            markersize=6, linewidth=2, label="Họp thật — test")
    ax.plot(vl, [val[x]["youtube"] for x in vl], "o--", color="#c0392b",
            markersize=4, linewidth=1.2, alpha=.55, label="Họp thật — val")
    ax.plot(tl, [test[x]["synthetic"] for x in tl], "s-", color="#2e6da4",
            markersize=6, linewidth=2, label="Tổng hợp — test")
    ax.plot(vl, [val[x]["synthetic"] for x in vl], "s--", color="#2e6da4",
            markersize=4, linewidth=1.2, alpha=.55, label="Tổng hợp — val")
    ax.plot(vl, [val[x]["ood"] for x in vl], "^-", color="#e8a33d",
            markersize=6, linewidth=2, label="VIVOS (giọng đọc, ngoài miền)")

    ax.axvline(args.chosen, color="#555", linestyle=":", linewidth=1.2)
    ax.annotate(f"λ = {args.chosen}", xy=(args.chosen, 13), fontsize=9,
                color="#555", ha="right", rotation=90, va="top")
    ax.annotate("test chưa đo ở λ=1,0", xy=(0.885, 5.0), fontsize=8,
                color="#777", ha="center", style="italic")

    ax.set_yscale("log")
    ax.set_yticks([1.5, 2, 3, 5, 8, 12, 16])
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("λ — hệ số hợp nhất adapter")
    ax.set_ylabel("CER (%, thang log)")
    ax.set_xticks(vl)
    ax.grid(alpha=.25, which="both")
    ax.legend(fontsize=8.5, framealpha=.95)
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
