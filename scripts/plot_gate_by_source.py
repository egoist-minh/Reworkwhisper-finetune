"""Plot tier1_in_domain's per-source (synthetic/youtube) CER: baseline vs candidate,
error bars from bootstrap CI, `delta_ci` + `verdict` annotated on each pair. Reads
ONLY `metrics/gate_results.json` -- `src/gate.py`'s `by_source` already computed
every number this needs (module docstring "by_source"); no CER is recomputed here.

228 YouTube test segments give a CI of roughly +/-1.5-2pp (youtube-data-pilot/
README.md step 8) -- a bare two-bar comparison would look like a clear win even
when `verdict` is INCONCLUSIVE, hence the error bars and the printed verdict on
every pair instead of just the two CER numbers.

    python scripts/plot_gate_by_source.py Outputs/v4-mixed-r16 \
        --out docs/training-curves/v4-mixed-r16_gate-by-source.png

Exits cleanly with no file written if `tier1_in_domain.by_source` is missing --
v3-r16 and earlier runs predate this field.
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    gate_path = args.run_dir / "metrics" / "gate_results.json"
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    by_source = gate.get("tier1_in_domain", {}).get("by_source")
    if not by_source:
        print(f"{gate_path}: tier1_in_domain.by_source missing -- this run predates "
              "the by_source breakdown, nothing to plot")
        sys.exit(0)

    sources = sorted(by_source)
    n = len(sources)
    width = 0.32
    fig, ax = plt.subplots(figsize=(2.6 * n + 3, 5.6), dpi=110)

    for i, src in enumerate(sources):
        entry = by_source[src]
        cer = entry["cer"] * 100
        lo, hi = entry["ci"]
        cand_err = [[cer - lo * 100], [hi * 100 - cer]]
        ax.bar(i + width / 2, cer, width, color="#e8a33d",
                label="candidate" if i == 0 else None)
        ax.errorbar(i + width / 2, cer, yerr=cand_err, fmt="none", ecolor="#1f2328", capsize=4)

        top = cer
        if "cer_baseline" in entry:
            b_cer = entry["cer_baseline"] * 100
            b_lo, b_hi = entry["ci_baseline"]
            base_err = [[b_cer - b_lo * 100], [b_hi * 100 - b_cer]]
            ax.bar(i - width / 2, b_cer, width, color="#7d8b99",
                    label="baseline" if i == 0 else None)
            ax.errorbar(i - width / 2, b_cer, yerr=base_err, fmt="none", ecolor="#1f2328", capsize=4)
            top = max(top, b_cer)

        verdict = entry.get("verdict", "")
        delta_ci = entry.get("delta_ci")
        annotation = f"Δ=[{delta_ci[0] * 100:.2f}, {delta_ci[1] * 100:.2f}]pp\n{verdict}" \
            if delta_ci else verdict
        ax.annotate(annotation, (i, top), xytext=(0, 10), textcoords="offset points",
                    ha="center", fontsize=8)

    ax.set_xticks(range(n))
    ax.set_xticklabels([f"{src}\n(n={by_source[src]['n_segments']})" for src in sources])
    ax.set_ylabel("CER %")
    ax.set_title(args.title or f"{args.run_dir.name} · tier1_in_domain CER theo source — "
                                "baseline vs candidate")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=9)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    print(f"wrote {args.out}  ({n} sources: {sources})")


if __name__ == "__main__":
    main()
