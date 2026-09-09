"""Emit the paper's result tables straight from the stored predictions.

Every number in the paper traces to a predictions CSV, not to a retyped figure --
docs/finetune-results-report-v4.md carried three stale lambda=0.25 values for weeks
because they were retyped. Run this instead of copying cells by hand.

    python scripts/paper_tables.py            # all tables
    python scripts/paper_tables.py --per-meeting
"""
import argparse
import csv
import sys
from pathlib import Path

from src.metrics import score, english_token_retention

sys.stdout.reconfigure(encoding="utf-8")  # Windows console defaults to cp1252

YOUTUBE = ("3nuCdzuyqng", "7B24A9GfHAo")
MODELS = {
    "Model nền": "Outputs/v4-mixed-r16/audit/predictions_baseline_test.csv",
    "Đang chạy (v3@0,5)": "Outputs/lambda075-metrics/v3-r16-lambda0.5/predictions_tier1_in_domain.csv",
    "v4@λ=0,5": "Outputs/v4-mixed-r16-lambda0.5/predictions_tier1_in_domain.csv",
    "v4@λ=0,75": "Outputs/lambda075-metrics/v4-mixed-r16-lambda0.75/audit/predictions_tier1_in_domain.csv",
}
VIVOS = {
    "Model nền": "Outputs/v4-mixed-r16/audit/predictions_baseline_ood.csv",
    "v4@λ=0,25": "Outputs/v4-mixed-r16/audit/predictions_tier2_ood.csv",
    "v4@λ=0,75": "Outputs/lambda075-metrics/v4-mixed-r16-lambda0.75/audit/predictions_tier2_ood.csv",
}


def load(path):
    with open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def cell(rows):
    """CER, WER, retention -- retention is None on a slice with no loanwords at all,
    which is the case for VIVOS."""
    s = score([r["ref"] for r in rows], [r["hyp"] for r in rows])
    t = english_token_retention([r["ref"] for r in rows], [r["hyp"] for r in rows])
    ret = None if t["retention"] is None else t["retention"] * 100
    return s["cer"] * 100, s["wer"] * 100, ret, t["n_retained"], t["n_candidates"]


def md_row(label, cells):
    return "| " + label + " | " + " | ".join(cells) + " |"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-meeting", action="store_true", help="only the per-meeting table")
    args = ap.parse_args()

    data = {k: load(v) for k, v in MODELS.items()}
    names = list(MODELS)
    head = md_row("", names) + "\n|" + "---|" * (len(names) + 1)

    if not args.per_meeting:
        print("### Kết quả chính (654 đoạn test)\n")
        print(head)
        slices = {
            "Họp thật": lambda rs: [r for r in rs if r["meeting_id"] in YOUTUBE],
            "Tổng hợp": lambda rs: [r for r in rs if r["meeting_id"] not in YOUTUBE],
            "Gộp 654": lambda rs: rs,
        }
        for metric, idx in (("CER", 0), ("WER", 1), ("Giữ từ mượn", 2)):
            for sl, pick in slices.items():
                vals = [cell(pick(data[n])) for n in names]
                print(md_row(f"{sl} · {metric}", [f"{v[idx]:.3f}%" for v in vals]))
        print("\n### VIVOS (760 câu, tập test chuẩn)\n")
        print(md_row("", list(VIVOS)) + "\n|" + "---|" * (len(VIVOS) + 1))
        vv = [cell(load(p)) for p in VIVOS.values()]
        print(md_row("CER", [f"{v[0]:.2f}%" for v in vv]))
        print(md_row("WER", [f"{v[1]:.2f}%" for v in vv]))
        print(md_row("Số token tiếng Anh", [str(v[4]) for v in vv]))
        print()

    print("### Theo từng buổi họp giữ riêng\n")
    print(md_row("Buổi họp · n", names) + "\n|" + "---|" * (len(names) + 1))
    for m in sorted({r["meeting_id"] for r in data[names[0]]}):
        groups = [[r for r in data[n] if r["meeting_id"] == m] for n in names]
        cells = [cell(g) for g in groups]
        print(md_row(f"`{m}` · {len(groups[0])}",
                     [f"{c[0]:.3f}% / {c[2]:.1f}%" for c in cells]))
    print("\nMỗi ô: CER / giữ từ mượn.")


if __name__ == "__main__":
    main()
