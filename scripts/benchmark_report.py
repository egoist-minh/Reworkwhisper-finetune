"""Score a benchmark matrix directory into a CER/WER table. CPU only, no re-decode.

Reads every `<model>.<suite>.persegment.jsonl` written by
`scripts/benchmark_run.py`, normalizes both sides with `src.normalize.Normalizer`
and scores with `src.metrics` -- the same scale as every gate number in this repo.

Two properties the table depends on, both enforced here rather than assumed:

  * **Same segments in every column.** A suite is scored on the intersection of
    the segment_ids present for all models. A row that one model failed on and
    another decoded would otherwise make the two columns measure different sets;
    the count of dropped rows is printed and stored.
  * **Same normalization in every column.** `--variant` picks one for the whole
    table. The ViMedCSS paper comparison uses `N0` (that is the variant which
    reproduced Table 4's base row on 2026-09-09, `Outputs/vimedcss_output/scores.json`);
    every other number in this project uses `N3`. They are not interchangeable --
    run the script twice rather than mixing variants inside one table.

Never compare across suites: five benchmark sets, five difficulties. Read down a
column, not across a row.

    python -m scripts.benchmark_report --dir Outputs/benchmark-2026-09-10 \\
        --baseline vinai/PhoWhisper-large
"""

import argparse
import json
import re
import sys
from pathlib import Path

from src.metrics import (bootstrap_ci, bootstrap_delta_ci, char_counts,
                         english_token_retention, rate, score, verdict)
from src.normalize import Normalizer

FILLERS = ["ừm", "ờm", "ehm", "uhm", "hmm"]   # ạ/à/ừ/ơ/dạ/vâng are lexical, never here

VARIANTS = {
    "N0": dict(strip_punctuation=False, lowercase=False, number_convention="as_written",    filler_tokens=[]),
    "N1": dict(strip_punctuation=True,  lowercase=True,  number_convention="as_written",    filler_tokens=[]),
    "N2": dict(strip_punctuation=True,  lowercase=True,  number_convention="word_to_digit", filler_tokens=[]),
    "N3": dict(strip_punctuation=True,  lowercase=True,  number_convention="word_to_digit", filler_tokens=FILLERS),
}

NAME = re.compile(r"^(?P<model>.+)\.(?P<suite>[^.]+)\.persegment\.jsonl$")


def collect(directory: Path) -> dict[tuple[str, str], list[dict]]:
    """{(model_slug, suite): rows}. The model key is the slug the run wrote, not
    the original repo id -- the file name is the only record of it."""
    out = {}
    for path in sorted(directory.glob("*.persegment.jsonl")):
        m = NAME.match(path.name)
        if not m:
            raise ValueError(f"{path.name} does not match <model>.<suite>.persegment.jsonl")
        with path.open(encoding="utf-8") as f:
            rows = [json.loads(x) for x in f if x.strip()]
        if not rows:
            raise ValueError(f"{path} is empty")
        out[(m["model"], m["suite"])] = rows
    if not out:
        raise FileNotFoundError(f"no *.persegment.jsonl under {directory}")
    return out


def normalize_pair(rows: list[dict], variant: str) -> tuple[list[str], list[str]]:
    """A fresh Normalizer per side: `stats` is mutable, so one shared instance
    would mix the two sides' conversion counters."""
    kw = VARIANTS[variant]
    nr, nh = Normalizer(**kw), Normalizer(**kw)
    return [nr(r["ref"]) for r in rows], [nh(r["hyp"]) for r in rows]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True, help="directory written by scripts.benchmark_run")
    ap.add_argument("--variant", default="N3", choices=sorted(VARIANTS),
                    help="normalization for the whole table (N3 = this repo's default)")
    ap.add_argument("--baseline", default=None,
                    help="model slug for the paired-bootstrap delta column, e.g. "
                         "vinai_phowhisper_large. Omit to skip the delta table.")
    ap.add_argument("--out", default=None, help="write scores.json here (default: <dir>/scores.json)")
    args = ap.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")
    directory = Path(args.dir)
    data = collect(directory)
    models = sorted({m for m, _ in data})
    suites = sorted({s for _, s in data})

    if args.baseline and args.baseline not in models:
        raise KeyError(f"--baseline {args.baseline!r} not among {models}")

    results, counts = {}, {}
    dropped = {}
    for suite in suites:
        present = {m: {r["segment_id"]: r for r in data[(m, suite)]}
                   for m in models if (m, suite) in data}
        common = sorted(set.intersection(*(set(v) for v in present.values())))
        if not common:
            raise RuntimeError(f"suite {suite}: no segment_id is present for all of {list(present)}")
        dropped[suite] = {m: sorted(set(present[m]) - set(common)) for m in present}
        n_drop = sum(len(v) for v in dropped[suite].values())
        print(f"{suite}: {len(common)} shared segments across {len(present)} model(s)"
              + (f" | {n_drop} row(s) not shared, excluded" if n_drop else ""))

        for model in present:
            rows = [present[model][sid] for sid in common]      # same order in every column
            refs, hyps = normalize_pair(rows, args.variant)
            s = score(refs, hyps)
            ret = english_token_retention(refs, hyps)
            cc = [char_counts(r, h) for r, h in zip(refs, hyps)]
            lo, hi = bootstrap_ci(cc)
            results[(model, suite)] = {
                "n_segments": s["n_segments"], "cer": s["cer"], "wer": s["wer"],
                "cer_ci": [lo, hi], "retention": ret["retention"],
                "n_candidates": ret["n_candidates"],
                # `duration_seconds` is what the 2026-09-09 ViMedCSS notebook wrote; those
                # files drop straight into a matrix directory and must not report 0 h.
                "audio_hours": sum(r.get("duration", r.get("duration_seconds", 0.0))
                                   for r in rows) / 3600,
                "missing_loanwords": dict(list(ret["missing"].items())[:15]),
            }
            counts[(model, suite)] = cc

    lines = [f"### Benchmark matrix -- normalization `{args.variant}`, corpus-level CER/WER", "",
             "Compare **down** a column (same audio, different model). Never across "
             "columns: the suites differ in difficulty.", ""]
    header = "| model | " + " | ".join(f"{s}<br>CER %" for s in suites) + " |"
    lines += [header, "|---|" + "---:|" * len(suites)]
    for model in models:
        cells = []
        for suite in suites:
            r = results.get((model, suite))
            cells.append("—" if r is None else f'{r["cer"] * 100:.2f}')
        lines.append(f"| `{model}` | " + " | ".join(cells) + " |")

    lines += ["", "#### Full detail", "",
              "| suite | model | n | audio h | CER % | CER 95% CI | WER % | loanword retention |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    for suite in suites:
        for model in models:
            r = results.get((model, suite))
            if r is None:
                continue
            ret = "—" if r["retention"] is None else f'{r["retention"] * 100:.2f}% (n={r["n_candidates"]})'
            lines.append(
                f'| {suite} | `{model}` | {r["n_segments"]} | {r["audio_hours"]:.2f} | '
                f'{r["cer"] * 100:.2f} | [{r["cer_ci"][0] * 100:.2f}, {r["cer_ci"][1] * 100:.2f}] | '
                f'{r["wer"] * 100:.2f} | {ret} |')

    deltas = {}
    if args.baseline:
        lines += ["", f"#### Paired bootstrap vs `{args.baseline}` "
                  "(positive dCER = the other model is better)", "",
                  "| suite | model | dCER pp | 95% CI | verdict |", "|---|---|---:|---:|---|"]
        for suite in suites:
            base = counts.get((args.baseline, suite))
            if base is None:
                continue
            for model in models:
                cand = counts.get((model, suite))
                if cand is None or model == args.baseline:
                    continue
                lo, hi = bootstrap_delta_ci(base, cand)
                d = rate(base) - rate(cand)
                v = verdict(lo, hi)
                deltas[f"{model}|{suite}"] = {"delta_cer_pp": d * 100,
                                              "ci_pp": [lo * 100, hi * 100], "verdict": v}
                lines.append(f"| {suite} | `{model}` | {d * 100:+.2f} | "
                             f"[{lo * 100:+.2f}, {hi * 100:+.2f}] | {v} |")

    table = "\n".join(lines)
    print()
    print(table)

    table_path = directory / f"benchmark-table.{args.variant}.md"
    table_path.write_text(table + "\n", encoding="utf-8")
    out_path = Path(args.out) if args.out else directory / "scores.json"
    out_path.write_text(json.dumps({
        "variant": args.variant,
        "normalization": VARIANTS[args.variant],
        "baseline": args.baseline,
        "results": {f"{m}|{s}": v for (m, s), v in results.items()},
        "deltas": deltas,
        "excluded_not_shared": dropped,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {table_path}\nwrote {out_path}")


if __name__ == "__main__":
    main()
