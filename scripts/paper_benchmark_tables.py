"""Emit the paper's benchmark tables from the scored matrix, not from retyped cells.

Input is the `scores.<variant>.json` pair written by `scripts.benchmark_report`;
this script only lays them out. Two facts about the 2026-09-10 matrix are encoded
here because they change what may appear in a table:

  * `scribe_v2` ran out of ElevenLabs quota mid-suite on `vimedcss-test`
    (609/1614) and entirely on `vivos` (0/760). Those two prediction files are
    kept out of the scored directory, so the cells read `--` rather than a
    number measured on a non-random 38% of the audio.
  * `N0` keeps case and punctuation, and VIVOS references are upper-case while
    every hypothesis is lower-case. The resulting 8x% CER measures the case
    mismatch, not the models, so the VIVOS row is dropped from the `N0` table.

    python scripts/paper_benchmark_tables.py --dir "Outputs/benchmark-2026-09-10 (3)/paper"
"""

import argparse
import csv
import json
import sys
from pathlib import Path

MODELS = ["vinai_phowhisper_large", "scribe_v2", "winhsss_reworkwhisper_large_v5"]
SUITES = ["vimedcss-test", "vimedcss-hard", "cross-domain", "youtube-test",
          "synthetic-test", "vivos"]
DASH = "--"


def load(path: Path) -> dict:
    d = json.loads(path.read_text(encoding="utf-8"))
    return {k: {tuple(kk.split("|")): vv for kk, vv in d[k].items()}
            for k in ("results", "deltas")} | {"variant": d["variant"],
                                               "normalization": d["normalization"]}


def hours(res: dict, suite: str) -> str:
    """Same segments in every column, so any present model gives the suite's hours.
    VIVOS carries no per-segment duration -- report that, do not report 0.00 h."""
    for m in MODELS:
        r = res.get((m, suite))
        if r:
            return f'{r["audio_hours"]:.2f}' if r["audio_hours"] else "n/a"
    return "n/a"


SUMMARY = [  # (label, suite, variant) -- ViMedCSS rows must be N0: that is the scale the
             # paper's own rows sit on, and the only one AG may be placed beside.
    ("ViMedCSS hard",            "vimedcss-hard",  "N0"),
    ("ViMedCSS test",            "vimedcss-test",  "N0"),
    ("YouTube test (họp thật)",  "youtube-test",   "N3"),
    ("Tổng hợp (TTS)",           "synthetic-test", "N3"),
    ("VIVOS (ngoài miền)",       "vivos",          "N3"),
    ("cross-domain-bench",       "cross-domain",   "N3"),
]
SUMMARY_COLS = ["PhoWhisper-large", "scribe v2", "Reworkwhisper-large-v5"]


def emit_tsv(name: str, header: list[str], rows: list[list[str]], out: Path) -> None:
    """Tab- and comma-separated twins of the summary, for the spreadsheet it came from.
    Neither can carry markdown bold, so the winner is named in its own column instead.
    `csv.writer` does the quoting: one legend cell contains a comma."""
    def cell(x: str) -> str:
        """A spreadsheet wants neither the markdown line break in a header nor `--` in a
        numeric column: one text cell makes the whole column non-numeric."""
        return "" if x == DASH else x.replace("<br>", " ")

    for ext, delim in ((".tsv", "\t"), (".csv", ",")):
        with (out / f"{name}{ext}").open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f, delimiter=delim)
            w.writerow([cell(h) for h in header])
            w.writerows([[cell(c) for c in r] for r in rows])
    print(f"wrote {out / name}.tsv + .csv  ({len(rows)} rows)")


def emit(name: str, header: list[str], rows: list[list[str]], title: str,
         notes: list[str], out: Path, text_cols: int = 1,
         csv_too: bool = True) -> None:
    """`text_cols` leading columns stay left-aligned; the numeric rest go right."""
    md = [f"### {title}", ""] + [n for n in notes] + ([""] if notes else [])
    md += ["| " + " | ".join(header) + " |",
           "|" + "|".join("---" if i < text_cols else "---:" for i in range(len(header))) + "|"]
    md += ["| " + " | ".join(r) + " |" for r in rows]
    (out / f"{name}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"wrote {out / name}.md  ({len(rows)} rows)")
    if csv_too:            # the summary writes its own pair, from rows without bold markers
        emit_tsv(name, header, rows, out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True, help="directory holding scores.N3.json and scores.N0.json")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    d = Path(args.dir)
    n3, n0 = load(d / "scores.N3.json"), load(d / "scores.N0.json")

    def matrix(sc: dict, suites: list[str]) -> list[list[str]]:
        rows = []
        for s in suites:
            cells = []
            for m in MODELS:
                r = sc["results"].get((m, s))
                cells += [DASH, DASH] if r is None else [f'{r["cer"] * 100:.2f}',
                                                         f'{r["wer"] * 100:.2f}']
            n = next((sc["results"][(m, s)]["n_segments"] for m in MODELS
                      if (m, s) in sc["results"]), 0)
            rows.append([s, str(n), hours(sc["results"], s)] + cells)
        return rows

    head = ["suite", "n", "audio h"]
    for m in MODELS:
        head += [f"{m}<br>CER %", f"{m}<br>WER %"]
    emit("table1-main-N3", head, matrix(n3, SUITES),
         "Table 1 -- corpus-level CER / WER, normalization `N3`",
         ["Read **down** a column (same audio, different system). Never across rows: "
          "the six suites differ in difficulty.",
          "",
          f"`N3` = {json.dumps(n3['normalization'], ensure_ascii=False)}.",
          "",
          f"`{DASH}` = not measured. `scribe_v2` exhausted its ElevenLabs quota part-way "
          "through `vimedcss-test` (609 of 1614 segments) and before any of `vivos`, and "
          "was never run on `synthetic-test`; a CER over that truncated, non-random subset "
          "is not comparable to the full-suite columns and is therefore omitted.",
          "",
          "`vivos` reports no audio hours: its prediction rows carry no per-segment duration.",
          "",
          "`youtube-test` is scored on the 214 segments shared by all three systems "
          "(14 of 228 failed on quota); `vimedcss-hard` and `vimedcss-test` are disjoint splits."],
         d)


    # ---- Table 2: the ViMedCSS paper's own Table 4, beside our rescoring of the same
    # split. `N0` is mandatory here: it is the variant that reproduced the paper's
    # PhoWhisper-Large row, and the paper never states its normalization.
    t4 = json.loads((d / "vimedcss-table4.json").read_text(encoding="utf-8"))
    rows = [[r["system"], "ViMedCSS Table 4, zero-shot", DASH,
             f'{r["cer"]:.2f}', f'{r["wer"]:.2f}'] for r in t4["rows"]]
    base = n0["results"][("vinai_phowhisper_large", "vimedcss-test")]
    for m in MODELS:
        r = n0["results"].get((m, "vimedcss-test"))
        if r is None:
            rows.append([m, "not measured (quota)", DASH, DASH, DASH])
            continue
        rows.append([m, "measured here, `N0`", str(r["n_segments"]),
                     f'{r["cer"] * 100:.2f}', f'{r["wer"] * 100:.2f}'])
    paper = next(r for r in t4["rows"] if r["system"] == t4["reproduced_row"])
    dc, dw = base["cer"] * 100 - paper["cer"], base["wer"] * 100 - paper["wer"]
    tol = t4["tolerance_pp"]
    g1 = "within" if abs(dc) <= tol["cer"] and abs(dw) <= tol["wer"] else "OUTSIDE"
    best = min((r["cer"] for r in t4["rows"]))
    ours = n0["results"][("winhsss_reworkwhisper_large_v5", "vimedcss-test")]["cer"] * 100
    emit("table2-vimedcss-vs-paper-N0",
         ["system", "source", "n", "CER %", "WER %"], rows,
         "Table 2 -- ViMedCSS `test` split against the paper's Table 4, normalization `N0`",
         [t4["_source"] + " Those three rows are the paper's own figures; the rows below "
          "them are this project's decoding of the same split.",
          "",
          f"Reproduction check: our `vinai_phowhisper_large` row lands {dc:+.2f} pp CER and "
          f"{dw:+.2f} pp WER from the paper's `{t4['reproduced_row']}` row -- {g1} the "
          f"+/-{tol['cer']} pp / +/-{tol['wer']} pp tolerance, so the two halves of this "
          "table may be read against each other. `N3` numbers may not: see Table 1.",
          "",
          f"`winhsss_reworkwhisper_large_v5` at {ours:.2f} CER is below the lowest CER in "
          f"Table 4 ({best:.2f}, `{min(t4['rows'], key=lambda r: r['cer'])['system']}`).",
          "",
          "n differs from the paper's split size: this run decoded 1614 of 1615 segments. "
          "Two of them (`Med_CS-4072_1-13`, `Med_CS-4072_1-18`) have audio that an earlier "
          "standalone run recorded as undecodable (`Outputs/vimedcss_output/"
          "decode_failures.json`, 'decoded 0.0s vs manifest 4.0s'); both models hallucinate "
          "unrelated text on them. Excluding the pair, that run scored 1612 segments at "
          "18.46 CER / 31.60 WER (base) and 15.66 / 25.65 (v5) -- 0.14 pp and 0.06 pp of CER "
          "away from the rows above, so the pair does not carry the comparison.",
          "",
          "The `hard` split is absent from this table by design: its Table 4 rows have never "
          "been extracted from the PDF. For `hard`, read Table 1, which has all three systems."],
         d, text_cols=2)

    # ---- Summary table in the layout of Outputs/bang-tong-hop/benchmark-v5-2026-09-09.tsv:
    # one row per suite, CER only, plus the paper's best fine-tuned system as the outside
    # baseline. `chuẩn hoá` is the column that spreadsheet lacked -- without it the ViMedCSS
    # rows (N0) and the rest (N3) look like one scale, and v5's ViMedCSS-test CER differs by
    # 1.5 pp between the two, enough to flip the comparison against AG.
    ag = t4["finetuned_best"]
    # scribe v2 is a closed commercial API, not a Vietnamese open model. It stays in the
    # table as a reference point but out of the ranking, so the winner columns answer
    # "best Vietnamese model", the category this work competes in. AG is IN that category:
    # it is a fine-tune in the ViMedCSS paper, so it still ranks.
    CER_COLS = ["CER % — PhoWhisper-large", "CER % — Baseline ngoài",
                "CER % — scribe v2 (thương mại, tham khảo)",
                "CER % — Reworkwhisper-large-v5"]
    RET_COLS = ["Giữ từ % — PhoWhisper-large",
                "Giữ từ % — scribe v2 (thương mại, tham khảo)",
                "Giữ từ % — Reworkwhisper-large-v5"]
    REFERENCE = {CER_COLS[2], RET_COLS[1]}
    head = (["Bộ đo", "n", "chuẩn hoá"] + CER_COLS[:2] + ["Baseline ngoài là ai"]
            + CER_COLS[2:] + ["CER tốt nhất (mô hình Việt)", "Ứng viên từ vay mượn"]
            + RET_COLS + ["Giữ từ tốt nhất (mô hình Việt)"])
    rows_md, rows_flat = [], []
    for label, suite, variant in SUMMARY:
        sc = {"N0": n0, "N3": n3}[variant]
        cer, ret, cand = {}, {}, None
        for cer_col, ret_col, m in zip([CER_COLS[0]] + CER_COLS[2:], RET_COLS, MODELS):
            r = sc["results"].get((m, suite))
            cer[cer_col] = None if r is None else r["cer"] * 100
            ret[ret_col] = None if r is None or r["retention"] is None else r["retention"] * 100
            if r is not None and r["retention"] is not None:
                cand = r["n_candidates"]
        outside = ag["cer"].get(suite)
        if outside is not None:
            cer[CER_COLS[1]] = outside          # AG reports CER only, never a retention
        n = next((sc["results"][(m, suite)]["n_segments"] for m in MODELS
                  if (m, suite) in sc["results"]), 0)
        # CER: lower is better. Retention: higher is better. Two directions, two winners.
        best_cer = min((k for k, v in cer.items()
                        if v is not None and k not in REFERENCE), key=lambda k: cer[k])
        best_ret = max((k for k, v in ret.items()
                        if v is not None and k not in REFERENCE),
                       key=lambda k: ret[k], default=None)
        cells = {k: ("" if v is None else f"{v:.2f}")
                 for k, v in list(cer.items()) + list(ret.items())}
        win_cer = ag["system"] if best_cer == CER_COLS[1] else best_cer.split("— ")[1]
        win_ret = "" if best_ret is None else best_ret.split("— ")[1]
        flat = [label, str(n), variant]
        for c in CER_COLS[:2]:
            flat.append(cells.get(c, ""))
        flat.append(ag["label"] if outside is not None else "")
        flat += [cells.get(c, "") for c in CER_COLS[2:]]
        flat += [win_cer, "" if cand is None else str(cand)]
        flat += [cells.get(c, "") for c in RET_COLS]
        flat.append(win_ret)
        rows_flat.append(flat)
        marks = {cells[best_cer]} | ({cells[best_ret]} if best_ret else set())
        ref_cells = {cells[c] for c in REFERENCE if cells.get(c)}
        # `--` means measurable but not measured. VIVOS has no loanword candidate at all,
        # so its whole retention block is blank rather than dashed.
        numeric = {3, 4, 6, 7} | ({10, 11, 12} if cand is not None else set())
        rows_md.append([f"**{c}**" if c and c in marks and c not in ref_cells
                        and i in numeric
                        else (c or (DASH if i in numeric else ""))
                        for i, c in enumerate(flat)])
    legend = [[""] * len(head),
              ["* in đậm = tốt nhất trong hàng", "", "CER: thấp hơn tốt hơn",
               "Giữ từ vay mượn: cao hơn tốt hơn",
               "Ô trống = chưa đo, không phải 0"] + [""] * (len(head) - 5)]
    emit_tsv("bang-tong-hop-2026-09-11", head, rows_flat + legend, d)
    emit("bang-tong-hop-2026-09-11", head, rows_md,
         "Bảng tổng hợp — CER % và giữ từ vay mượn %, một hàng một bộ đo",
         ["Hai khối cột, **hai chiều tốt ngược nhau**: CER thấp hơn là tốt hơn, giữ từ "
          "vay mượn cao hơn là tốt hơn. In đậm = tốt nhất trong hàng của khối đó. "
          "`--` = chưa đo, không phải 0.",
          "",
          "**`scribe v2` không tham gia xếp hạng.** Đây là API thương mại đóng của "
          "ElevenLabs, không phải mô hình tiếng Việt mở, nên nó nằm trong bảng làm mốc "
          "tham khảo còn hai cột `tốt nhất` chỉ xét nhóm mô hình Việt. **`AG` thì có tham "
          "gia**: đó là một bản fine-tune trong bài ViMedCSS, cùng nhóm với công trình này. "
          "Vì vậy hàng `ViMedCSS test` do AG dẫn (14.73 so với 15.72), không phải v5.",
          "",
          f"**Baseline ngoài** = {ag['label']} — chỉ có CER, không có giữ từ vay mượn. "
          "Chỉ có cho hai split ViMedCSS. Bài báo không nói nó chuẩn hoá thế nào, nên hai "
          "hàng ViMedCSS ở đây dùng `N0`, biến thể đã dựng lại được dòng PhoWhisper-Large "
          "của bài báo trong sai số ±1.0 pp CER (xem Bảng 2). Bốn hàng còn lại dùng `N3`, "
          "thang của mọi số khác trong repo. **Không so ngang hai nhóm chuẩn hoá.**",
          "",
          "**Giữ từ vay mượn** = tỉ lệ token không mang hình thái âm tiết tiếng Việt trong "
          "câu tham chiếu còn sống sót trong câu máy đọc ra. `Ứng viên từ vay mượn` là số "
          "token như vậy trong tham chiếu, đếm sau khi chuẩn hoá — nên nó đổi theo biến thể, "
          "và không so được giữa hàng `N0` và hàng `N3`. `VIVOS` không có token nào thuộc "
          "loại này, nên cả khối để trống.",
          "",
          "`ViMedCSS test` ở đây là 1614 đoạn giải mã lại ngày 2026-09-10, không phải 1612 "
          "như bảng cũ: hai đoạn `Med_CS-4072_1-13` và `Med_CS-4072_1-18` có audio hỏng, "
          "lần chạy cũ loại chúng. Giữ hay bỏ cặp đó thì AG vẫn thắng hàng này "
          "(18.46 / 15.66 nếu bỏ). Xem chú thích Bảng 2.",
          "",
          "`YouTube test` đo trên 214 đoạn chung cho cả ba hệ; 14 trong 228 đoạn hỏng vì "
          "`scribe_v2` hết quota. Bỏ Scribe ra và chấm đủ 228 đoạn thì cột CER của hàng này "
          "là 15.93 → 5.93, đúng như bảng ngày 2026-09-09.",
          "",
          "Hai ô CER `cross-domain-bench` của bảng 2026-09-09 (Scribe 6.54, v5 8.44) không "
          "dựng lại được từ đường chấm điểm nào trong repo này; số ở đây là `N3` từ "
          "`scripts.benchmark_report`, cùng một lần chấm với mọi ô khác."],
         d, text_cols=3, csv_too=False)

    rows = [[s, m, f'{v["delta_cer_pp"]:+.2f}',
             f'[{v["ci_pp"][0]:+.2f}, {v["ci_pp"][1]:+.2f}]', v["verdict"]]
            for s in SUITES for m in MODELS
            for k, v in [((m, s), n3["deltas"].get((m, s)))] if v]
    emit("table3-paired-bootstrap-N3",
         ["suite", "model", "dCER pp", "95% CI", "verdict"], rows,
         "Table 3 -- paired bootstrap vs `vinai_phowhisper_large`, normalization `N3`",
         ["Positive `dCER` = the listed system has the lower CER. Paired over the same "
          "segments; the verdict is the sign the whole 95% interval agrees on."], d,
         text_cols=2)

    rows = []
    for s in SUITES:
        cells, n_cand = [], DASH
        for m in MODELS:
            r = n3["results"].get((m, s))
            if r is None or r["retention"] is None:
                cells.append(DASH)
                continue
            cells.append(f'{r["retention"] * 100:.2f}')
            n_cand = str(r["n_candidates"])
        rows.append([s, n_cand] + cells)
    emit("table4-loanword-retention-N3",
         ["suite", "candidates"] + [f"{m}<br>retention %" for m in MODELS], rows,
         "Table 4 -- loanword retention, normalization `N3`",
         ["Share of the reference's non-Vietnamese-shaped tokens that survive in the "
          "hypothesis. `candidates` is the reference token count under `N3`; it shifts "
          "with the variant, so never place an `N3` and an `N0` retention figure in one column.",
          "", f"`{DASH}` = the slice contains no loanword candidate (`vivos`) or was not measured."], d)

    emit("tableA-main-N0", head, matrix(n0, [s for s in SUITES if s != "vivos"]),
         "Table A1 -- corpus-level CER / WER, normalization `N0` (appendix)",
         ["`N0` keeps case, punctuation and numbers as written -- the variant that "
          "reproduced the ViMedCSS paper's base row on 2026-09-09. Use it only to compare "
          "against that paper; every other number in this work uses `N3`.",
          "",
          "The `vivos` row is omitted deliberately: its references are upper-case and all "
          "hypotheses are lower-case, so an `N0` score over it (8x% CER for both systems) "
          "measures the case mismatch rather than the systems. See Table 1 for VIVOS."], d)
    print("verdict column, N3: "
          + ", ".join(f"{s}/{m.split('_')[0]}={v['verdict']}"
                      for (m, s), v in n3["deltas"].items()))


if __name__ == "__main__":
    main()
