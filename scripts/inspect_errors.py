"""Reusable error-inspection tool over any predictions_*.csv written by
src/gate.py's write_predictions (segment_id, meeting_id, ref, hyp). Consolidates
SESSIONS.md E5's ad hoc scratch analyses so future runs can rerun them instead
of re-deriving from scratch:

  1. rank_by_edit_chars -- rank segments by absolute edit-char count, not
     per-segment CER rate (a short 100%-wrong segment can have fewer edit
     chars than a long 20%-wrong one; ranking by rate hides which segment
     actually drives the corpus-level CER).
  2. loanword_dropped_segments -- flag segments where a candidate English
     loanword in ref (Vietnamese syllable-shape test, PROJECT_CORE.md §4)
     doesn't appear anywhere in hyp.
  3. digit_mismatch_segments -- flag segments where ref contains digit(s)
     but hyp's digit sequence differs (convention drift or a real miss).

Real-bench predictions (segment_ids like seg_0074_2) are rejoined to their
parent segment first (src/gate.py:rejoin_real_chunks) -- a no-op for
non-chunked predictions files.
"""

import argparse
import csv
import re
from pathlib import Path

from src.gate import rejoin_real_chunks
from src.metrics import char_counts

# strip_tone / is_vietnamese_shaped and the syllable grammar behind them moved to
# src/normalize.py so src/metrics.py can use them too -- importing them from here
# would be circular, since this module imports src.metrics above. Re-exported
# because plot_youtube_stats.py, probe_youtube_captions.py and
# tests/test_inspect_errors.py already import them from this module.
from src.normalize import is_vietnamese_shaped, strip_tone  # noqa: E402,F401


def load_predictions(csv_path: str | Path) -> list[dict]:
    with open(csv_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rejoin_real_chunks(rows)


def rank_by_edit_chars(rows: list[dict]) -> list[dict]:
    ranked = [{**r, "edit_chars": char_counts(r["ref"], r["hyp"]).edits} for r in rows]
    ranked.sort(key=lambda r: r["edit_chars"], reverse=True)
    return ranked


def top_segment_edit_share(rows: list[dict]) -> dict:
    ranked = rank_by_edit_chars(rows)
    total = sum(r["edit_chars"] for r in ranked)
    top = ranked[0]
    return {
        "segment_id": top["segment_id"],
        "meeting_id": top["meeting_id"],
        "edit_chars": top["edit_chars"],
        "total_edit_chars": total,
        "share": top["edit_chars"] / total if total else 0.0,
    }


def loanword_dropped_segments(rows: list[dict]) -> dict:
    n_candidates = 0
    dropped_segment_ids = []
    for r in rows:
        candidates = [w for w in r["ref"].split() if not w.isdigit() and not is_vietnamese_shaped(w)]
        if not candidates:
            continue
        n_candidates += 1
        hyp_words = set(r["hyp"].split())
        if any(c not in hyp_words for c in candidates):
            dropped_segment_ids.append(r["segment_id"])
    return {
        "n_candidate_segments": n_candidates,
        "n_dropped_segments": len(dropped_segment_ids),
        "dropped_segment_ids": dropped_segment_ids,
    }


def digit_mismatch_segments(rows: list[dict]) -> dict:
    n_with_digits = 0
    mismatch_segment_ids = []
    for r in rows:
        ref_digits = re.findall(r"\d+", r["ref"])
        if not ref_digits:
            continue
        n_with_digits += 1
        if ref_digits != re.findall(r"\d+", r["hyp"]):
            mismatch_segment_ids.append(r["segment_id"])
    return {
        "n_segments_with_digits": n_with_digits,
        "n_mismatch_segments": len(mismatch_segment_ids),
        "mismatch_segment_ids": mismatch_segment_ids,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("csv_path", type=Path)
    args = ap.parse_args()

    rows = load_predictions(args.csv_path)

    top = top_segment_edit_share(rows)
    print(f"top edit-char segment: {top['meeting_id']}/{top['segment_id']} "
          f"({top['edit_chars']} chars, {top['share']:.1%} of {top['total_edit_chars']} total edit-chars)")

    loanword = loanword_dropped_segments(rows)
    print(f"loanword-dropped: {loanword['n_dropped_segments']}/{loanword['n_candidate_segments']} "
          "segments with a candidate loanword lost entirely from hyp")

    digits = digit_mismatch_segments(rows)
    print(f"digit-mismatch: {digits['n_mismatch_segments']}/{digits['n_segments_with_digits']} "
          "segments where ref's digit sequence differs from hyp's")


if __name__ == "__main__":
    main()
