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
import unicodedata
from pathlib import Path

from src.gate import rejoin_real_chunks
from src.metrics import char_counts

# Vietnamese tone diacritics (sắc, huyền, hỏi, ngã, nặng) as NFD combining
# marks -- stripped before syllable-shape matching so tone doesn't matter,
# while the six extra vowel LETTERS (ă â ê ô ơ ư, not accents) are kept.
_TONE_MARKS = frozenset("̣́̀̉̃")


def strip_tone(word: str) -> str:
    decomposed = unicodedata.normalize("NFD", word)
    return unicodedata.normalize("NFC", "".join(c for c in decomposed if c not in _TONE_MARKS))


# Vietnamese syllable grammar: optional onset + required nucleus + optional coda.
# PROJECT_CORE.md §4's illustrative pattern only allows a single consonant
# letter and a single vowel letter, which flags ~99% of real Vietnamese words
# ("không", "được", "nhưng", ...) as non-Vietnamese -- unusable as literally
# written. This extends it with digraph onsets and diphthong/triphthong nuclei.
_ONSETS = ["ngh", "ng", "nh", "ph", "th", "tr", "ch", "kh", "gi", "qu", "gh",
           "b", "c", "d", "đ", "g", "h", "k", "l", "m", "n", "p", "q", "r",
           "s", "t", "v", "x", "y"]
_NUCLEI = ["oai", "oay", "uao", "uay", "uôi", "ươi", "ươu", "iêu", "yêu",
           "uyu", "uya", "oeo", "uyê",
           "ia", "ya", "iê", "yê", "ua", "uô", "ưa", "ươ", "oa", "oe", "uy",
           "uơ", "uâ", "oă", "uê",
           "ai", "ay", "ây", "ao", "au", "âu", "eo", "êu", "oi", "ôi", "ơi",
           "ui", "ưi", "iu", "ưu",
           "a", "ă", "â", "e", "ê", "i", "o", "ô", "ơ", "u", "ư", "y"]
_CODAS = ["ng", "nh", "ch", "c", "m", "n", "p", "t"]


def _alternation(options: list[str]) -> str:
    return "|".join(sorted(options, key=len, reverse=True))


_SYLLABLE = re.compile(
    f"^(?:{_alternation(_ONSETS)})?(?:{_alternation(_NUCLEI)})(?:{_alternation(_CODAS)})?$"
)


def is_vietnamese_shaped(token: str) -> bool:
    return bool(_SYLLABLE.fullmatch(strip_tone(token.lower())))


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
