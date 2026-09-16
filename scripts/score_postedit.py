"""Score an LLM post-edit pass against the reference, next to the raw decode it edited.

The only number that matters here is the DELTA against the same system's unedited
output on the same segments -- absolute CER says nothing about whether the post-editor
earned its place. Paired bootstrap over segments, because the two systems saw identical
audio and the pairing is what makes a 1-point difference readable.

    PYTHONPATH=. python scripts/score_postedit.py \
        --edited Outputs/v6_l075-postedit-output.jsonl \
        --key Outputs/v6_l075-postedit-key.json
"""

import argparse
import json
import sys
from pathlib import Path

from scripts.benchmark_report import VARIANTS
from src.metrics import (bootstrap_delta_ci, char_counts, english_token_retention,
                          score, verdict, word_counts)
from src.normalize import Normalizer


def load_edited(path: Path) -> dict[str, str]:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        text = r.get("ban_sua")
        if text is None:
            text = r.get("ban_dung")          # tolerate the blind-refine field name
        if not isinstance(text, str):
            raise ValueError(f"{r.get('segment_id')} has no `ban_sua` string")
        out[r["segment_id"]] = text
    return out


def main() -> None:
    # The row labels are Vietnamese and the default Windows console codepage is not:
    # without this the script dies on its own output after doing all the work.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--edited", required=True)
    ap.add_argument("--key", required=True)
    ap.add_argument("--variant", default="N3", choices=sorted(VARIANTS))
    args = ap.parse_args()

    key = json.loads(Path(args.key).read_text(encoding="utf-8"))
    edited = load_edited(Path(args.edited))

    missing = [s for s in key if s not in edited]
    extra = [s for s in edited if s not in key]
    if missing:
        print(f"WARNING: {len(missing)} segment(s) missing from the edited file "
              f"-- falling back to the raw decode for those: {missing[:5]}")
    if extra:
        raise ValueError(f"{len(extra)} segment_id(s) in the edited file are not in the key")

    kw = VARIANTS[args.variant]
    seg_ids = sorted(key)
    refs, raws, news = [], [], []
    for sid in seg_ids:
        refs.append(Normalizer(**kw)(key[sid]["ref"]))
        raws.append(Normalizer(**kw)(key[sid]["hyp_goc"]))
        news.append(Normalizer(**kw)(edited.get(sid, key[sid]["hyp_goc"])))

    rows = []
    for name, hyps in (("decode gốc", raws), ("sau hậu xử lý", news)):
        s = score(refs, hyps)
        r = english_token_retention(refs, hyps)
        rows.append((name, s["cer"], s["wer"], r["retention"]))

    print(f"normalization {args.variant} -- {len(seg_ids)} segments\n")
    print(f"{'':16} {'CER':>8} {'WER':>8} {'retention':>11}")
    for name, cer, wer, ret in rows:
        print(f"{name:16} {100 * cer:7.2f}% {100 * wer:7.2f}% {100 * ret:10.2f}%")
    print(f"{'delta':16} {100 * (rows[1][1] - rows[0][1]):+7.2f}  "
          f"{100 * (rows[1][2] - rows[0][2]):+7.2f}  {100 * (rows[1][3] - rows[0][3]):+10.2f}")

    base_c = [char_counts(r, h) for r, h in zip(refs, raws)]
    cand_c = [char_counts(r, h) for r, h in zip(refs, news)]
    lo, hi = bootstrap_delta_ci(base_c, cand_c)
    print(f"\npaired bootstrap dCER vs decode gốc: [{100 * lo:+.2f}, {100 * hi:+.2f}] "
          f"-> {verdict(lo, hi)}")
    base_w = [word_counts(r, h) for r, h in zip(refs, raws)]
    cand_w = [word_counts(r, h) for r, h in zip(refs, news)]
    lo_w, hi_w = bootstrap_delta_ci(base_w, cand_w)
    print(f"paired bootstrap dWER vs decode gốc: [{100 * lo_w:+.2f}, {100 * hi_w:+.2f}] "
          f"-> {verdict(lo_w, hi_w)}")

    changed = sum(1 for sid in seg_ids
                  if edited.get(sid, key[sid]["hyp_goc"]).strip() != key[sid]["hyp_goc"].strip())
    print(f"\nsegments the post-editor touched: {changed}/{len(seg_ids)}")


if __name__ == "__main__":
    main()
