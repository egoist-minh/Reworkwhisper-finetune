"""Rescore the 3-meeting held-out benchmark with this repo's normalizer.

The 2026-08-27 run scored ElevenLabs Scribe v2 against Reworkwhisper with
viet-speech's `normalize_vi`, which has no number convention -- so a spoken "tám
mươi sáu" against a written "86" counted as a total miss on both models, and the
medical meeting's numbers pushed its WER up for reasons that are scoring, not
recognition. This rescores the stored hypotheses with `src.normalize.Normalizer`
(the same one the gate uses) so the numbers sit on the paper's own scale, and adds
english_token_retention, which that run did not measure at all.

No inference is re-run: hypotheses come verbatim from the archived per-segment JSONL.

    python -m scripts.rescore_pilot_benchmark
"""
import argparse
import json
import sys
import zipfile
from pathlib import Path

from src.metrics import score, english_token_retention
from src.normalize import Normalizer

sys.stdout.reconfigure(encoding="utf-8")

ZIP = "Outputs/review_youtube_data_pilot-20260827T054523Z-1-001.zip"
MEETINGS = ("IGZYBrbDUEw", "ySdJ3sg_2lk", "z-nODwLyhA0")
MODELS = {"ElevenLabs Scribe v2": "elevenlabs_scribe_v2",
          "Reworkwhisper-large-v5": "reworkwhisper_large_v5"}
FILLERS = ["ừm", "ờm", "ehm", "uhm", "hmm"]


def read(zf, model, meeting):
    name = f"review_youtube_data_pilot/{model}.{meeting}.persegment.jsonl"
    return [json.loads(x) for x in zf.read(name).decode("utf-8").splitlines()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default=ZIP)
    args = ap.parse_args()

    norm = Normalizer(strip_punctuation=True, lowercase=True,
                      number_convention="word_to_digit", filler_tokens=FILLERS)
    zf = zipfile.ZipFile(args.zip)
    rows = {}
    for label, slug in MODELS.items():
        for m in MEETINGS:
            recs = read(zf, slug, m)
            rows[(label, m)] = ([norm(r["ref"]) for r in recs],
                                [norm(r["hyp"]) for r in recs])

    print("### 3 buổi giữ riêng, chấm lại bằng Normalizer của repo này\n")
    print("| Buổi họp | n | Model | CER | WER | Giữ từ mượn |")
    print("|---|---:|---|---:|---:|---:|")
    for m in MEETINGS:
        for label in MODELS:
            refs, hyps = rows[(label, m)]
            s, t = score(refs, hyps), english_token_retention(refs, hyps)
            ret = "—" if t["retention"] is None else f'{t["retention"]*100:.2f}% ({t["n_retained"]}/{t["n_candidates"]})'
            print(f'| `{m}` | {s["n_segments"]} | {label} | {s["cer"]*100:.3f}% | {s["wer"]*100:.3f}% | {ret} |')
    print()
    print("| Gộp 3 buổi | n | Model | CER | WER | Giữ từ mượn |")
    print("|---|---:|---|---:|---:|---:|")
    for label in MODELS:
        refs = [x for m in MEETINGS for x in rows[(label, m)][0]]
        hyps = [x for m in MEETINGS for x in rows[(label, m)][1]]
        s, t = score(refs, hyps), english_token_retention(refs, hyps)
        print(f'| — | {s["n_segments"]} | {label} | **{s["cer"]*100:.3f}%** | **{s["wer"]*100:.3f}%** | '
              f'**{t["retention"]*100:.2f}%** ({t["n_retained"]}/{t["n_candidates"]}) |')


if __name__ == "__main__":
    main()
