"""H2(a) follow-up (SESSIONS.md, "kiếm thêm audio thật cùng domain") -- score
BOTH v3-r16 (HF v4, lambda=0.5) and v4-mixed-r16 (HF v5, lambda=0.25) against
a second held-out real-audio reference: 20 segments of `NZiW4QH83CI`
(29:00-35:49, chosen for highest code-switch density in a 10-min window --
5.24% vs 4.15% for a naive first-10-min cut), hand-corrected by the user in
`D:\\corrections.NZiW4QH83CI.json`. 6 of the original 26 corrected segments
(seg_0009/0019/0020/0023/0024/0025) contain `<??>` -- portions the user could
not make out -- and are EXCLUDED here rather than scored against a guess.

Domain: programming/RPA tech talk (`api`, `web`, `html`, `css`, `javascript`,
`chatbot`, `platform` -- see youtube-data-pilot/caption-probe.md), a THIRD
domain distinct from `first10.wav` (career webinar) and tier4a (ML/VAE
lecture) -- neither of which alone settled whether v5 regresses on real
audio (SESSIONS.md H2, the three-way contradiction).

`NZiW4QH83CI` is not one of the 7 `youtube-meetings` videos mixed-noisy-v1
was built from (dataset/youtube-meetings/raw/ as of 2026-08-17), so this is
held out from both models' training exactly like first10.wav was.

Mirrors eval_v3_on_youtube_test.py's structure: raw (pre-lambda-bake)
checkpoints + in-memory `set_lambda`, same as `stage_sweep_gate` uses for
every lambda in a sweep. `--v4mix-lambda` defaults to 0.25 (SESSIONS.md,
"v5 dùng lambda=0.25, thấp hơn v4 (lambda=0.5)") -- the published v5's own
lambda, not a guess.

Must run on GPU (Kaggle T4) -- UNTESTED on this machine (no torch/peft here).

    python -m scripts.eval_v3_v5_on_corrected_segments \\
        --audio-root /kaggle/input/datasets/<user>/youtube-meetings/youtube-meetings/audio \\
        --out-prefix predictions_NZiW4QH83CI
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

SKIP_UNINTELLIGIBLE = {"seg_0009", "seg_0019", "seg_0020", "seg_0023", "seg_0024", "seg_0025"}
MAX_SEGMENT = 25  # user corrected only up to seg_0025 ("audio khá khó nghe")


def load_segments(manifest_path: Path, corrections_path: Path) -> list[dict]:
    records = {r["segment_id"]: r for r in
               (json.loads(l) for l in manifest_path.read_text(encoding="utf-8").splitlines())}
    corrections = json.loads(corrections_path.read_text(encoding="utf-8"))

    out = []
    for seg_id, corrected_text in corrections.items():
        n = int(seg_id.split("_")[1])
        if n > MAX_SEGMENT or seg_id in SKIP_UNINTELLIGIBLE:
            continue
        if "<??>" in corrected_text:
            raise ValueError(f"{seg_id} contains <??> but is not in SKIP_UNINTELLIGIBLE -- "
                              "update the skip set, don't score a guess")
        rec = dict(records[seg_id])
        rec["text"] = corrected_text
        out.append(rec)
    if not out:
        raise RuntimeError(f"no usable segments in {corrections_path}")
    return sorted(out, key=lambda r: r["segment_id"])


def _run(run_dir: Path, lam: float, ds, base_model: str, eval_cfg, normalizer, desc: str):
    from src.asr import load_for_eval
    from src.gate import _eval_split, write_predictions
    from src.lora import set_lambda

    checkpoint_dir = run_dir / "checkpoints" / "best"
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"{checkpoint_dir} missing")
    model, processor = load_for_eval(base_model, checkpoint_dir)
    n_scaled = set_lambda(model, lam)
    print(f"{run_dir.name}: set_lambda {n_scaled} LoRA layers to lambda={lam}")
    result = _eval_split(model, processor, ds, normalizer, eval_cfg, desc=desc)
    predictions = result.pop("_predictions")
    return result, predictions


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default="dataset/youtube-meetings/manifest.NZiW4QH83CI.jsonl")
    ap.add_argument("--corrections", default="D:/corrections.NZiW4QH83CI.json")
    ap.add_argument("--audio-root", required=True,
                    help="youtube-meetings's audio/ dir as mounted on this machine")
    ap.add_argument("--v3-run-dir", default="Outputs/v3-r16")
    ap.add_argument("--v3-lambda", type=float, default=0.5)
    ap.add_argument("--v4mix-run-dir", default="Outputs/v4-mixed-r16")
    ap.add_argument("--v4mix-lambda", type=float, default=0.25)
    ap.add_argument("--out-prefix", default="predictions_NZiW4QH83CI")
    args = ap.parse_args()

    from src import compat
    compat.apply()

    from src.data import ManifestDataset
    from src.gate import write_predictions
    from src.metrics import english_token_retention
    from src.normalize import Normalizer

    segments = load_segments(Path(args.manifest), Path(args.corrections))
    print(f"{len(segments)} usable corrected segments "
          f"(<= seg_{MAX_SEGMENT:04d}, excluding {sorted(SKIP_UNINTELLIGIBLE)})")

    v3_cfg = json.loads((Path(args.v3_run_dir) / "config.json").read_text(encoding="utf-8"))
    ds = ManifestDataset(records=segments, audio_root=Path(args.audio_root))
    norm_cfg = v3_cfg["normalization"]
    normalizer = Normalizer(
        strip_punctuation=norm_cfg["strip_punctuation"], lowercase=norm_cfg["lowercase"],
        number_convention=norm_cfg["number_convention"], filler_tokens=norm_cfg["filler_tokens"],
    )
    eval_cfg = SimpleNamespace(**v3_cfg["eval"])

    v3_result, v3_preds = _run(Path(args.v3_run_dir), args.v3_lambda, ds, v3_cfg["base_model"],
                                eval_cfg, normalizer, "v3-r16_on_NZiW4QH83CI")
    write_predictions(v3_preds, f"{args.out_prefix}_v3.csv")

    v4mix_cfg = json.loads((Path(args.v4mix_run_dir) / "config.json").read_text(encoding="utf-8"))
    v4mix_result, v4mix_preds = _run(Path(args.v4mix_run_dir), args.v4mix_lambda, ds,
                                      v4mix_cfg["base_model"], eval_cfg, normalizer,
                                      "v4-mixed-r16_on_NZiW4QH83CI")
    write_predictions(v4mix_preds, f"{args.out_prefix}_v4mix.csv")

    refs = [p["ref"] for p in v3_preds]
    v3_ret = english_token_retention(refs, [p["hyp"] for p in v3_preds])
    v4mix_ret = english_token_retention(refs, [p["hyp"] for p in v4mix_preds])

    print(f"v3-r16   (lambda={args.v3_lambda}):    cer={v3_result['cer']:.4f}  "
          f"retention={v3_ret['retention']}  ({v3_ret['n_retained']}/{v3_ret['n_candidates']})")
    print(f"v4-mixed-r16 (lambda={args.v4mix_lambda}): cer={v4mix_result['cer']:.4f}  "
          f"retention={v4mix_ret['retention']}  ({v4mix_ret['n_retained']}/{v4mix_ret['n_candidates']})")
    print(f"wrote {args.out_prefix}_v3.csv / {args.out_prefix}_v4mix.csv")


if __name__ == "__main__":
    main()
