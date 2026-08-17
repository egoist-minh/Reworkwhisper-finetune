"""One-off measurement, SESSIONS.md H6 ("PHEP DO DUT DIEM, chay truoc moi row
khac con lai"): score v3-r16 (HF `winhsss/Reworkwhisper-large-v4`, lambda=0.5)
on the exact 228 `source: youtube` segments of v4-mixed-r16's tier1_in_domain
test split (HF `winhsss/Reworkwhisper-large-v5`'s own CER there is 0.0764).

Every existing gate tier compares v4-mixed-r16 only against its own base-model
baseline -- nothing has ever scored v3-r16 on this slice, because the slice
did not exist when v3-r16 ran (v3-r16 trained on paid-dataset only, no
YouTube). This is the one comparison that is simultaneously real speech,
human-reviewed, and same-domain as the production regression under
investigation (SESSIONS.md, "Hoi quy Reworkwhisper-large-v5 trong production").

Uses `Outputs/v4-mixed-r16/validated_manifest.jsonl` (already resolved and
split-assigned by that run) to select the 228 segments, rather than
recomputing `resolve_splits` here -- avoids any chance of drifting from the
exact set v4-mixed-r16 was gated on. `--audio-root` must point at the same
`mixed-noisy-v1` dataset mounted wherever this runs, since the manifest only
carries relative `audio_filepath`s.

Baking lambda=0.5 reuses `src.lora.set_lambda` in-memory (PROJECT_CORE.md
lambda contract) against `Outputs/v3-r16/checkpoints/best` -- the raw
pre-lambda-bake adapter, same source `stage_sweep_gate` scales for every
lambda in a sweep -- not `Outputs/v3-r16/adapter` (already baked to
lambda=1.0, the gate's original "largest within budget" pick before the
docs/finetune-results-report-v3.md grill chose lambda=0.5 instead).

Must run on GPU (Kaggle T4) with `mixed-noisy-v1` attached -- UNTESTED on this
machine (no torch/peft/GPU here; CLAUDE.md memory "Kaggle code never works
first try" applies, expect at least one iteration).

    python -m scripts.eval_v3_on_youtube_test \\
        --audio-root /kaggle/input/datasets/<user>/mixed-noisy-v1/mixed-noisy-v1/audio \\
        --out predictions_v3-r16_youtube228.csv
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default="Outputs/v4-mixed-r16/validated_manifest.jsonl",
                    help="resolved manifest to select the 228 test/youtube segments from")
    ap.add_argument("--audio-root", required=True,
                    help="mixed-noisy-v1's audio/ dir as mounted on this machine")
    ap.add_argument("--run-dir", default="Outputs/v3-r16",
                    help="v3-r16 run dir: reads config.json for base_model/normalization, "
                    "checkpoints/best for the raw (pre-lambda) adapter")
    ap.add_argument("--lam", type=float, default=0.5)
    ap.add_argument("--out", default="predictions_v3-r16_youtube228.csv")
    args = ap.parse_args()

    from src import compat
    compat.apply()

    from src.asr import load_for_eval
    from src.data import ManifestDataset
    from src.gate import _eval_split, write_predictions
    from src.lora import set_lambda
    from src.metrics import english_token_retention
    from src.normalize import Normalizer

    run_dir = Path(args.run_dir)
    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    checkpoint_dir = run_dir / "checkpoints" / "best"
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"{checkpoint_dir} missing -- v3-r16's raw checkpoint is required")

    records = [json.loads(l) for l in Path(args.manifest).read_text(encoding="utf-8").splitlines()]
    youtube_test = [r for r in records if r["split"] == "test" and r["source"] == "youtube"]
    if not youtube_test:
        raise RuntimeError(f"no split=test/source=youtube records in {args.manifest}")
    print(f"{len(youtube_test)} youtube test segments selected from {args.manifest}")

    ds = ManifestDataset(records=youtube_test, audio_root=Path(args.audio_root))

    model, processor = load_for_eval(cfg["base_model"], checkpoint_dir)
    n_scaled = set_lambda(model, args.lam)
    print(f"set_lambda: {n_scaled} LoRA layers scaled to lambda={args.lam}")

    norm_cfg = cfg["normalization"]
    normalizer = Normalizer(
        strip_punctuation=norm_cfg["strip_punctuation"],
        lowercase=norm_cfg["lowercase"],
        number_convention=norm_cfg["number_convention"],
        filler_tokens=norm_cfg["filler_tokens"],
    )
    eval_cfg = SimpleNamespace(**cfg["eval"])

    result = _eval_split(model, processor, ds, normalizer, eval_cfg,
                         desc=f"h6:v3-r16_lambda{args.lam}_on_youtube_test")
    predictions = result.pop("_predictions")
    write_predictions(predictions, args.out)

    retention = english_token_retention([p["ref"] for p in predictions],
                                         [p["hyp"] for p in predictions])
    print(f"v3-r16 (lambda={args.lam}) on {len(predictions)} youtube-test segments: "
          f"cer={result['cer']:.4f} english_token_retention={retention['retention']}")
    print("v4-mixed-r16 on the same slice (already measured): cer=0.0764 -- "
          "see SESSIONS.md H3 for its english_token_retention.")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
