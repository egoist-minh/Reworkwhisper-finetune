"""Score one model on `data.cross_domain_path` and print the three numbers
`gates.cross_domain_*` bound (docs/v6-ondomain-plan.md §4.1, §4.3).

Exists because those thresholds are the production adapter's own measurements,
and a threshold is only meaningful against a number produced by the same decode
path it will be compared with. That path is `src.gate._eval_split` plus
`src.gate.score_cross_domain` -- the gate's own -- not `scripts/benchmark_run.py`,
which chunks audio at 30 s and re-joins the pieces.

`--model` is anything `src.asr.load_for_eval` takes: an HF model id (the
published adapter is a merged model, e.g. winhsss/Reworkwhisper-large-v5) or a
local dir. `--adapter` adds a PEFT adapter on top of `--model`, for scoring a
run's own `checkpoints/best` against the same bench.

    python -m scripts.score_cross_domain_model \
        --model winhsss/Reworkwhisper-large-v5 \
        --cross-domain-path dataset/cross-domain-bench \
        --out Outputs/cross-domain-v5.json
"""

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None, help="optional PEFT adapter dir on top of --model")
    ap.add_argument("--cross-domain-path", required=True)
    ap.add_argument("--config", default="configs/experiment.yaml",
                    help="source of the normalization and eval settings the numbers are "
                         "measured under -- must be the run's own config")
    ap.add_argument("--batch-size", type=int, default=None, help="override eval.batch_size")
    ap.add_argument("--out", required=True, help="where to write the score json")
    args = ap.parse_args()

    from src import compat
    compat.apply()

    from src.asr import load_for_eval
    from src.config import load as load_config
    from src.data import ManifestDataset, load_manifests
    from src.gate import _eval_split, score_cross_domain
    from src.normalize import Normalizer

    cfg = load_config(args.config)
    if args.batch_size:
        cfg.eval.batch_size = args.batch_size
    normalizer = Normalizer(
        strip_punctuation=cfg.normalization.strip_punctuation,
        lowercase=cfg.normalization.lowercase,
        number_convention=cfg.normalization.number_convention,
        filler_tokens=cfg.normalization.filler_tokens,
    )
    records = load_manifests(args.cross_domain_path)
    ds = ManifestDataset(records=records, audio_root=Path(args.cross_domain_path) / "audio")

    model, processor = load_for_eval(args.model, args.adapter)
    metrics = _eval_split(model, processor, ds, normalizer, cfg.eval, desc="cross_domain_bench")
    predictions = metrics.pop("_predictions")
    result = score_cross_domain(predictions)
    result["model"] = args.model
    result["adapter"] = args.adapter
    result["eval_batch_size"] = cfg.eval.batch_size

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"score": result, "predictions": predictions},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"cer                {result['cer']:.4f}   (ci {result['ci'][0]:.4f}-{result['ci'][1]:.4f})")
    print(f"retention          {result['retention']:.4f}   ({result['n_candidates']} candidates)")
    print(f"cer_no_loanword    {result['cer_no_loanword']:.4f}   "
          f"({result['n_segments_no_loanword']} segments)")
    print(f"cer_with_loanword  {result['cer_with_loanword']:.4f}   "
          f"({result['n_segments_with_loanword']} segments)")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
