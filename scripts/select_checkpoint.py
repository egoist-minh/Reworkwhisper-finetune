"""Score every `checkpoints/step-N` of a run on one bench, at one or more lambdas,
in a single process -- the checkpoint+lambda picker for a run whose val CER is
known not to see what the pick is about (docs/v6-phase2-mixed-plan.md §0.3, §5).

Why not 13 runs of `scripts/score_cross_domain_model.py`: that loads
PhoWhisper-large per invocation, ~1 GPU minute each, to rank adapters that all
sit on the same frozen base. Here the base loads once and each candidate is a
`load_adapter` + `set_lambda`, so the GPU spends its time decoding.

Decode path is `src.gate._eval_split` + `src.gate.score_cross_domain`, the same
pair the gate thresholds were measured under -- NOT `scripts/benchmark_run.py`,
which chunks at 30 s and re-joins. Selecting on one path and gating on the other
makes the winner a property of the path as much as of the adapter.

    python -m scripts.select_checkpoint \
        --run-dir outputs/v6-phase2-mixed \
        --bench dataset/youtube-test \
        --batch-size 64 \
        --out outputs/v6-phase2-mixed/metrics/selection.json
"""

import argparse
import json
import re
from pathlib import Path


def pick_best(rows: list[dict]) -> dict:
    """The pre-committed rule: lowest CER wins, ties go to the higher retention,
    then to the earlier step. Module-level and tested (tests/test_select_checkpoint.py)
    rather than eyeballed off the printed table -- a rule read off a table after
    seeing it is a rule that can follow the numbers."""
    return min(rows, key=lambda r: (r["cer"], -r["retention"], r["step"]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--bench", required=True, help="manifest dir to score on (needs audio/)")
    ap.add_argument("--steps", nargs="*", type=int, default=None,
                    help="step numbers to score; default = every checkpoints/step-N on disk")
    ap.add_argument("--lambdas", nargs="*", type=float, default=[1.0],
                    help="lambda per candidate; 1.0 = the adapter as trained")
    ap.add_argument("--batch-size", type=int, default=None, help="override eval.batch_size")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from src import compat
    compat.apply()

    from src.asr import load_for_eval
    from src.config import Config, _build
    from src.data import ManifestDataset, load_manifests
    from src.gate import _eval_split, score_cross_domain
    from src.lora import set_lambda
    from src.normalize import Normalizer

    run_dir = Path(args.run_dir)
    cfg = _build(Config, json.loads((run_dir / "config.json").read_text(encoding="utf-8")))
    if args.batch_size:
        cfg.eval.batch_size = args.batch_size

    dirs = sorted((d for d in (run_dir / "checkpoints").glob("step-*") if d.is_dir()),
                  key=lambda d: int(re.fullmatch(r"step-(\d+)", d.name).group(1)))
    if args.steps:
        keep = set(args.steps)
        dirs = [d for d in dirs if int(d.name[5:]) in keep]
    if not dirs:
        raise SystemExit(f"no checkpoints/step-N under {run_dir} matching {args.steps}")

    normalizer = Normalizer(
        strip_punctuation=cfg.normalization.strip_punctuation,
        lowercase=cfg.normalization.lowercase,
        number_convention=cfg.normalization.number_convention,
        filler_tokens=cfg.normalization.filler_tokens,
    )
    ds = ManifestDataset(records=load_manifests(args.bench),
                         audio_root=Path(args.bench) / "audio")
    print(f"{len(ds)} segments from {args.bench}; "
          f"{len(dirs)} checkpoints x {len(args.lambdas)} lambda(s), batch {cfg.eval.batch_size}")

    model, processor = load_for_eval(cfg.base_model, dirs[0])
    names = {dirs[0].name: "default"}
    for d in dirs[1:]:
        model.load_adapter(str(d), adapter_name=d.name)
        names[d.name] = d.name

    rows = []
    for d in dirs:
        model.set_adapter(names[d.name])
        for lam in args.lambdas:
            set_lambda(model, lam)
            preds = _eval_split(model, processor, ds, normalizer, cfg.eval,
                                desc=f"{d.name}@{lam}").pop("_predictions")
            s = score_cross_domain(preds)
            rows.append({"step": int(d.name[5:]), "lambda": lam, "cer": s["cer"],
                         "retention": s["retention"], "cer_no_loanword": s["cer_no_loanword"],
                         "ci": s["ci"]})
            print(f"  {d.name:>10} lam={lam:<5} cer={s['cer']:.4f} "
                  f"retention={s['retention']:.4f} cer_no_loanword={s['cer_no_loanword']:.4f}")

    best = pick_best(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"bench": args.bench, "n_segments": len(ds),
                               "eval_batch_size": cfg.eval.batch_size,
                               "rows": rows, "best": best}, indent=2), encoding="utf-8")
    print(f"\nbest: step-{best['step']} @ lambda={best['lambda']} "
          f"retention={best['retention']:.4f} cer={best['cer']:.4f}\nwrote {out}")


if __name__ == "__main__":
    main()
