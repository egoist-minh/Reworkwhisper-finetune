"""Run the full gate on an already-trained run at a lambda a human chose, and
bake the adapter at that lambda -- producing a run dir `scripts/merge_and_push.py`
accepts unchanged.

This is deliberately NOT a flag on `src/pipeline.py`'s sweep-gate stage.
`select_lambda` picking the lambda is the pipeline's contract; a person picking
a different one is a separate, recorded act, and keeping it in its own script
means a normal run can never quietly bypass the rule.

Why it exists (SESSIONS.md, the v5 production regression): `select_lambda` walks
val CER against OOD CER, and val CER cannot see English-token retention -- the
axis production actually regressed on. Measuring v4-mixed-r16's own weights at
several lambdas found 0.75 better than the rule's 0.5 on all six in-domain and
real cells (CER 0.0593 / 0.0161 / 0.2440, retention 0.8547 / 0.8622 / 0.5678),
for +0.72pp of OOD CER that stays inside the budget. The repo has precedent in
the other direction: v3-r16's gate picked lambda=1.0 and review shipped 0.5.

The gate re-measures tier 2 OOD at this lambda rather than copying the sweep's
number, because `merge_and_push.check_provenance` cross-checks the two: greedy
decode over the same OOD split should reproduce the sweep row exactly, and if
it does not, that disagreement is a finding, not a rounding detail.

Refuses a lambda that is not a row in the run's `lambda_sweep.csv` -- provenance
would reject the adapter later anyway, and failing here costs no GPU time.

Must run on GPU (Kaggle T4) -- UNTESTED on this machine (no torch/peft/GPU
here; CLAUDE.md memory "Kaggle code never works first try" applies). Inference
only, nothing is trained and nothing is pushed.

    python -m scripts.gate_at_lambda --lam 0.75 \\
        --run-dir /kaggle/input/.../v4-mixed-r16 \\
        --audio-root /kaggle/working/dataset/mixed-noisy-v1/audio \\
        --ood-eval-path dataset/vivos \\
        --real-bench-path /kaggle/input/.../real-meetings-bench \\
        --out-dir /kaggle/working/v4-mixed-r16-lambda0.75
"""

import argparse
import csv
import json
import shutil
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True,
                    help="source run artifacts: config.json, checkpoints/best, "
                    "validated_manifest.jsonl, metrics/{baseline.json,lambda_sweep.csv}, audit/")
    ap.add_argument("--lam", type=float, required=True,
                    help="the lambda to gate and bake -- must be a row in lambda_sweep.csv")
    ap.add_argument("--audio-root", required=True,
                    help="audio/ dir of the run's training corpus, as mounted here")
    ap.add_argument("--ood-eval-path", required=True,
                    help="OOD eval corpus root (tier 2); scripts/fetch_vivos.py writes one")
    ap.add_argument("--real-bench-path", default=None,
                    help="real-meetings-bench root; tier 4a is skipped without it, which "
                    "leaves that tier `pass: None` and still allows overall_pass")
    ap.add_argument("--out-dir", required=True, help="destination run dir for the gated lambda")
    args = ap.parse_args()

    from src import compat
    compat.apply()

    from src.asr import load_for_eval
    from src.config import Config, _build
    from src.data import ManifestDataset, load_manifests
    from src.gate import run_gate, write_gate_results
    from src.lora import save_with_lambda
    from src.normalize import Normalizer

    run_dir, out_dir = Path(args.run_dir), Path(args.out_dir)
    cfg = _build(Config, json.loads((run_dir / "config.json").read_text(encoding="utf-8")))
    cfg.data.ood_eval_path = args.ood_eval_path
    cfg.data.real_bench_path = args.real_bench_path

    sweep_csv = run_dir / "metrics" / "lambda_sweep.csv"
    with open(sweep_csv, encoding="utf-8") as f:
        sweep = [{k: float(v) for k, v in row.items() if v != ""} for row in csv.DictReader(f)]
    row = next((r for r in sweep if abs(r["lambda"] - args.lam) < 1e-9), None)
    if row is None:
        raise SystemExit(
            f"lambda={args.lam} is not a row in {sweep_csv} "
            f"({sorted(r['lambda'] for r in sweep)}) -- merge_and_push's provenance check "
            "would reject the baked adapter, so this run would be wasted GPU time.")
    print(f"lambda={args.lam} found in the sweep: val_cer={row['val_cer']:.5f} "
          f"ood_cer={row['ood_cer']:.6f}")
    print(f"NOTE: this is a human override of select_lambda's pick, not the rule's output. "
          f"The gate's tier2_ood must reproduce {row['ood_cer']:.6f} or provenance will reject it.")

    checkpoint_dir = run_dir / "checkpoints" / "best"
    if not checkpoint_dir.exists():
        raise SystemExit(f"{checkpoint_dir} missing -- the raw pre-lambda-bake adapter is required")
    baseline = json.loads((run_dir / "metrics" / "baseline.json").read_text(encoding="utf-8"))

    validated = [json.loads(l) for l in
                 (run_dir / "validated_manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    test_ds = ManifestDataset(records=[r for r in validated if r["split"] == "test"],
                              audio_root=Path(args.audio_root))
    ood_ds = ManifestDataset(records=load_manifests(args.ood_eval_path),
                             audio_root=Path(args.ood_eval_path) / "audio")
    real_ds = None
    if args.real_bench_path:
        real_ds = ManifestDataset(records=load_manifests(args.real_bench_path),
                                  audio_root=Path(args.real_bench_path) / "audio")
    else:
        print("tier 4a skipped -- pass --real-bench-path to gate it")

    normalizer = Normalizer(
        strip_punctuation=cfg.normalization.strip_punctuation,
        lowercase=cfg.normalization.lowercase,
        number_convention=cfg.normalization.number_convention,
        filler_tokens=cfg.normalization.filler_tokens,
    )

    model, processor = load_for_eval(cfg.base_model, checkpoint_dir)
    # save_with_lambda calls set_lambda itself, so the model is at `lam` for the gate below.
    adapter_dir = save_with_lambda(model, args.lam, out_dir / "adapter")
    print(f"baked adapter at lambda={args.lam} -> {adapter_dir}")

    results = run_gate(cfg, model, processor, normalizer, test_ds, ood_ds, real_ds, baseline,
                       baseline_real_csv=run_dir / "audit" / "predictions_baseline_real.csv",
                       baseline_test_csv=run_dir / "audit" / "predictions_baseline_test.csv")
    results["_lambda"] = args.lam
    results["_lambda_source"] = (
        "human override of select_lambda, recorded in SESSIONS.md -- the rule optimises "
        "val CER against OOD CER and cannot see english_token_retention")
    gate_path = write_gate_results(results, out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(run_dir / "config.json", out_dir / "config.json")
    shutil.copy2(sweep_csv, out_dir / "metrics" / "lambda_sweep.csv")

    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    tier2 = gate["tier2_ood"]["cer"]
    print(f"\ntier2_ood measured {tier2:.9f}; sweep row says {row['ood_cer']:.9f}; "
          f"delta {abs(tier2 - row['ood_cer']):.2e}")
    if abs(tier2 - row["ood_cer"]) > 1e-9:
        print("MISMATCH -- merge_and_push.check_provenance will refuse this adapter. Do not "
              "paper over it by editing either file; find out why the same greedy decode over "
              "the same OOD split gave a different number.")
    print(f"overall_pass={gate['overall_pass']}")
    print(f"\nrun dir ready for merge_and_push: {out_dir}")


if __name__ == "__main__":
    main()
