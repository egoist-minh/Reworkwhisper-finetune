# v4-mixed-r16 at λ=0.75 — provenance copy

Byte-identical copies from the λ=0.75 gate run (`scripts/gate_at_lambda.py`,
2026-08-18), whose full output lives in `Outputs/lambda075-metrics.zip` and
`Outputs/lambda075-adapter.zip`. They live here because the Kaggle runner clones
this repo from GitHub `main` — it never sees the local `Outputs/` tree.

This is the run dir `scripts/merge_and_push.py` consumes, **not**
`experiments/v4-mixed-r16/`, which is the λ=0.25 gate that produced the
currently-published adapter.

| File | Used for |
|---|---|
| `config.json` | base model, `lora.rank` / `lora.alpha` to recover λ, run id |
| `metrics/gate_results.json` | `overall_pass` check, model-card gate table, `_lambda_source` |
| `metrics/lambda_sweep.csv` | cross-check that the adapter's baked λ is the row the gate scored |
| `audit/predictions_tier1_in_domain.csv` | candidate side of `check_no_regression_vs_production` |

λ=0.75 is a **human override** of `select_lambda`, which still picks 0.5 — the
rule optimises val CER against OOD CER and cannot see `english_token_retention`.
`gate_results.json` records that in `_lambda_source`; SESSIONS.md has the full
curve. The adapter weights are bit-identical to training at every λ; λ is baked
into `adapter_config.json`'s `lora_alpha` (24.0 = 0.75 × 32).
