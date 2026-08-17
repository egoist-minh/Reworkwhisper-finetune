# v4-mixed-r16 — provenance copy

Byte-identical copies of three files from `outputs/v4-mixed-r16/`, which is
gitignored (the full run dir is ~1.1 GB of checkpoints and audit CSVs). They
live here because `scripts/merge_and_push.py` needs them to verify and describe
the adapter, and the Kaggle runner clones this repo from GitHub `main` — it
never sees the local `outputs/` tree.

| File | Used for |
|---|---|
| `config.json` | base model, `lora.rank` / `lora.alpha` to recover λ, run id |
| `metrics/gate_results.json` | `overall_pass` check, model-card gate table |
| `metrics/lambda_sweep.csv` | cross-check that the adapter's baked λ is the row the gate scored |

The full run artifacts stay in `Outputs/outputs_v4-mixed-r16.zip`.
