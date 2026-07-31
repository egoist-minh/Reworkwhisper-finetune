# SESSIONS

Task-tracking breakdown of the PhoWhisper pipeline build. For the stable spec (module
contracts, data flow, config schema, gate rules), see `PROJECT_CORE.md`. For behavioral
rules, see `CLAUDE.md`. For the deadline and the facts a prior session already verified
live, see the handoff note referenced from memory.

**How to use this file**: before starting a session, read `PROJECT_CORE.md` (esp. §2.1
DATA FLOW) + `CLAUDE.md` + this session's row only. Mark Status when you finish.

## Status legend
`todo` → not started · `in-progress` → partially done · `done` → deliverable exists and
meets Acceptance · `blocked` → waiting on a dependency · `deferred` → explicitly cut from
this deadline, do not start without re-opening scope with the user

## Deadline scope cut (2026-08-02) — read this before picking up any `todo` row

Decided 2026-07-31, confirmed by the user 2026-07-31 (second pass — real-meeting bench
moved from deferred to required). Do not re-litigate without user sign-off:

- **`lora.rank: 16`** for the first run (not 64 — the PROJECT_CORE §3 example is
  illustrative, not the live default). `configs/experiment.yaml` is the source of truth.
- **Tier 3 (RTF) dropped, tier 4b (long-form) deferred.** The gate that ships for this
  deadline is tiers **1, 2, 4a only**. `src/gate.py` and `configs/experiment.yaml:gates`
  already reflect this — do not add `rtf_threshold`/`longform_*` back in without a scope
  conversation first.
- **Tier 4a (real-meeting bench) is REQUIRED, not opportunistic** (confirmed by the user
  — this reverses the first pass's "opportunistic" call). `data.real_bench_path` is set
  to `dataset/real-meetings-bench`, ingested from `done/` via
  `scripts/ingest_real_bench.py` — see A2b below, **done**.
- **`training.full_finetune: false`** for the first run, so the SVD reconstruction path
  (§6 Stage 3 Path B, `min_retained_energy` guard) is **not needed** to ship. It stays a
  documented but unbuilt path.
- **Only `paid-dataset` is used.** `unpaid-dataset` and `dataset_by_task` are explicitly
  out of scope for this deadline (confirmed by the user) — do not profile or wire them in.
- **Dataset is NOT migrated into `Fine_tune_wf/dataset/`.** `data.dataset_path` points
  straight at `D:/phowhisper-finetune-exp/dataset/paid-dataset` (read-only) — copying it
  is not on the critical path per CLAUDE.md. Getting the same bytes onto Kaggle is a Track
  B launcher concern (upload as a Kaggle Dataset attachment), not a Track A migration.
  (`dataset/real-meetings-bench/` is the one exception — it's generated fresh by A2b, not
  copied wholesale, and is small enough to attach directly.)
- **The pipeline must actually push and persist evidence, not just print numbers**
  (confirmed by the user). `src/hub.py:push_adapter` is built and wired into
  `pipeline.py`'s `sweep-gate` stage (fires only on `overall_pass` and `hub.push: true`).
  `src/gate.py` now writes per-segment predictions (`audit/predictions_{tier}.csv`) for
  every eval it runs, at both baseline and gate time — not just the aggregate CER.
- **Module boundaries differ from the original split below**, and the differences are
  final for this deadline, not an oversight: `normalize.py` (not `metrics.py`) owns the
  §6 Stage 4 contract; LoRA λ scaling lives in `lora.py` (Path A only, no separate
  `model.py`); sweep selection is inlined in `pipeline.py`'s `sweep-gate` stage rather than
  a standalone `sweep.py`; `report.py` (provenance.md/ledger) is **not built** — not
  required for a passing run, revisit only if time remains after D2.

## Actual status as of 2026-07-31 (post-handoff continuation)

| Track | What it is | GPU |
|:---|:---|:---|
| **A** | Foundation — pure-Python modules, CPU-testable | no |
| **B** | GPU bring-up — smoke test, baseline numbers, trained checkpoint | yes |
| **D** | Integration — real sweep + gate exercised for real | yes |

Track C from the original plan is folded into Track A/D above: gate + sweep decision
logic already exist as pure-enough functions inside `gate.py`/`pipeline.py` and are
covered by the GPU integration session (D1) rather than a separate CPU session, since
splitting them out further does not buy back any Kaggle queue time before the deadline.

| # | Session | Env | Depends on | Output | Acceptance | Status |
|---|---------|-----|-----------|--------|------------|--------|
| **A1** | **Bootstrap** — `git init`, `.gitignore`, `requirements.txt`, `src/__init__.py`, GitHub remote | CPU | none | repo initialized + pushed | `git status` clean; `configs/experiment.yaml` has no path back that breaks on another machine except `data.dataset_path` itself | **done** — repo live at `github.com/egoist-minh/Reworkwhisper-finetune` (private), initial commit pushed to `main` |
| **A3** | **Config schema** — `src/config.py`, `configs/experiment.yaml` | CPU | A1 | `src/config.py`, `configs/experiment.yaml` | Loads YAML → validated dataclass; rejects unknown keys, bad rank/alpha, lexical filler tokens, tier-4 leak, empty sweep grid | **done** (validated by hand via `python -c`); `tests/test_config.py` **not yet written** — `todo`, do before trusting further config edits |
| **A4** | **Data module** — `src/data.py`: manifest merge, split resolution, 24k→16k resample | CPU | A1, A3 | `src/data.py`, `tests/test_data.py` | Validates real `paid-dataset` manifests: **1316/250/236** train/val/test (corrected from an earlier wrong 1566/250/236 note — verified live, see handoff), val meetings resolve to `[0001, 0002, 0011]`, meeting_id conflict raises | **done** — 4 tests pass against the real predecessor dataset (read-only) |
| **A5** | **Normalization + metrics** — `src/normalize.py` (§6 Stage 4 contract, word→digit), `src/metrics.py` (Levenshtein, CER/WER, bootstrap CI, paired delta CI) | CPU | A1 | `src/normalize.py`, `src/metrics.py`, `tests/test_normalize.py` | word→digit correct on the arithmetic/digit-string cases incl. the `hai ba nghìn`→`23000` bug fixed this session; symmetric normalization verified; CER/WER corpus-level, not per-segment mean; delta CI paired | **done** — 9 tests pass. `code-switch tagging` (syllable-shape regex) and `measure_rtf()` **not built** — deferred, not needed for tiers 1/2/4a |
| **A6** | **LoRA + lambda (Path A only)** — `src/lora.py`: build config, `set_lambda` (filters `LoraLayer`, not `hasattr`), `save_with_lambda` (bakes into `adapter_config.json`, verifies via reload within 1e-5) | CPU (logic) | A1, A3 | `src/lora.py` | No literal rank anywhere (`cfg.lora.rank` only); λ mechanism proven live on Kaggle T4 prior to this session (three routes agree to 0.000e+00) | **done**, but **UNVERIFIED end-to-end in this session** — no torch/peft on this machine. Needs a Kaggle smoke run. Path B (full-FT SVD) is **deferred**, out of scope per this deadline |
| **A2** | **VIVOS (OOD)** — `scripts/fetch_vivos.py`: parquet route (primary) + tarball route (fallback) | CPU (script), needs network | A1 | `scripts/fetch_vivos.py` | Writes `dataset/vivos/{audio/,manifest.jsonl}` in the §4 schema | **in-progress** — script written this session, **UNTESTED** (no `pyarrow`/`huggingface_hub` locally). Run `python scripts/fetch_vivos.py --smoke` on Kaggle next, inspect the printed schema before trusting the manifest |
| **A2b** | **Real-meetings benchmark** — `scripts/ingest_real_bench.py`: ingest `done/` as `data.real_bench_path`, re-segment >30s spans (silence-detected audio splits + proportional text splits, snapped to word boundaries) | CPU | A1 | `dataset/real-meetings-bench/` | Every segment ≤30s; concatenating a re-segmented span's text reproduces the original modulo whitespace; leak guard passes | **done** — ran against the real `done/` source: 196 raw segments → **264** after re-segmenting the 20 that exceeded 30s (real_0001: 146, real_0002: 118), 29,579 chars. 6 tests in `tests/test_ingest_real_bench.py` pass, incl. one against the real 16 kHz wav. `configs/experiment.yaml:data.real_bench_path` now set. **Caveat carried forward**: text/audio correspondence inside a re-segmented span is a *proportional-duration approximation*, not forced alignment — fine for tier 4a's aggregate CER, not for anything claiming word-level timing |
| **A2c** | **Reference audit + freeze** | CPU | A2b | frozen reference v1 | — | **deferred** — the convention-sensitivity check and version freeze aren't required for a first passing run; the six §4 caveats (post-edited-by-PhoWhisper reference, mixed number convention, etc.) already apply and are documented, just not re-verified against the newly re-segmented copy |
| **A7** | **Hub push + prediction evidence** — `src/hub.py:push_adapter` (HF upload, model card with gate CER table, no token in any artifact); `src/gate.py` now returns and writes per-segment predictions (`audit/predictions_{tier}.csv`) at both baseline and gate time | CPU (logic) | A6 | `src/hub.py`, `src/gate.py` predictions | Push gated on `gate_results.json` overall PASS + `hub.push: true`; predictions CSV has segment_id/meeting_id/ref/hyp for every eval split run | **done**, **UNVERIFIED end-to-end** — no `huggingface_hub` push has been exercised in this session. Needs a Kaggle run with `HF_TOKEN` set and `hub.push: true` to confirm |
| **B1** | **Kaggle smoke test** — `python -m src.pipeline --stage smoke` on Kaggle; attach `paid-dataset` as a Kaggle Dataset; verify compat patch, manifest load, split resolve, normalize all run before spending any real GPU time | GPU | A1, A3, A4, A5 | first Kaggle run log | `compat.apply()` neutralizes the torchao probe if needed and prints version table; split_stats prints 1316/250/236; no exception | **todo — next action**. `src/pipeline.py --stage smoke` already written; this is the first time any of `compat.py`'s Kaggle-specific logic gets exercised for real this session |
| **B2** | **Baseline eval** — `python -m src.pipeline --stage baseline`: base model over test + OOD + real-bench → `metrics/baseline.json` + per-split prediction CSVs | GPU | A2, A2b, A6, B1 | `outputs/{run_id}/metrics/baseline.json`, `outputs/{run_id}/audit/predictions_baseline_*.csv` | `cer_test`, `cer_ood`, `cer_real` all present; OOD CER sane vs historical 1.78%; prediction CSVs non-empty | **todo**, blocked on B1 passing and A2 (vivos) landing on Kaggle. `dataset/real-meetings-bench/` needs to reach Kaggle too (small enough to commit or attach directly, unlike `paid-dataset`) |
| **B3** | **Train** — `python -m src.pipeline --stage train`: LoRA SFT (rank 16), early stopping, OOD eval every epoch | GPU | A5, A6, B2 | `checkpoints/best/`, `metrics/training.csv` | Training completes at least 1 epoch without OOM (batch 8 measured to fit in 9.75 GiB on T4); early stopping wired; `training.csv` has per-step loss/LR/val_cer/ood_cer | **todo**. `src/train.py` written this session but **UNVERIFIED**: the transformers 5.0.0 `Trainer(eval_dataset=dict)` multi-eval-set API it depends on has not been exercised anywhere in this repo yet — smoke this specifically before trusting a long run |
| **D1** | **Sweep + Gate + Push (combined)** — `python -m src.pipeline --stage sweep-gate`: λ grid eval → `select λ*` (hard-fails if none in budget) → save adapter with λ baked in → gate tiers 1/2/4a (writes `audit/predictions_tier{1,2,4a}.csv`) → HF push if PASS | GPU | A2b, A6, A7, B3 | `metrics/lambda_sweep.csv`, `adapter/`, `metrics/gate_results.json`, `audit/predictions_*.csv`, HF repo (if PASS) | λ* selection matches §6 exactly, raises (not falls back) when no λ fits `sweep.ood_cer_budget`; `save_with_lambda`'s reload-verify passes within 1e-5; gate tiers computed correctly incl. tier 4a on the real bench; predictions CSVs written for all three tiers; HF push only fires on overall PASS and `hub.push: true` | **todo**, blocked on B3. Set `hub.push: true` + `hub.repo_id` + `HF_TOKEN` env var before this run if the push should fire for real — currently `false`/`null` in `configs/experiment.yaml`, confirm the repo id with the user before flipping it (outward-facing action). This session's `stage_sweep_gate` in `pipeline.py` is **UNVERIFIED end-to-end** — combines what the original plan split into C1/D1/C2/D2/C4, so this single Kaggle run needs to prove all of: λ mechanism in the trained (not synthetic) delta, gate math incl. real tier 4a, the fail-hard path, and the HF push |
| **D2** | **E2E confirmation** — re-run `sweep-gate` on a config deliberately set to fail one gate tier (e.g. raise `gates.min_improvement_pct`), confirm halt + no push | GPU | D1 | a second `outputs/{run_id}/` showing the halt | Halts with non-zero exit, `gate_results.json` shows the failed tier, no `hub` push attempted | **todo**, blocked on D1 |

## Critical path to 2026-08-02

```
A2 (vivos, on Kaggle) ──┐
A2b (done) ─────────────┼──► B1 (smoke) ──► B2 (baseline) ──► B3 (train) ──► D1 (sweep-gate+push) ──► D2 (fail-path check)
A6, A7 already done ────┘
```

Everything else in this file (A2c, `tests/test_config.py`, code-switch tagging,
`measure_rtf()`, Path B SVD, `report.py`, a standalone `sweep.py`) is either `deferred`
or non-blocking `todo` — pick it up only after D2 passes, and only if time remains
before 2026-08-02.

## Notes

- **Nothing past A1/A3/A4/A5/A6 has touched a GPU yet.** Every claim about `torch`,
  `transformers`, `peft`, `pyarrow`, or `huggingface_hub` in this repo is unverified until
  B1 runs. Say so explicitly rather than marking anything `done` on the strength of local
  syntax-checking alone — CLAUDE.md's "test in the real environment" rule applies here
  even though the environment is Kaggle, not a browser.
- Tier 4's reference (if A2b ever lands) is post-edited PhoWhisper-small output — it
  flatters PhoWhisper checkpoints. See `PROJECT_CORE.md` §4 before reporting any tier-4
  number.
- `unpaid-dataset` / `dataset_by_task` remain unprofiled and out of scope for this
  deadline; `configs/experiment.yaml:data.dataset_path` uses `paid-dataset` only.
