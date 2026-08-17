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
- **`paid-dataset` IS migrated into `Fine_tune_wf/dataset/`** (reversed 2026-07-31 — the
  first pass deferred this as "not on the critical path"; the user asked for it directly).
  `data.dataset_path: dataset/paid-dataset`, 2.0 GB, checksum-verified identical to the
  predecessor repo's copy via `scripts/checksum_dataset.py` → `dataset/CHECKSUMS.txt`
  (2,237 files covering both `paid-dataset` and `real-meetings-bench`; the only file
  under `dataset/` tracked in git). Getting these same bytes onto Kaggle is still a Track
  B concern (zip + attach as a Kaggle Dataset, then verify with `checksum_dataset.py
  --mode verify` before trusting the mounted copy) — the point of CHECKSUMS.txt is
  proving *that* step didn't corrupt anything, not avoiding the copy.
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
| **A1** | **Bootstrap** — `git init`, `.gitignore`, `requirements.txt`, `src/__init__.py`, GitHub remote, migrate `paid-dataset` + `dataset/CHECKSUMS.txt` | CPU | none | repo initialized + pushed, `dataset/paid-dataset/` (2.0 GB, gitignored) + `dataset/CHECKSUMS.txt` (tracked) | `git status` clean; `configs/experiment.yaml` has no path back to `D:/phowhisper-finetune-exp`; `scripts/checksum_dataset.py --mode verify` passes | **done** — repo live at `github.com/egoist-minh/Reworkwhisper-finetune` (private). `paid-dataset` copied in 2026-07-31, checksum-verified byte-identical to the source (spot-checked one file's sha256 by hand too), split resolution re-confirmed against the local copy: 1316/250/236 |
| **A3** | **Config schema** — `src/config.py`, `configs/experiment.yaml` | CPU | A1 | `src/config.py`, `configs/experiment.yaml` | Loads YAML → validated dataclass; rejects unknown keys, bad rank/alpha, lexical filler tokens, tier-4 leak, empty sweep grid | **done** (validated by hand via `python -c`); `tests/test_config.py` **not yet written** — `todo`, do before trusting further config edits |
| **A4** | **Data module** — `src/data.py`: manifest merge, split resolution, 24k→16k resample | CPU | A1, A3 | `src/data.py`, `tests/test_data.py` | Validates real `paid-dataset` manifests: **1316/250/236** train/val/test (corrected from an earlier wrong 1566/250/236 note — verified live, see handoff), val meetings resolve to `[0001, 0002, 0011]`, meeting_id conflict raises | **done** — 4 tests pass against the real predecessor dataset (read-only) |
| **A5** | **Normalization + metrics** — `src/normalize.py` (§6 Stage 4 contract, word→digit), `src/metrics.py` (Levenshtein, CER/WER, bootstrap CI, paired delta CI) | CPU | A1 | `src/normalize.py`, `src/metrics.py`, `tests/test_normalize.py` | word→digit correct on the arithmetic/digit-string cases incl. the `hai ba nghìn`→`23000` bug fixed this session; symmetric normalization verified; CER/WER corpus-level, not per-segment mean; delta CI paired | **done** — 9 tests pass. `code-switch tagging` (syllable-shape regex) and `measure_rtf()` **not built** — deferred, not needed for tiers 1/2/4a |
| **A6** | **LoRA + lambda (Path A only)** — `src/lora.py`: build config, `set_lambda` (filters `LoraLayer`, not `hasattr`), `save_with_lambda` (bakes into `adapter_config.json`, verifies via reload within 1e-5) | CPU (logic) | A1, A3 | `src/lora.py` | No literal rank anywhere (`cfg.lora.rank` only); λ mechanism proven live on Kaggle T4 prior to this session (three routes agree to 0.000e+00) | **done**, but **UNVERIFIED end-to-end in this session** — no torch/peft on this machine. Needs a Kaggle smoke run. Path B (full-FT SVD) is **deferred**, out of scope per this deadline |
| **A2** | **VIVOS (OOD)** — `scripts/fetch_vivos.py`: parquet route (primary) + tarball route (fallback) | CPU (script), needs network | A1 | `scripts/fetch_vivos.py` | Writes `dataset/vivos/{audio/,manifest.jsonl}` in the §4 schema | **in-progress** — script written this session, **UNTESTED** (no `pyarrow`/`huggingface_hub` locally). Run `python scripts/fetch_vivos.py --smoke` on Kaggle next, inspect the printed schema before trusting the manifest |
| **A2b** | **Real-meetings benchmark** — `scripts/ingest_real_bench.py`: ingest `done/` as `data.real_bench_path`, re-segment >30s spans (silence-detected audio splits + proportional text splits, snapped to word boundaries) | CPU | A1 | `dataset/real-meetings-bench/` | Every segment ≤30s; concatenating a re-segmented span's text reproduces the original modulo whitespace; leak guard passes | **done** — ran against the real `done/` source: 196 raw segments → **264** after re-segmenting the 20 that exceeded 30s (real_0001: 146, real_0002: 118), 29,579 chars. 6 tests in `tests/test_ingest_real_bench.py` pass, incl. one against the real 16 kHz wav. `configs/experiment.yaml:data.real_bench_path` now set. **Caveat carried forward**: text/audio correspondence inside a re-segmented span is a *proportional-duration approximation*, not forced alignment — fine for tier 4a's aggregate CER, not for anything claiming word-level timing |
| **A2c** | **Reference audit + freeze** | CPU | A2b | frozen reference v1 | — | **deferred** — the convention-sensitivity check and version freeze aren't required for a first passing run; the six §4 caveats (post-edited-by-PhoWhisper reference, mixed number convention, etc.) already apply and are documented, just not re-verified against the newly re-segmented copy |
| **A7** | **Hub push + prediction evidence** — `src/hub.py:push_adapter` (HF upload, model card with gate CER table, no token in any artifact); `src/gate.py` now returns and writes per-segment predictions (`audit/predictions_{tier}.csv`) at both baseline and gate time | CPU (logic) | A6 | `src/hub.py`, `src/gate.py` predictions | Push gated on `gate_results.json` overall PASS + `hub.push: true`; predictions CSV has segment_id/meeting_id/ref/hyp for every eval split run | **done**, **UNVERIFIED end-to-end** — no `huggingface_hub` push has been exercised in this session. Needs a Kaggle run with `HF_TOKEN` set and `hub.push: true` to confirm |
| **B1** | **Kaggle smoke test** — `python -m src.pipeline --stage smoke` on Kaggle; attach `paid-dataset` as a Kaggle Dataset; verify compat patch, manifest load, split resolve, normalize all run before spending any real GPU time | GPU | A1, A3, A4, A5 | first Kaggle run log | `compat.apply()` neutralizes the torchao probe if needed and prints version table; split_stats prints 1316/250/236; no exception | **done** — 2026-08-01, live on Kaggle T4 |
| **B2** | **Baseline eval** — `python -m src.pipeline --stage baseline`: base model over test + OOD + real-bench → `metrics/baseline.json` + per-split prediction CSVs | GPU | A2, A2b, A6, B1 | `outputs/{run_id}/metrics/baseline.json`, `outputs/{run_id}/audit/predictions_baseline_*.csv` | `cer_test`, `cer_ood`, `cer_real` all present; OOD CER sane vs historical 1.78%; prediction CSVs non-empty | **done** — 2026-08-01. `cer_test=0.1117`, `cer_ood=0.0067` (VIVOS text-column bug fixed first — see Notes), `cer_real=0.4676` |
| **B3** | **Train** — `python -m src.pipeline --stage train`: LoRA SFT (rank 16), early stopping, OOD eval every epoch | GPU | A5, A6, B2 | `checkpoints/best/`, `metrics/training.csv` | Training completes at least 1 epoch without OOM (batch 8 measured to fit in 9.75 GiB on T4); early stopping wired; `training.csv` has per-step loss/LR/val_cer/ood_cer | **done** — 2026-08-01, full 3 epochs, `train_loss=2.388`. Known residual bug: `metric_for_best_model="val_cer"` doesn't match the actual multi-eval-dataset-prefixed key, so `EarlyStoppingCallback` logs "did not find eval_val_cer" and disables itself — harmless this run only because `patience=3 == epochs=3` (early stop is mathematically impossible either way); unverified whether `load_best_model_at_end` shares the same lookup miss. Revisit if a future run uses more epochs |
| **D1** | **Sweep + Gate + Push (combined)** — `python -m src.pipeline --stage sweep-gate`: λ grid eval → `select λ*` (hard-fails if none in budget) → save adapter with λ baked in → gate tiers 1/2/4a (writes `audit/predictions_tier{1,2,4a}.csv`) → HF push if PASS | GPU | A2b, A6, A7, B3 | `metrics/lambda_sweep.csv`, `adapter/`, `metrics/gate_results.json`, `audit/predictions_*.csv`, HF repo (if PASS) | λ* selection matches §6 exactly, raises (not falls back) when no λ fits `sweep.ood_cer_budget`; `save_with_lambda`'s reload-verify passes within 1e-5; gate tiers computed correctly incl. tier 4a on the real bench; predictions CSVs written for all three tiers; HF push only fires on overall PASS and `hub.push: true` | **done** — 2026-08-01. λ sweep found a λ within OOD budget (tier2 pass, 2.00% ≤ 2.6676% bound); tier1 pass (2.28% CER); **tier4a FAIL** (55.81% vs baseline-locked bound 46.76%, zero-tolerance `real_cer_regression_pp: 0.0`) → `overall_pass: false` → automatic HF push correctly did not fire. User made an informed decision to push the adapter manually via `src.hub.push_adapter` outside the gate (model card still shows the honest FAIL table) rather than loosen the gate threshold |
| **D2** | **E2E confirmation** — re-run `sweep-gate` on a config deliberately set to fail one gate tier (e.g. raise `gates.min_improvement_pct`), confirm halt + no push | GPU | D1 | a second `outputs/{run_id}/` showing the halt | Halts with non-zero exit, `gate_results.json` shows the failed tier, no `hub` push attempted | **done, by real accident rather than a deliberate test** — 2026-08-01's actual D1 run failed tier4a for real and correctly wrote `gate_results.json` with `overall_pass: false` and did not auto-push. Satisfies the row's intent (prove the halt+no-push path fires correctly) without a synthetic misconfiguration; revisit with an explicit synthetic-fail test only if time remains |

## Critical path to 2026-08-02

```
A2 (vivos, on Kaggle) ──┐
A2b (done) ─────────────┼──► B1 (smoke) ──► B2 (baseline) ──► B3 (train) ──► D1 (sweep-gate+push) ──► D2 (fail-path check)
A6, A7 already done ────┘
```

**Critical path cleared 2026-08-01** — B1 through D2 all `done` (see rows above). The
deadline deliverable exists: a pipeline that ran end-to-end for real, with an adapter
pushed to HF Hub. Note D1's gate result is `overall_pass: false` (tier 4a real-meeting
bench regressed 9.05pp vs. the zero-tolerance bound) — the push that happened was a
deliberate manual override by the user (`src.hub.push_adapter` called directly, gate
config left untouched), not an automatic PASS. The honest FAIL is recorded in both
`gate_results.json` and the pushed model card.

Everything else in this file (A2c, `tests/test_config.py`, code-switch tagging,
`measure_rtf()`, Path B SVD, `report.py`, a standalone `sweep.py`) is either `deferred`
or non-blocking `todo` — pick it up only if time remains before 2026-08-02.

## Notes

- **B1 through D2 have now run for real on Kaggle T4 (2026-08-01)**, closing out the
  critical path below. Getting there took 9 real bugs, every one found only from an
  actual Kaggle traceback (never from local static review — no torch/transformers/peft
  on the dev machine): (1) VIVOS manifest filename not matching `load_manifests`' glob,
  (2) `ManifestDataset.__getitem__` requiring keys VIVOS records don't have, (3)
  `compat.apply()` only called in `stage_smoke`, (4) `Trainer(remove_unused_columns=True)`
  stripping collator input keys, (5) missing `enable_input_require_grads()` for PEFT +
  gradient checkpointing, (6) `fetch_vivos.py` picking `speaker_id` as the text column
  instead of `sentence` (baseline `cer_ood` read 358% until fixed), (7) an infinite loop
  in `normalize.py:words_to_digits` on a lone zero-filler word like the name "Linh", (8)
  `PeftModel.generate()` rejecting a positional arg `transcribe_batch` passed positionally,
  (9) `_verify_saved_adapter`'s dummy tensor defaulting to fp32 against an fp16 model. See
  memory `kaggle-code-never-works-first-try` — this reconfirms it hard.
- Tier 4's reference (if A2b ever lands) is post-edited PhoWhisper-small output — it
  flatters PhoWhisper checkpoints. See `PROJECT_CORE.md` §4 before reporting any tier-4
  number.
- `unpaid-dataset` / `dataset_by_task` remain unprofiled and out of scope for this
  deadline; `configs/experiment.yaml:data.dataset_path` uses `paid-dataset` only.

## Next round: pipeline fixes for the next fine-tune (post v3-r16 grill, 2026-08-04)

Source: a full grill session on v3-r16's results (`docs/finetune-results-report-v3.md`,
already updated with the λ explanation and forgetting example). Found by manually
inspecting predictions CSVs, not by any existing pipeline tooling — that gap is E5 below.
Scope for this round is **pipeline code only** — fixing v3-r16's own report numbers
(e.g. rescoring real CER after excluding the bad segment in E1) is a separate one-off
task, not tracked here. All rows below are `todo`; user codes them one session at a time.

| # | Session | Env | Depends on | Output | Acceptance | Status |
|---|---------|-----|-----------|--------|------------|--------|
| **E1** | **Ingest duration sanity-check** — `scripts/ingest_real_bench.py`: after building each segment's record, check `len(text) / duration` against a max plausible Vietnamese speech rate; raise (fail loud, don't silently drop) if any segment exceeds it | CPU | A2b | updated `ingest_real_bench.py` | Running ingest against `done/` raises on `real_0002/seg_0074` (measured 475.7 chars/sec vs. the corpus' next-worst 39.5) with a message identifying the segment and both numbers; a `tests/test_ingest_real_bench.py` case with a synthetic over-rate segment triggers it too | **done** — `_check_speech_rate` added, `MAX_CHARS_PER_SEC=60.0` (headroom above corpus' 39.5 worst-genuine, well below the 475.7 bug case). Live run against `done/` confirmed: raises exactly on `real_0002/seg_0074` after real_0001's 146 segments write clean. 8 tests pass incl. 2 new synthetic + 1 real-source repro. Underlying rejoin/proportional-split bug that *causes* seg_0074's bad split is NOT fixed here — out of scope for E1, which only had to detect+raise |
| **E2** | **Cost/benefit λ selection** — `src/pipeline.py:stage_sweep_gate`: replace "largest λ within OOD budget" with an elbow rule (delta_ood_cer / delta_val_cer between adjacent grid points; stop before the ratio jumps past a configurable multiple of the previous step). New field on `src.config.Sweep` (e.g. `elbow_ratio_threshold`), not hardcoded | CPU (logic) | D1 | updated `pipeline.py` + `config.py` | Feeding v3-r16's own `metrics/lambda_sweep.csv` values through the new selection function (pure recompute, no GPU) picks λ=0.5, not λ=1.0 — matches this session's manual cost/benefit finding (ratio jumps ~12x at 0.75, ~31x at 1.0); `tests/test_pipeline.py` covers the selection function directly | **done** — `select_lambda()` added to `pipeline.py`, wired into `stage_sweep_gate` (old inline "largest within budget" tracking removed); `Sweep.elbow_ratio_threshold: float = 10.0` new config field, also added to `configs/experiment.yaml`. Verified against v3-r16's real sweep CSV: step-to-step ratio jump measures 8.9x at λ=0.5 (accepted, under threshold) vs. 12.6x at λ=0.75 (rejected) — confirms the ~12x figure exactly; picks λ=0.5, not 1.0, even though 1.0 alone is still budget-safe (0.0186 < 0.02). Could not reproduce the "~31x at 1.0" figure from the raw CSV under any adjacent-step formula tried (closest: 2.55x jump 0.75→1.0) — likely a rounding/different-basis artifact from the original manual grill analysis; not blocking since it isn't part of the acceptance test. 5 new tests in `tests/test_pipeline.py` pass |
| **E3** | **Wire `audit_conversions` for tier2_ood** — `src/gate.py:run_gate`: same `normalization_check` block already present for tier1/tier4a, added for tier2_ood | CPU (logic), GPU to verify | A7 | updated `gate.py` | A gate run's `gate_results.json` has `tier2_ood.normalization_check` with a `delta_pp`; run against v3-r16's checkpoint to see whether VIVOS's 42/183 (23%) digit-mismatch rate (found this session) is convention-driven or a real listening gap | todo |
| **E4** | **Persist per-λ predictions during sweep** — `src/pipeline.py:stage_sweep_gate`: call `write_predictions` for val/ood after each λ's `_eval_split`, not only for the officially-selected λ | CPU (logic), GPU to verify | D1 | updated `pipeline.py` | After a sweep-gate run, `audit/predictions_{tier1,tier2}_lambda{L}.csv` exists for every `L` in `cfg.sweep.lambdas`, without any extra GPU inference (reuses the eval already run for the sweep, just also writes it) | todo |
| **E5** | **Reusable error-inspection tool** — consolidate this session's scratch analyses (rank predictions by absolute edit-char count, not per-segment rate; EN-loanword-dropped detector; ref-chars/duration sanity) into `scripts/inspect_errors.py`, taking any `predictions_*.csv` | CPU | A7 | new `scripts/inspect_errors.py` | Running it against v3-r16's existing CSVs reproduces this session's numbers: real-bench top segment = 23% of total edit-chars, test-set 83/361 loanword-drop segments, VIVOS digit-mismatch 42/183 | **done, with an honest gap on 2 of 3 numbers** — `scripts/inspect_errors.py` built with `rank_by_edit_chars`/`top_segment_edit_share`, `loanword_dropped_segments` (Vietnamese syllable-shape test, extended — PROJECT_CORE.md §4's literal 1-consonant/1-vowel pattern flags ~99% of real words like "không"/"được" as non-Vietnamese, so digraph onsets + diphthong/triphthong nuclei were added to make it usable at all), `digit_mismatch_segments`. **Digit-mismatch reproduces exactly**: 42/183 on `predictions_tier2_ood.csv` (VIVOS, gate-time). **The other two don't reproduce bit-exact** — this session's original scratch code is gone (summarized-away context), only the numbers survived in this file. Real-bench top-segment share (rejoined to parent-segment level, `real_0002/seg_0074` — the same 4.2s/1998-char segment E1's speech-rate check now catches) measures 19.0%, not 23%. Test-set loanword-drop measures 59/266 (22.2% rate — close to the noted 23%, but different absolute counts, likely a looser original candidate filter). 10 tests in `tests/test_inspect_errors.py` pass (5 synthetic pinning exact behavior, 3 against real v3-r16 CSVs incl. the exact digit-mismatch figure, 1 smoke bound on loanword count) |

Suggested order: **E1** and **E5** first (pure CPU, no GPU queue time, and E5 makes
verifying E2/E3 faster). **E2** next (also pure CPU/logic against already-downloaded
sweep data). **E3** and **E4** need a GPU run to verify, so batch them into the same
Kaggle session as the next real training run's sweep-gate stage.

## YouTube data pilot (2026-08-11) — collection only

What/how-much/why is settled in `youtube-data-pilot/README.md` (6–8 online meetings ×
15–20 min, split by `meeting_id`, ASR draft fully human-reviewed). These rows are the
code for README steps 1–5 (fetch → draft → segment → review → manifest). **Scope is
collection only**: README step 6 (`dataset/mixed-v1/`), steps 7b/7c (code-switch gate
tier, long-form eval), and step 8 (train + gate) are **not** in this round.

Two deltas from that README, decided 2026-08-11:

- **Draft labels come from YouTube auto-captions (Google ASR), not ElevenLabs Scribe.**
  Free, and json3 carries **word-level timing** — which removes the whole bug class
  README step 2 warns about (`done/` has segment-level timing only, so
  `ingest_real_bench.py` splits text *proportionally*, and that approximation is what
  produced `real_0002/seg_0074` at 475.7 chars/sec). Scribe stays swappable at any time
  via `scripts/draft_sources.py` — see F1/F3.
- **Number convention flips to digits for clearly-numeric values** (README §5 said
  words). Measured on `dataset/paid-dataset-v2`: 529/4946 segments (10.7%) contain
  digits, 1022 (20.7%) contain number-words, 183 contain both — e.g. `"tăng từ 200ms lên
  gần 3 giây"`, `"tầm 95% ạ"`, `"delay 24 tiếng"`. That is `05-annotation-guideline.md`'s
  §Numbers rule, and both draft sources emit digits, so words would mean the reviewer
  hand-converting every one. CER is unaffected either way (`number_convention:
  word_to_digit` is applied symmetrically to ref and hyp); the difference is purely
  reviewer keystrokes and a `digits_to_words` helper that does not exist.

F1–F4 all run **local** (CPU + network). Not Kaggle — its network is off by default and
it only consumes the finished, zipped dataset.

| # | Session | Env | Depends on | Output | Acceptance | Status |
|---|---------|-----|-----------|--------|------------|--------|
| **F1** | **Caption probe + draft-source layer** — `scripts/draft_sources.py`: `Word(text, start, end, speaker=None)` plus `parse_json3` / `parse_scribe`, both returning `list[Word]`; everything downstream consumes only that, so swapping labeller is one `--draft-source` flag. `scripts/probe_youtube_captions.py`: for each URL the user supplies, report what `yt-dlp` offers in **both** of its caption dicts — `automatic_captions` (Google speech recognition; the one segmentation uses) and `subtitles` (typed or uploaded by the channel owner) — then pull the Vietnamese json3 from `automatic_captions` and report its shape. No audio download, no dataset written. No channel enumeration: candidate URLs come from the user, so the script screens a supplied list and nothing more | CPU + network | none | `scripts/draft_sources.py`, `scripts/probe_youtube_captions.py`, `youtube-data-pilot/caption-probe.md` | Per URL prints: duration, the **full key list of both caption dicts** plus each entry's `name` field (printed raw — the mechanism for spotting a translated track has to be confirmed against real output, not assumed from a guessed field shape), whether `tOffsetMs` word offsets are present, wpm per 5-min bucket, punctuation rate, digit rate, **lexical-particle rate** (`src.config.LEXICAL_PARTICLES` — this is what decides review effort, and whether Scribe is worth switching to), English-token density per minute (`inspect_errors.is_vietnamese_shaped`), and the exact bracketed sound-label strings (`[Âm nhạc]`…). **Three reject rules, all enforced here rather than discovered after downloading:** (1) no Vietnamese entry in `automatic_captions` → reject; (2) the Vietnamese entry is a **machine translation** of speech in another language rather than the recognition original → reject, because that pairs Vietnamese text with non-Vietnamese audio (YouTube auto-translates its captions into ~100 languages, so a `vi` key alone proves nothing — a 100+ key `automatic_captions` dict is mostly translations); (3) wpm collapses across buckets → reject and investigate. **Known-answer tests:** on `E5dAymt68-0` the `automatic_captions` wpm must be **flat** (last bucket within ±30% of first) because speech recognition does not summarise, while the 169→32 wpm collapse `viet-speech`'s `ground_truth.py:20-26` records comes from the owner-supplied `subtitles`; and any English-spoken video must trip reject rule 2. **When a URL has both dicts populated**, also report the CER between the two full transcripts (each source's text concatenated into one string, `src.normalize.Normalizer` + `src.metrics.score`, no alignment needed) — cheap agreement signal available before any audio is fetched: high agreement means Google heard this audio well and review will be light. Read it together with the wpm buckets, since a summary-style `subtitles` also produces a high CER for an unrelated reason `parse_scribe` **raises** when word-level timing is absent — no proportional-split fallback, since that fallback is exactly the `seg_0074` bug | **done, with two acceptance items that could not be tested and one measured surprise** — `scripts/draft_sources.py` (`Word`, `parse_json3`, `parse_scribe`, `load`, `transcript`; both parsers raise on missing word-level timing) and `scripts/probe_youtube_captions.py` built; report at `youtube-data-pilot/caption-probe.md`. **Reject rule 2's mechanism resolved from real output, not guessed**: the recognition original is a separate key suffixed `-orig` whose entries' `name` ends with `" (Original)"` (`vi-orig` = `"Vietnamese (Original)"`, plain `vi` = `"Vietnamese"`), and `automatic_captions` carries **157 keys**, so a `vi` key alone proves nothing. All 4 supplied URLs expose `vi-orig` → all 4 pass rules 1–2. **Known-answer test 2 passes**: `aircAruvnKk` (English-spoken) rejects with "recognition original is ['en-orig']". **Known-answer test 1 half-passes**: `E5dAymt68-0`'s `automatic_captions` wpm is flat across 11 buckets (184 first → 221 last, +20%, inside ±30%), confirming recognition captions don't summarise; but that video now exposes **`subtitles: {}`**, so the 169→32 wpm collapse `viet-speech`'s `ground_truth.py:20-26` recorded cannot be reproduced from this source — and since **none of the 5 videos probed has `subtitles` populated at all**, the consensus-CER path has synthetic test coverage only. **Measured surprise: the caption endpoint is nondeterministic.** Six fresh extractions of `dGT3YW0AdD8`'s `vi-orig` returned the plain revision 5× (1196 events, 4866 words, 4 commas, `"mãng"`, `"Zo 11"`) and a punctuated, better one 1× (1714 events, 5218 words, 209 commas, `"embedded"`, `"Zero 11"`) — same key, minutes apart; punctuation swings 4.0 → 14.1 per 100 words, so every probe number is a one-fetch snapshot. Re-requesting the signed URL alone does not vary it and hits **HTTP 429** fast (which is why `--attempts` defaults to 1 and 429 is caught with a named message). This is the measured reason F2 must store the exact json3 and F3 must parse the stored file rather than re-fetch. **json3 gives word START times only**: the first seg of each event carries no `tOffsetMs` (implicitly 0 — 5459 segs − 1195 events = 4264 with offsets, exact), and `dDurationMs` is a caption *display* duration that overruns the next event (582/597 neighbouring real events overlap in time), so `Word.end` is inferred as next-start clipped by display-end — **F3 must therefore cut on start-to-start deltas, not on `next.start − cur.end`**. **Answer to question 3 (particle retention):** the drafts keep 0.58–1.67% lexical particles vs **2.87%** in `dataset/real-meetings-bench` (real speech, human-edited, 7,168 words) and **7.12%** in `dataset/paid-dataset-v2` (synthetic, 74,542 words) — a 1.7×–5× shortfall against a reference that is itself a floor (post-edited PhoWhisper-small output), so review must add particles back; non-Vietnamese-shaped words run 8.1% on `dGT3YW0AdD8` vs 8.75% / 7.23% in those two corpora, i.e. code-switch density is already comparable. Rule 3's baseline had to become the first bucket **with speech** — `rCd8DSMk3-c` opens with a near-silent 5 minutes (4 wpm) that otherwise disables the check. `noplaylist: True` is required: URL 2 carries `&list=…&index=8` and without it yt-dlp walks the playlist and dies on a private member (`BWtN8agEenA`). Rate limiting also forced two behaviours: a 429 on the caption endpoint keeps the rules-1–2 verdict and records `draft_error` instead of losing the row (metadata extraction is not what gets limited), and a run where **no** URL yielded measurements refuses to overwrite an existing report — learned by overwriting a good one. 22 offline tests in `tests/test_draft_sources.py` + `tests/test_probe_youtube_captions.py` pass; full suite 95 pass. Not this script's call, but worth flagging for source selection: all 4 URLs are one channel (EngineerPro), 2 are titled "Webinar" — which README §3 rejects as one-presenter talks — and durations are 27–100 min against §2's 15–20 min target |
| **F2** | **Fetch audio + captions** — `scripts/fetch_youtube.py`: read `youtube-data-pilot/sources.jsonl`, download bestaudio, convert to **mono 16 kHz at extraction time**, store raw json3 and `provenance.json`. Needs `yt-dlp` (new, unpinned) + `ffmpeg` (system dep, not pip) | CPU + network + `ffmpeg` | F1 | `dataset/youtube-meetings/raw/<meeting_id>/{audio.wav, captions.json3, provenance.json}` | `soundfile.read` on each wav → `data.ndim == 1` and `sr == 16000` (YouTube audio is stereo by default and `load_audio_16k` **raises** on stereo, `src/data.py:92`); `provenance.json` has all 11 README §5 fields, none null, `label_source: "google_asr"`, `asr_draft_model` recording the real track id; wav `sha256` recorded; a second run **does not re-download** but verifies the checksum and skips; missing `ffmpeg` raises a message naming `ffmpeg` rather than letting yt-dlp dump a trace; **the uncut full file is kept for the 2 test meetings** (tier 4b material, README step 1) | **done** — `scripts/fetch_youtube.py` built and run for real on all 4 sources; `youtube-data-pilot/sources.jsonl` created (4 rows, `meeting_id` = the real YouTube video id, no invented naming). **Every acceptance item verified, none skipped:** all 4 `dataset/youtube-meetings/raw/<meeting_id>/audio.wav` are mono, 16000 Hz (`soundfile.read`); all 4 `provenance.json` carry the 11 README §5 fields with no field null, `label_source: "google_asr"`, `asr_draft_model: "vi-orig"` (the real track key from F1, not a placeholder); recorded `sha256` matches a fresh hash of each wav; a second run against `dGT3YW0AdD8` printed `skip ...: already fetched, checksum verified` and did not re-download (verified both by the offline test suite with a monkeypatched network call, and by a real second CLI run). `require_ffmpeg()` raises naming `ffmpeg` before touching the network when missing — exercised for real, since this machine had no `ffmpeg` at session start; installing it needed `& "$env:LOCALAPPDATA\Microsoft\WindowsApps\winget.exe"` by full path (bare `winget` is not on this shell's PATH even though the alias file exists), and the resulting PATH change lands in the user registry, not the current shell — every later command in this session had to re-export it manually. **Design decision made here, not left ambiguous:** the raw file is the full, uncut download for **every** meeting, not just the 2 that will end up as test — README step 2's "keep the middle" is a step-3 (cut) decision, not a fetch-time one, so storing the uncut file for all 4 is a strict superset of "keep it for the 2 test meetings" and needs no split assignment yet (`val_meetings`/train-test split is still open per §9, safely deferred). `yt_start`/`yt_end` in this raw-level `provenance.json` are `0.0`/full duration (the whole downloaded file), not a trimmed range — per-segment `yt_start`/`yt_end` are a step-5 manifest field, not a step-1 one. `reviewed_by`/`review_date` are written as `""`, not `null`, since no human review has happened yet at fetch time — satisfies "none null" honestly rather than by placeholder. All 4 fetches landed on json3's plain revision (1196/2480/4182/2748 events — identical to F1's probe counts), not the punctuated one F1 also observed; expected per F1's measured nondeterminism, not a bug. 6 new offline tests in `tests/test_fetch_youtube.py` (network entirely monkeypatched); full suite 102 pass. **Cập nhật 2026-08-12 (cùng ngày, sau khi tìm thêm nguồn):** 3 meeting nữa fetch bằng cùng script, không sửa code — `7B24A9GfHAo`, `xKDHjUoUN54`, `3nuCdzuyqng` — đưa tổng lên **7 nguồn**, khớp mục tiêu 6–8 của README §2 (ban đầu chỉ 4, thiếu). `7B24A9GfHAo` và `3nuCdzuyqng`'s draft opening tự xưng `"buổi webinar"` trong tiêu đề/lời mở đầu; người dùng xác nhận trực tiếp cả hai vẫn nhiều người nói thật (không phải một người trình bày như README §3 loại) nên vẫn giữ — quyết định của người dùng, không phải Claude tự suy ra từ chữ "webinar" trong tên |
| **F3** | **Draft → segments + manifest** — `scripts/ingest_youtube.py --draft-source {json3,scribe}`: parse the draft into `list[Word]`, cut at inter-word gaps, write per-segment wavs + `manifest.<meeting_id>.jsonl` with `verified: false` | CPU | F2 | `dataset/youtube-meetings/{audio/,manifest.*.jsonl}`, `tests/test_ingest_youtube.py` | No segment > 30 s, targeting 10–25 s; `_check_speech_rate` clean across the corpus; concatenating one meeting's segment texts **reproduces** the draft text modulo whitespace (the same invariant `ingest_real_bench.py` holds); a synthetic over-rate case **does** raise; non-speech segments dropped by the exact bracket labels F1 found; `resolve_splits` + `split_stats` run on the output without raising; `ManifestDataset.__getitem__` returns mono 16 kHz on a random sample. **Swap test:** a synthetic Scribe fixture through `--draft-source scribe` yields the same segment count as the equivalent json3 fixture, and a Scribe fixture **missing** word timing **raises** instead of falling back | **done** — `scripts/ingest_youtube.py` built and run for real against all 4 of F2's stored raw meetings (no re-fetch, per F1/F2's nondeterminism finding). Cutting reuses `ingest_real_bench.py`'s `_choose_splits`/`_check_speech_rate` verbatim (same `MAX_SEGMENT_SEC`/`MAX_CHARS_PER_SEC`); the only new piece is `gap_candidates`, which feeds it inter-word **start-to-start** gaps (`MIN_GAP_SEC = 0.3`, reusing `ingest_real_bench.MIN_SILENCE_SEC`'s value) instead of RMS silence runs, since `Word.end` is inferred (F1). **Every acceptance item verified on the real corpus, none skipped:** 854 segments across the 4 meetings (105/201/340/208), max duration exactly **30.0 s** (the bound holds, not just "under" it), median 15.6 s, **99.3% land inside the 10–25 s target**, 0.47% run 25–30 s, 0.12% (1 segment) is under 5 s — no `_check_speech_rate` raise anywhere in the real run. Concatenating each meeting's segment texts reproduces `transcript()` of the bracket-filtered draft, word-for-word, verified against all 4 real `captions.json3` files (not just the synthetic fixture). `resolve_splits([...], val_meetings=[])` + `split_stats` run clean on the combined 854 records (all resolve to `train`, since no `val_meetings` is set yet — §9 is still open); `ManifestDataset.__getitem__` returns mono 16 kHz on a real sample. Non-speech words (`[Âm nhạc]`, `[Hắng giọng]` — F1's exact measured strings) are dropped by full-string match before cutting, so a span left with only those has no words and is written to neither manifest nor disk; confirmed no output segment's text contains `"nhạc"` or the sound-label brackets. **Swap test passes on the synthetic fixture only** — same caveat F1 already recorded: there is no ElevenLabs API key here, so `parse_scribe`'s field names (`words`/`text`/`start`/`end`/`type`/`speaker_id`) remain unverified against a real Scribe response; a fixture **missing** `start`/`end` raises `"missing start/end timing"` through the full `ingest_meeting` path, not just at the parser unit level. **Not decided here, left open on purpose:** every record's `split` is written as `"demo"` — this pilot has only 4 meetings, not the 6–8 README §2 planned, so which become `val`/`test` is a real decision, not a mechanical default; deferring it is the same "safe to postpone, only blocks `--stage train`" reasoning §9 already applies to `val_meetings`. 11 new tests in `tests/test_ingest_youtube.py` (7 offline/synthetic, 1 running the real corpus through the full pipeline in a scratch `tmp_path`, not overwriting the real output); full suite 113 pass. **Cập nhật 2026-08-12 (cùng ngày, sau khi F1/F2 thêm 3 nguồn):** chạy lại `ingest_youtube.py` (không sửa code) trên cả **7 meeting** — 1816 segment tổng (335/421/105/201/340/208/206), max vẫn đúng 30.0s, median 15.62s, **99.7% trong 10–25s**. `val_meetings` đã quyết là `rCd8DSMk3-c` (README §9); train/test cho 6 meeting còn lại **vẫn chưa quyết**, không phải việc F3 tự chọn.

**Cập nhật 2026-08-12 (lần 3, cùng ngày) — 2 quyết định mới của người dùng, đã code + chạy thật:**

1. **Trim mỗi meeting về cửa sổ giữa 30 phút** trước khi cắt segment (`TRIM_TARGET_SEC = 1800.0`), bỏ đầu/cuối trên video dài hơn — đúng tinh thần README §2 "lấy đoạn giữa" nhưng 30 phút, không phải 15–20 như bản gốc (quyết định của người dùng, không phải Claude). `middle_window()` + `window_words()` mới trong `ingest_youtube.py`; `raw/<meeting_id>/audio.wav` của F2 **không đổi** (vẫn đủ cho tier 4b), chỉ ảnh hưởng cái gì được cắt thành manifest. `yt_start`/`yt_end` mỗi record vẫn ghi toạ độ **tuyệt đối trong video gốc** (cộng lại offset cửa sổ), không phải toạ độ trong cửa sổ.
2. **Test = 2 meeting khó nhì/khó ba** (`7B24A9GfHAo`, `3nuCdzuyqng`), xếp theo composite rank 3 chỉ số `caption-probe.md` (EN words/min, tỉ lệ non-VN-shaped, particle rate nghịch) trên cả 7 — đúng "test khó hơn train một chút", ngay dưới val (`rCd8DSMk3-c`, khó nhất). `--test-meetings` (mặc định `DEFAULT_TEST_MEETINGS`) mới; `split` giờ ghi `"test"` cho 2 meeting đó, `"demo"` cho 5 còn lại (train + `rCd8DSMk3-c` chờ wire val).

**Số đo thật sau khi chạy lại:** 790 segment tổng (giảm từ 1816, vì đã trim) — 105 (`dGT3YW0AdD8`, không đổi, gốc đã <30 phút) + 113–115 mỗi meeting còn lại (đều bị trim về đúng 30.0 phút). Tổng thời lượng dùng cho manifest: **207.4 phút (≈3.46 giờ)** — so với 234.9 phút của **riêng 4 meeting cũ chưa trim** trước đó, và gần hơn nhiều tới ngân sách README §2 gốc (1.5–2.4 giờ cho 6–8 meeting×15–20p) so với việc dùng nguyên độ dài thô. Max duration segment giờ 22.92s (giảm từ 30.0s đúng biên trước đó — hợp lý vì tổng audio giảm), median 15.64s, 99.6% trong 10–25s. 4 test mới trong `tests/test_ingest_youtube.py` (2 cho `middle_window`, 1 cho `window_words`, 1 cho `is_test`→`split: "test"`); test corpus-thật cập nhật để tính lại theo cửa sổ đã trim. Full suite **117 pass**.

**Bug tìm được và sửa 2026-08-12 (lúc chuẩn bị đưa audio sang cho team, xem F4 "Cập nhật... đưa review sang viet-speech"):** `ingest_meeting` không xoá `audio/<meeting_id>/*.wav` cũ trước khi ghi — mỗi lần chạy lại với số segment ít hơn (thêm trim/đổi cắt) để lại file mồ côi từ lần chạy trước. Đo được thật: `audio/` nặng **889MB** trước sửa, đĩa có 335–421 file/meeting trong khi manifest chỉ còn 105–115 — lệch hoàn toàn. Sửa: `ingest_meeting` xoá hết `seg_*.wav` cũ trong `audio_dir` trước khi ghi segment mới. Chạy lại thật: `audio/` xuống **382MB**, số file mỗi thư mục khớp đúng số record manifest (790 tổng, không hơn không kém). 1 test mới (`test_ingest_meeting_clears_stale_wavs_from_a_prior_run_with_more_segments`) mô phỏng đúng tình huống (10 file giả từ "lần chạy trước", chỉ 2 record thật) — xác nhận không còn file thừa. Full suite **131 pass** |
| **F4** | **Review round-trip + unverified guard** — `scripts/review_youtube.py`: `--emit` writes an **HTML worksheet with a per-segment `<audio>` element** next to the input box (playback in place is the slow part of reviewing, so a CSV is the wrong shape), `--apply` reads corrected text back into the manifest, `--check` is the guard | CPU | F3 | `scripts/review_youtube.py`, `youtube-data-pilot/style-guide.md` | `--emit` then `--apply` changes neither segment count nor any `audio_filepath`; a blank box raises naming the segment (never silently keeps the draft); `--apply` sets `verified: true`, `reviewed_by`, `review_date`; `--check` **raises** while any row still has `verified: false` — mechanising `05-annotation-guideline.md`'s "held-out test set is independently verified in full" instead of relying on discipline; drafts are shown with bracket labels stripped and casing normalised so those never have to be hand-fixed | **done, tooling verified, human review not performed this session** — `scripts/review_youtube.py` (`emit_worksheet`/`build_html`, `apply_corrections`, `check_verified`) and `youtube-data-pilot/style-guide.md` built. **Casing rule resolved from measurement, not guessed**: `dataset/real-meetings-bench` (real speech, already in this repo) is **all-lowercase, no sentence casing at all**, vs. `paid-dataset-v2` (synthetic) which keeps normal written casing — the YouTube pilot is real speech, so `normalize_for_review` lowercases the draft to match, confirmed against a real, measured artifact rather than a hypothetical: `7B24A9GfHAo/seg_0000`'s raw caption capitalises "Là sao Bạn" mid-clause with no preceding punctuation and no proper noun. Bracket-label stripping in the emitted draft reuses the identical `_BRACKETED` regex F1 validated (defensive — F3 already drops whole bracket-label words before segmenting, so none should reach here). **Every acceptance item verified, all offline except emit which also ran for real:** `--emit dGT3YW0AdD8` produced a real 49,352-char HTML with exactly 105 `<audio controls src="file:///...">` elements, one per real segment, each pointing at an existing wav; `--check` run against the real corpus (no code path skipped) raises correctly, naming 790 unreviewed segments train **and** test alike. The full emit→apply→check round trip (segment count and `audio_filepath` unchanged, `verified`/`reviewed_by`/`review_date` set, blank/missing correction raises naming the segment) is verified both on synthetic fixtures and against a **scratch copy** of one real meeting's manifest (never the real files on disk — the real corpus is still untouched, still all `verified: false`). **What this session did not do, and is not claiming to have done:** no human actually listened to any of the 790 segments and typed a real correction — `--apply`'s test fixtures simulate a reviewer who accepts the lowercased draft verbatim, which exercises the tool's mechanics but is not a stand-in for real annotation quality. 13 new tests in `tests/test_review_youtube.py`; full suite 130 pass |

Order: **the probe row (F1) runs first, and its findings can change how the fetch (F2) and
segmentation (F3) rows are written.** It is the cheap one — no audio downloaded — and the
three things it answers are prerequisites for everything after it:

1. Does the video have Vietnamese `automatic_captions`, is that entry the recognition
   original rather than a machine translation, and is the text a transcript rather than a
   summary?
2. Do the json3 events carry per-word `tOffsetMs` offsets? If not, cutting at true word
   boundaries is impossible and the whole `draft_sources.Word` design loses its point.
3. What fraction of Vietnamese lexical particles (`ạ à ừ ơ`) does the draft keep?

Number 3 matters most. If Google's captions strip those particles at a high rate, the
draft violates README step 4 rule 1 directly (`viet-speech` measured **107**
`filler_convention` errors out of ~380 counted), review effort climbs, and that measured
number — not a guess — is what decides whether paying for Scribe is worth it.

## YouTube data report (2026-08-12) — viết báo cáo, không thu thập thêm

Nguồn: một phiên grill về yêu cầu "báo cáo dữ liệu YouTube có biểu đồ phân phối đàng hoàng".
**Scope là viết báo cáo**: không fetch thêm video, không sửa manifest, không chạy review thủ
công, không train, không đo CER/WER. Mọi số ở mục "Số đã đo" dưới đây **đã đo thật trong phiên
grill** — session mới dùng lại, không đo lại.

**Ba chỗ phiên grill đọc sai rồi tự sửa — ghi lại cho khỏi lặp:**

1. `audio/` nặng **382MB**, đúng như bug-fix note ở F3. Con số **1.345 GB** là **cả thư mục**
   `dataset/youtube-meetings/` (818 file): 790 segment wav + 7 `raw/<meeting_id>/audio.wav`
   video đầy đủ (1337.4MB tổng wav) + 7 `captions.json3` (6.8MB) + 7 `provenance.json` + 7
   manifest. Hai con số khác phạm vi, không xung đột. README **không** cũ.
2. Cửa sổ 1800.0s là **một đoạn 30 phút liên tục căn giữa video**, đúng như `TRIM_TARGET_SEC`
   / `middle_window()` mô tả ở update lần 3. `yt_start`/`yt_end` là toạ độ **tuyệt đối trong
   video gốc** nên chạy tới 3550s — đó **không** phải bằng chứng cửa sổ bị đứt đoạn. Kiểm lại
   bằng số học trên cả 7 meeting: tâm cửa sổ khớp tâm video, sai số < 1s.
3. Test chọn theo **composite rank 3 chỉ số** (EN words/min, tỉ lệ non-VN-shaped, particle
   rate nghịch), không phải riêng particle rate — comment `ingest_youtube.py:69-72` chỉ nhắc
   particle rate nên dễ đọc thiếu.

**Quyết định của người dùng trong phiên grill, không tự đổi:** báo cáo **trung tính, không có
mục khuyến nghị** (nêu đủ phép so sánh, không kết luận nên/không nên dùng dữ liệu này); tiếng
Việt; Markdown; **6 biểu đồ PNG**; **không** `stats.json`; chạy **cả hai** phân tích audio
trước khi viết.

**Số đã đo, dùng trực tiếp — không đo lại:**

- Quy mô: 7 meeting (`meeting_id == video_id` mọi row), 790 segment, **207.42 phút = 3.457
  giờ**, **41.210 từ**. Duration: min 3.024s, max 22.920s, mean 15.753s, median 15.640s, **0
  segment vượt `MAX_SEGMENT_SEC`**, 1 segment dưới 5s. Words/segment: 8–97, mean 52.16,
  median 53. Per-meeting duration: 6 meeting đúng 1800.0s, `dGT3YW0AdD8` 1645.0s (video gốc
  chỉ 1645s).
- Split: `test` 228 segment (`3nuCdzuyqng`, `7B24A9GfHAo`, 114 mỗi cái) / `demo` 562 (5
  meeting còn lại). Không meeting nào nằm hai bên.
- Nhãn: `label_source` = `google_asr`, `verified` = `false`, `reviewed_by`/`review_date` = `""`
  cho **cả 790**. `download_date` một giá trị duy nhất `2026-08-12`.
- Code-switch, đo bằng `inspect_errors.is_vietnamese_shaped` + filter của
  `probe_youtube_captions.text_stats`: **2.748 / 41.043 token = 6.70%**, **708 type**;
  **709/790 = 89.7% segment** có ít nhất một. Phân bố per-segment: 0→81, 1→112, 2→138, 3→121,
  4→108, 5+→230 (5→63, 6→63, 7→42, 8→27, 9→13, 10→11, 11→5, 12→3, 13→1, 15→1, 16→1). Trên
  cùng bộ token: lexical-particle **1.22%** (500), digit-bearing **0.53%** (217).
- Top type ≥3 meeting (81 type, 1.365 lần, 3.33% token), dạng `token(count, meetings)`:
  ok(140,6) design(118,7) code(96,7) interview(76,6) system(60,6) team(44,5) level(40,7)
  solution(32,4) google(27,5) coding(26,4) engineer(25,6) grap(24,4) list(24,4) project(23,5)
  test(22,5) interviewer(21,4) cv(21,4) review(20,6) pro(20,4) junior(19,3) bigtech(19,4)
  skill(19,4) backend(18,4) senior(15,5) fail(15,3) th(14,6) master(14,3) scope(14,3)
  requirement(14,3) cs(14,3) software(13,3) network(13,3) technical(12,4) case(12,3) big(11,4)
  round(11,3) time(10,5) focus(10,3) meta(9,3) grab(9,4). `grap(24)` vs `grab(9)` và `th(14)`
  là lỗi nhãn `google_asr` thấy được ngay — dùng làm ví dụ cụ thể ở mục chất lượng nhãn.
- Đối chiếu (đã có từ F1): `paid-dataset-v2` 74.542 từ → non-VN-shaped **7.23%**, particle
  **7.12%**; `dataset/real-meetings-bench` 7.168 từ → **8.75%** / **2.87%**. Tức YouTube pilot
  ở 6.70% / 1.22% là **thấp hơn cả hai** trên cả hai chỉ số.
- Chủ đề: cả 7 tiêu đề (`youtube-data-pilot/urls.txt`) đều là **phỏng vấn tuyển dụng kỹ thuật
  / nghề Big Tech**, cùng hệ sinh thái EngineerPro/MentorPro. Không có trường chủ đề trong 18
  key của record, và không cần gán — corpus chỉ có một chủ đề, nên **không có phân phối chủ đề
  để vẽ**. Top từ vựng ở trên xác nhận (interview/cv/interviewer/junior/senior/bigtech/dsa).

**Môi trường cho G1/G2 — đã dò, không cần dò lại:** `d:\viet-speech\.venv\Scripts\python.exe`,
Python 3.10.9, torch 2.13.0+cu126 **CUDA khả dụng, 1 GPU**, `pyannote.audio` 4.0.7,
`speechbrain`, `torchaudio`, `sklearn`. `HF_TOKEN` trong `d:\viet-speech\.env`. Trọng số gated
đã cache ở `d:\viet-speech\.cache\huggingface\hub`: `pyannote/segmentation-3.0`,
`pyannote/embedding`, `speechbrain/spkrec-ecapa-voxceleb`. **Gotcha đã biết: `torchcodec` cài
lỗi nên đường đọc file mặc định của pyannote sẽ fail** — nạp wav bằng `soundfile`, truyền
`{'waveform': tensor(channel,time), 'sample_rate': int}`, đúng cách warning của nó chỉ ra.
`pyannote.audio` 4.0.7 đã bỏ `OverlappedSpeechDetection`, nên module 2 dẫn overlap từ
`segmentation-3.0` chứ không gọi class đó. **Không sửa file nào trong `d:\viet-speech`; không
cài gì vào Fine_tune_wf** (env local ở đây chỉ có numpy/scipy/sklearn/soundfile/matplotlib,
không có torch) — script phân tích để ở scratchpad, gọi interpreter kia.

| # | Session | Env | Depends on | Output | Acceptance | Status |
|---|---------|-----|-----------|--------|------------|--------|
| **G1** | **Đo chồng tiếng** — chạy module 2 của viet-speech (`backend/adapters/overlap/pyannote_segmentation.py`, class `PyannoteSegmentationOverlapDetector`, model `pyannote/segmentation-3.0`) trên **790 segment wav** ở `dataset/youtube-meetings/audio/<meeting_id>/`. **Không** chạy trên 7 `raw/*/audio.wav`: file raw là video đầy đủ tới 110 phút, corpus chỉ là cửa sổ 30 phút giữa, nên đo trên raw là đo phần lớn nội dung không thuộc corpus; chạy theo segment còn cho gán kết quả về từng `segment_id`, khớp manifest | GPU (`.venv` của viet-speech) | F3 | `scratchpad/overlap.jsonl`, mỗi dòng `{meeting_id, segment_id, duration, overlap_sec, overlap_ratio}` | Đúng **790 dòng**, không segment nào lỗi; in tỷ lệ chồng tiếng từng meeting và toàn corpus. **Đối chiếu sai số công cụ trước khi tin số**: viet-speech đo module 2 ra 16.4% so với ground truth 17.2%, và ghi nhận `pyannote_osd.py` đo vượt lên 36.9%. Nếu corpus ra gần 0% hoặc gần 40% thì dừng kiểm lại cách gọi, đừng ghi vào báo cáo. **Đọc `pyannote_segmentation.py` trước khi viết** — chữ ký hàm và tên class chưa được xác nhận, không đoán | **done — gate trip là BÁO ĐỘNG GIẢ, đã truy ra nguyên nhân.** 790/790 dòng, 0 lỗi. Số đầu tiên 0.89% (giây chồng tiếng / thời lượng clip) trip gate "gần 0%". Tự kiểm chứng sau đó tìm ra: số tham chiếu 16.4%/17.2% do `experiments/real-domain-gap-diagnosis/osd_rate.py` sinh ra bằng **công thức khác hẳn** — `OverlapLabel.overlapping` là **boolean**, nên tử số của nó là *thời lượng đơn vị VAD bị gắn cờ*, không phải số giây chồng tiếng. Hai đại lượng khác nhau, gate so nhầm loại. Chạy lại đúng công thức tham chiếu (silero-VAD trong từng clip, 3217 đơn vị / 10366s, gắn cờ theo cùng timeline): **corpus 10.57%**, per-meeting 3.73%–21.39%, trong đó `rIFrrmm8ILY` 21.39% và `dGT3YW0AdD8` 17.12% **vượt cả ground truth 17.2%**. Cùng bậc độ lớn với tham chiếu, không hề "gần 0". Cũng đã bác 2 nghi vấn khác: đuôi clip **không** bị bỏ sót (Inference phủ tới 19.98s, 0s mất), và re-implementation khớp `detect()` công khai trên clip mẫu. Báo cáo đưa **cả hai** chỉ số kèm giải thích. `scratchpad/{overlap.jsonl, apples.json}` |
| **G2** | **Đo trùng người nói giữa `test` và `demo`** — ECAPA embedding qua `backend/adapters/speaker_embedding/ecapa.py` (model `speechbrain/spkrec-ecapa-voxceleb`), so cosine theo cách của `backend/core/speaker_id.py`. **Không** dùng embedder MFCC mặc định của `SpeakerStore`. Lấy mẫu **40 segment ít chồng tiếng nhất mỗi meeting** dựa trên G1 — đây là cách giảm trực tiếp nhiễu "segment 15.75s có thể lẫn nhiều giọng" | GPU (`.venv` của viet-speech) | G1 | `scratchpad/speaker_sim.json`: ma trận cosine 7×7 (mean + max mỗi cặp meeting) + kết quả gom cụm agglomerative trên 280 embedding kèm nhãn meeting mỗi cụm | Ma trận 7×7 đầy đủ, không ô lỗi. **Cổng bắt buộc: tương đồng trong cùng một meeting phải cao hơn tương đồng giữa hai meeting khác nhau.** Không đạt = embedder không hoạt động → **bỏ hẳn mục 10 và biểu đồ 6, ghi thành câu hỏi mở**, không đưa số không đáng tin vào báo cáo. Đọc kết quả: cặp `test`–`demo` có cosine cao bất thường, hoặc cụm chứa cả segment `test` và `demo`, là dấu hiệu trùng người nói. Hai meeting `test` là `3nuCdzuyqng` và `7B24A9GfHAo`. **Rủi ro đã biết: người nói chưa từng là tiêu chí chia split** (chỉ có độ khó), nên rủi ro này không được thiết kế ngăn — đúng lỗi `voice_id` overlap mà `CLAUDE.md` ghi nhận ở `paid-dataset` v1 | **done, cổng kiểm ĐẠT** — within-meeting 0.5411 > cross-meeting 0.3160. Ma trận 7×7 đủ, 0 ô lỗi. Phát hiện: `xKDHjUoUN54` (demo) × `3nuCdzuyqng` (test) = 0.4899, cặp cross cao nhất; 3/18 cụm chứa cả segment test lẫn `xKDHjUoUN54`. **Tự kiểm chứng phát hiện lỗi chọn mẫu**: 49–103 segment/meeting có overlap đúng 0 nên sort ổn định hoà theo thứ tự manifest → mẫu "40 ít chồng tiếng nhất" thật ra là ~40 segment ĐẦU theo thời gian (idx 0–43 ở 4/7 meeting), không rải đều. Chạy lại với mẫu ngẫu nhiên seed 42: within 0.5389 > cross 0.3101 (vẫn đạt), `3nuCdzuyqng`×`xKDHjUoUN54` vẫn đứng đầu (0.4381) — kết luận bền, không do cách chọn mẫu. `scratchpad/{speaker_sim.json, speaker_sim_random.json}` |
| **G3** | **Script sinh biểu đồ** — `scripts/plot_youtube_stats.py`, chạy bằng Python của Fine_tune_wf (chỉ `numpy` + `matplotlib`, **không cần torch**). Đọc 7 manifest; dùng lại `is_vietnamese_shaped` từ `scripts/inspect_errors.py` và `LEXICAL_PARTICLES` từ `src/config.py` — **không tự viết bộ nhận diện, không dùng whitelist từ tiếng Anh** (`CLAUDE.md` ghi nhận whitelist có 72% dương tính giả). Kết quả G1/G2 nhúng thành **hằng số đầu file kèm chú thích ghi ngày đo, model, số mẫu** — vì người dùng đã bỏ `stats.json` và `dataset/` thì bị `.gitignore` chặn, nên nếu không nhúng thì chạy lại script sau này sẽ lỗi ở 2 biểu đồ cuối | CPU | G1, G2 | `scripts/plot_youtube_stats.py` + 6 PNG trong `docs/youtube-data-charts/`: `codeswitch-per-segment` (histogram, bin 0→16) · `codeswitch-vocab` (bar ngang, top 25, kèm số meeting mỗi từ xuất hiện) · `corpus-comparison` (grouped bar, 3 corpus × 2 chỉ số) · `segment-duration` (histogram, bin 1s, 3–23s) · `overlap-per-meeting` (bar 7 meeting + đường tham chiếu 16.4%/17.2%) · `speaker-similarity` (heatmap 7×7, tô viền 2 meeting `test`) | Sinh đúng 6 file. Script in ra các số then chốt và chúng **khớp mục "Số đã đo"** ở trên: 6.70%, 89.7%, 41.210 từ, 207.42 phút, 790 segment. Nếu lệch, script sai — không sửa số trong SESSIONS.md để khớp script | **done** — `scripts/plot_youtube_stats.py` + 6 PNG. In số khớp: 790 segment, 207.42 phút, 41.210 từ, 89.7% segment code-switch, đúng distribution 0→16 và top-25 vocab (kiểm từng số, khớp 100%). Riêng "6.70%" lệch thành 6.67% — tử số 2.748 khớp chính xác nhưng mẫu SESSIONS ghi 41.043, không dựng lại được từ manifest hiện tại (41.210 từ thật); phương pháp đã tự kiểm chứng đúng vì áp lên paid-dataset-v2/real-meetings-bench ra đúng 7.23%/7.12% và 8.75%/2.87% khớp 100% — kết luận 41.043 là số cũ còn sót trong ghi chú grill, không sửa SESSIONS, ghi caveat trong output script. Phát hiện thêm: phân phối duration cực hẹp, 739/790 (93.5%) nằm trong 15–17s — có thật (kiểm bằng percentile), không phải lỗi bin |
| **G4** | **Viết báo cáo** — `docs/youtube-data-report.md`, tiếng Việt, 11 mục: 1 tóm tắt · 2 nguồn (7 video kèm tiêu đề, 3 luật sàng lọc của F1, kết quả 7 nhận/0 loại) · 3 kiểm kê · 4 quy trình (5 script; cửa sổ 30 phút liên tục căn giữa; **điểm cắt lấy từ khoảng hở giữa thời điểm bắt đầu các từ trong caption, không từ năng lượng audio** — nên biên segment phụ thuộc độ chính xác caption Google) · 5 cách chia test/demo + **bảng bốn tầng độc lập** (xem dưới) · 6 code-switch + biểu đồ 1,2 · 7 đối chiếu 3 corpus + biểu đồ 3 · 8 đặc tính segment + biểu đồ 4 · 9 chồng tiếng + biểu đồ 5 · 10 trùng người nói + biểu đồ 6 · 11 trạng thái nhãn + **vấn đề nhãn cùng nguồn** + phần chưa thực hiện. **Không mục khuyến nghị** | CPU | G3 | `docs/youtube-data-report.md` | Mỗi con số trong báo cáo chỉ được về **một file cụ thể trong repo hoặc một lần đo ở G1–G3**. Không câu nào phát biểu điều dữ liệu không đo được. Mục 10 **phải** có đủ 3 giới hạn: segment mean 15.75s có thể lẫn nhiều giọng; không có nhãn định danh nên chỉ kết luận được "giọng meeting X giống giọng meeting Y"; ngưỡng tương đồng do Claude chọn nên đưa cả ma trận thay vì một câu có/không. Mục 11 nói thật: **0/790 `verified`**, 1 worksheet đã emit (`review/review.dGT3YW0AdD8.html`, 105 segment), **0 file `corrections.*.json`** — chưa ai review; CER/WER chưa đo vì cần một `predictions_*.csv` mà `inspect_errors.py` chưa có gì để đọc | **done** — `docs/youtube-data-report.md`, 11 mục, tiếng Việt, trung tính, không mục khuyến nghị. Mục 10 đủ 3 giới hạn. Mục 11 nói thật 0/790 verified + 0 corrections + CER/WER chưa đo. Đã nêu điểm lệch `rCd8DSMk3-c` val/demo ở cuối mục 11 |

**Bốn tầng độc lập train/test — chỉ một tầng đạt.** Phát biểu đúng là test đo *"cùng chủ đề,
cùng từ vựng, giọng chưa rõ, session khác"*, **không** phải một tập test độc lập:

| Tầng | Trạng thái | Bằng chứng |
|---|---|---|
| `meeting_id` / file audio | **đạt** | test = `3nuCdzuyqng`, `7B24A9GfHAo`; demo = 5 meeting còn lại; không meeting nào nằm hai bên, kiểm trên cả 790 record |
| Người nói | **chưa kiểm** | không có trường `speaker` trong 18 key; cả 7 video cùng hệ sinh thái EngineerPro/MentorPro nên rất có thể cùng người dẫn lặp lại — đây đúng là việc của G2 |
| Chủ đề | **trùng hoàn toàn** | cả 7 tiêu đề đều là phỏng vấn tuyển dụng kỹ thuật; đã đo, không phải nghi vấn |
| Từ vựng | **trùng, đo được** | `design`, `code`, `level` xuất hiện ở **cả 7 meeting** nên chắc chắn có ở cả hai bên; 81 type xuất hiện ở ≥3 meeting |

**Vấn đề nặng hơn cả bốn tầng trên, đúng bất kể chia split thế nào:** nhãn của cả hai bên đến
từ **cùng một nguồn `google_asr`**, và **0/790 segment đã được người kiểm**. Fine-tune trên nhãn
`google_asr` rồi đo CER trên nhãn `google_asr` là đo *"model học nhái Google ASR tốt tới đâu"*,
không đo độ chính xác phiên âm — model phiên âm ra `grap` thay vì `Grab` được **0 lỗi**, vì tham
chiếu cũng ghi `grap`. Lỗi hệ thống của Google có ở cả hai bên nên triệt tiêu nhau thay vì bị
phát hiện. Corpus **không có ground truth ở đâu cả**, nên F4 (review thủ công) không phải việc
làm cho đẹp quy trình: chưa có nó thì mọi con số CER trên corpus này không đọc được. `grap(24)`
so với `grab(9)`, và `th(14)` ở 6 meeting, là bằng chứng lỗi đó đang tồn tại trong nhãn.

**Một chỗ đang lệch trạng thái, không phải trung lập:** `rCd8DSMk3-c` đã quyết là val meeting
(README §9) nhưng manifest ghi `split: "demo"` cho nó, và `configs/experiment.yaml:data.val_meetings`
vẫn chỉ trỏ vào ba meeting của `paid-dataset-v2`. Merge manifest YouTube vào training hôm nay thì
meeting dự định làm val sẽ nằm trong train. Việc wire còn treo — không thuộc scope G1–G4, nhưng
báo cáo phải nói ra.

Thứ tự bắt buộc là **G1 → G2 → G3 → G4**: G2 chọn mẫu dựa trên output của G1, G3 cần hằng số
từ cả hai, G4 nhúng PNG của G3. Không có row nào cần Kaggle.

**Hai cổng dừng, không được đi vòng:** (1) nếu cách vòng `soundfile` cho `torchcodec` không
chạy, dừng và báo — **không tự đổi sang model khác**; (2) nếu cổng kiểm G2 thất bại, bỏ mục 10
+ biểu đồ 6 và ghi thành câu hỏi mở, **không** hạ ngưỡng cho tới khi ra kết quả đẹp.

## Hồi quy Reworkwhisper-large-v5 trong production (2026-08-17)

**Triệu chứng.** Người dùng đổi model ASR production từ `reworkwhisper-large-v4-remote` sang
`reworkwhisper-large-v5-remote` và pipeline 8 module ở `d:\viet-speech` cho kết quả **tệ hơn**.
Hai run so được vì chỉ khác model ASR (module 1–3, 5–6 hành vi trùng, điểm định danh giống hệt
0.7856 / 0.7341 / 0.8433): `artifacts/web-1786700346805/` (v4) và `artifacts/web-1786942786371/`
(v5). Trên `04_asr.json`: chữ hoa **15 → 1**, `team` 7 → 1 (thành `tim` ×5), `build` 2 → 0
(thành `bill` ×2), `developer`/`standard`/`realize`/`nus` 1/1/1/2 → **0/0/0/0**. Timestamp lành
mạnh ở cả hai run, nên **không phải** truncation hay lỗi decode timestamp. Module 8 mất chức
danh (Backend Engineer, TL) và lộ trình nghề Singapore/Shopee/Canada→US.

> **Đọc triệu chứng trên cho đúng những gì nó là (sửa 2026-08-17, sau H3).** Toàn bộ phép so
> sánh production là **đếm token và chữ hoa, không có reference nào**. Nó kết luận "tệ hơn" dựa
> trên giả định ngầm rằng những từ tiếng Anh đó ở v4 là **đúng** — nhưng không ai phiên âm audio
> production, nên nếu v4 hallucinate `team` ở chỗ người nói không nói `team` thì nhiều từ tiếng
> Anh hơn không có nghĩa là tốt hơn. Phép đo này chứng minh hai model **khác nhau**, chưa chứng
> minh v5 **tệ hơn**. Mọi row dưới phải giữ đúng phân biệt đó.

**Nguyên nhân chắc chắn — quy trình, không phải weights (kết luận 2026-08-17).**
Gate **không bao giờ so run mới với model đang chạy production**; nó chỉ so mỗi run với baseline
của base model. Nên v3-r16 và v4-mixed-r16 đều `overall_pass: true` trên tiêu chí riêng, trong
khi trên **đúng 426 segment synthetic dùng chung, `ref` trùng từng byte**, CER của chúng là
**0.0171 (v3-r16) so với 0.0258 (v4-mixed-r16)** — tệ hơn **51% tương đối**. Không phải nhiễu,
đo được từ artifact có sẵn, và **không có cơ chế nào trong pipeline có thể phát hiện**. Đây là
lỗi đã cho v5 ra production, và nó đúng bất kể nguyên nhân hành vi là gì → row **H4**.

**Hành vi của weights là trade-off, không phải bug.** v4 train 100% synthetic TTS; v5 train ~60%
YouTube speech thật + 40% synthetic. v5 tệ hơn trên synthetic, **tốt hơn** trên
`real-meetings-bench` (H3). Đúng như dự đoán: model bị kéo về phân phối của nguồn train chiếm đa
số. Đó là cái giá đã trả để được speech thật, không phải thứ cần "sửa".

**Ánh xạ tên.** `Reworkwhisper-large-v5` = run `v4-mixed-r16` (λ=0.25) · `Reworkwhisper-large-v4`
= run `v3-r16` (λ=0.5). Tên repo HF lệch một bậc so với run id — đã ghi trong memory, không phải
lỗi đánh máy.

**Đã loại trừ, không điều tra lại.**
- *Preprocessing borrowed từ PhoWhisper-large*: giả thuyết của người dùng, đã bị bác ở phiên
  trước (handoff). `processor_config.json` khớp `preprocessor_config.json` của
  `vinai/PhoWhisper-large` từng giá trị; `generation_config.json` khớp byte-for-byte trừ
  `transformers_version`; `tokenizer.json` là whisper-large-v2 gốc, **0 added token**. Thiếu
  `preprocessor_config.json` chỉ vì transformers 5.0.0 đổi tên file.
- *Nhãn YouTube chưa soát*: **sai, đã bác phiên này.** Cả 790 record ghi `verified: true`,
  `reviewed_by: Quang`, `review_date: 2026-08-13/14`. Và
  [scripts/build_mixed_dataset.py](scripts/build_mixed_dataset.py) có gate raise nếu bất kỳ
  record nào `verified != true`, nên `mixed-noisy-v1` build được đã chứng minh nhãn đã soát.
  Sai sót gốc: đọc `docs/youtube-data-report.md` mục 11 (viết **trước** khi soát xong) thay vì
  đo manifest. **Bài học áp cho mọi row dưới: đọc số từ artifact, không từ tài liệu mô tả
  artifact.**
- *λ / rank*: v5 dùng λ=0.25, **thấp hơn** v4 (λ=0.5), tức ít adapter hơn mà vẫn tệ hơn trên
  synthetic. Hạ λ tiếp chỉ tiến về baseline. *(Tỉ lệ trộn thì KHÔNG loại trừ — xem D3.)*
- *Chất lượng nội dung nhãn YouTube*: **đã đo 2026-08-17, nhãn tốt, không cần soát lại.** So token
  không mang hình dạng tiếng Việt giữa `raw/*/captions.json3` và nhãn đã soát, khớp theo khoảng
  `yt_start`/`yt_end` trên cả 790 segment: người soát **bỏ 491** token lỗi của Google (`grap` 21,
  `prom` 10, `btech` 5, cùng rác vụn `th`/`ph`/`nh`/`ch`) và **thêm 788** token tiếng Anh đúng
  chính tả (`system` 38, `grab` 34, `leetcode` 34, `design` 21, `backend` 14, `engineer` 13).
  Quy ước 4 của style guide được tuân thủ. Việc sửa data còn lại **chỉ là casing**, và hẹp hơn dự
  đoán ban đầu: 788 token người soát tự gõ **không có nguồn casing** trong `captions.json3`, mà
  phần lớn chúng vốn viết thường trong tiếng Anh đúng (`system`, `design`, `backend`, `mock`,
  `behavior`) — chỉ tên riêng cần hoa (`Grab`, `LeetCode`, `Shopee`, `Axon`), ước 20–40 mục.

**Casing — truy được đến dòng code, nhưng độ lớn nhỏ, đừng đề cao quá.** v4 train trên `paid-dataset-v2` (đọc
`Outputs/v3-r16/config.json`), v5 trên `mixed-noisy-v1` = paid-dataset-v2 + `youtube-meetings`
(đọc `experiments/v4-mixed-r16/config.json`). Nhãn YouTube có **0 chữ hoa trên 178.075 ký tự**
(đo trực tiếp 7 manifest), so với 6.610/333.272 của paid-dataset-v2 — và youtube chiếm ~60% ký
tự (`by_source` trong gate: 49.260 vs 32.329). Nguồn: `normalize_for_review` tại
[scripts/review_youtube.py:78](scripts/review_youtube.py#L78) là `text.lower()` áp cho toàn bộ.
Đây là **quy ước cố ý**, ghi rõ trong [youtube-data-pilot/style-guide.md](youtube-data-pilot/style-guide.md)
mục Casing, lý do là caption Google viết hoa ngẫu nhiên giữa câu — hợp lý cho CER, nhưng chính
nó làm module 8 mất thực thể. **Giới hạn của giả thuyết này:** v4 cũng chỉ xuất **15 ký tự hoa
trên 8.515**, tức bản thân v4 gần như không viết hoa. Chênh 15 so với 1 là thật nhưng số tuyệt
đối rất nhỏ — casing khó là toàn bộ nguyên nhân, và không được viết như thể nó là.

**Vì sao gate báo `overall_pass: true`.** Bốn điểm mù. Điểm 4 là điểm nặng nhất và mới tìm ra;
ba điểm đầu đều mang tính định nghĩa chứ không phải ngưỡng đặt sai:
1. `normalization.lowercase: true` và `Normalizer` áp cho **cả hyp lẫn ref**
   ([src/train.py:94-95](src/train.py#L94-L95)) → 14 ký tự hoa bị mất đóng góp **đúng 0** vào CER.
2. CER cân đều mọi ký tự. Toàn bộ khác biệt handoff đo được ước chừng 30–36 ký tự edit distance
   trên 8.515 → **~0,4pp**, nằm gọn trong bề rộng CI tier1-youtube 1,35pp (0,0701–0,0836).
   *(Số tính tay từ bảng handoff, không phải chạy metric — đúng bậc độ lớn, đừng trích như số đo.)*
3. **Gate code-switch chưa từng được xây.** `style-guide.md` quy ước 4 tuyên bố chính tả tiếng
   Anh là "ground truth cho gate code-switch (README bước 7b)", nhưng `src/gate.py` grep
   `code_switch|english` ra **0 kết quả**. Gate chỉ có tier1 / tier2 OOD / tier4a, toàn bộ CER.
4. **Không có phép so nào giữa run mới và model đang chạy production.** Mọi tier so với baseline
   của base model. Hai run cách nhau 51% tương đối về CER trên cùng 426 segment mà cả hai vẫn
   `pass`. Ba điểm trên là điểm mù về *chỉ số*; điểm này là điểm mù về *đối tượng so sánh*, và nó
   là thứ duy nhất đúng bất kể nguyên nhân hành vi là gì → **H4**.

**Nghi vấn còn mở — đây là thứ có thể làm chẩn đoán trên sập.** Xếp theo mức nguy hiểm:

| # | Nghi vấn | Nếu đúng thì sao | Row xử lý |
|---|---|---|---|
| N1 | **Chưa chứng minh v5 được publish bằng `scripts/merge_and_push.py`.** Bằng chứng duy nhất là docstring [merge_and_push.py:24](scripts/merge_and_push.py#L24) ghi `--adapter winhsss/Reworkwhisper-large-v5` — đó là *ý định*, không phải *log thực thi* | Nếu v5 publish bằng notebook cũ hoặc bằng tay, toàn bộ phần loại trừ NF4-merge và double-merge **mất hiệu lực**, và giả thuyết merge của handoff sống lại | **H1** |
| N2 | **Chưa giải thích được cơ chế `team`→`tim`.** Casing truy được đến dòng code; từ mượn thì không: nhãn chứa `team` **32 lần**, `tim` **0 lần**, và quy ước 4 bảo vệ chính tả tiếng Anh | Chỉ có tương quan "cùng xuất hiện ở v5", cộng một cơ chế chưa kiểm chứng (mất casing làm mất mỏ neo chính tả). Nếu cả H2 và H5 đều không phục hồi được từ mượn thì nguyên nhân nằm ở chỗ chưa nhìn tới | **H2** phân đôi |
| N3 | **Chưa tự đo số nào của v4/v5 trên audio production**, mọi số so sánh đến từ handoff. **Phần gán run→model đã ĐÓNG 2026-08-17:** `04_asr.json` ghi `model_id` lồng trong từng speaker (`web-1786700346805` = `reworkwhisper-large-v4-remote`, `web-1786942786371` = `reworkwhisper-large-v5-remote`, khớp `07_transcript.json:asr_model_id`). Nhãn đó do client dán ([remote.py:79](d:/viet-speech/backend/adapters/asr/remote.py#L79)), không echo từ server, và hai id trỏ cùng URL ngrok — nhưng đổi id ở client **không** đổi weights, mà output khác thật (8515→8608 ký tự), nên weights trên server **đã đổi** giữa hai run. Phép so sánh hợp lệ. *(Ghi nhận sai sót: lượt trước tôi báo "không ghi model id ở đâu cả" — script chỉ in key top-level nên bỏ sót key lồng. Lỗi đo, không phải thiếu sót artifact.)* | Dư một khả năng nhỏ: server có thể đã load một model thứ ba. Xác suất thấp, signature khớp đúng dữ liệu train của `v4-mixed-r16` | **H2** vẫn đo lại 4 chỉ số để độc lập với handoff |

**N4 — nghi vấn mới, do chính H3 sinh ra (2026-08-17). Nặng hơn N2.** Chỉ số retention cho
phán quyết **trái chiều theo domain**: v5 tệ hơn v4 trên synthetic (0.7599 vs 0.8643) nhưng
**tốt hơn** trên real-meetings-bench (0.4545 vs 0.4083), và CER cùng chiều đó (tier4a 264 chunk:
0.4395 vs 0.4801). Audio production là YouTube họp thật, **gần tier4a hơn tier1-synthetic** —
nên nếu tier4a nói v5 tốt hơn mà production nói v5 tệ hơn thì có mâu thuẫn chưa giải thích được.

Ba cách hoà giải, chưa cái nào được kiểm:
1. **Casing** — vẫn là giải thích duy nhất nhất quán với cả ba nguồn số, chính vì **không bộ test
   nào đo được nó** (ref của mọi tier đều đã hạ chữ thường, xem H3). Nếu tổn thất production
   nằm gần hết ở casing thì tier4a "tốt hơn" và production "tệ hơn" không xung đột.
2. **Ref của `real-meetings-bench` không phải ground truth thật** — theo `CLAUDE.md` nó là bản
   post-edit output PhoWhisper-small, nên retention ở tier4a đo mức khớp với một nguồn ASR khác.
3. **Domain production khác cả hai** — YouTube tuyển dụng kỹ thuật, cùng domain với dữ liệu train
   `youtube-meetings`, nên đáng lẽ v5 phải *tốt hơn* ở đó. Chưa có bộ test nào đại diện nó.

**Hệ quả cho plan:** H2 quan trọng hơn trước, và phần (a) — tự đo lại trên audio production —
không còn là thủ tục kiểm chứng handoff mà là **phép đo duy nhất đứng cùng phía với triệu chứng**.

**Môi trường — đã dò phiên này, không dò lại.** `HF_TOKEN` **không** có trong shell và
`huggingface_hub` **không** cài trong `/c/Program Files/Python310/python`. Dùng
`d:\viet-speech\.venv\Scripts\python.exe` (`huggingface_hub` 1.23.0) + `HF_TOKEN` trong
`d:\viet-speech\.env`. Repo HF `winhsss/Reworkwhisper-large-v4` và `-v5` là **private/gated** —
`curl` tới `/raw/main/` trả HTTP 401. `dataset/mixed-noisy-v1` **không** có trên máy này (build
trên Kaggle); artifact đầy đủ nằm trong `Outputs/outputs_v4-mixed-r16.zip`. Console cp1252 →
đặt `PYTHONIOENCODING=utf-8` cho mọi script in tiếng Việt.

| # | Session | Env | Depends on | Output | Acceptance | Status |
|---|---------|-----|-----------|--------|------------|--------|
| **H0** | **Rollback production** — ~~đổi `config/models.yaml` ở `d:\viet-speech` về `reworkwhisper-large-v4-remote`~~. **Tiền đề sai, đã sửa 2026-08-17 sau khi đọc file:** `config/models.yaml` **không** chọn v5 ở đâu cả. `default: true` đã nằm trên `reworkwhisper-large-v4` (entry cuda, dòng 72); không entry `-remote` nào có `default`. Nặng hơn: `reworkwhisper-large-v4-remote` (dòng 108) và `reworkwhisper-large-v5-remote` (dòng 126) trỏ **cùng một URL ngrok** `https://eskimo-grandma-copious.ngrok-free.dev`, nên model thật sự chạy do `ASR_MODEL_SIZE` phía Kaggle quyết định ([remote_asr_server.py:113](d:/viet-speech/scripts/remote_asr_server.py#L113)), **không** do id trong config. Sửa models.yaml một mình **không đổi được model nào**. Rollback thật = khởi động lại server Kaggle với `ASR_MODEL_SIZE=winhsss/Reworkwhisper-large-v4` (và URL ngrok kia đã chết theo handoff). **Đã truy tiếp qua N3: cả hai phía đều được đổi khi lên v5** — client đổi id, server đổi `ASR_MODEL_SIZE` — nên rollback là **thao tác vận hành trên session Kaggle của người dùng, không phải sửa code**. Không có file nào trong repo cần đổi | — | Người dùng khởi động lại server Kaggle | Server load v4 | **not-my-action.** Không sửa `models.yaml`: `default: true` đã ở v4 và đổi id client không đổi được weights. Không xoá entry v5 (H2 cần dùng). Không sửa URL ngrok đã chết | **user action** |
| **H1** | **Đóng nghi vấn N1** — xác minh repo HF `winhsss/Reworkwhisper-large-v5` do `scripts/merge_and_push.py` tạo. Dấu hiệu kiểm được: model card mà script này viết ([merge_and_push.py:141-176](scripts/merge_and_push.py#L141-L176)) chứa dòng ``Pipeline commit `<sha>`, run `v4-mixed-r16` `` **và** bảng gate 3 tier | CPU + `.venv` của viet-speech | — | Ghi chú kết quả vào row này | **Cổng phân nhánh.** Có dòng provenance + bảng gate → N1 đóng. Không có → dừng, thêm row đo delta-norm per-layer | **done — N1 ĐÓNG.** README của v5 ghi đúng ``Pipeline commit `9fea278`, run `v4-mixed-r16` `` (9fea278 = HEAD hiện tại) + bảng gate 3 tier khớp `experiments/v4-mixed-r16/metrics/gate_results.json` (0.0564 / 0.0226 / 0.2890). **Merge từ NF4 và double-merge bị loại trừ dứt điểm.** 8 file, `adapter_*` đã bị `--delete-remote-adapter` xoá, 2 commit (initial 08-16 14:18, upload 08-17 04:04). Kiểm thêm và **bác** một lead mới: `generation_config.json` của v4 và v5 **giống hệt nhau**, nên `forced_decoder_ids` thiếu `language`/`task` mà handoff nêu đúng cho **cả hai** và không giải thích được khác biệt — đường decode đồng nhất. `config.json` chỉ lệch `dtype` fp16→fp32 (cố ý, [merge_and_push.py:8-12](scripts/merge_and_push.py#L8-L12)) và `tie_word_embeddings` False→True (v5 khớp mặc định base; khác transformers version lúc lưu). Ghi nhận phụ: v4 publish bằng đường khác (README của nó là bảng gate trần của `push_adapter`, repo còn cả `adapter_*`), v4 `private=False gated=manual` vs v5 `private=True` | done |
| **H6** | **PHÉP ĐO DỨT ĐIỂM, chạy trước mọi row khác còn lại** — chạy **v3-r16 (=HF v4)** trên đúng **228 segment `source: youtube`** của tier1 test, rồi so với con số v4-mixed-r16 (=HF v5) đã có: **CER 0.0764**, retention (H3). Đây là bộ test duy nhất vừa là **speech thật**, vừa có **nhãn người soát**, vừa **cùng domain với audio production** (phỏng vấn tuyển dụng kỹ thuật). Hiện `gate_results.json` của v4-mixed-r16 chỉ so 0.0764 với baseline PhoWhisper (0.1593), **không** so với v4 — nên chưa ai biết v4 đạt bao nhiêu trên đó. Lấy adapter từ `Outputs/v3-r16/adapter/`, λ=0.5, chạy qua `src.gate._eval_split` với đúng `eval` config của v4-mixed-r16 để hai số so được | GPU Kaggle T4 (chỉ inference, không train) | — | `predictions_v3-r16_youtube228.csv` + CER và retention trên 228 segment đó | **Cổng phân nhánh cho toàn bộ phần còn lại.** v4 CER **thấp hơn** 0.0764 → v5 hồi quy thật trên domain production, và việc sửa data (D1/D3) có cơ sở. v4 CER **cao hơn** → v5 tốt hơn trên domain production, triệu chứng module 8 là vấn đề **casing convention** chứ không phải chất lượng phiên âm, và D3 (đổi tỉ lệ trộn) mất phần lớn lý do tồn tại. **Không** đọc kết quả này bằng cách so với baseline PhoWhisper — phép so duy nhất có nghĩa ở đây là v4 với v5 | **script + notebook sẵn sàng, chưa chạy trên Kaggle — vẫn todo.** [notebooks/eval-v3-on-youtube-test.ipynb](notebooks/eval-v3-on-youtube-test.ipynb) mới (13 cell, theo đúng khuôn `merge-and-push.ipynb`/`fine-tune-workflow.ipynb`: clone → cài đặt → probe `/kaggle/input` → set path → env → chạy → đọc kết quả). Đòi hỏi 2 Kaggle Dataset đính kèm chưa có sẵn tên slug xác định: dataset `youtube-meetings` (đã dùng để train v4-mixed-r16, chỉ cần vì audio 228 segment nằm y nguyên đó, không cần build lại `mixed-noisy-v1`) và một dataset mới người dùng tự zip từ `Outputs/` (gitignored, chưa từng push GitHub): `v3-r16/checkpoints/best/` (111MB) + `v3-r16/config.json` + `v4-mixed-r16/validated_manifest.jsonl` (3MB). Đã tính thêm mốc so sánh mới (không có trong SESSIONS trước đây): retention của v4-mixed-r16 trên đúng 228 segment này = **0.7094** (874 candidate, 620 giữ được), tính trực tiếp từ `Outputs/v4-mixed-r16/audit/predictions_tier1_in_domain.csv` lọc theo khoá manifest, không phải số cũ. [scripts/eval_v3_on_youtube_test.py](scripts/eval_v3_on_youtube_test.py): chọn đúng 228 record từ `Outputs/v4-mixed-r16/validated_manifest.jsonl` (đếm lại xác nhận 228, không phải giả định), nạp `Outputs/v3-r16/checkpoints/best` (adapter thô trước khi bake λ — **không** dùng `Outputs/v3-r16/adapter/` vì thư mục đó đã bake sẵn λ=1.0, là lựa chọn tự động ban đầu của gate trước khi báo cáo v3 chọn λ=0.5), `set_lambda(model, 0.5)` trong bộ nhớ theo đúng cách `stage_sweep_gate` dùng cho mỗi λ trong sweep, rồi `_eval_split` + `english_token_retention`. Đọc `base_model`/`normalization`/`eval` từ chính `Outputs/v3-r16/config.json`, không hardcode. **Chưa chạy được vì cần GPU + `mixed-noisy-v1` gắn trên Kaggle, máy này không có torch/peft** — đã kiểm phần không cần torch (đếm 228 record khớp, đọc config khớp, `checkpoints/best` tồn tại, `--help` chạy được) nhưng phần `_eval_split` thật chưa test — theo đúng cảnh báo memory "Kaggle code never works first try", khả năng cao cần sửa ít nhất một lần trên Kaggle |
| **H2** | **Đo lại N3 + phân đôi N2** — (a) tự đếm lại 4 chỉ số của handoff trên `artifacts/web-1786700346805/04_asr.json` và `.../web-1786942786371/04_asr.json`, xác nhận audio đầu vào hai run đồng nhất (so `01_*`/`02_*`); (b) chạy v5 với `prompt_ids` = `processor.get_prompt_ids("<glossary có casing>")` truyền vào `generate()` trong [remote_asr_server.py](d:/viet-speech/scripts/remote_asr_server.py), A/B trên **cùng file audio**. **Hạ ưu tiên 2026-08-17:** phần (b) chỉ đáng chạy nếu H6 xác nhận có hồi quy thật, vì nếu không thì không có gì cần "cứu" | GPU (Kaggle, model chạy remote) | H6 | Bảng số tự đo cho (a); transcript A/B cho (b) | **Cổng phân đôi N2.** Prompt phục hồi được `team`/`build` → thông tin còn trong weights, lệch prior, sửa ở decode là đủ. Không phục hồi → đã nướng vào weights. Rủi ro đã biết: prompt conditioning của Whisper dễ gây lặp/hallucinate — nếu output lặp thì đó là **fail của phép thử**, không phải kết quả âm tính; báo và đổi cách seed. Nếu (a) lệch số handoff thì dừng, N3 thành vấn đề chính | todo |
| **H3** | **Xây thang đo còn thiếu** — (1) **tỉ lệ giữ token tiếng Anh**; (2) CER có phân biệt hoa thường. Dùng lại `is_vietnamese_shaped`, **không** tự viết bộ nhận diện, **không** whitelist (72% dương tính giả) | CPU | — | `english_token_retention` trong `src/metrics.py` + 6 test | Phải phân biệt được v3-r16 với v4-mixed-r16 trên `predictions_*.csv`; không phân biệt được thì thiết kế lại, **đừng hạ ngưỡng** | **done cho chỉ số (1), cổng ĐẠT nhưng kết quả BÁC MỘT PHẦN chẩn đoán — xem N4.** `english_token_retention(refs, hyps)`: đếm token trong ref có `len>1`, `isalpha()`, và **không** khớp `is_vietnamese_shaped`; khớp theo multiset trên cả segment (đổi chỗ vẫn tính, nói hai lần phiên một lần tính một). Cùng bộ lọc `plot_youtube_stats.py:126` và `probe_youtube_captions.py:169` để ba số so được với nhau. 6 test mới, `pytest -q` **159 passed**. **Đo trên đúng segment dùng chung, khoá `(meeting_id, segment_id)`, `ref` trùng từng byte:** tier1 synthetic 426 segment → v3-r16 **0.8643** (414/479) vs v4mix **0.7599** (364/479), **giảm 10.4pp**; token mất thêm gồm `job`(6) `checklist`(5) `rollback`(4) `host`(4) **`team`(3)** `report`(3) — `team` khớp đúng triệu chứng production, bằng chứng độc lập trên artifact có sẵn. tier4a real 264 chunk → v3-r16 0.4083 vs v4mix **0.4545**, tức **v5 tốt hơn**. **Chỉ số (2) không làm được hồi tố:** cả 4 CSV audit đều `upper=0, punct=0` ở **cả `ref` lẫn `hyp`** vì `_eval_split` normalize trước khi ghi ([gate.py:92,96,98-102](src/gate.py#L92-L102)) — casing bị phá trước khi tới đĩa, đòi chạy lại eval. **Chưa gắn tier vào `gate.py`**: chỉ số cho phán quyết trái chiều giữa synthetic và real, nên "slice nào được quyền chặn push" là câu hỏi thiết kế thật, không tự chọn (CLAUDE.md §1). **Đổi kèm theo, bị ép chứ không tuỳ ý:** `strip_tone`/`is_vietnamese_shaped` + bảng âm tiết dời từ `scripts/inspect_errors.py` sang `src/normalize.py` vì `inspect_errors` đã `from src.metrics import char_counts` → để nguyên chỗ cũ là vòng import. Re-export ở `inspect_errors` nên 3 caller cũ không phải sửa; bỏ `import unicodedata` thành mồ côi | done |
| **H4** | **Bịt điểm mù số 4 — so run mới với model đang chạy production.** Đây là **row quan trọng nhất của plan**, vì nó là lỗi duy nhất chắc chắn và đúng bất kể nguyên nhân hành vi là gì. Hai phần: (a) `scripts/merge_and_push.py` nhận thêm một tham chiếu tới run/adapter **đang chạy production** và raise nếu ứng viên hồi quy so với nó — cạnh `check_provenance`, đúng nguyên tắc "mọi điều kiện đều kiểm trước khi push, không giả định" mà script đã có; (b) tier mới trong `src/gate.py` dùng `english_token_retention` của H3. Phần (b) chờ H6 để biết slice nào được quyền chặn push | CPU | H3, và (b) chờ H6 | Hàm check mới + raise khi hồi quy | Chạy lại trên adapter v4-mixed-r16 với tham chiếu là v3-r16 phải **raise** — trên 426 segment synthetic dùng chung, 0.0258 so với 0.0171 là hồi quy 51% tương đối. Không raise = check không hoạt động. **Không dùng segment không dùng chung**: khoá join là `(meeting_id, segment_id)`, `ref` phải trùng từng byte, nếu không thì đang so hai bộ test khác nhau (chi tiết ở H3) | **(a) done, (b) todo (chờ H6).** `check_no_regression_vs_production` mới trong [scripts/merge_and_push.py](scripts/merge_and_push.py#L48) — join `(meeting_id, segment_id)`, raise nếu `ref` lệch giữa hai file, raise nếu CER ứng viên tệ hơn production trên tập chung; wired vào `main()` qua `--production-predictions` (tuỳ chọn), chạy **trước** bước merge fp32 CPU tốn thời gian. **Acceptance đúng như đề**: chạy thật trên `Outputs/v4-mixed-r16/audit/predictions_tier1_in_domain.csv` (ứng viên) so với `Outputs/v3-r16/...` (production) raise đúng, in ra candidate cer=0.0258 vs production cer=0.0171 trên 426 segment chung — khớp số H3 tuyệt đối. 5 test mới `tests/test_merge_and_push.py` (3 synthetic pinning hành vi raise/không-raise/ref-lệch/không-chung-key, 1 case đúng số thật này), không cần torch/peft. Full suite 164 pass. (b) chưa làm — cần biết slice nào (H6) trước khi thêm tier vào `gate.py` |
| **D1** | **Phục hồi casing tên riêng trong nhãn YouTube** — quét token tiếng Anh trong nhãn đã soát, viết hoa lại **chỉ tên riêng** (`Grab`, `LeetCode`, `Shopee`, `Axon`), ước 20–40 mục. **Phạm vi đã thu hẹp so với bản plan đầu:** `captions.json3` chỉ phục hồi được casing cho token Google từng viết đúng; **788 token người soát tự gõ không có nguồn casing**, và phần lớn chúng (`system`, `design`, `backend`, `mock`, `behavior`) vốn viết thường trong tiếng Anh đúng nên **không** được viết hoa. Tiếng Việt giữ chữ thường. Không nghe lại audio | CPU | H6 | Nhãn đã sửa casing + ghi chú kèm ngày vào `style-guide.md` | **Lợi ích nhỏ, đừng đề cao:** v4 cũng chỉ xuất 15 ký tự hoa trên 8.515. Danh sách tên riêng phải do người xác nhận, **không** tự suy từ tần suất. Ghi vào `style-guide.md` theo đúng luật versioning của chính file đó — quy ước 4 vốn *đòi* chính tả tiếng Anh đúng nên không đối kháng, nhưng phải nói ra. **Giới hạn ghi vào báo cáo: không phục hồi được tên riêng mà cả người soát lẫn Google đều gõ thường** | todo |
| **D3** | **Đổi tỉ lệ trộn, train lại** — hiện `mixed-noisy-v1` là ~60% ký tự YouTube / 40% synthetic TTS. Hạ hoặc bỏ hẳn phần synthetic rồi train lại, để trả lời trực tiếp "60% synthetic TTS có kéo model khỏi speech thật hay không". Rẻ nhất trong nhóm data: không cần data mới, chỉ build lại dataset qua `scripts/build_mixed_dataset.py` và train lại | CPU (build) + Kaggle T4 (~6.5h ở rank 16) | **H6** | Dataset mới + run mới | So với v4-mixed-r16 bằng H4(a) trên segment dùng chung, **không** bằng baseline riêng. **H6 quyết row này có đáng chạy hay không:** nếu H6 cho thấy v5 đã tốt hơn v4 trên domain production thì D3 mất phần lớn lý do tồn tại | todo |
| **D2** | **Tăng lượng dữ liệu thật** — YouTube hiện chỉ **3,46 giờ / 790 segment** mà chiếm 60% ký tự train. Thêm video cùng domain qua `scripts/draft_sources.py` → `fetch_youtube.py` → `ingest_youtube.py` → `review_youtube.py`. Đòn bẩy lớn nhất trong nhóm data nhưng cũng đắt nhất: mỗi meeting mới phải soát tay | CPU + công soát tay | H6 | Meeting mới đã soát, `verified: true` | 3 luật sàng lọc của bước F1 (`youtube-data-pilot/README.md` mục 2). `meeting_id` và thư mục audio phải disjoint với 7 meeting hiện có, nếu không `build_mixed_dataset.py` raise. **Không** đưa nhãn chưa soát vào train — gate `verified` là thứ duy nhất chặn việc đó | todo |

**Thứ tự, sau khi cập nhật 2026-08-17.** **H6 trước tất cả** — nó là phép đo duy nhất đứng cùng
phía với triệu chứng, và nó quyết định D1/D3 có cơ sở hay không. Sửa data hay đổi tỉ lệ trộn
trước khi biết kết quả H6 là làm mù. **H4(a) chạy song song được ngay**, không chờ H6: lỗi "không
so với model đang production" đúng bất kể H6 ra gì. H4(b) chờ H6. H2(b) chỉ đáng chạy nếu H6 xác
nhận có hồi quy thật. D1/D3/D2 sau H6, theo thứ tự đó nếu H6 xác nhận hồi quy.

**Bốn cổng dừng, không được đi vòng:** (1) ~~H1 không tìm được dòng provenance~~ — đã đóng, N1
xong; (2) H2(b) ra output lặp/hallucinate → đó là phép thử fail, **không** được đọc thành "prompt
không cứu được"; (3) H3 chỉ số không phân biệt được v4/v5 → thiết kế lại chỉ số, **không** hạ
ngưỡng cho tới khi ra kết quả đẹp; (4) **H6 cho thấy v5 tốt hơn v4 trên domain production → dừng
D3, viết lại kết luận, không đi tìm cách khác để chứng minh v5 tệ.** Cổng này tồn tại vì cả phiên
này đã đi theo khung "v5 là hồi quy" do handoff đặt ra, mà khung đó dựa trên phép đếm không có
reference (xem ô ghi chú ở đầu mục).
