# PHOWHISPER FINE-TUNE: PROJECT CORE

**Status**: Production-Ready Pipeline  
**Approach**: Structural Restart (Module-Driven)  
**Last Updated**: 2026-07-31

---

## 0. WHY THIS PROJECT STRUCTURE? (Agent Context)

### Mục đích dự án
Biến quy trình fine-tune PhoWhisper từ **"thủ công + notebook"** thành **"ASR Pipeline as Code"**.

### Vấn đề cần giải quyết

1. **Notebook Bottleneck**  
   - **Hiện trạng**: Mỗi lần thay đổi hyperparameter, phải sửa cell → upload lại notebook lên Kaggle → chạy lại toàn bộ.
   - **Giải pháp**: `configs/experiment.yaml` là single source of truth. Chỉ cần `git push` config → Kaggle tự động `git pull` và chạy.

2. **Catastrophic Forgetting**  
   - **Hiện trạng**: Fine-tune tăng in-domain CER (1.3%) nhưng làm VIVOS OOD tệ đi (1.78% → 4.58%).
   - **Giải pháp**: Lambda Sweep tự động tìm λ* cân bằng giữa "in-domain gain" và "OOD preservation".

3. **Reproducibility Gaps**  
   - **Hiện trạng**: Không biết run cũ dùng config gì, commit nào, đã pass gate nào.
   - **Giải pháp**: Mọi run tự động tạo `provenance.md` (commit hash, timestamp, config snapshot) + ledger row.

4. **Safety Risks**  
   - **Hiện trạng**: Model "nhìn tốt" trên test nhưng thực tế hallucinate/truncate trên real audio.
   - **Giải pháp**: Eval Gate 5 check (in-domain, OOD, RTF, real-audio segmented, real-audio long-form) → FAIL ngay nếu 1 check sai. Chỉ tier 4 đo trên tiếng nói thật, nên chỉ tier 4 có ý nghĩa tuyệt đối cho production.

### Triết lý thiết kế (Design Philosophy)

| Nguyên tắc | Ý nghĩa |
|:---|:---|
| **Config-Driven** | YAML là "cửa ngõ" duy nhất để thay đổi hành vi. Code không bao giờ hardcode magic numbers. |
| **Atomic Pipeline** | Mỗi module (`data.py`, `train.py`, `sweep.py`, `eval.py`) là black box độc lập: Input rõ ràng → Output rõ ràng. |
| **Fail Loudly** | `pipeline.py` phải dừng ngay lập tức (raise exception) nếu bất kỳ gate nào báo đỏ. Không silent skip. |
| **Data-Docs-as-Code** | Không để data docs trôi nổi ngoài repo. Schema manifest, warnings (voice overlap, synthetic bias) phải nằm trong `PROJECT_CORE.md` và `src/data.py` docstrings. |
| **Evidence-Based** | Mọi claim ("model improves X%") phải có artifact path chứng minh (`outputs/{run_id}/metrics/*.json`). |

### Khi bạn (Agent) làm việc ở đây

- ✅ **DO**: Luôn hỏi "Cấu hình này đã có trong `configs/experiment.yaml` chưa?" trước khi code.
- ✅ **DO**: Mọi tính năng mới đi kèm unit test (`tests/`) + validation logic (`src/config.py`).
- ✅ **DO**: Port logic từ notebook cũ → `src/` modules. Không refactor notebook cũ.
- ❌ **DON'T**: Hardcode hyperparameters trong code.
- ❌ **DON'T**: Sửa notebook cũ (`finetune-paid-dataset.ipynb`) → nó là historical reference.
- ❌ **DON'T**: Tạo file docs riêng lẻ → gộp vào `PROJECT_CORE.md`.

### Quan hệ với `D:\phowhisper-finetune-exp`

`Fine_tune_wf` là **repo chính thức mới**, kế thừa nhưng độc lập hoàn toàn với `phowhisper-finetune-exp`:
- **Dataset**: copy cả 4 thư mục sang `Fine_tune_wf/dataset/`; sau khi copy không còn phụ thuộc repo cũ. Nhưng **chúng không cùng loại**: `paid-dataset`, `unpaid-dataset`, `dataset_by_task` là data train (synthetic), còn **`done/` là benchmark audio thật** (2 recording + transcript người hiệu chỉnh) → ingest thành `data.real_bench_path`, **không bao giờ dùng làm `dataset_path`** (§4).
- **Code cũ** (`src/`, `notebooks/`): chỉ dùng để **tham khảo logic** khi viết `src/` module mới ở đây — không migrate nguyên trạng, không copy outputs/docs cụ thể của các run cũ (vd `v1c-lambda-sweep-valfix`, `v1c_lambda05_*`). Nếu cần đối chiếu số liệu lịch sử, tra trực tiếp trong `phowhisper-finetune-exp` (vẫn giữ nguyên, không xóa).
- **Model**: không hardcode PhoWhisper — `base_model` là field cấu hình, nhận bất kỳ checkpoint HF nào tương thích kiến trúc Whisper.

### Tại sao không refactor notebook cũ?

| Lý do | Giải thích |
|:---|:---|
| **Technical Debt** | Cell dependencies rối, biến toàn cục lẫn lộn, khó test. |
| **Evidence Loss** | Notebook cũ có outputs đã chạy (training curves, metrics) → là bằng chứng lịch sử cho `v1c`. |
| **Time Cost** | Dọn sạch notebook tốn thời gian hơn xây khung mới. |

**Kết luận**: Giữ notebook cũ làm reference, xây `src/` mới sạch sẽ.

---

## 1. STRATEGIC PLAN

### Objective
Triệt tiêu bottleneck thủ công (notebook upload), tự động hóa Lambda Sweep, tăng reproducibility.

### Approach: Structural Restart
- **KHÔNG** refactor notebook cũ (technical debt rối).
- **KHÔNG** dựng từ đầu (mất logic đã debug).
- **LÀM**: Xây khung mới (`src/`, `configs/`), port logic từ notebook.

### Phases

| Phase | Duration | Deliverable |
|:---|:---|:---|
| **1. Skeleton** | 2 days | `src/data.py`, `src/config.py`, `configs/experiment.yaml` |
| **2. Engine** | 3 days | `src/train.py`, `src/sweep.py`, dry-run on Kaggle |
| **3. Registry** | 2 days | `src/eval.py`, `src/report.py`, auto-ledger |
| **4. Orchestrate** | 1 day | `src/pipeline.py`, `run_pipeline.ipynb`, E2E test |

---

## 2. MODULE RESPONSIBILITIES

| Module | Input | Output | Key Logic |
|:---|:---|:---|:---|
| `src/data.py` | `manifest.*.jsonl` (N files, one per meeting) | `validated_manifest.jsonl`, `split_stats.json` | Manifest merge, split resolution, disjointness check, code-switch tagging |
| `src/config.py` | `experiment.yaml` | Pydantic config object | Schema validation, hyperparam constraints |
| `src/model.py` | Checkpoint + base | `delta_weights.pt`, `adapter_config.json` | Delta extraction, SVD reconstruction |
| `src/train.py` | Config + data | Best checkpoint, `training_metrics.csv` | SFT loop, early stopping, eval per step |
| `src/sweep.py` | Delta + datasets | `lambda_sweep.csv`, `adapter/` (λ*) | Lambda grid [0.25-1.0], selection rule |
| `src/eval.py` | Model + datasets | `gate_results.json`, `error_top100.csv` | Gate: in-domain, OOD, RTF, real-audio segmented (4a), real-audio long-form (4b) |
| `src/report.py` | All metrics | `provenance.md`, `ledger_row.txt` | Auto-gen artifacts |
| `src/hub.py` | Adapter | HF repo URLs | Push to HuggingFace |
| `src/pipeline.py` | `experiment.yaml` | `outputs/{run_id}/` | Orchestrator (resume-capable) |

### 2.1 DATA FLOW

Mọi handoff giữa các stage là **một file trên disk** — không có state truyền in-memory qua ranh giới stage. Đó là điều kiện để `pipeline.py` resume được, và cũng là điều kiện để build/test được từng module mà không cần chạy lại upstream.

#### Stage-by-stage

```
INPUTS (on disk, user-supplied)
  configs/experiment.yaml
  {dataset_path}/manifest.*.jsonl   + {dataset_path}/audio/**.wav
  {ood_eval_path}/ , {real_bench_path}/         (same manifest schema; real_clip_path optional)
  base_model                                    (HF hub id or local checkpoint)

STAGE 1 — INIT
  experiment.yaml ──► config.py ──► ExperimentConfig ──► outputs/{run_id}/config.json
  manifest.*.jsonl + ExperimentConfig ──► data.py ──► validated_manifest.jsonl
                                                    └► metrics/split_stats.json
  base_model + validated_manifest[val,test] + ood ──► metrics.py ──► metrics/baseline.json

STAGE 2 — TRAIN
  ExperimentConfig + validated_manifest[train,val] + ood(subset) + base_model
      ──► train.py ──► checkpoints/best/
                    └► metrics/training.csv          (per step: loss, LR, val_cer, ood_cer)

STAGE 3 — SWEEP
  checkpoints/best/ + base_model ──► model.py:extract_delta ──► delta_weights.pt
  delta_weights.pt + grid + validated_manifest[val] + ood
      ──► sweep.py ──► metrics/lambda_sweep.csv
  lambda_sweep.csv + metrics/baseline.json + ood_budget_pp
      ──► sweep.py:select_lambda ──► λ*        │ no λ within budget ──► HARD FAIL, stop
  delta_weights.pt × λ* ──► model.py:apply_lambda ──► adapter/
                            (scale_lora if LoRA · svd_project if full-FT — §6 Stage 3)

STAGE 4 — GATE
  adapter/ + base_model + validated_manifest[test] + ood + real_clips + metrics/baseline.json
      ──► eval.py ──► metrics/gate_results.json
                   └► audit/error_top100.csv        │ any tier FAIL ──► halt, no push

STAGE 5 — PUBLISH
  outputs/{run_id}/** + git HEAD ──► report.py ──► provenance.md, ledger_row.txt
  adapter/ + gate_results.json(PASS) ──► hub.py ──► HF repo URL
```

#### Producer → consumer table

| Artifact | Written by | Read by | Why it must be a file |
|:---|:---|:---|:---|
| `config.json` | `config.py` (Stage 1) | every later stage, `report.py` | Resume must use the config the run *started* with, not a later edit to `experiment.yaml` |
| `validated_manifest.jsonl` | `data.py` | `train.py`, `sweep.py`, `eval.py` | Carries the **resolved** split — raw manifests only say `demo`/`test`, train/val is derived (see below) |
| `metrics/baseline.json` | `metrics.py` (Stage 1) | `sweep.py` (budget), `eval.py` (tiers 1–3), `report.py` | Base numbers are computed **once** and never recomputed during the gate; also the fixture that lets Stage 3–4 logic be developed without a GPU |
| `checkpoints/best/` | `train.py` | `model.py:extract_delta` | Only artifact crossing the train→sweep boundary; enables restarting a sweep without retraining |
| `delta_weights.pt` | `model.py` | `sweep.py`, `model.py:apply_lambda` | Δ is reused for every λ in the grid — extract once |
| `metrics/lambda_sweep.csv` | `sweep.py` | `select_lambda`, `report.py` | Makes the λ decision auditable after the fact: the row set is the evidence for λ* |
| `adapter/` | `model.py:svd_to_lora` | `eval.py`, `hub.py` | The deliverable; gate must score the exact bytes that get pushed |
| `metrics/gate_results.json` | `eval.py` | `hub.py` (push precondition), `report.py` | Push is a pure function of this file's status |
| `.pipeline_state.json` | `pipeline.py` | `pipeline.py` on restart | Records the last completed stage; resume re-reads that stage's artifacts, never recomputes |

#### Split resolution (non-obvious)

Manifests ship `"split": "demo"` for train-side meetings and `"split": "test"` for test meetings — **train/val is not in the data**. `data.py` resolves it:

```
split == "test"                          ──► test
split == "demo" and meeting_id in val_meetings ──► val     # val_meetings from §4, seed=42
split == "demo" otherwise                ──► train
```

`validated_manifest.jsonl` is the only artifact that carries the resolved split. Nothing downstream re-derives it — re-deriving is how a val/train leak gets in (cf. the `valfix` suffix on the old repo's runs).

#### Invariants

1. **Stage boundaries are files.** No stage returns a Python object to the next stage. If you can't restart the pipeline at a stage boundary from disk alone, the boundary is wrong.
2. **Downstream never reads `configs/experiment.yaml`.** Only `outputs/{run_id}/config.json`. Editing the YAML mid-run must not change a running or resumed run.
3. **`baseline.json` is the single source of "base" numbers.** Tiers 1–3 and the OOD budget all read it. Never re-measure the base model inside the gate.
4. **One normalization function.** `metrics.py` owns text normalization; `train.py`'s eval, `sweep.py`'s λ scoring, and `eval.py`'s gate all call it. If λ* is selected under one normalization and the gate applies another, λ* is meaningless.
5. **Failure writes nothing downstream.** A hard fail (no valid λ, any gate tier red) leaves no `adapter/` selected and no HF push — the run's artifacts record the failure instead of a partial success.

---

## 3. CONFIGURATION (Single Source of Truth)

### `configs/experiment.yaml`
```yaml
# Run identity
run_id: v4-r64
base_model: vinai/PhoWhisper-large    # any HF Whisper-family checkpoint — not hardcoded
seed: 42                              # split sampling + training; §7 requires it fixed

# Data (all fields are user-supplied paths — pipeline is data-agnostic)
data:
  dataset_path: dataset/paid-dataset   # training data: paid / unpaid / dataset_by_task — NEVER the bench
  ood_eval_path: dataset/vivos          # default OOD benchmark, swappable
  real_bench_path: dataset/real-meetings-bench   # Tier-4 real audio + reference transcripts (§4)
  real_clip_path: null                  # optional: extra unlabelled real clips for eyeballing. Tier 4
                                        # is satisfied by real_bench_path; null means "skip", not "fail"

# LoRA
lora:
  rank: 64
  alpha: 128
  target_modules: [q_proj, k_proj, v_proj, out_proj, fc1, fc2]
  dropout: 0.1
  use_rslora: true
  min_retained_energy: 0.99   # full-FT SVD path only (§6 Stage 3 Path B)

# Training
training:
  learning_rate: 2e-4
  epochs: 3
  batch_size: 8
  grad_accum_steps: 2
  full_finetune: false        # true → §6 Stage 3 Path B (SVD projection) instead of Path A (scale)

# Lambda sweep
lambda_sweep:
  grid: [0.25, 0.4, 0.5, 0.6, 0.75, 1.0]
  ood_budget_pp: 0.20

# Gates
gates:
  min_improvement_pct: 10
  ood_budget_pp: 0.20
  rtf_threshold: 1.05
  # Tier 4 — real audio
  real_cer_regression_pp: 0.0     # 4a: real CER must not get worse than base
  longform_repetition_max: 0.02   # 4b: repeated-ngram share of output
  longform_dropped_max: 0.05      # 4b: reference spans with no aligned hypothesis
  longform_length_ratio: [0.9, 1.1]  # 4b: len(hyp)/len(ref), catches truncation and runaway

# Scoring normalization (§6 Stage 4 contract) — applied identically to hyp and ref
normalization:
  strip_punctuation: true
  lowercase: true
  filler_tokens: [ừm, ờm, ehm, uhm, hmm]   # restricted list; ạ/à/ừ/ơ/dạ/vâng are lexical, never here
  number_convention: word_to_digit           # decided 2026-07-31 — applied to hyp AND ref, symmetrically
  audit_conversions: true                    # log per-token conversion counts (§6 Stage 4)

# Publish
hub:
  push_enabled: true
  repo_id: null               # null → derive from run_id; push still requires gate PASS
```

### Validation Rules (in `src/config.py`)
- `alpha == 2 * rank` if `use_rslora: false`
- `target_modules` must include `q_proj, k_proj, v_proj, out_proj` for Whisper
- `task_type` must be `SEQ_2_SEQ_LM` (explicit, not default)
- `learning_rate` in range [1e-5, 5e-4]
- `data.dataset_path`, `data.ood_eval_path`, `data.real_bench_path` must exist on disk — no default falls back to a bundled dataset. `data.real_clip_path` is optional (`null` → skip)
- `data.real_bench_path` must not overlap `data.dataset_path` — raise if any `audio_filepath` appears in both (tier 4 leak guard)
- no literal LoRA rank anywhere: SVD/reconstruction rank is always `lora.rank` (§6 Stage 3)
- `normalization.filler_tokens` must not contain any of `ạ à ừ ơ dạ vâng` — raise, these are lexical (§6 Stage 4)
- `training.full_finetune: true` requires `lora.min_retained_energy` to be set (Path B needs it)

### Platform Independence
- `src/pipeline.py` and all modules assume **no specific execution environment**. No hardcoded Kaggle paths (`/kaggle/input/...`), no notebook-only assumptions.
- Must run identically via `python src/pipeline.py --config ...` on Kaggle, local machine, Colab, or a dedicated server — only Python + declared deps required.
- Kaggle notebook (§5) is one *launcher* among several, not the only supported one.

---

## 4. DATA METADATA (Internal Backup)

> Pipeline is dataset-agnostic (see §3 `data.dataset_path`). This section documents `paid-dataset` specifically as the best-understood example — its warnings (voice overlap, synthetic bias, digit bias) are properties of *this* dataset, not universal pipeline assumptions. `unpaid-dataset` and `dataset_by_task` still need their own metadata notes here once profiled. `done/` **has** been profiled and is not a training set at all — it is the real-audio benchmark, documented below.

### Dataset: `dataset/paid-dataset/`

| Property | Value |
|:---|:---|
| Segments | 1,802 (train: 1,566 / val: 250 / test: 236) |
| Duration | 2.17 hours |
| TTS Engine | ElevenLabs `eleven_v3` |
| Voices | 10 fixed (reused in train/test) |
| Language | Vietnamese + English code-switch (~48%) |

### Manifest Schema (JSONL)
```json
{
  "audio_filepath": "paid_meeting_0001/raw_turns/seg_0000.wav",
  "meeting_id": "paid_meeting_0001",
  "segment_id": "seg_0000",
  "speaker": "LEAD",
  "start": 0.0,
  "end": 3.5,
  "duration": 3.5,
  "text": "Chúng ta bắt đầu meeting nhé.",
  "lang": "vi",
  "quality": "clean",
  "overlap": false,
  "source": "synthetic",
  "tts_engine": "elevenlabs",
  "tts_model": "eleven_v3",
  "voice_id": "21m00Tcm4TlvDq8ikWAM",
  "split": "demo"
}
```

### Critical Warnings ⚠️

1. **Voice ID Overlap**: All 10 voices appear in train AND test. Test CER = "same voices, new content", NOT "unseen voice generalization".
2. **Synthetic Data**: 100% LLM-generated. Absolute CER does NOT transfer to real speech. Use only for relative comparison.
3. **Forgetting on VIVOS**: Base VIVOS CER 1.78% → FT can go to 2.2-4.6%. Mitigate via Lambda Sweep.
4. **Digit Bias**: 62% of in-domain improvement comes from digit normalization, not acoustic gain.

### Dataset: real-meetings benchmark (`data.real_bench_path`) — tier 4

**Source: `done/` in the predecessor repo.** `done/` is *not* a fourth training set — it is the real-audio benchmark. It holds 2 recordings, each as 48 kHz master + 16 kHz mono copy, plus a `*.draft.json` transcript. Profiled 2026-07-31:

| Property | `real_0001` | `real_0002` |
|:---|:---|:---|
| Audio (16 kHz mono, 16-bit) | 22.13 min | 23.55 min |
| Transcribed span | 22.1 min (~100%) | 21.1 min (~90%) |
| Segments | 115 | 81 |
| Characters | 13,673 | 16,169 |
| Speaker labels in segments | 0–3 (4) | 0–3 (4) |
| `num_speakers_detected` (pyannote meta) | 6 | 8 |
| Segment duration min / median / max | 0.2 / 3.0 / **212.0** s | 0.4 / 4.0 / **175.6** s |
| Content | ML/technical walkthrough | student research presentation (VAE) |

**Totals: 196 segments, 29,842 chars, 43.2 min transcribed, 11.5 char/s.**

#### ⚠️ Six properties that limit what tier 4 can claim

1. **The reference is ASR output that a human edited, not an independent transcription.** `_draft_meta.asr_model_id = phowhisper-small`, `status = "Đã hiệu chỉnh"`. Errors the editor did not catch remain *PhoWhisper-shaped* errors. Scoring a PhoWhisper checkpoint against a PhoWhisper-small-derived reference **systematically understates its CER**, and the bias runs in the flattering direction. It also unfairly penalises any non-PhoWhisper `base_model`. Usable as a gate; not usable as an absolute accuracy claim, and not usable to compare across model families.
2. **Number convention is mixed, and not fixable by a normalizer.** `real_0001`: 104 digit tokens vs 29 number-words. `real_0002`: 7 vs 88 — opposite directions in the same set, and mixed inside single segments. Worse, the words are ambiguous: `một` is usually the indefinite article ("một bộ trọng số" = "a set of weights", not "1 set"), and `năm` means both *five* and *year*. A word→digit normalizer will corrupt real content. See §6 Stage 4.
3. **Vietnamese "fillers" are lexical.** The reference keeps them: `ờ` 70, `ạ` 68, `ừ` 26, `à` 24. But `ạ` is the politeness particle (`vâng ạ`), `à` a question particle (`thế à?`), `ơ`/`ừ` real words. Blanket filler deletion removes meaning, not noise.
4. **Segment durations are pathologically skewed.** Median 3–4 s but max 212 s. A handful of segments hold a large share of all characters, so a naive segment-level bootstrap is dominated by them. And a 212 s segment **cannot be fed to Whisper at all** (30 s window) — re-segmentation is required before tier 4a can run, not optional.
5. **Domain mismatch with the training data.** The benchmark is ML/academic talks (`sub layer`, `variational auto encoder`, `thầy`, `các bạn`); `paid-dataset` is business meetings (LEAD/BE/SM, deploy, sprint). Tier 4 is therefore a *safety* gate, not an in-domain gain measurement. A tier-4 regression cannot be attributed to forgetting without separating out domain mismatch.
6. **The JSON metadata is unreliable.** `audio_duration` says 4035 s for `real_0001` whose wav is 1328 s — stale by 3×. `num_speakers_detected` (6, 8) contradicts the actual label set (4, 4). `status` is free text with a typo across the two files. Read durations from the wav; treat every `_draft_meta` field as a hint, not a fact.

#### Statistical power — corrected

196 segments over 43.2 min, ~30k characters. At a real-speech CER around 8–15% that is a few thousand character errors — real signal, but the bootstrap unit count is **196, not the ~500 a 40-minute set would give at normal segment lengths**, and property 4 makes the effective count lower still. Expect a 95% CI on the order of **±1–1.5 pp**, so a difference below roughly **1.5–2 pp (≈15–20% relative at CER 10%)** is not resolvable. Set `gates.real_cer_regression_pp` with that in mind, report the CI beside every tier-4 number, and treat anything inside the CI as INCONCLUSIVE rather than PASS.

This is still a stronger instrument than the synthetic test set, where CER ≈1.3% over 236 short segments yields only a couple hundred character errors and the 10%-relative gate sits well inside the noise. Higher error rate means more signal per minute — but it does not manufacture segments.

What the set does **not** give: acoustic diversity. 2 recordings = 2 rooms, 2 mic chains, ~8 speakers, 1 topic area. It supports one honest pass/fail per run. It does not support tuning, ablations, or per-condition breakdowns.

**Growth path**: more audio is expected. Extend the set rather than replacing it, keep these two recordings as a frozen slice so historical runs stay comparable, and record the split in this table. Prioritise new recordings that are (a) business-meeting domain and (b) transcribed from scratch rather than post-edited from ASR — that fixes properties 1 and 5, the two that most limit the current set.

### Code-Switch Detection
```python
# Vietnamese syllable shape test (NO whitelist - false positive 72%)
def is_english_token(token: str) -> bool:
    pattern = r"^[bcdđghklmnpqrstvxyz]?[aăâeêiơoôuư][cmnpt]?$"
    return not re.fullmatch(pattern, token.lower())

# Usage
english_ratio = sum(is_english_token(t) for t in text.split()) / len(text.split())
```

### Disjointness
- `meeting_id`: ✅ Disjoint (train/val/test)
- `voice_id`: ❌ NOT disjoint (10 voices overlap)

### Val Split (Reproducible)
```python
# seed=42, fixed 3 meetings for validation
val_meetings = ["paid_meeting_0001", "paid_meeting_0002", "paid_meeting_0011"]
```

---

## 5. OPERATIONAL WORKFLOW

### Daily Workflow
1. **Local**: Edit `configs/experiment.yaml` → `git push`
2. **Kaggle**: `git pull` → `python src/pipeline.py --config configs/experiment.yaml`
3. **Output**: Download `outputs/{run_id}/` (self-contained evidence)

### Minimal Kaggle Notebook
```python
# Cell 1: Setup
!git clone https://github.com/{user}/phowhisper-finetune.git
%cd phowhisper-finetune
!git pull origin main

# Cell 2: Run
!python src/pipeline.py --config configs/experiment.yaml --run_id v4-r64
```

### Artifact Structure
```
outputs/{run_id}/
├── config.json              # Exact config used
├── adapter/                # LoRA adapter (λ*-optimized)
├── metrics/
│   ├── baseline.json      # Pre-FT metrics
│   ├── training.csv        # Per-step loss/LR
│   ├── lambda_sweep.csv   # λ vs CER
│   └── gate_results.json  # PASS/FAIL status
├── audit/
│   ├── error_top100.csv
│   └── training_curve.png
├── provenance.md           # Commit hash, timestamps
└── ledger_row.txt        # Paste to task_ledger.md
```

---

## 6. PIPELINE STAGES

### Stage 1: Initialization
- Load config → validate schema
- Load manifest → disjointness check
- Tag code-switch
- **Baseline eval**: Run base model on Val + Test + VIVOS → save `metrics/baseline.json` (canonical name, per §5 artifact structure)

### Stage 2: SFT Training
- Build LoRA config (assert `task_type=SEQ_2_SEQ_LM`)
- Train with HF Trainer
- **Checkpoint strategy**: Save if `val_cer` improves
- **Early stopping**: 3 eval patience
- **Eval loop**: Run VIVOS OOD every eval (detect forgetting early)

### Stage 3: Lambda Optimization
- Extract Δ = finetuned - base (assert only the configured `target_modules` weight types differ)
- **Lambda grid**: from `lambda_sweep.grid` (default `[0.25, 0.4, 0.5, 0.6, 0.75, 1.0]`)
- For each λ: eval on Val + OOD
- **Selection rule** (pre-registered):
  ```
  λ* = argmax(λ) s.t. CER_ood(λ) ≤ CER_ood(base) + OOD_BUDGET
  ```
- **Fallback rule**: If NO λ in the grid satisfies the constraint (all candidates exceed the OOD budget), the pipeline **fails hard** — no adapter is selected, no push to HF, run is marked FAIL. Fail Loudly applies here too: there is no silent fallback to λ=0 or to the least-bad λ.

#### Applying λ* — two paths, `rank` is never hardcoded

`r = cfg.lora.rank` everywhere. No literal rank appears in code or in this spec.

**Path A — LoRA-trained checkpoint (the default, and what the pipeline does today).**
No SVD. A LoRA delta is already low-rank by construction:

```
Δ = (alpha / r) · B · A        ⇒        λ·Δ = ((λ · alpha) / r) · B · A
```

So applying λ is scaling one factor: `B ← λ · B` (equivalently `alpha ← λ · alpha`). Exact, no reconstruction error, no truncation, nothing to verify against a tolerance. `model.py:scale_lora`.

**Shipping contract (verified by live run on Kaggle T4, 2026-07-31, `src/lora.py`).** Three routes to a given λ — in-memory `LoraLayer.scaling`, multiplying `lora_B`, halving `lora_alpha` — were measured to agree to `0.000e+00` after save/reload; λ=1 vs λ=0.5 differ by `4.56e+01`, so λ genuinely changes the model (this rules out the `√r` denominator being a confound: `λ·(alpha/denom) = (λ·alpha)/denom`, linear in alpha regardless of the rsLoRA denominator). The pipeline bakes λ into `adapter_config.json`'s `lora_alpha` and leaves `lora_B` bit-identical to training — this is the one route that needs no runtime hook to reproduce. `save_with_lambda` (`src/lora.py`) then reloads the saved adapter fresh and asserts its logits match the in-memory λ within **1e-5, not `== 0`** — λ values that are not exact binary fractions (e.g. 0.75) drift ~1e-7 through the save/reload round trip, so an exact-equality check would fail loud on a non-bug.

**Path B — full-parameter fine-tune (only if `training.full_finetune: true`).**
Δ is dense and must be projected onto rank `r`:

```
U, S, Vᵗ = SVD(λ* · Δ)
B = U[:, :r] @ diag(S[:r])
A = Vᵗ[:r, :]
```

Truncation here is **lossy** — `‖B·A − λ*Δ‖ < 1e-6` only holds when `numerical_rank(Δ) ≤ r`. So Path B must:
- compute numerical rank as `count(σ_i > tol · σ_max)`,
- report `retained_energy = Σ_{i<r} σ_i² / Σ_i σ_i²` into `metrics/lambda_sweep.csv`,
- **fail loud** if `retained_energy < cfg.lora.min_retained_energy` (default 0.99) — do not silently ship a truncated delta.

> **Historical note**: earlier notebook code hardcoded rank 16 (from the `v1c-r16` run) and asserted `‖B·A − Δ_opt‖ < 1e-6`. With `lora.rank: 64` that assertion cannot hold — a rank-64 Δ is not representable at rank 16. The hardcode is removed; rank comes from config, and the tolerance check belongs to Path B only.

### Stage 4: Evaluation Gate (5 checks)

| Tier | Check | Gate Rule |
|:---|:---|:---|
| 1 | In-domain (synthetic) | `CER_test ≤ 0.9 * CER_base` (10% improvement) |
| 2 | OOD | `CER_ood ≤ CER_ood(base) + 0.20pp` |
| 3 | RTF | `RTF ≤ 1.05 * RTF_base` |
| 4a | **Real audio, segmented** | `CER_real ≤ CER_real(base)` — no regression on real meeting audio |
| 4b | **Real audio, long-form** | Whole-meeting decode: repetition rate, dropped-span ratio, and length ratio all within `gates.longform_*` bounds |

If ANY tier fails → **Pipeline halts**, no push to HF.

**Tier 4 is the only tier measured on real speech, and therefore the only tier whose absolute numbers mean anything for production.** Tiers 1–2 are relative signals on synthetic / read speech. Do not report tier 1 as the headline result.

#### Tier 4 rules of use (`data.real_bench_path`)

The real-audio benchmark is small and non-renewable. It is a **gate, not a tuning signal**:

- **Never trained on.** Never in `validated_manifest.jsonl`.
- **Never used to rank λ.** λ* is selected on synthetic val + OOD only (§6 Stage 3). Sweeping λ against 40 minutes of one or two rooms fits λ to those rooms.
- **Read once per run, at gate time.** Every extra look burns the set's independence. `provenance.md` records how many times the run touched it.
- **Not split into dev/test.** Splitting a set this small halves its statistical power for no benefit — tuning happens on synthetic data instead.
- **4b is nearly free**: long-form decode of a whole meeting needs only the full reference transcript, no segment-level timestamps. Prefer building 4b first.

#### Normalization contract (decided 2026-07-31)

Human transcripts carry conventions that dominate CER. Because ~62% of past in-domain gain was digit normalization (§4), the reference's conventions decide the number more than acoustics do. `metrics.py` applies **one** normalization function to both hypothesis and reference.

| Convention | Decision | State in the data (§4) | Implementation note |
|:---|:---|:---|:---|
| Numbers | **word→digit, symmetric** (`mười lăm` → `15`) | ⚠️ mixed: 111 digit vs 117 number-word tokens, opposite ratios per meeting | Applied to hyp **and** ref — see below |
| Punctuation / casing | **strip both** | ✅ already near-absent (29 punctuation marks in 30k chars; text already lowercase) | Trivial, low risk |
| Fillers | **remove** | ⚠️ present and lexically ambiguous | **Restricted list only** — see below |
| Code-switch | **keep English as-is** | ✅ matches (`sub layer`, `pain point`, `variational auto encoder`) | No-op |
| Speaker labels / timestamps | *undecided* → **moot** | ✅ they live in separate JSON fields (`speaker`, `start`, `end`), never inline in `text` | Nothing to strip |

**Numbers — word→digit, applied symmetrically (decided 2026-07-31).** This continues the `v1c` convention, so historical numbers stay comparable.

Vietnamese number words are ambiguous with non-numeric uses: `một` is normally the indefinite article (`một bộ trọng số` = "a set of weights", not "1 set"), `năm` is both *five* and *year*. **Symmetry is what makes this safe enough**: `một bộ` → `1 bộ` on both sides cancels out and contributes nothing to CER. Damage only occurs where hypothesis and reference disagree at a converted token. A one-sided normalizer would be genuinely unsafe; this is not one.

On the real bench the conversion is actively *helpful* — the reference contradicts itself (104 digit vs 29 word tokens in `real_0001`, 7 vs 88 in `real_0002`), and normalizing both sides to digits removes that inconsistency.

Required safeguards, all cheap:

- **Symmetric or nothing.** The same function, same config, applied to hypothesis and reference in the same call path. Never normalize one side.
- **Audit the conversions** (`normalization.audit_conversions`): log per-token conversion counts into `audit/normalization_counts.json`, with `một` and `năm` broken out separately. If those two dominate, the number is being moved by grammar rather than by numerals and the reference should be hand-fixed (A2c).
- **Baseline reports both.** `metrics/baseline.json` carries `cer_test` (normalized) **and** `cer_test_raw` (as-written), so the "how much of the gain is the normalizer" question from §4 warning 4 stays answerable for every run. This is what keeps the accounting honest under this choice.
- **Convention-sensitivity check** (A2c): score tier 4 under both conventions once. If the two CERs differ by more than the effect size being gated on, the number is normalization-dominated and must not be reported as a model result.

**Fillers — restricted deletion.** Delete only unambiguous hesitation tokens: `ừm`, `ờm`, `ehm`, `uhm`, `hmm`. Explicitly **do not** delete `ạ`, `à`, `ừ`, `ơ`, `dạ`, `vâng` — these are politeness/question particles and real words (`ạ` alone occurs 68× and is almost always the politeness particle). Blanket deletion removes meaning, not noise. The deletion list is a config field, not a constant, and every token in it must be justified in this table.

**Repetition interacts with tier 4b.** The reference itself transcribes genuine disfluency (`em nói nói là`). A repeated-ngram detector measured in absolute terms will flag correct output. Tier 4b's `longform_repetition_max` must be computed as repetition **relative to the reference's own repetition rate**, not absolute.

---

## 7. CODING CONVENTIONS (VibeCode Instructions)

### General Rules
- **Config is king**: All hyperparameters from YAML, never hardcoded.
- **Fail loud**: Raise exceptions, no silent skips.
- **Type hints everywhere**: `def load_manifest(path: str) -> List[Dict]`.
- **Docstrings**: Every function has input/output docs.
- **Logging, not prints**: Use `logging` module.

### Error Prevention
```python
# Always validate BEFORE running
def validate_config(cfg: ExperimentConfig) -> None:
    if cfg.lora.alpha != 2 * cfg.lora.rank and not cfg.lora.use_rslora:
        raise ValueError(f"alpha must be 2*rank if use_rslora=false")
    if "fc1" not in cfg.lora.target_modules:
        logger.warning("MLP (fc1/fc2) not in target_modules - may hurt performance")
```

### Reproducibility
```python
# Always use fixed seeds for splits
random.seed(42)
val_meetings = sorted(random.sample(all_meetings, k=3))
# Assert deterministic: must output [0001, 0002, 0011]
```

### Artifact Traceability
```python
# Every run MUST generate provenance
provenance = {
    "commit": subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip(),
    "timestamp": datetime.utcnow().isoformat(),
    "config": cfg.dict(),
    "metrics": results
}
```

---

## 8. FILES TO CREATE

### Folder Structure
```
phowhisper-finetune-exp/
├── configs/
│   ├── base_config.yaml         # Template defaults
│   └── experiment.yaml          # Active config (edit this)
├── src/
│   ├── __init__.py
│   ├── config.py                # Pydantic schema + validation
│   ├── data.py                  # Manifest validation
│   ├── model.py                 # LoRA config, delta, SVD
│   ├── train.py                 # SFT loop
│   ├── sweep.py                 # Lambda optimization
│   ├── eval.py                  # gate: 5 checks (§6 Stage 4)
│   ├── hub.py                   # HF push
│   ├── report.py                # Provenance + ledger
│   └── pipeline.py              # Orchestrator
├── notebooks/
│   └── run_pipeline.ipynb       # Minimal Kaggle runner (2 cells)
├── dataset/                     # NOT tracked in git — supplied per platform; CHECKSUMS.txt IS tracked
│   ├── CHECKSUMS.txt            # ← committed; only proof two platforms ran on identical bytes
│   ├── paid-dataset/            # migrated from phowhisper-finetune-exp
│   ├── unpaid-dataset/
│   ├── dataset_by_task/
│   ├── vivos/                   # OOD benchmark (data.ood_eval_path)
│   ├── real-meetings-bench/     # tier 4 — ingested from old repo's done/ — NEVER a dataset_path
│   │                            #   (data.real_bench_path; real audio + post-edited transcripts)
│   └── real-clips-*/            # unlabelled real clips (data.real_clip_path)
├── experiments/
│   └── task_ledger.md          # (auto-appended)
├── outputs/                    # (auto-generated)
├── PROJECT_CORE.md              # ← THIS FILE
└── README.md                   # Entry point
```

---

## 9. SUCCESS CRITERIA

- [ ] No manual notebook editing (config-driven)
- [ ] Reproducibility (provenance.md on every run)
- [ ] Automation (Lambda sweep, SVD, gate all auto)
- [ ] Traceability (every artifact has source)
- [ ] Model versioning (HF Hub with metrics)

---

## 10. REFERENCES

- **Base Model**: `vinai/PhoWhisper-large`
- **Current Best**: `v1c-r16-valfix` (1.286% test CER, λ=0.5)
- **Lambda Sweep Logic**: `outputs/v1c-lambda-sweep-valfix/`
- **LoRA Paper**: Hu et al. (arXiv:2106.09685)
- **rsLoRA**: (arXiv:2312.03732)

---

*Last Updated: 2026-07-31*
*Status: Ready for Implementation*
