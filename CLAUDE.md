# CLAUDE.md

Behavioral guidelines to reduce common LLM coding mistakes, merged with project-specific instructions for the PhoWhisper fine-tune pipeline.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them — don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.
- Config-specific: before writing code, check "is this already a field in `configs/experiment.yaml`?" — see [PROJECT_CORE.md §3](PROJECT_CORE.md).

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested — the exception is the flexibility already decided in this project (base_model, dataset_path, ood_eval_path, real_clip_path are config fields by design, not speculative additions).
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it — don't delete it.
- Never edit `phowhisper-finetune-exp` (the predecessor repo) — reference only, don't touch its files.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"
- Pipeline-specific: any stage's success criteria is a gate (§6 in `PROJECT_CORE.md`), not "looks right" — a stage isn't done until its gate check passes or explicitly fails loud.

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

## 5. Name Things By Their Real Names

**Don't coin terms. Don't use IDs as nouns.**

- **Use the name the tool, API, or codebase actually uses.** If you don't know it, look it up or describe the thing in plain words. Never invent a label and then reuse it as though it were established vocabulary. Real example of getting this wrong: "track upload" and `kind: asr` for YouTube captions — the first appears in no documentation anywhere, and the second is a YouTube Data API field while the code uses `yt-dlp`, whose actual field names are `automatic_captions` and `subtitles`. A coined name is also an unverifiable claim: the reader can't check it, so it reads as authority it hasn't earned.
- **Row/tier/step IDs are pointers, not names.** `F2`, `tier4a`, `E5`, "bước 6" mean nothing to a reader who isn't holding the table open. On first use in a message, say what the thing does — "bước tải audio (F2)" — after which the bare ID is fine for the rest of that message.
- **Separate identifiers that exist from ones you just made up.** `tier4a` is in `src/gate.py`; `real_0002/seg_0074` is in the corpus; a phrase invented this session is in neither. Presenting all three in the same tone makes the invented one look like a fact.
- **Terse is not the same as compressed-by-labelling.** Swapping a description for a short coined label saves the writer tokens and costs the reader comprehension — a net loss, and the opposite of what a brevity instruction asks for. This holds under any style directive, including an active terse/caveman mode.

The test: could the reader grep the term and find it — in this repo, in a dependency, or in that tool's own docs? If not, use the real term or spell out what it means.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.

---

## Project Context: Fine_tune_wf

Full design spec, module contracts, config schema, and gate rules live in `PROJECT_CORE.md` — read it before touching pipeline code. **How to actually run either pipeline — every command, artifact and gate, in order — lives in `docs/runbook.md`.** This section holds what neither doc does: operational facts, gotchas, and things that will bite you if assumed.

### State right now

As of 2026-09-09, `src/`, `configs/`, `scripts/`, `tests/`, `dataset/` and two runs' artifacts (`Outputs/v3-r16/`, `Outputs/v4-mixed-r16/`) all exist. Both have run end-to-end on Kaggle T4 and both are published: `Reworkwhisper-large-v4` = `v3-r16` at λ=0.5, `Reworkwhisper-large-v5` = `v4-mixed-r16` at λ=0.75 (in production since 2026-08-18). Repo names lag run ids — don't map them by number.

`data.dataset_path` still points at `dataset/mixed-noisy-v1` (the `v4-mixed-r16` corpus) — that one is now present and checksum-verified on this machine too, not Kaggle-only as before. `dataset/youtube-meetings/manifest.*.jsonl` here is still the **pre-review** snapshot (all `verified: false`) even though the corpus that trained `v4-mixed-r16` was reviewed 790/790. Still check before assuming a specific module or config field is present — anything described below as "planned" or "todo" (see `SESSIONS.md`) does not exist on disk until it's built.

**The next run trains on `v6-corpus`, not `mixed-noisy-v1`.** Built 2026-09-12 by
`scripts/build_v6_corpus.py` (commit `367d88a`) from two HuggingFace train-only drops
(`rework-whisper-v6-org/youtube-meeting-1`, `rework-whisper-v6-org/50h-elevenlab-data`)
plus the local val/test splits at `Outputs/splits-export/{val,test}`. Published **private**
as `rework-whisper-v6-org/v6-corpus` (one file, `v6-corpus.tar`, 14.7 GB) — it is not on
this machine or on any GPU box right now, only on HF. 35,544 records: train 34,525 seg /
99.28 h / 286 meetings (50.3% synthetic / 49.7% YouTube), val 365 seg / 0.79 h, test 654
seg / 1.58 h. `configs/experiment.yaml` still says `run_id: v4-mixed-r16` and
`dataset_path: dataset/mixed-noisy-v1` — a v6 run overrides both at the command line
(`docs/server-finetune.md` §4's `R=v6-<mô tả>` recipe), it does not edit the config file.
Test voice_ids are disjoint from train's; val's are not (all 7 of val's appear in train) —
see `v6-corpus-merge-blockers.md` in memory for the full list of things that differ from
what the two source repos' own cards claim.

### Relationship to `D:\phowhisper-finetune-exp`

That's the predecessor repo — real git history, real training runs, real evidence (`outputs/v1c-lambda-sweep-valfix/`, `outputs/v1c_lambda05_*`, `docs/*.md`). This repo does **not** inherit its files wholesale.

- **Dataset**: copy all four folders into `Fine_tune_wf/dataset/`, then this repo is fully independent — no path back to the old repo. They are **not the same kind of thing**: `paid-dataset`, `unpaid-dataset`, `dataset_by_task` are synthetic training data; **`done/` is the real-audio tier-4 benchmark** (2 recordings, 43.2 min transcribed, human-edited ASR draft) → becomes `data.real_bench_path` and must never be used as `dataset_path`. See `PROJECT_CORE.md` §4.
- **Code / notebooks**: reference only. Its `src/` is empty — the real logic lives in `notebooks/finetune-paid-dataset.ipynb`, `lambda-sweep-v1c.ipynb`, `build-publish-lambda-model.ipynb`, and its own `CLAUDE.md`. Read those when porting logic into `src/` here. Do not copy old `outputs/`/`docs/` run artifacts into this repo — they stay in the old repo as the historical record.
- If a number here ever needs checking against a past run, go look in the old repo — don't recreate it from memory.

### Non-obvious design decisions (settled, don't re-litigate)

- **Nothing is hardcoded to PhoWhisper or VIVOS.** `base_model`, `data.dataset_path`, `data.ood_eval_path`, `data.real_bench_path` are all config fields (`configs/experiment.yaml`); `data.real_clip_path` is optional. PhoWhisper-large / VIVOS / paid-dataset are just the current defaults, not assumptions baked into code. No literal LoRA rank anywhere either — always `lora.rank`.
- **λ* sweep has no soft fallback.** If no λ in the grid keeps OOD CER within budget, the pipeline fails hard — no adapter selected, nothing pushed to HF. Never silently fall back to λ=0 or "closest λ".
- **Platform-independent by requirement.** `src/pipeline.py` must run the same on Kaggle, local, Colab, or a future dedicated server — Python + declared deps only. No `/kaggle/input/...` paths or notebook-only assumptions baked into `src/`. (The old repo's notebook is Kaggle-only; don't carry that constraint forward.)
- **Versioning is git commit + run_id + `provenance.md` + `experiments/task_ledger.md`.** No DVC/MLflow/W&B — deliberately out of scope.
- **Fail loud everywhere.** Any gate tier failing, any config validation failing → raise, stop the pipeline. No silent skips, no partial success states.

### Dataset facts inherited from paid-dataset (verify before generalizing to the other 3 sets)

- This was true of `paid-dataset` (v1): 10 ElevenLabs voices reused across every train/test meeting — `meeting_id` disjoint but `voice_id` not, so test CER measured "same voices, new content," not voice generalization. **Verified fixed in `paid-dataset-v2`** (2026-08-04, run v3-r16's `validated_manifest.jsonl`: train has 10 voice_ids, test has 9, zero overlap) — v2's test CER does measure voice generalization. Don't assume either way for a new run; check `voice_id` overlap between splits in that run's manifest before citing this.
- 100% synthetic (LLM-scripted + TTS). Absolute CER doesn't transfer to real speech — relative comparisons only.
- ~62% of past in-domain CER improvement traced to digit normalization, not acoustic learning — don't attribute gains to the model without checking this first.
- Code-switch detection: don't whitelist English words (72% false-positive rate observed). Use the Vietnamese syllable-shape regex test instead (see `PROJECT_CORE.md` §4).
- `unpaid-dataset`, `dataset_by_task` haven't been profiled yet — don't assume these same warnings apply until someone checks. `done/` has been profiled (2026-07-31) and has its own set of caveats — reference transcript is post-edited PhoWhisper-small output, mixed number conventions, segments up to 212 s. See `PROJECT_CORE.md` §4 before using any tier-4 number.

### "Tiến hành fine tune" = chạy trên server GPU thuê

Cách chạy hiện tại là thuê một máy GPU theo giờ và chạy `src/pipeline.py` qua SSH — không
phải Kaggle. Toàn bộ quy trình, từ sửa IP tới chép kết quả về, nằm trong
**`docs/server-finetune.md`**; đọc file đó trước khi gõ lệnh. Bốn điều hay quên nhất:

- **IP đổi mỗi lần thuê.** Host `speech-agent-gpu` trong `~/.ssh/config` trỏ tới lần thuê
  trước. Hỏi người dùng IP mới, sửa `HostName`, rồi khảo sát máy — đừng giả định phần cứng.
- **`pip install torch` hay lấy sai bản CUDA** so với driver trên máy. Kiểm
  `torch.cuda.is_available()` trước khi chạy bất cứ thứ gì.
- **`dataset/vivos` không nằm trong gói corpus**, phải dựng bằng `scripts/fetch_vivos.py`.
  Thiếu nó là `--stage baseline` chết.
- **Chép `outputs/<run_id>/` về trước khi trả máy.** Đã mất một lượt đo vì chép muộn.
- **Run đứt thì `--stage train --resume`, đừng xoá `checkpoints/`.** Chạy lại stage train
  khi đã có checkpoint là `FileExistsError` cố ý, và thông báo đưa ra cả hai lựa chọn —
  xoá là mất số giờ GPU đã trả. Chỉ xoá khi thật sự muốn train lại từ step 1.

Số liệu chi phí đo được trên H200 (0,728 giây mỗi step, ngân sách cho corpus 100 giờ, so
sánh `eval.batch_size`): `docs/h200-timing-probe-2026-09-11.md`.

### Kaggle runner notebooks — two files, don't mix them up

`notebooks/fine-tune-workflow.ipynb` is the **current template**: 35 cells, zero execution counts, `RUN_ID = "v4-mixed-r16"`. This is the one to edit and run. `docs/runbook.md` Part B walks it cell by cell.

`notebooks/fine-tune-workflow.v3-r16-executed.ipynb` is the **executed record** of the v3-r16 session: 30 cells, 14 with real execution counts and preserved outputs. Reference only.

- **Real mount paths are nested**: `/kaggle/input/datasets/<user>/<slug>/<slug>`, not `/kaggle/input/<slug>`. The `!ls -la /kaggle/input` cell exists to confirm this before `DATASET_PATH` / `REAL_BENCH_PATH` are set — always re-run it, the shape has bitten this project once already.
- **`TRANSFORMERS_AUTO_CONVERSION=0` is required**, set twice: as an env var in the clone cell and inline on the `--stage baseline` command. Don't drop either when editing.
- **The executed notebook's source and its own outputs disagree on the run's name.** Its `RUN_ID` cell was later edited to `v4-mixed-r16`, but its 47 preserved output paths still say `outputs/v0-r16/`; the same run's stored artifacts here are `Outputs/v3-r16/` (whose `config.json` says `run_id: v3-r16`). Three labels, one run — never cite a directory name from that notebook.
- **Its `gate_results.json` output is pre-rejoin-fix**: tier4a real CER 48.01% vs. baseline 45.76%, 146+118 segments. The authoritative rescored numbers are in `Outputs/v3-r16/metrics/gate_results.json` (35.07% vs. 31.57%, 115+81 segments, `_note` records the 2026-08-04 fix). Quote the stored file, not the notebook cell.
- Run cost for scale: 3 epochs, 801 steps, **6 h 31 m** on a single T4 for PhoWhisper-large at `lora.rank=16`, batch 2 × grad_accum 8.

### Whisper/LoRA specifics

- Target modules: `q_proj, k_proj, v_proj, out_proj, fc1, fc2` with `task_type=SEQ_2_SEQ_LM`. LLM-guide defaults (`o_proj`, `gate_proj`, `CAUSAL_LM`, etc.) don't exist on Whisper and fail silently if copied in.
- CER is the headline metric for Vietnamese ASR, not WER.
