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

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.

---

## Project Context: Fine_tune_wf

Full design spec, module contracts, config schema, and gate rules live in `PROJECT_CORE.md` — read it before touching pipeline code. This section holds what that doc doesn't: operational facts, gotchas, and things that will bite you if assumed.

### State right now

As of 2026-07-31, this repo has **only `PROJECT_CORE.md` and this file** — no `src/`, no `configs/`, no dataset yet. Nothing described below as "planned" exists on disk until it's built. Check before assuming a module, config, or dataset path is present.

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

- 10 ElevenLabs voices reused across every train/test meeting — `meeting_id` is disjoint but `voice_id` is not. Test CER measures "same voices, new content," not voice generalization.
- 100% synthetic (LLM-scripted + TTS). Absolute CER doesn't transfer to real speech — relative comparisons only.
- ~62% of past in-domain CER improvement traced to digit normalization, not acoustic learning — don't attribute gains to the model without checking this first.
- Code-switch detection: don't whitelist English words (72% false-positive rate observed). Use the Vietnamese syllable-shape regex test instead (see `PROJECT_CORE.md` §4).
- `unpaid-dataset`, `dataset_by_task` haven't been profiled yet — don't assume these same warnings apply until someone checks. `done/` has been profiled (2026-07-31) and has its own set of caveats — reference transcript is post-edited PhoWhisper-small output, mixed number conventions, segments up to 212 s. See `PROJECT_CORE.md` §4 before using any tier-4 number.

### Whisper/LoRA specifics

- Target modules: `q_proj, k_proj, v_proj, out_proj, fc1, fc2` with `task_type=SEQ_2_SEQ_LM`. LLM-guide defaults (`o_proj`, `gate_proj`, `CAUSAL_LM`, etc.) don't exist on Whisper and fail silently if copied in.
- CER is the headline metric for Vietnamese ASR, not WER.
