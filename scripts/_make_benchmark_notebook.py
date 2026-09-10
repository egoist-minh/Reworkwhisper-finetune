"""Generate notebooks/benchmark-matrix.ipynb. Run once; edit the notebook after.

A generator rather than a hand-written .ipynb because the JSON is unreadable in a
diff -- same reason the repo keeps notebook logic in scripts/ and the notebook thin.

    python scripts/_make_benchmark_notebook.py
"""
import json
from pathlib import Path

CELLS: list[tuple[str, str]] = []


def md(text):
    CELLS.append(("markdown", text.strip("\n")))


def code(text):
    CELLS.append(("code", text.strip("\n")))


md(r"""
# Benchmark matrix — 3 models × 5 benchmark sets (Kaggle free tier)

Scores `vinai/PhoWhisper-large`, `winhsss/Reworkwhisper-large-v5` and ElevenLabs Scribe v2
on `vimedcss-test`, `vimedcss-hard`, `youtube-test`, `synthetic-test`, `vivos` and
`cross-domain` — one decode path, one normalizer, one table.

All logic lives in the repo (`scripts/benchmark_suites.py`, `scripts/benchmark_run.py`,
`scripts/benchmark_report.py`). **This notebook is the only place a `/kaggle/input/...`
path appears.** Cell 2 clones from GitHub `main`, so commit and push before running.

## Before running

1. Settings: `Accelerator = GPU T4 x2` (one GPU is used), `Internet = On`.
2. Add-ons → Secrets → `HF_TOKEN` (required: `Reworkwhisper-large-v5` is private) and,
   only if you run the paid backend, `ELEVENLABS_API_KEY`.
3. Add Data — attach these Kaggle Datasets:
   - `paid-dataset-v2` and `youtube-meetings` (merged into `mixed-noisy-v1` by cell 6;
     this is where `youtube-test` and `synthetic-test` come from)
   - `cross-domain-bench` — build the upload with
     `python -m scripts.zip_for_kaggle --src dataset/youtube-data-pilot-package --out
     cross-domain-bench.zip` (3 meetings / 299 segments / 78 min). Do NOT zip it with a
     Windows tool: a backslash separator inside the archive extracts as one flat file
     named `IGZYBrbDUEw\seg_0000.wav` and every audio path then misses.
   VIVOS is fetched from the Hub by cell 7; ViMedCSS is streamed from the Hub.

## Cost, and how not to pay it twice

| | segments | audio | ~T4 time per model |
|---|---:|---:|---:|
| `vimedcss-test` | 1,615 | 3.38 h | ~1.5 h |
| `vimedcss-hard` | 658 | 1.38 h | ~0.6 h |
| `youtube-test` | 228 | 1.00 h | ~0.3 h |
| `synthetic-test` | 426 | 0.58 h | ~0.2 h |
| `vivos` | 760 | ~0.75 h | ~0.3 h |
| `cross-domain` | 299 | 1.30 h | ~0.4 h |

Two GPU models over everything is ~6.6 h — inside one 12 h session, but it eats a fifth
of the weekly free-tier quota. The ViMedCSS cell defaults to `REUSE_VIMEDCSS = False`, so
both splits are decoded fresh in this run. Set it to `True` to import the 2026-09-09
decodes from `Outputs/vimedcss_output/` instead and spend ~4.2 h less.

ElevenLabs is billed by audio hour (~$0.22/h): the full matrix is ~8.4 h ≈ **$1.9**.
Every backend resumes by `segment_id`, so an interrupted session never re-decodes — and
never re-bills — what it already wrote.
""")

md("## 1. Clone the repo")

code(r"""
import os

# Single GPU: nothing here is distributed, and a second visible device only invites
# accelerate to wrap things in DataParallel.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# transformers' safetensors auto-conversion probe 403s on vinai/PhoWhisper-* (discussions
# disabled). Must be set before transformers is imported anywhere.
os.environ["TRANSFORMERS_AUTO_CONVERSION"] = "0"

REPO_URL = "https://github.com/egoist-minh/Reworkwhisper-finetune.git"
REPO_DIR = "/kaggle/working/Reworkwhisper-finetune"

if os.path.exists(REPO_DIR):
    %cd {REPO_DIR}
    !git pull origin main
else:
    !git clone {REPO_URL} {REPO_DIR}
    %cd {REPO_DIR}
""")

code(r"""
!pip install -q -r requirements.txt
# `datasets` is deliberately not in requirements.txt (see its header); only the two
# Hub-hosted suites need it. `requests` ships with the Kaggle image.
!pip install -q datasets
""")

code(r"""
# Feature-detect, never branch on version strings: Kaggle patches images in place.
import sys
import torch

for name in ("torch", "transformers", "datasets", "numpy", "scipy", "soundfile",
             "huggingface_hub", "requests"):
    try:
        print(f"{name:18s} {__import__(name).__version__}")
    except Exception as e:
        print(f"{name:18s} <{type(e).__name__}>")
print("python", sys.version.split()[0])
print("cuda:", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no gpu",
      "| capability:", torch.cuda.get_device_capability(0) if torch.cuda.is_available() else "-")

try:
    from kaggle_secrets import UserSecretsClient
    secrets = UserSecretsClient()
    os.environ["HF_TOKEN"] = secrets.get_secret("HF_TOKEN")
    print("HF_TOKEN loaded")
except Exception as e:
    print(f"HF_TOKEN NOT loaded ({type(e).__name__}) -- Reworkwhisper-large-v5 will 401")
""")

md("""
## 2. Locate the attached datasets

Kaggle mount paths nest one level deeper than the dataset name
(`/kaggle/input/datasets/<user>/<slug>/<slug>`). Re-run this and read it — the shape has
bitten this project before.
""")

code("!ls -la /kaggle/input")

md("""
## 3. Config — the only cell to edit

Every path below is passed to the scripts as `--path <suite>=<dir>`. Nothing in `src/`
or `scripts/` knows a Kaggle path.
""")

code(r"""
from pathlib import Path

RUN_DATE = "2026-09-10"
OUT_DIR = f"/kaggle/working/benchmark-{RUN_DATE}"     # outside the repo: `git pull` must not touch it
Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

BASE_MODEL = "vinai/PhoWhisper-large"
CAND_MODEL = "winhsss/Reworkwhisper-large-v5"
EL_MODEL   = "scribe_v2"                # ElevenLabs model_id; cell 12 probes it first

# --- suite -> directory. Edit the two /kaggle/input paths to match cell 5's listing. ---
MIXED_PATH  = "/kaggle/working/dataset/mixed-noisy-v1"                       # written by cell 6
VIVOS_PATH  = "/kaggle/working/Reworkwhisper-finetune/dataset/vivos"          # written by cell 7
CROSS_PATH  = "/kaggle/input/datasets/winhkento/cross-domain-bench"   # confirmed 2026-09-10: this one is NOT doubled

PATHS = (f"--path youtube-test={MIXED_PATH} "
         f"--path synthetic-test={MIXED_PATH} "
         f"--path vivos={VIVOS_PATH} "
         f"--path cross-domain={CROSS_PATH}")

GPU_SUITES = "vimedcss-test,vimedcss-hard,youtube-test,synthetic-test,vivos,cross-domain"
BATCH_SIZE = 8          # eval batch on one T4, same as the gate's eval.batch_size
LIMIT      = ""         # "--limit 8" for a smoke run of the whole notebook first

print(OUT_DIR)
print(PATHS)
""")

md("""
## 4. Build the local corpora

`mixed-noisy-v1` supplies `youtube-test` (228) and `synthetic-test` (426) — the same test
split `v4-mixed-r16` was gated on. VIVOS comes from the Hub.
""")

code(r"""
# Merge the two attached corpora. Edit both --sources to match cell 5's listing.
# Re-running on a non-empty destination is refused: rm -rf it first if this dies partway.
!python -m scripts.build_mixed_dataset \
    --sources /kaggle/input/datasets/winhkento/paid-dataset-v2/paid-dataset-v2 \
              /kaggle/input/datasets/winhkento/youtube-meetings/youtube-meetings \
    --out {MIXED_PATH}
""")

code(r"""
# VIVOS: --smoke first, read the printed schema, then the full test split.
!python scripts/fetch_vivos.py --out dataset/vivos --smoke
""")

code(r"""
!python scripts/fetch_vivos.py --out dataset/vivos
""")

md("""
## 5. Reuse the ViMedCSS decodes this project already has

`Outputs/vimedcss_output/` holds base + v5 on both ViMedCSS splits, decoded 2026-09-09 by
`notebooks/vimedcss-eval.ipynb` with the identical decode path this harness uses: greedy
(`num_beams=1`), `language="vi"`, `task="transcribe"`, batch 8, 30 s chunking, no n-gram
ban. Copying them in costs nothing and saves ~4.2 GPU hours; the run below then skips
those two suites for those two models because every segment_id is already present.

`--force` on a later run, or deleting these files, re-decodes them from scratch.
""")

code(r"""
import shutil

REUSE_VIMEDCSS = False    # True -> reuse the 2026-09-09 decodes instead (~4.2 T4 hours cheaper)

# The stored files are named `<model>.<split>`; this harness keys suites `vimedcss-<split>`.
RENAME = {
    "vinai_phowhisper_large.test.persegment.jsonl":            "vinai_phowhisper_large.vimedcss-test.persegment.jsonl",
    "vinai_phowhisper_large.hard.persegment.jsonl":            "vinai_phowhisper_large.vimedcss-hard.persegment.jsonl",
    "winhsss_reworkwhisper_large_v5.test.persegment.jsonl":    "winhsss_reworkwhisper_large_v5.vimedcss-test.persegment.jsonl",
    "winhsss_reworkwhisper_large_v5.hard.persegment.jsonl":    "winhsss_reworkwhisper_large_v5.vimedcss-hard.persegment.jsonl",
}

if REUSE_VIMEDCSS:
    src_dir = Path("Outputs/vimedcss_output")
    for old, new in RENAME.items():
        s, d = src_dir / old, Path(OUT_DIR) / new
        if not s.exists():
            print(f"MISSING {s} -- this suite will be decoded from scratch")
        elif d.exists():
            print(f"kept    {d.name} (already in the run dir)")
        else:
            shutil.copyfile(s, d)
            print(f"copied  {d.name} ({sum(1 for _ in d.open(encoding='utf-8'))} rows)")
else:
    stale = sorted(Path(OUT_DIR).glob("*.vimedcss-*.persegment.jsonl"))
    print("REUSE_VIMEDCSS = False -- ViMedCSS is decoded fresh in this session "
          "(~4.2 T4 hours for two models over both splits)")
    if stale:
        # Resume matches on segment_id, so leftovers from an earlier session would be
        # kept and only the gaps re-decoded -- not a fresh run. Say so; deleting a
        # decode is the operator's call, not this cell's.
        print(f"\n{len(stale)} ViMedCSS file(s) ALREADY in {OUT_DIR}:")
        for f in stale:
            print(f"  {f.name} ({sum(1 for _ in f.open(encoding='utf-8'))} rows)")
        print("Resume will keep these. For a genuinely fresh decode, delete them first:")
        print(f"  !rm {OUT_DIR}/*.vimedcss-*.persegment.jsonl")
""")

md("""
### Hypotheses already paid for

`Outputs/benchmark-2026-09-10/` holds decodes tracked in git precisely because they cost
money or GPU hours and `/kaggle/working` does not survive the session. Copying them in
makes the runs below skip those segments.
""")

code(r"""
import shutil          # re-imported: this cell must work even if cell 14 was skipped

PAID_DIR = "Outputs/benchmark-2026-09-10"

for src in sorted(Path(PAID_DIR).glob("*.persegment.jsonl")):
    dst = Path(OUT_DIR) / src.name
    if dst.exists():
        print(f"kept   {dst.name} (already in the run dir)")
    else:
        shutil.copyfile(src, dst)
        print(f"copied {dst.name} ({sum(1 for _ in dst.open(encoding='utf-8'))} rows)")
""")

md("""
## 6. Decode — baseline, then the candidate

Two separate cells, one model each: a Kaggle session that dies between them keeps the
first model's output, and both are resumable by `segment_id` anyway.
""")

code(r"""
!python -m scripts.benchmark_run \
    --backend hf --model {BASE_MODEL} \
    --suites {GPU_SUITES} {PATHS} \
    --out {OUT_DIR} --batch-size {BATCH_SIZE} {LIMIT}
""")

code(r"""
!python -m scripts.benchmark_run \
    --backend hf --model {CAND_MODEL} \
    --suites {GPU_SUITES} {PATHS} \
    --out {OUT_DIR} --batch-size {BATCH_SIZE} {LIMIT}
""")

md("""
## 7. ElevenLabs Scribe v2 — PAID, opt-in

Needs `Internet = On` and `ELEVENLABS_API_KEY`. No GPU: if the GPU cells are done, this
can run in a separate CPU session on the same `OUT_DIR` to save quota.

**Probe one segment first.** `scribe_v2` is the model_id this project's earlier
comparisons used; if the account or the API disagrees, the probe fails on one segment
instead of after an hour of billed calls.

**On a free-tier key:** the plan has a monthly credit cap and a low concurrency limit.
This backend sends one request at a time, so concurrency is not a problem, but the cap
almost certainly is — the full matrix is ~8.4 h of audio. Run one suite at a time,
smallest first. A 429 is retried three times with backoff and then written to
`<model>.<suite>.failures.jsonl`; whatever did land is kept, and the next run resumes
from there. Check the plan's commercial-use terms before a number from it goes on a
slide.
""")

code(r"""
# Self-contained: cell 4's `secrets` may not exist if the HF_TOKEN load failed there.
# Kaggle Secrets first; typed input as the fallback so a local/Colab run also works.
try:
    from kaggle_secrets import UserSecretsClient
    os.environ["ELEVENLABS_API_KEY"] = UserSecretsClient().get_secret("ELEVENLABS_API_KEY")
    print("ELEVENLABS_API_KEY loaded from Kaggle Secrets")
except Exception as e:
    from getpass import getpass
    print(f"Kaggle Secrets unavailable ({type(e).__name__})")
    os.environ["ELEVENLABS_API_KEY"] = getpass("ELEVENLABS_API_KEY: ")

# The key is only ever read from the environment -- never written to a notebook cell,
# an output file, or the run directory.
print("key set:", bool(os.environ.get("ELEVENLABS_API_KEY")))
""")

code(r"""
# PROBE: one segment, one call. Writes to a throwaway directory, not OUT_DIR.
!python -m scripts.benchmark_run \
    --backend elevenlabs --model {EL_MODEL} \
    --suites cross-domain {PATHS} \
    --out /kaggle/working/_el_probe --limit 1
# `--limit 1` is what makes this a probe. Removing it sends the WHOLE suite to the
# paid API -- that is how 299 cross-domain segments got billed on 2026-09-10.
!cat /kaggle/working/_el_probe/*.persegment.jsonl
""")

code(r"""
# ONE SUITE PER RUN. Not GPU_SUITES: a free-tier key runs out of credit mid-way, and a
# half-decoded suite narrows the shared segment set for EVERY model on that suite,
# not just this one. Finish a suite, save it, then change this line.
#   youtube-test 1.00 h -> synthetic-test 0.58 h -> cross-domain 1.30 h (done)
#   -> vivos 0.75 h -> vimedcss-test 3.38 h -> vimedcss-hard 1.38 h
EL_SUITE = "youtube-test"

!python -m scripts.benchmark_run \
    --backend elevenlabs --model {EL_MODEL} \
    --suites {EL_SUITE} {PATHS} \
    --out {OUT_DIR} {LIMIT}
""")

md("""
## 8. The table

`N3` is this repo's default normalization — the scale every other number in
`docs/so-lieu-tong-hop.md` sits on. `N0` is the raw variant that reproduced ViMedCSS's
Table 4 base row; the paper comparison must use that one. They are two runs of the same
script, never one mixed table.
""")

code(r"""
!python -m scripts.benchmark_report --dir {OUT_DIR} --variant N3 --baseline vinai_phowhisper_large
""")

code(r"""
!python -m scripts.benchmark_report --dir {OUT_DIR} --variant N0 \
    --baseline vinai_phowhisper_large --out {OUT_DIR}/scores.N0.json
""")

code(r"""
import shutil
zip_path = shutil.make_archive(f"/kaggle/working/benchmark-{RUN_DATE}", "zip", OUT_DIR)
print(zip_path, Path(zip_path).stat().st_size / 1e6, "MB")
""")


def main() -> None:
    nb = {
        "cells": [
            {"cell_type": kind, "metadata": {},
             **({"execution_count": None, "outputs": []} if kind == "code" else {}),
             "source": (src + "\n").splitlines(keepends=True)}
            for kind, src in CELLS
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }
    out = Path("notebooks/benchmark-matrix.ipynb")
    out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {out} ({len(CELLS)} cells)")


if __name__ == "__main__":
    main()
