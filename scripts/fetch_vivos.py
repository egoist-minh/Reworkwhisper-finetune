"""Fetch VIVOS (OOD benchmark) and write it in this repo's manifest schema.

`datasets.load_dataset("AILAB-VNUHCM/vivos")` is dead: the dataset repo is
script-based, and `datasets` 5.0 removed script support (see CLAUDE.md).
`datasets` is therefore NOT a dependency of this repo -- everything here goes
through `huggingface_hub` + `pyarrow` (primary) or a raw tarball (fallback).

Primary route: the repo's `refs/convert/parquet` revision, confirmed present
on 2026-07-31 (files: default/test/0000.parquet, default/train/*.parquet).
Schema confirmed on Kaggle 2026-08-01: columns are
`speaker_id, path, audio, sentence` (audio is a struct with `bytes`/`path`
sub-fields). Column names are hardcoded below -- an earlier "detect by shape"
heuristic picked `speaker_id` as the text column (it was the first
string-typed column in schema order), which is wrong; the real transcript
column is `sentence`. See ~/.claude memory `kaggle-code-never-works-first-try`.

Fallback route: the repo's default (script) revision ships raw files
`data/vivos.tar.gz` + `data/prompts-test.txt.gz` (confirmed present
2026-07-31), the original VIVOS release layout:
    test/waves/<speaker>/<utt_id>.wav
    prompts-test.txt  -- lines "<utt_id> <SENTENCE IN UPPERCASE>"
Used only if the parquet route raises.

Both routes are UNVERIFIED end-to-end (no torch/transformers/pyarrow
available in this dev environment -- see handoff). Run `--smoke` on Kaggle
first and inspect the printed schema before trusting the manifest it writes.
"""

import argparse
import gzip
import json
import shutil
import tarfile
from pathlib import Path

REPO_ID = "AILAB-VNUHCM/vivos"
PARQUET_REV = "refs/convert/parquet"
# Must match src/data.py:load_manifests's glob "manifest.*.jsonl" -- a bare
# "manifest.jsonl" does NOT match that pattern (the glob requires a non-empty
# middle component), which is exactly the bug that crashed stage_baseline's
# load_ood() on the first real Kaggle run.
MANIFEST_FILENAME = "manifest.vivos.jsonl"
# Confirmed via table.schema on Kaggle 2026-08-01 (parquet revision above).
AUDIO_COLUMN = "audio"
TEXT_COLUMN = "sentence"


def _from_parquet(out_dir: Path, split: str, limit: int | None) -> int:
    import io

    import pyarrow.parquet as pq
    import soundfile as sf
    from huggingface_hub import hf_hub_download

    remote = f"default/{split}/0000.parquet"
    local = hf_hub_download(REPO_ID, remote, repo_type="dataset", revision=PARQUET_REV)
    table = pq.read_table(local)

    audio_dir = out_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    n = min(limit, table.num_rows) if limit else table.num_rows

    manifest_path = out_dir / MANIFEST_FILENAME
    with open(manifest_path, "w", encoding="utf-8") as mf:
        for i in range(n):
            row_audio = table.column(AUDIO_COLUMN)[i].as_py()
            text = table.column(TEXT_COLUMN)[i].as_py()
            audio_bytes = row_audio["bytes"]
            wav_name = f"vivos_{split}_{i:05d}.wav"
            data, sr = sf.read(io.BytesIO(audio_bytes))
            sf.write(audio_dir / wav_name, data, sr)
            mf.write(json.dumps({
                "audio_filepath": wav_name,
                "segment_id": f"vivos_{split}_{i:05d}",
                "text": text,
                "lang": "vi",
                "source": "vivos",
                "split": "test" if split == "test" else "train",
            }, ensure_ascii=False) + "\n")
    return n


def _from_tarball(out_dir: Path, split: str, limit: int | None) -> int:
    from huggingface_hub import hf_hub_download

    prompts_gz = hf_hub_download(REPO_ID, f"data/prompts-{split}.txt.gz", repo_type="dataset")
    tar_gz = hf_hub_download(REPO_ID, "data/vivos.tar.gz", repo_type="dataset")

    work = out_dir / "_raw"
    work.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_gz) as tf:
        tf.extractall(work)  # nosec -- trusted HF dataset repo, not user input
    with gzip.open(prompts_gz, "rt", encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f if ln.strip()]
    if limit:
        lines = lines[:limit]

    audio_dir = out_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    waves_root = work / "vivos" / split / "waves"

    manifest_path = out_dir / MANIFEST_FILENAME
    n = 0
    with open(manifest_path, "w", encoding="utf-8") as mf:
        for line in lines:
            utt_id, _, text = line.partition(" ")
            speaker = utt_id.rsplit("_", 1)[0]
            src_wav = waves_root / speaker / f"{utt_id}.wav"
            if not src_wav.exists():
                raise FileNotFoundError(f"prompts-{split}.txt references missing wav: {src_wav}")
            wav_name = f"{utt_id}.wav"
            shutil.copy(src_wav, audio_dir / wav_name)
            mf.write(json.dumps({
                "audio_filepath": wav_name,
                "segment_id": utt_id,
                "text": text.lower(),
                "lang": "vi",
                "source": "vivos",
                "split": "test" if split == "test" else "train",
            }, ensure_ascii=False) + "\n")
            n += 1
    shutil.rmtree(work, ignore_errors=True)
    return n


def fetch(out_dir: Path, split: str = "test", limit: int | None = None) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        n = _from_parquet(out_dir, split, limit)
        print(f"parquet route: wrote {n} segments")
    except Exception as e:
        print(f"parquet route failed ({type(e).__name__}: {e}); falling back to tarball")
        n = _from_tarball(out_dir, split, limit)
        print(f"tarball route: wrote {n} segments")
    return out_dir / MANIFEST_FILENAME


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="dataset/vivos")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=None,
                     help="cap segments -- use a small number for --smoke")
    ap.add_argument("--smoke", action="store_true",
                     help="limit=5 and print schema, for a first Kaggle sanity check")
    args = ap.parse_args()
    limit = 5 if args.smoke else args.limit
    manifest = fetch(Path(args.out), args.split, limit)
    print(f"manifest: {manifest}")
