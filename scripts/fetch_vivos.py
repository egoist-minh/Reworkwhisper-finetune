"""Fetch VIVOS (OOD benchmark) and write it in this repo's manifest schema.

`datasets.load_dataset("AILAB-VNUHCM/vivos")` is dead: the dataset repo is
script-based, and `datasets` 5.0 removed script support (see CLAUDE.md).
`datasets` is therefore NOT a dependency of this repo -- everything here goes
through `huggingface_hub` + `pyarrow` (primary) or a raw tarball (fallback).

Primary route: the repo's `refs/convert/parquet` revision, confirmed present
on 2026-07-31 (files: default/test/0000.parquet, default/train/*.parquet).
Column names are NOT hardcoded -- HF's auto-conversion schema has not been
observed directly in this repo, so the audio/text columns are detected by
shape (an audio column is a struct with a `bytes` sub-field) rather than by
assumed name. See ~/.claude memory `kaggle-code-never-works-first-try`.

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


def _find_audio_text_columns(table) -> tuple[str, str]:
    """Detect the audio struct column and the text column by shape, not name."""
    audio_col = text_col = None
    for name in table.column_names:
        field = table.schema.field(name)
        t = field.type
        if str(t.__class__.__name__) == "StructType" or (
            hasattr(t, "names") and "bytes" in getattr(t, "names", [])
        ):
            audio_col = name
        elif str(t) in ("string", "large_string"):
            text_col = text_col or name
    if audio_col is None or text_col is None:
        raise RuntimeError(
            f"could not detect audio/text columns from schema: {table.schema}"
        )
    return audio_col, text_col


def _from_parquet(out_dir: Path, split: str, limit: int | None) -> int:
    import io

    import pyarrow.parquet as pq
    import soundfile as sf
    from huggingface_hub import hf_hub_download

    remote = f"default/{split}/0000.parquet"
    local = hf_hub_download(REPO_ID, remote, repo_type="dataset", revision=PARQUET_REV)
    table = pq.read_table(local)
    audio_col, text_col = _find_audio_text_columns(table)
    print(f"detected columns: audio={audio_col!r} text={text_col!r}")

    audio_dir = out_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    n = min(limit, table.num_rows) if limit else table.num_rows

    manifest_path = out_dir / "manifest.jsonl"
    with open(manifest_path, "w", encoding="utf-8") as mf:
        for i in range(n):
            row_audio = table.column(audio_col)[i].as_py()
            text = table.column(text_col)[i].as_py()
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

    manifest_path = out_dir / "manifest.jsonl"
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
    return out_dir / "manifest.jsonl"


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
