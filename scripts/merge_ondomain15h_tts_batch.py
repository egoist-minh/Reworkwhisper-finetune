"""Merge dataset/v6-ondomain-15h (48 train meetings, "lõi dữ liệu đậm" per
docs/yeu-cau-lo-tts-10h.md §1) with the delivered TTS batch
dataset/paid-meeting-vi-0247-0296 into one self-contained corpus.

Not a bare rerun of scripts/build_mixed_dataset.py: dataset/v6-ondomain-15h/audio
is a Windows junction to the WHOLE dataset/v6-corpus/audio pool (14.7 GB,
scripts/filter_corpus_density.py:_write), so copying "everything under audio/"
would pull all of v6-corpus, not just the ~48 meetings v6-ondomain-15h's own
manifests reference. This copies only the audio each source's own manifest
records point at.

The TTS batch's raw manifests (dataset/paid-meeting-vi-0247-0296/dataset/
manifests/<meeting_id>.jsonl) use `"split": "train"`, which src.data.resolve_splits
rejects outright (it only accepts "demo"/"test") -- remapped to "demo" here, same
as scripts/ingest_paid_dataset_v2.py:remap_dot2_split, so resolve_splits assigns
every one of these meetings to actual train (none are in data.val_meetings).

    python -m scripts.merge_ondomain15h_tts_batch --out dataset/v6-ondomain-15h-tts5h
"""

import argparse
import json
import shutil
from datetime import date
from pathlib import Path

from src.config import load as load_config
from src.data import load_manifests, resolve_splits, split_stats

ONDOMAIN_SRC = Path("dataset/v6-ondomain-15h")
TTS_SRC = Path("dataset/paid-meeting-vi-0247-0296/dataset")


def _copy_referenced_audio(records: list[dict], audio_root: Path, out_audio: Path) -> None:
    for r in records:
        src = audio_root / r["audio_filepath"]
        dst = out_audio / r["audio_filepath"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def _ingest_ondomain15h(out: Path) -> list[dict]:
    all_records = []
    for manifest_path in sorted(ONDOMAIN_SRC.glob("manifest.*.jsonl")):
        records = [json.loads(l) for l in
                   manifest_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        shutil.copy2(manifest_path, out / manifest_path.name)
        _copy_referenced_audio(records, ONDOMAIN_SRC / "audio", out / "audio")
        all_records.extend(records)
    return all_records


def _ingest_tts_batch(out: Path) -> list[dict]:
    all_records = []
    for manifest_path in sorted((TTS_SRC / "manifests").glob("*.jsonl")):
        meeting_id = manifest_path.stem
        records = [json.loads(l) for l in
                   manifest_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        remapped = []
        for r in records:
            if r["split"] != "train":
                raise ValueError(f"expected split='train' on TTS batch record, "
                                  f"got {r['split']!r} ({meeting_id}/{r['segment_id']})")
            remapped.append({**r, "split": "demo"})
        (out / f"manifest.{meeting_id}.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in remapped) + "\n",
            encoding="utf-8")
        _copy_referenced_audio(remapped, TTS_SRC / "audio", out / "audio")
        all_records.extend(remapped)
    return all_records


def _write_provenance(out: Path, ondomain_records: list[dict], tts_records: list[dict]) -> None:
    lines = [
        f"# {out.name} -- provenance",
        "",
        f"Built {date.today().isoformat()} by scripts/merge_ondomain15h_tts_batch.py from:",
        "",
        f"- `{ONDOMAIN_SRC}` (\"lõi dữ liệu đậm\", docs/yeu-cau-lo-tts-10h.md §1): "
        f"{len(ondomain_records)} records, audio copied per-record (source audio/ is a "
        "junction to the shared v6-corpus pool, not copied whole).",
        f"- `{TTS_SRC}` (data card: `{TTS_SRC.parent}/docs/"
        f"datacard-paid-meeting-0247-0296.md`, delivered 2026-09-15, 5,37 h / 50 cuộc "
        "against a 10 h ask): all 50 meetings, `split` remapped train->demo.",
        "",
        "Acceptance check against docs/yeu-cau-lo-tts-10h.md §3: 6/7 thresholds pass; "
        "#7 (opening-sentence duplication) fails -- 12/50 meetings (24%) open with the "
        "same literal sentence, vs a <=5% threshold. Included anyway per user decision "
        "2026-09-15; not deduplicated.",
    ]
    (out / "provenance.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def merge(out: Path, config_path: str = "configs/experiment.yaml",
          dry_run: bool = False) -> dict:
    ondomain_records = load_manifests(ONDOMAIN_SRC)
    ondomain_meetings = {r["meeting_id"] for r in ondomain_records}
    tts_meetings = {p.stem for p in (TTS_SRC / "manifests").glob("*.jsonl")}
    overlap = ondomain_meetings & tts_meetings
    if overlap:
        raise ValueError(f"meeting_id collision between sources: {overlap}")

    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"{out} already exists and is not empty -- refusing to merge twice")

    cfg = load_config(config_path)

    if dry_run:
        tts_raw = []
        for p in sorted((TTS_SRC / "manifests").glob("*.jsonl")):
            tts_raw.extend(json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip())
        tts_records = [{**r, "split": "demo"} for r in tts_raw]
        resolved = resolve_splits(ondomain_records + tts_records, cfg.data.val_meetings)
        print("split_stats (dry-run):", split_stats(resolved))
        return split_stats(resolved)

    out.mkdir(parents=True, exist_ok=True)
    ondomain_records = _ingest_ondomain15h(out)
    tts_records = _ingest_tts_batch(out)
    resolved = resolve_splits(ondomain_records + tts_records, cfg.data.val_meetings)
    stats = split_stats(resolved)
    print("split_stats:", stats)
    _write_provenance(out, ondomain_records, tts_records)
    print(f"wrote {out}")
    return stats


def main() -> None:
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    merge(args.out, args.config, args.dry_run)


if __name__ == "__main__":
    main()
