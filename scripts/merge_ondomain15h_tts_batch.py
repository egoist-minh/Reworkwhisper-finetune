"""Merge dataset/v6-ondomain-15h (48 train meetings, "lõi dữ liệu đậm" per
docs/yeu-cau-lo-tts-10h.md §1) with the delivered TTS batches
dataset/paid-meeting-vi-0247-0296 and dataset/paid-meeting-vi-0297-0339 into one
self-contained corpus.

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

    python -m scripts.merge_ondomain15h_tts_batch --out dataset/v6-ondomain-15h-tts10h

Pass --tts-src once per batch to merge a different set; the default is both
delivered batches (9,94 h timeline / 9,06 h speech, 93 meetings).
"""

import argparse
import json
import shutil
from datetime import date
from pathlib import Path

from src.config import load as load_config
from src.data import load_manifests, resolve_splits, split_stats

ONDOMAIN_SRC = Path("dataset/v6-ondomain-15h")
TTS_SRCS = [Path("dataset/paid-meeting-vi-0247-0296/dataset"),
            Path("dataset/paid-meeting-vi-0297-0339/dataset")]


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


def _ingest_tts_batch(out: Path, tts_src: Path) -> list[dict]:
    all_records = []
    for manifest_path in sorted((tts_src / "manifests").glob("*.jsonl")):
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
        _copy_referenced_audio(remapped, tts_src / "audio", out / "audio")
        all_records.extend(remapped)
    return all_records


def _write_provenance(out: Path, ondomain_records: list[dict],
                      tts_srcs: list[Path], tts_records: dict[Path, list[dict]]) -> None:
    lines = [
        f"# {out.name} -- provenance",
        "",
        f"Built {date.today().isoformat()} by scripts/merge_ondomain15h_tts_batch.py from:",
        "",
        f"- `{ONDOMAIN_SRC}` (\"lõi dữ liệu đậm\", docs/yeu-cau-lo-tts-10h.md §1): "
        f"{len(ondomain_records)} records, audio copied per-record (source audio/ is a "
        "junction to the shared v6-corpus pool, not copied whole).",
    ]
    for src in tts_srcs:
        card = next(iter(sorted((src.parent / "docs").glob("datacard-*.md"))), None)
        lines.append(
            f"- `{src}` (data card: `{card}`): "
            f"{len({r['meeting_id'] for r in tts_records[src]})} meetings, "
            f"{len(tts_records[src])} records, `split` remapped train->demo.")
    lines += [
        "",
        "Acceptance check against docs/yeu-cau-lo-tts-10h.md §3 (scratch script, not "
        "committed), per batch:",
        "",
        "- `paid-meeting-vi-0247-0296` (4,91 h speech / 50 cuộc): 6/7 thresholds pass; "
        "#7 (opening-sentence duplication) fails -- 23/50 meetings (46%) share an opening "
        "sentence with another meeting, vs a <=5% threshold. Included anyway per user "
        "decision 2026-09-15; not deduplicated.",
        "- `paid-meeting-vi-0297-0339` (4,15 h speech / 43 cuộc): opening-sentence "
        "duplication is 0%; every rate threshold passes (1.695 foreign token/h, 14,2% "
        "median per-meeting density, 250 type/h, 6,78 token/type). The only miss is the "
        "absolute >=1.200 type/lô floor (1.038), which is written for a 10 h batch -- at "
        "4,15 h this batch is over twice the per-hour type rate the floor implies.",
        "",
        "Together the two batches are 9,94 h timeline / 9,06 h speech across 93 meetings, "
        "against the 10 h ask. Code-switch density differs between them (21,5% vs 14,0% "
        "of tokens); they are merged without resampling.",
    ]
    (out / "provenance.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def merge(out: Path, config_path: str = "configs/experiment.yaml",
          dry_run: bool = False, tts_srcs: list[Path] | None = None) -> dict:
    tts_srcs = tts_srcs or TTS_SRCS
    ondomain_records = load_manifests(ONDOMAIN_SRC)
    seen = {m: str(ONDOMAIN_SRC) for m in {r["meeting_id"] for r in ondomain_records}}
    for src in tts_srcs:
        for manifest_path in sorted((src / "manifests").glob("*.jsonl")):
            if manifest_path.stem in seen:
                raise ValueError(f"meeting_id collision: {manifest_path.stem} in both "
                                 f"{seen[manifest_path.stem]} and {src}")
            seen[manifest_path.stem] = str(src)

    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"{out} already exists and is not empty -- refusing to merge twice")

    cfg = load_config(config_path)

    if dry_run:
        tts_raw = []
        for src in tts_srcs:
            for manifest_path in sorted((src / "manifests").glob("*.jsonl")):
                tts_raw.extend(json.loads(l) for l in
                               manifest_path.read_text(encoding="utf-8").splitlines() if l.strip())
        tts_records = [{**r, "split": "demo"} for r in tts_raw]
        resolved = resolve_splits(ondomain_records + tts_records, cfg.data.val_meetings)
        print("split_stats (dry-run):", split_stats(resolved))
        return split_stats(resolved)

    out.mkdir(parents=True, exist_ok=True)
    ondomain_records = _ingest_ondomain15h(out)
    tts_records = {src: _ingest_tts_batch(out, src) for src in tts_srcs}
    all_tts = [r for src in tts_srcs for r in tts_records[src]]
    resolved = resolve_splits(ondomain_records + all_tts, cfg.data.val_meetings)
    stats = split_stats(resolved)
    print("split_stats:", stats)
    _write_provenance(out, ondomain_records, tts_srcs, tts_records)
    print(f"wrote {out}")
    return stats


def main() -> None:
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--tts-src", action="append", type=Path, dest="tts_srcs",
                    help="TTS batch dataset/ dir; repeatable, defaults to both delivered batches")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    merge(args.out, args.config, args.dry_run, args.tts_srcs)


if __name__ == "__main__":
    main()
