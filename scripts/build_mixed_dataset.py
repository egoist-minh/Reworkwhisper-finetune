"""Merge N dataset directories into one mixed corpus (SESSIONS.md, plan step 6
of youtube-data-pilot/README.md). Reused as-is by src/pipeline.py once
`configs/experiment.yaml:data.dataset_path` points at the merged directory --
this script has no runtime role, it only assembles a directory once.

Gates run BEFORE copying a single byte, all fail loud (no soft fallback):
  - `meeting_id` must be disjoint across sources.
  - The top-level directory name under each source's `audio/` must be disjoint
    across sources -- this is what lets `audio_filepath` stay untouched (both
    corpora already use paths relative to `<root>/audio`).
  - Any source whose records carry a `verified` field must have every record
    `verified: true` (scripts.review_youtube.check_verified) -- the only gate
    standing between this merge and training on unreviewed labels.
  - The destination must not already exist non-empty -- refuses to silently
    merge a second time on top of a first.

`--dry-run` runs every gate and prints the same train/val/test table (split by
`source`) without copying anything -- usable before review is finished, to
gate-check and estimate size ahead of time.

    python -m scripts.build_mixed_dataset --sources <dir1> <dir2> \
        --out dataset/mixed-noisy-v1 [--dry-run]

Run as a module (`-m scripts.build_mixed_dataset`), not a bare file path -- this
imports scripts.review_youtube, same convention as youtube-data-pilot/style-guide.md's
`python -m scripts.review_youtube`.

Copies real files, no symlinks. Does NOT copy `raw/` (only used later for
tier 4b, out of scope here). `data.val_meetings` for the printed split table
comes from `--config` (default `configs/experiment.yaml`) -- already the
field this project uses for that split decision, not a new parameter.
"""

import argparse
import shutil
from datetime import date
from pathlib import Path

from src.config import load as load_config
from src.data import load_manifests, resolve_splits, split_stats
from scripts.review_youtube import check_verified


def _check_meeting_ids_disjoint(per_source: list[tuple[Path, list[dict]]]) -> None:
    seen: dict[str, Path] = {}
    for src, records in per_source:
        for r in records:
            mid = r["meeting_id"]
            if mid in seen and seen[mid] != src:
                raise ValueError(
                    f"meeting_id {mid!r} appears in both {seen[mid]} and {src} -- "
                    "meeting_id must be disjoint across sources"
                )
            seen[mid] = src


def _check_audio_dirs_disjoint(per_source: list[tuple[Path, list[dict]]]) -> None:
    seen: dict[str, Path] = {}
    for src, records in per_source:
        for r in records:
            top = Path(r["audio_filepath"]).parts[0]
            if top in seen and seen[top] != src:
                raise ValueError(
                    f"audio top-level directory {top!r} appears in both {seen[top]} and "
                    f"{src} -- audio_filepath would collide after merging"
                )
            seen[top] = src


def _check_verified(per_source: list[tuple[Path, list[dict]]]) -> None:
    for src, records in per_source:
        if any("verified" in r for r in records):
            check_verified(manifest_dir=src)


def _check_dest_empty(out: Path) -> None:
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(
            f"{out} already exists and is not empty -- refusing to merge a second time"
        )


def _by_source_report(records: list[dict]) -> str:
    """train/val/test table, each split broken out by the record's own `source`
    field (not the input directory) -- generalizes past the synthetic/youtube
    case to however many sources were passed."""
    lines = []
    for split in ("train", "val", "test"):
        split_records = [r for r in records if r["split"] == split]
        total_dur = sum(r["duration"] for r in split_records) / 60
        lines.append(f"{split}: {len(split_records)} segments, {total_dur:.1f} min")
        for src_name in sorted({r.get("source", "unknown") for r in split_records}):
            src_records = [r for r in split_records if r.get("source", "unknown") == src_name]
            src_dur = sum(r["duration"] for r in src_records) / 60
            pct = 100 * src_dur / total_dur if total_dur else 0.0
            lines.append(f"  {src_name}: {len(src_records)} seg / {src_dur:.1f} min ({pct:.1f}%)")
    return "\n".join(lines)


def _label_source_summary(records: list[dict]) -> str:
    values = sorted({r["label_source"] for r in records if "label_source" in r})
    return ", ".join(values) if values else "n/a (no label_source field)"


def _copy_source(src: Path, out: Path) -> None:
    for f in sorted(src.glob("manifest.*.jsonl")):
        shutil.copy2(f, out / f.name)
    out_audio = out / "audio"
    out_audio.mkdir(parents=True, exist_ok=True)
    for child in sorted((src / "audio").iterdir()):
        dest = out_audio / child.name
        if child.is_dir():
            shutil.copytree(child, dest)
        else:
            shutil.copy2(child, dest)


def _verify_audio_exists(records: list[dict], out: Path) -> None:
    missing = [r["audio_filepath"] for r in records
               if not (out / "audio" / r["audio_filepath"]).exists()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} audio_filepath missing after copy, e.g. {missing[:5]}"
        )


def _write_provenance(out: Path, per_source: list[tuple[Path, list[dict]]]) -> Path:
    lines = [
        f"# {out.name} -- provenance",
        "",
        f"Built {date.today().isoformat()} by scripts/build_mixed_dataset.py from:",
        "",
        "| source | records | duration (min) | label_source |",
        "|---|---:|---:|---|",
    ]
    for src, records in per_source:
        dur = sum(r["duration"] for r in records) / 60
        lines.append(f"| {src} | {len(records)} | {dur:.1f} | {_label_source_summary(records)} |")
    path = out / "provenance.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def build_mixed_dataset(sources: list[Path], out: Path,
                         config_path: str = "configs/experiment.yaml",
                         dry_run: bool = False) -> None:
    if not dry_run:
        _check_dest_empty(out)

    per_source = [(src, load_manifests(src)) for src in sources]
    _check_meeting_ids_disjoint(per_source)
    _check_audio_dirs_disjoint(per_source)
    _check_verified(per_source)

    cfg = load_config(config_path)
    all_records = [r for _, records in per_source for r in records]
    resolved = resolve_splits(all_records, cfg.data.val_meetings)
    print("split_stats:", split_stats(resolved))
    print(_by_source_report(resolved))

    if dry_run:
        return

    out.mkdir(parents=True, exist_ok=True)
    for src, _ in per_source:
        _copy_source(src, out)
    _verify_audio_exists(all_records, out)
    _write_provenance(out, per_source)
    print(f"wrote {out}")


def main() -> None:
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")     # Vietnamese output on a cp1252 console
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sources", nargs="+", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--config", default="configs/experiment.yaml",
                    help="source of data.val_meetings for the printed split table")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    build_mixed_dataset(args.sources, args.out, args.config, args.dry_run)


if __name__ == "__main__":
    main()
