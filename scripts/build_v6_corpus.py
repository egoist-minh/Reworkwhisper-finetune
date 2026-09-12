"""Assemble the v6 corpus: two HuggingFace train-only drops plus the local
val/test splits, into one directory `src.data.load_manifests` can read.

    python -m scripts.build_v6_corpus \
        --youtube    raw/yt/youtube-meeting-merged/dataset \
        --elevenlabs raw/el/dataset \
        --val raw/local/val --test raw/local/test \
        --out dataset/v6-corpus [--dry-run]

Why this is not `scripts.build_mixed_dataset`: that script gates on every record
being `verified: true` (the YouTube drop is 3,128 unreviewed google_asr rows and
is wanted anyway, as pseudo-label train), it reads one layout only (the
ElevenLabs drop keeps its manifests under `manifests/`), and it has no notion of
dropping a meeting from train because val claims it.

Three transforms, each of which exists because of something measured in the
sources, not out of taste:

  - `split` is rewritten to the two values `src.data.resolve_splits` accepts.
    `train`/`demo`/`val` all become `demo`; `test` stays `test`. Which of the
    `demo` rows end up as val is then decided the way this project already
    decides it -- `data.val_meetings` in the config, which already names exactly
    the four meetings the local val split holds. The pre-merge value is kept as
    `split_orig`, the only remaining record of which rows were the ElevenLabs
    vendor's own held-out demo subset.
  - Any `meeting_id` claimed by val or test is dropped from the train sources.
    The ElevenLabs drop ships `paid_meeting_0001`, `_0002` and `_0011`, which the
    local val split also holds (in a QC-reviewed form) -- keeping both would
    train on val audio.
  - `manifest.all.jsonl` in the YouTube drop duplicates its 43 per-meeting
    manifests. `load_manifests` globs `manifest.*.jsonl`, so copying both would
    count all 11,139 rows twice.

Every gate fails loud, and all of them run before a byte is copied except the
audio check, which can only run after. `--dry-run` runs the gates and prints the
same table without copying.
"""

import argparse
import json
import shutil
from collections import defaultdict
from datetime import date
from pathlib import Path

from src.config import load as load_config
from src.data import load_manifests, resolve_splits, split_stats

SPLIT_REMAP = {"train": "demo", "demo": "demo", "val": "demo", "test": "test"}


def _read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _load_source(root: Path, name: str) -> list[dict]:
    """Handles the two layouts the sources actually ship: manifests beside
    `audio/`, or under `manifests/`."""
    nested = sorted((root / "manifests").glob("*.jsonl"))
    if nested:
        return [r for f in nested for r in _read_jsonl(f)]
    flat = [f for f in sorted(root.glob("manifest.*.jsonl")) if f.name != "manifest.all.jsonl"]
    if not flat:
        raise FileNotFoundError(
            f"{name}: no manifest.*.jsonl and no manifests/*.jsonl under {root}"
        )
    return [r for f in flat for r in _read_jsonl(f)]


def _check_meeting_ids_disjoint(per_source: list[tuple[str, list[dict]]]) -> None:
    seen: dict[str, str] = {}
    for name, records in per_source:
        for r in records:
            mid = r["meeting_id"]
            if seen.setdefault(mid, name) != name:
                raise ValueError(
                    f"meeting_id {mid!r} appears in both {seen[mid]} and {name} -- "
                    "meeting_id must be disjoint across sources"
                )


def _check_audio_dirs_disjoint(per_source: list[tuple[str, list[dict]]]) -> None:
    seen: dict[str, str] = {}
    for name, records in per_source:
        for r in records:
            top = Path(r["audio_filepath"]).parts[0]
            if seen.setdefault(top, name) != name:
                raise ValueError(
                    f"audio top-level directory {top!r} appears in both {seen[top]} and "
                    f"{name} -- audio_filepath would collide after merging"
                )


def _check_dest_empty(out: Path) -> None:
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"{out} exists and is not empty -- refusing to merge on top of it")


def _remap(records: list[dict]) -> list[dict]:
    out = []
    for r in records:
        raw = r["split"]
        if raw not in SPLIT_REMAP:
            raise ValueError(f"{r['meeting_id']}/{r['segment_id']}: unknown split {raw!r}")
        out.append({**r, "split": SPLIT_REMAP[raw], "split_orig": raw})
    return out


def _report(records: list[dict]) -> str:
    lines = []
    for split in ("train", "val", "test"):
        rows = [r for r in records if r["split"] == split]
        total = sum(r["duration"] for r in rows) / 3600
        lines.append(f"{split}: {len(rows)} seg, {total:.2f} h, "
                     f"{len({r['meeting_id'] for r in rows})} meetings")
        by = defaultdict(list)
        for r in rows:
            by[r.get("source", "unknown")].append(r)
        for src_name in sorted(by):
            dur = sum(r["duration"] for r in by[src_name]) / 3600
            pct = 100 * dur / total if total else 0.0
            lines.append(f"    {src_name}: {len(by[src_name])} seg / {dur:.2f} h ({pct:.1f}%)")
    return "\n".join(lines)


def _copy(records: list[dict], audio_root: Path, out: Path) -> None:
    """Write one `manifest.<meeting_id>.jsonl` per meeting rather than copying the
    source manifests -- the records carry the rewritten `split`, and the
    ElevenLabs drop names its manifests `<meeting_id>.jsonl`, which
    `load_manifests` would not glob.

    Audio is copied one referenced `audio_filepath` at a time, not by copying
    each meeting's directory: the ElevenLabs drop keeps a 9.05 GB set of 246
    whole-meeting `mixed.wav` files next to the 8.69 GB of segments, and no
    manifest row points at any of them."""
    by_meeting = defaultdict(list)
    for r in records:
        by_meeting[r["meeting_id"]].append(r)
    for mid, rows in by_meeting.items():
        with open(out / f"manifest.{mid}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    for r in records:
        src = audio_root / r["audio_filepath"]
        if not src.is_file():
            raise FileNotFoundError(f"{src} missing -- referenced by {r['meeting_id']}"
                                    f"/{r['segment_id']}")
        dest = out / "audio" / r["audio_filepath"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)


def _verify_audio(records: list[dict], out: Path) -> None:
    missing = [r["audio_filepath"] for r in records
               if not (out / "audio" / r["audio_filepath"]).exists()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} audio_filepath missing after copy, e.g. {missing[:5]}"
        )


def _write_provenance(out: Path, per_source: list[tuple[str, list[dict]]],
                      dropped: dict[str, list[str]], resolved: list[dict]) -> None:
    lines = [f"# {out.name} -- provenance", "",
             f"Built {date.today().isoformat()} by scripts/build_v6_corpus.py.", "",
             "| source | records | hours | meetings |", "|---|---:|---:|---:|"]
    for name, records in per_source:
        dur = sum(r["duration"] for r in records) / 3600
        lines.append(f"| {name} | {len(records)} | {dur:.2f} | "
                     f"{len({r['meeting_id'] for r in records})} |")
    lines += ["", "## Meetings dropped from the train sources", ""]
    if any(dropped.values()):
        for name, mids in dropped.items():
            for mid in mids:
                lines.append(f"- `{mid}` from {name} -- claimed by val/test")
    else:
        lines.append("None.")
    lines += ["", "## Resolved splits", "", "```", _report(resolved), "```", ""]
    (out / "provenance.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build(youtube: Path, elevenlabs: Path, val: Path, test: Path, out: Path,
          config_path: str, dry_run: bool) -> None:
    if not dry_run:
        _check_dest_empty(out)

    sources = [("youtube", youtube), ("elevenlabs", elevenlabs), ("val", val), ("test", test)]
    loaded = {name: _load_source(root, name) for name, root in sources}

    held = {r["meeting_id"] for name in ("val", "test") for r in loaded[name]}
    dropped: dict[str, list[str]] = {}
    for name in ("youtube", "elevenlabs"):
        hit = sorted({r["meeting_id"] for r in loaded[name]} & held)
        if hit:
            dropped[name] = hit
            loaded[name] = [r for r in loaded[name] if r["meeting_id"] not in held]
            print(f"dropped from {name}: {', '.join(hit)} -- claimed by val/test")

    per_source = [(name, _remap(loaded[name])) for name, _ in sources]
    _check_meeting_ids_disjoint(per_source)
    _check_audio_dirs_disjoint(per_source)

    all_records = [r for _, records in per_source for r in records]
    cfg = load_config(config_path)
    resolved = resolve_splits(all_records, cfg.data.val_meetings)
    print("split_stats:", split_stats(resolved))
    print(_report(resolved))
    for split in ("train", "val", "test"):
        if not any(r["split"] == split for r in resolved):
            raise ValueError(f"resolved {split} split is empty -- check data.val_meetings "
                             f"in {config_path}")

    if dry_run:
        return

    (out / "audio").mkdir(parents=True, exist_ok=True)
    for (name, root), (_, records) in zip(sources, per_source):
        _copy(records, root / "audio", out)
    _verify_audio(all_records, out)
    _write_provenance(out, per_source, dropped, resolved)
    print(f"reloaded from disk: {len(load_manifests(out))} records")
    print(f"wrote {out}")


def main() -> None:
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__)
    for flag in ("youtube", "elevenlabs", "val", "test", "out"):
        ap.add_argument(f"--{flag}", required=True, type=Path)
    ap.add_argument("--config", default="configs/experiment.yaml",
                    help="source of data.val_meetings for the resolved split")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    build(args.youtube, args.elevenlabs, args.val, args.test, args.out,
          args.config, args.dry_run)


if __name__ == "__main__":
    main()
