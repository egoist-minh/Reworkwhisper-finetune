"""Pack the delivered TTS batches as an add-on that extracts over dataset/v6-corpus.

`v6-corpus.tar` on the Hub is the 2026-09-12 build and contains neither
`paid-meeting-vi-0247-0296` nor `paid-meeting-vi-0297-0339` -- 9.06 h of speech
at 21.5% and 14.0% loanword density, against the corpus train split's 2.34%.
Rebuilding and re-uploading the whole 15 GB corpus to add them would cost an
afternoon of home uplink; this writes only the new part, in the corpus's own
layout, so the rented box extracts it on top of what it already downloaded:

    python -m scripts.build_v6_corpus_addon --out dataset/v6-corpus-addon
    tar -cf v6-corpus-addon.tar -C dataset v6-corpus-addon

`src.data.load_manifests` globs `manifest.*.jsonl`, so a corpus directory with
the add-on unpacked into it needs no merge step and no config change.

The batches' own manifests carry `"split": "train"`, which `resolve_splits`
rejects outright (it accepts only "demo"/"test") -- remapped to "demo" here, the
same remap `scripts/merge_ondomain15h_tts_batch.py` and
`scripts/ingest_paid_dataset_v2.py` do. None of these meetings is in
`data.val_meetings`, so every one lands in train.
"""

import argparse
import json
import shutil
from datetime import date
from pathlib import Path

from src.config import load as load_config
from src.data import load_manifests, resolve_splits, split_stats

CORPUS = Path("dataset/v6-corpus")
TTS_SRCS = [Path("dataset/paid-meeting-vi-0247-0296/dataset"),
            Path("dataset/paid-meeting-vi-0297-0339/dataset")]


def _remap(records: list[dict], meeting_id: str) -> list[dict]:
    out = []
    for r in records:
        if r["split"] != "train":
            raise ValueError(f"expected split='train' on TTS batch record, got "
                             f"{r['split']!r} ({meeting_id}/{r['segment_id']})")
        out.append({**r, "split": "demo"})
    return out


def _check_no_collision(srcs: list[Path], corpus: Path) -> None:
    have = {p.name[len("manifest."):-len(".jsonl")]
            for p in corpus.glob("manifest.*.jsonl")}
    seen: dict[str, Path] = {}
    for src in srcs:
        for manifest_path in sorted((src / "manifests").glob("*.jsonl")):
            mid = manifest_path.stem
            if mid in have:
                raise ValueError(f"meeting_id {mid} is already in {corpus} -- extracting "
                                 "this add-on would silently replace it")
            if mid in seen:
                raise ValueError(f"meeting_id collision between {seen[mid]} and {src}: {mid}")
            seen[mid] = src


def build(out: Path, srcs: list[Path] | None = None,
          config_path: str = "configs/experiment.yaml") -> dict:
    srcs = srcs or TTS_SRCS
    _check_no_collision(srcs, CORPUS)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"{out} exists and is not empty -- refusing to build twice")
    (out / "audio").mkdir(parents=True, exist_ok=True)

    added, per_src = [], {}
    for src in srcs:
        records = []
        for manifest_path in sorted((src / "manifests").glob("*.jsonl")):
            mid = manifest_path.stem
            remapped = _remap([json.loads(l) for l in
                               manifest_path.read_text(encoding="utf-8").splitlines()
                               if l.strip()], mid)
            (out / f"manifest.{mid}.jsonl").write_text(
                "\n".join(json.dumps(r, ensure_ascii=False) for r in remapped) + "\n",
                encoding="utf-8")
            for r in remapped:
                dst = out / "audio" / r["audio_filepath"]
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src / "audio" / r["audio_filepath"], dst)
            records.extend(remapped)
        per_src[src] = records
        added.extend(records)

    _write_provenance(out, per_src)

    # The gate that matters: the corpus PLUS this add-on has to split the way the
    # run plan says. Checked here, on the machine that built it, rather than on
    # the rented box after the tar is already up.
    cfg = load_config(config_path)
    merged = load_manifests(CORPUS) + added
    stats = split_stats(resolve_splits(merged, cfg.data.val_meetings))
    bytes_ = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    return {"records": len(added),
            "meetings": len({r["meeting_id"] for r in added}),
            "hours": sum(r["duration"] for r in added) / 3600,
            "bytes": bytes_, "split_stats_with_default_val": stats}


def _write_provenance(out: Path, per_src: dict[Path, list[dict]]) -> None:
    lines = [f"# {out.name} -- provenance", "",
             f"Built {date.today().isoformat()} by scripts/build_v6_corpus_addon.py.",
             "", "Extracts over `dataset/v6-corpus/`; adds meetings, replaces nothing.",
             "`split` remapped train->demo so `src.data.resolve_splits` accepts it.", ""]
    for src, records in per_src.items():
        card = next(iter(sorted((src.parent / "docs").glob("datacard-*.md"))), None)
        lines.append(f"- `{src}` (data card: `{card}`): "
                     f"{len({r['meeting_id'] for r in records})} meetings, "
                     f"{len(records)} records, "
                     f"{sum(r['duration'] for r in records) / 3600:.2f} h.")
    (out / "provenance.addon.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--src", type=Path, action="append", dest="srcs",
                    help="a delivered batch's dataset/ dir; repeat for several "
                         "(default: both 2026-09 batches)")
    args = ap.parse_args()
    r = build(args.out, args.srcs)
    print(f"{r['meetings']} meetings, {r['records']} records, {r['hours']:.2f} h, "
          f"{r['bytes'] / 1e9:.2f} GB")
    print(f"corpus + add-on splits to {r['split_stats_with_default_val']}")


if __name__ == "__main__":
    main()
