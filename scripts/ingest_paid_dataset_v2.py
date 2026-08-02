"""Consolidate the dot2 data drop + repurposed legacy test meetings into
dataset/paid-dataset-v2/, the new data.dataset_path. PROJECT_CORE.md §3, §4.

Two sources:
  - dataset/paid-tts-dataset-dot2/dataset/ -- "dot2" batch (docs/bao-cao-dot2.md,
    2026-08-02): train meetings paid_meeting_0001..0037 (0001-0015 confirmed
    byte-identical to the old paid-dataset by hand; 0016-0037 new content), plus
    a corrected test set paid_meeting_test_0001..0003 using 10 voices never used
    in train -- dot2's own report flags batch 1's test set as reusing 3/10 train
    voices. dot2 mixes two split-label conventions across meetings (checked by
    hand): 0001-0015 are literal copies of the old paid-dataset and kept its
    `"demo"|"test"` labels, while 0016-0037 (new content) use `"train"|"test"`
    instead. Both remapped here to this project's `"demo"|"test"` convention
    (src/data.py:resolve_splits only accepts "demo"/"test").
  - dataset/paid-dataset/ (old) -- only its 6 test meetings are pulled in, per
    the user's 2026-08-02 decision to fold the old (flawed, same-voice-as-train)
    test set into train now that dot2 supplies a clean replacement test set.
    Renamed paid_meeting_test_000N -> paid_meeting_legacy_000N so it doesn't
    collide with dot2's own paid_meeting_test_000N namespace (dot2 will
    eventually deliver test_0004-0006 too, under those same IDs).

Only `raw_turns/*.wav` is copied (the only thing manifests reference) --
mixed.wav, speaker_tracks/, rttm/ stay in dataset/paid-tts-dataset-dot2/ as
reference, not needed by the ASR training pipeline.
"""

import argparse
import json
import re
import shutil
from pathlib import Path

LEGACY_TEST_RE = re.compile(r"^paid_meeting_test_(\d+)$")


def remap_dot2_split(raw_split: str) -> str:
    if raw_split in ("train", "demo"):
        return "demo"
    if raw_split == "test":
        return "test"
    raise ValueError(f"unexpected dot2 split {raw_split!r} (expected 'train', 'demo', or 'test')")


def rename_legacy_meeting(meeting_id: str) -> str:
    m = LEGACY_TEST_RE.match(meeting_id)
    if not m:
        raise ValueError(f"expected a legacy paid_meeting_test_NNNN id, got {meeting_id!r}")
    return f"paid_meeting_legacy_{m.group(1)}"


def _copy_raw_turns(src_meeting_dir: Path, dst_meeting_dir: Path) -> None:
    src_turns = src_meeting_dir / "raw_turns"
    dst_turns = dst_meeting_dir / "raw_turns"
    dst_turns.mkdir(parents=True, exist_ok=True)
    for wav in src_turns.glob("*.wav"):
        shutil.copyfile(wav, dst_turns / wav.name)


def ingest_dot2(dot2_root: Path, out_root: Path) -> dict:
    """Returns {meeting_id: n_records} for every dot2 meeting ingested."""
    counts = {}
    for manifest_path in sorted((dot2_root / "manifests").glob("*.jsonl")):
        meeting_id = manifest_path.stem
        records = [json.loads(l) for l in manifest_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        remapped = [{**r, "split": remap_dot2_split(r["split"])} for r in records]

        _copy_raw_turns(dot2_root / "audio" / meeting_id, out_root / "audio" / meeting_id)
        out_manifest = out_root / f"manifest.{meeting_id}.jsonl"
        with open(out_manifest, "w", encoding="utf-8") as f:
            for r in remapped:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        counts[meeting_id] = len(remapped)
    return counts


def ingest_legacy_test_as_train(legacy_root: Path, out_root: Path) -> dict:
    """Returns {new_meeting_id: n_records} for every legacy test meeting folded into train."""
    counts = {}
    for manifest_path in sorted(legacy_root.glob("manifest.paid_meeting_test_*.jsonl")):
        old_meeting_id = manifest_path.stem.removeprefix("manifest.")
        new_meeting_id = rename_legacy_meeting(old_meeting_id)
        records = [json.loads(l) for l in manifest_path.read_text(encoding="utf-8").splitlines() if l.strip()]

        remapped = []
        for r in records:
            if r["split"] != "test":
                raise ValueError(f"expected split='test' on legacy {old_meeting_id}, got {r['split']!r}")
            new_audio_filepath = r["audio_filepath"].replace(old_meeting_id, new_meeting_id, 1)
            remapped.append({**r, "split": "demo", "meeting_id": new_meeting_id,
                              "audio_filepath": new_audio_filepath})

        _copy_raw_turns(legacy_root / "audio" / old_meeting_id, out_root / "audio" / new_meeting_id)
        out_manifest = out_root / f"manifest.{new_meeting_id}.jsonl"
        with open(out_manifest, "w", encoding="utf-8") as f:
            for r in remapped:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        counts[new_meeting_id] = len(remapped)
    return counts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dot2-src", default="dataset/paid-tts-dataset-dot2/dataset")
    ap.add_argument("--legacy-src", default="dataset/paid-dataset")
    ap.add_argument("--out", default="dataset/paid-dataset-v2")
    args = ap.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    dot2_counts = ingest_dot2(Path(args.dot2_src), out_root)
    legacy_counts = ingest_legacy_test_as_train(Path(args.legacy_src), out_root)

    overlap = set(dot2_counts) & set(legacy_counts)
    if overlap:
        raise RuntimeError(f"meeting_id collision after renaming: {overlap}")

    n_dot2_train = sum(n for mid, n in dot2_counts.items() if not mid.startswith("paid_meeting_test_"))
    n_dot2_test = sum(n for mid, n in dot2_counts.items() if mid.startswith("paid_meeting_test_"))
    n_legacy = sum(legacy_counts.values())

    print(f"dot2 train meetings: {sum(1 for m in dot2_counts if not m.startswith('paid_meeting_test_'))}"
          f" ({n_dot2_train} segments)")
    print(f"dot2 test meetings: {sum(1 for m in dot2_counts if m.startswith('paid_meeting_test_'))}"
          f" ({n_dot2_test} segments)")
    print(f"legacy test->train meetings: {len(legacy_counts)} ({n_legacy} segments)")
    print(f"total train segments (dot2 train + legacy): {n_dot2_train + n_legacy}")
    print(f"total meetings written: {len(dot2_counts) + len(legacy_counts)} -> {out_root}")


if __name__ == "__main__":
    main()
