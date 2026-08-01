"""Manifest loading and split resolution. PROJECT_CORE.md §2.1, §4.

Raw manifests only distinguish `split in {"demo", "test"}` -- train/val is not
in the data, it is derived here from `data.val_meetings` (§2.1):

    split == "test"                                 -> test
    split == "demo" and meeting_id in val_meetings   -> val
    split == "demo" otherwise                        -> train

`validated_manifest.jsonl` is the only artifact that carries the resolved
split; nothing downstream re-derives it (that is how a val/train leak gets
in -- see the `valfix` note in PROJECT_CORE.md).
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

TARGET_SR = 16000


def load_manifests(dataset_path: str | Path) -> list[dict]:
    """Merge every `manifest.*.jsonl` at the root of `dataset_path` into one list."""
    root = Path(dataset_path)
    files = sorted(root.glob("manifest.*.jsonl"))
    if not files:
        raise FileNotFoundError(f"no manifest.*.jsonl under {root}")
    records = []
    for f in files:
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


def resolve_splits(records: list[dict], val_meetings: list[str]) -> list[dict]:
    """Return records with `split` overwritten by the resolved train/val/test.

    Raises if a meeting_id straddles two resolved splits -- that would mean
    `val_meetings` names a meeting that also has `split: "test"` rows, which
    is a data or config error, not something to silently paper over.
    """
    val_set = set(val_meetings)
    resolved = []
    seen: dict[str, str] = {}
    for r in records:
        raw_split = r["split"]
        meeting_id = r["meeting_id"]
        if raw_split == "test":
            new_split = "test"
        elif raw_split == "demo":
            new_split = "val" if meeting_id in val_set else "train"
        else:
            raise ValueError(f"unknown raw split {raw_split!r} for {meeting_id}")
        prior = seen.setdefault(meeting_id, new_split)
        if prior != new_split:
            raise ValueError(
                f"meeting_id {meeting_id!r} resolves to both {prior!r} and "
                f"{new_split!r} -- val_meetings/manifest disagree"
            )
        resolved.append({**r, "split": new_split})
    return resolved


def split_stats(records: list[dict]) -> dict:
    counts = {"train": 0, "val": 0, "test": 0}
    for r in records:
        counts[r["split"]] += 1
    return counts


def write_validated_manifest(records: list[dict], out_path: str | Path) -> Path:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return out


def load_audio_16k(wav_path: str | Path) -> np.ndarray:
    """Read a wav and resample to TARGET_SR if needed. Mono only -- the dataset
    is single-channel by construction (§4); a stereo file is a data error."""
    data, sr = sf.read(str(wav_path), dtype="float32")
    if data.ndim != 1:
        raise ValueError(f"{wav_path}: expected mono audio, got shape {data.shape}")
    if sr == TARGET_SR:
        return data
    from math import gcd
    g = gcd(sr, TARGET_SR)
    return resample_poly(data, TARGET_SR // g, sr // g).astype("float32")


@dataclass
class ManifestDataset:
    """Thin `torch.utils.data.Dataset` over a resolved manifest split.

    Kept dependency-light: only `__getitem__`/`__len__` are used by the HF
    Trainer's default collation path, so this does not need to subclass
    torch.utils.data.Dataset to work with it, but the pipeline should always
    hand a torch.utils.data.Dataset to Trainer -- see src/train.py.
    """
    records: list[dict]
    audio_root: Path

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        r = self.records[idx]
        audio = load_audio_16k(self.audio_root / r["audio_filepath"])
        return {"audio": audio, "sampling_rate": TARGET_SR, "text": r["text"],
                "segment_id": r.get("segment_id"), "meeting_id": r.get("meeting_id")}

    def filter_split(self, split: str) -> "ManifestDataset":
        return ManifestDataset(
            records=[r for r in self.records if r["split"] == split],
            audio_root=self.audio_root,
        )
