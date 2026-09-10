"""Benchmark suite registry: name -> list of `Segment`, one per scored utterance.

One place that knows what each benchmark set *is*, so `scripts/benchmark_run.py`
(decode) and `scripts/benchmark_report.py` (score) stay model- and
dataset-agnostic. Adding a benchmark set means adding a row to `SUITE_SPECS`,
not touching either of those.

Two kinds of source:

  * `manifest` -- a directory of `manifest.*.jsonl` + `audio/`, this repo's own
    schema (`src/data.py`). Used by every locally-built set. The `filter` field
    selects rows by the raw manifest fields; `mixed-noisy-v1` carries both
    `source: synthetic` and `source: youtube` rows in one directory, which is
    why `youtube-test` and `synthetic-test` are two suites over one path.
  * `hf` -- a Hugging Face Hub dataset, loaded with `decode=False` and read
    through soundfile. `Audio(sampling_rate=...)` routes decoding via torchcodec,
    which raises `RuntimeError: No audio frames were decoded` on some ViMedCSS
    rows (hit on Kaggle 2026-09-09).

Audio is always returned as 16 kHz mono float32, decoded lazily -- ViMedCSS test
alone is 3.4 h, and a full matrix run must not hold every benchmark set in RAM.
"""

import io
import json
from dataclasses import dataclass, field
from math import gcd
from pathlib import Path
from typing import Callable

import numpy as np

TARGET_SR = 16000

SUITE_SPECS: dict[str, dict] = {
    "vimedcss-test":   {"kind": "hf", "dataset": "tensorxt/ViMedCSS", "split": "test",
                        "text_col": "segment_text", "id_col": "segment_id"},
    "vimedcss-hard":   {"kind": "hf", "dataset": "tensorxt/ViMedCSS", "split": "hard",
                        "text_col": "segment_text", "id_col": "segment_id"},
    "youtube-test":    {"kind": "manifest", "filter": {"split": "test", "source": "youtube"}},
    "synthetic-test":  {"kind": "manifest", "filter": {"split": "test", "source": "synthetic"}},
    "vivos":           {"kind": "manifest", "filter": {"split": "test"}},
    "cross-domain":    {"kind": "manifest", "filter": {}},
}


@dataclass
class Segment:
    """One scored utterance. `segment_id` must be unique within its suite -- it is
    the resume key and the join key the report uses to align models."""
    suite: str
    segment_id: str
    group_id: str
    text: str
    duration: float
    _load: Callable[[], np.ndarray] = field(repr=False)

    def audio(self) -> np.ndarray:
        return self._load()

    def wav_bytes(self) -> bytes:
        """The segment re-encoded as a 16 kHz mono wav, for API backends that
        upload a file. Re-encoding rather than shipping the original bytes keeps
        every backend on the identical audio the GPU backends see."""
        import soundfile as sf

        buf = io.BytesIO()
        sf.write(buf, self.audio(), TARGET_SR, format="WAV", subtype="PCM_16")
        return buf.getvalue()


def _to_16k_mono(data: np.ndarray, sr: int) -> np.ndarray:
    if data.ndim > 1:
        data = data.mean(axis=1)
    data = data.astype("float32", copy=False)
    if sr == TARGET_SR:
        return data
    from scipy.signal import resample_poly

    g = gcd(int(sr), TARGET_SR)
    return resample_poly(data, TARGET_SR // g, int(sr) // g).astype("float32")


def _read_wav(path: Path) -> np.ndarray:
    import soundfile as sf

    data, sr = sf.read(str(path), dtype="float32", always_2d=False)
    return _to_16k_mono(data, sr)


def _read_bytes(raw: bytes) -> np.ndarray:
    import soundfile as sf

    data, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
    return _to_16k_mono(data, sr)


def chunk_audio(audio: np.ndarray, max_sec: float = 30.0) -> list[np.ndarray]:
    """Split anything longer than Whisper's 30 s window. Whisper TRUNCATES silently
    past the window, so a 60 s segment scored whole would count its second half as
    a total deletion -- a scoring artefact, not a recognition result."""
    n = int(max_sec * TARGET_SR)
    if len(audio) <= n:
        return [audio]
    return [audio[i:i + n] for i in range(0, len(audio), n)]


def _load_manifest_suite(name: str, spec: dict, path: Path) -> list[Segment]:
    from src.data import load_manifests

    records = load_manifests(path)
    for key, want in spec["filter"].items():
        records = [r for r in records if r.get(key) == want]
    if not records:
        raise RuntimeError(f"suite {name}: no records under {path} match {spec['filter']}")

    unverified = [r["segment_id"] for r in records if r.get("verified") is False]
    if unverified:
        raise RuntimeError(
            f"suite {name}: {len(unverified)} record(s) have verified=false "
            f"(first: {unverified[:3]}) -- an unreviewed reference is not a benchmark")

    audio_root = Path(path) / "audio"
    if not audio_root.is_dir():
        raise FileNotFoundError(f"suite {name}: {audio_root} missing")

    segments = []
    for r in records:
        wav = audio_root / r["audio_filepath"]
        # `segment_id` is unique only within a meeting ("seg_0000" repeats in every
        # one), so the resume/join key is `<meeting_id>/<segment_id>` -- the same
        # shape the corpus itself uses.
        group = r.get("meeting_id") or r.get("source") or name
        sid = r["segment_id"] if "meeting_id" not in r else f'{r["meeting_id"]}/{r["segment_id"]}'
        segments.append(Segment(
            suite=name, segment_id=sid,
            group_id=group,
            text=r["text"], duration=float(r.get("duration") or 0.0),
            _load=(lambda p=wav: _read_wav(p)),
        ))
    return segments


def _load_hf_suite(name: str, spec: dict) -> list[Segment]:
    # `datasets` is deliberately absent from requirements.txt (see its header) and is
    # imported here, in scripts/, only for Hub-hosted benchmark sets. The Kaggle
    # runner installs it; nothing in src/ depends on it.
    try:
        from datasets import Audio, load_dataset
    except ImportError as e:
        raise ImportError(
            f"suite {name} needs `datasets` (pip install datasets) -- it is not in "
            "requirements.txt on purpose") from e

    ds = load_dataset(spec["dataset"], split=spec["split"])
    ds = ds.cast_column("audio", Audio(decode=False))
    id_col, text_col = spec["id_col"], spec["text_col"]
    ids = ds[id_col]
    texts = ds[text_col]
    durations = ds["duration_seconds"] if "duration_seconds" in ds.column_names else [0.0] * len(ds)

    def loader(i: int):
        def _load():
            row = ds[i]["audio"]
            raw = row["bytes"] if isinstance(row, dict) else None
            if not raw:
                raise RuntimeError(f"{ids[i]}: dataset row carries no audio bytes")
            return _read_bytes(raw)
        return _load

    return [Segment(suite=name, segment_id=str(ids[i]), group_id=spec["split"],
                    text=texts[i], duration=float(durations[i]), _load=loader(i))
            for i in range(len(ds))]


def load_suite(name: str, path: str | Path | None = None, limit: int | None = None) -> list[Segment]:
    """Materialize a suite's segment list. `path` is required for `manifest` suites
    and ignored for `hf` ones -- a Kaggle mount path never belongs in this file."""
    if name not in SUITE_SPECS:
        raise KeyError(f"unknown suite {name!r}; known: {sorted(SUITE_SPECS)}")
    spec = SUITE_SPECS[name]
    if spec["kind"] == "manifest":
        if path is None:
            raise ValueError(f"suite {name} needs --path {name}=<dir> (a manifest directory)")
        segments = _load_manifest_suite(name, spec, Path(path))
    else:
        segments = _load_hf_suite(name, spec)

    seen = set()
    for s in segments:
        if s.segment_id in seen:
            raise RuntimeError(f"suite {name}: duplicate segment_id {s.segment_id!r}")
        seen.add(s.segment_id)
    return segments[:limit] if limit else segments


def parse_path_args(pairs: list[str]) -> dict[str, Path]:
    """`--path youtube-test=/kaggle/working/dataset/mixed-noisy-v1` -> {suite: dir}."""
    out = {}
    for p in pairs:
        if "=" not in p:
            raise ValueError(f"--path expects suite=dir, got {p!r}")
        suite, _, d = p.partition("=")
        if suite not in SUITE_SPECS:
            raise KeyError(f"--path names unknown suite {suite!r}; known: {sorted(SUITE_SPECS)}")
        out[suite] = Path(d)
    return out
