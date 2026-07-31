"""Ingest done/ (tier-4 real-meeting benchmark) into dataset/real-meetings-bench/.
PROJECT_CORE.md §4, §6 Stage 4 tier 4a. Source: D:/phowhisper-finetune-exp/dataset/done/
(read-only, per CLAUDE.md -- this script only reads from it).

Segments over 30s (Whisper's window) are re-segmented. There is no word-level
timestamp in `*.draft.json` -- only a per-segment [start, end] and text -- so
this is NOT forced alignment:

  1. Audio is split at genuine silence (RMS-energy-based) so each audio chunk
     is acoustically a real pause boundary, never a mid-word cut.
  2. Text is split proportionally to each audio chunk's share of the segment's
     duration, snapped to the nearest whitespace so no word is cut and no
     character is dropped (concatenating the chunks' text always reproduces
     the original segment's text exactly).

This means the per-chunk text/audio correspondence is an approximation, not
verified alignment -- acceptable for tier 4a (CER over the concatenated
segment set) but do not use these chunk boundaries for anything claiming
word-level timing.

`audio_duration` in the JSON is stale (§4 warning 6) -- durations always come
from the 16 kHz wav, never from the JSON.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import soundfile as sf

MAX_SEGMENT_SEC = 30.0
MIN_SILENCE_SEC = 0.3
FRAME_MS = 30
HOP_MS = 15
SILENCE_PERCENTILE = 15


def _rms_frames(audio: np.ndarray, sr: int) -> tuple[np.ndarray, int]:
    frame = int(sr * FRAME_MS / 1000)
    hop = int(sr * HOP_MS / 1000)
    n = max(1, (len(audio) - frame) // hop + 1)
    rms = np.empty(n, dtype=np.float64)
    for i in range(n):
        chunk = audio[i * hop: i * hop + frame]
        rms[i] = float(np.sqrt(np.mean(chunk.astype(np.float64) ** 2))) if len(chunk) else 0.0
    return rms, hop


def _silence_split_times(audio: np.ndarray, sr: int) -> list[float]:
    """Seconds, relative to the start of `audio`, of the midpoint of every
    silence run at least MIN_SILENCE_SEC long."""
    rms, hop = _rms_frames(audio, sr)
    if len(rms) == 0:
        return []
    threshold = np.percentile(rms, SILENCE_PERCENTILE)
    is_silent = rms <= threshold

    runs = []
    start = None
    for i, s in enumerate(is_silent):
        if s and start is None:
            start = i
        elif not s and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(is_silent) - 1))

    min_frames = MIN_SILENCE_SEC * sr / hop
    times = []
    for a, b in runs:
        if (b - a + 1) >= min_frames:
            mid_frame = (a + b) / 2
            times.append(mid_frame * hop / sr)
    return sorted(times)


def _choose_splits(duration: float, candidates: list[float]) -> list[float]:
    """Greedily walk candidate silence times, cutting whenever the chunk since
    the last cut would otherwise exceed MAX_SEGMENT_SEC. Falls back to a hard
    cut at MAX_SEGMENT_SEC if no candidate silence exists in time -- documented
    here, not silent: this is the one case that can still split mid-word."""
    cuts = []
    last_cut = 0.0
    for t in candidates:
        if t - last_cut > MAX_SEGMENT_SEC:
            forced = last_cut + MAX_SEGMENT_SEC
            cuts.append(forced)
            last_cut = forced
        elif t - last_cut >= MAX_SEGMENT_SEC * 0.5:
            cuts.append(t)
            last_cut = t
    while duration - last_cut > MAX_SEGMENT_SEC:
        forced = last_cut + MAX_SEGMENT_SEC
        cuts.append(forced)
        last_cut = forced
    return cuts


def _split_text_proportionally(text: str, chunk_durations: list[float]) -> list[str]:
    """Split `text` into len(chunk_durations) pieces sized by each chunk's
    share of total duration, snapped to whitespace. Concatenation with single
    spaces reproduces `text` exactly -- see test_ingest_real_bench.py."""
    words = text.split()
    total_dur = sum(chunk_durations)
    total_words = len(words)
    if total_words == 0:
        return ["" for _ in chunk_durations]

    pieces = []
    word_idx = 0
    consumed_dur = 0.0
    for i, dur in enumerate(chunk_durations):
        consumed_dur += dur
        if i == len(chunk_durations) - 1:
            end_idx = total_words
        else:
            end_idx = round(total_words * consumed_dur / total_dur)
            end_idx = max(word_idx, min(end_idx, total_words))
        pieces.append(" ".join(words[word_idx:end_idx]))
        word_idx = end_idx
    return pieces


def resegment(audio_slice: np.ndarray, sr: int, text: str) -> list[tuple[np.ndarray, str]]:
    """Split one over-long (audio_slice, text) segment into chunks each
    <= MAX_SEGMENT_SEC. Returns [(sub_audio, sub_text), ...]."""
    duration = len(audio_slice) / sr
    candidates = _silence_split_times(audio_slice, sr)
    cut_times = _choose_splits(duration, candidates)
    bounds = [0.0] + cut_times + [duration]
    chunk_durs = [bounds[i + 1] - bounds[i] for i in range(len(bounds) - 1)]
    texts = _split_text_proportionally(text, chunk_durs)

    out = []
    for i in range(len(bounds) - 1):
        s = int(bounds[i] * sr)
        e = int(bounds[i + 1] * sr)
        out.append((audio_slice[s:e], texts[i]))
    return out


def ingest_recording(draft_path: Path, wav_16k_path: Path, meeting_id: str,
                      out_root: Path) -> list[dict]:
    draft = json.loads(draft_path.read_text(encoding="utf-8"))
    audio, sr = sf.read(str(wav_16k_path), dtype="float32")
    if sr != 16000:
        raise ValueError(f"{wav_16k_path}: expected 16 kHz, got {sr}")

    audio_dir = out_root / "audio" / meeting_id
    audio_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for i, seg in enumerate(draft["segments"]):
        start, end = seg["start"], seg["end"]
        s_idx, e_idx = int(start * sr), int(end * sr)
        seg_audio = audio[s_idx:e_idx]
        duration = end - start

        if duration <= MAX_SEGMENT_SEC:
            chunks = [(seg_audio, seg["text"])]
        else:
            chunks = resegment(seg_audio, sr, seg["text"])

        for j, (chunk_audio, chunk_text) in enumerate(chunks):
            seg_id = f"seg_{i:04d}" if len(chunks) == 1 else f"seg_{i:04d}_{j}"
            wav_name = f"{seg_id}.wav"
            sf.write(audio_dir / wav_name, chunk_audio, sr)
            records.append({
                "audio_filepath": f"{meeting_id}/{wav_name}",
                "meeting_id": meeting_id,
                "segment_id": seg_id,
                "speaker": seg.get("speaker"),
                "duration": len(chunk_audio) / sr,
                "text": chunk_text,
                "lang": "vi",
                "source": "real",
            })
    return records


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="D:/phowhisper-finetune-exp/dataset/done")
    ap.add_argument("--out", default="dataset/real-meetings-bench")
    args = ap.parse_args()

    src = Path(args.src)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    all_records = []
    for meeting_id in ("real_0001", "real_0002"):
        draft = src / f"{meeting_id}.draft.json"
        wav16k = src / f"{meeting_id}_16k.wav"
        records = ingest_recording(draft, wav16k, meeting_id, out_root)
        print(f"{meeting_id}: {len(records)} segments after re-segmentation")
        all_records.extend(records)

    n_long = sum(1 for r in all_records if r["duration"] > MAX_SEGMENT_SEC)
    if n_long:
        raise RuntimeError(f"{n_long} segments still exceed {MAX_SEGMENT_SEC}s after re-segmentation")

    manifest_path = out_root / "manifest.real-meetings-bench.jsonl"
    with open(manifest_path, "w", encoding="utf-8") as f:
        for r in all_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"wrote {len(all_records)} segments -> {manifest_path}")
    print(f"total chars: {sum(len(r['text']) for r in all_records)}")


if __name__ == "__main__":
    main()
