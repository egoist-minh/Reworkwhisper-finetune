"""Draft -> segments + manifest for the YouTube pilot (SESSIONS.md F3,
youtube-data-pilot/README.md steps 3 and 5). Reads F2's stored
dataset/youtube-meetings/raw/<meeting_id>/{audio.wav, captions.json3,
provenance.json}, parses the draft via scripts.draft_sources (json3 by
default, --draft-source scribe to swap sources without touching the code
below), cuts on inter-word pause gaps, and writes
dataset/youtube-meetings/{audio/<meeting_id>/seg_*.wav,
manifest.<meeting_id>.jsonl}.

Segmentation reuses scripts/ingest_real_bench.py's `_choose_splits` /
`_check_speech_rate` (same MAX_SEGMENT_SEC=30.0, MAX_CHARS_PER_SEC=60.0), but
the pause candidates come from word-START gaps, not RMS audio energy: json3
gives word start times only (draft_sources.py's module docstring measured
this on real videos), so `Word.end` is inferred and carries no pause
information inside an event. A gap is therefore measured
`next.start - cur.start`, never `next.start - cur.end`.

Non-speech spans (bracket sound labels like `[Âm nhạc]`, `[Hắng giọng]` --
the exact strings youtube-data-pilot/caption-probe.md measured) are dropped:
any Word whose entire text is a bracket label is removed from the word
stream before cutting, using the identical regex F1's
scripts/probe_youtube_captions.py already validated against real captions.
A cut span left with no words after that filtering (a genuine silence, or a
span that was only sound labels) is written to neither the manifest nor
disk.

Every record's provenance fields are copied from F2's provenance.json for
that meeting -- video_id, video_url, download_date, sha256, source,
label_source, asr_draft_model -- except yt_start/yt_end, which here are the
segment's own span within the full downloaded file, not the whole-file
0.0/duration that provenance.json carries at fetch time.

A meeting longer than TRIM_TARGET_SEC (30 min, a user decision -- README §2's
own original target was 15-20 min) is cut down to a single **middle** window
of that length before segmentation -- head/tail dropped, per README §2's
"lấy đoạn giữa, bỏ đầu/cuối". A meeting already shorter than the target is
used whole. This only affects what gets segmented into manifest records;
F2's raw/<meeting_id>/audio.wav stays the full, uncut download regardless
(tier 4b long-form material), so trimming here loses nothing already on
disk.

`split` is written as `"test"` for meetings named in `--test-meetings`,
`"demo"` otherwise (train, or val once `data.val_meetings` picks it up).
`verified: false` for every record -- the human review that flips it is F4,
not this one's job.

Re-running a meeting clears its `audio/<meeting_id>/seg_*.wav` before writing
new ones -- a re-run after a trim/cut change writes fewer or renumbered
segments than before, and the previous run's leftover files are not
implicitly still valid (measured: re-running after TRIM_TARGET_SEC was added
left 1000+ stale wav files across the corpus, silently inflating
`dataset/youtube-meetings/audio/` past what any manifest referenced).
"""

import argparse
import json
from pathlib import Path

import soundfile as sf

from scripts.draft_sources import Word, load as load_draft, transcript
from scripts.ingest_real_bench import MAX_SEGMENT_SEC, _check_speech_rate, _choose_splits
from scripts.probe_youtube_captions import _BRACKETED

RAW_ROOT = Path("dataset/youtube-meetings/raw")
OUT_ROOT = Path("dataset/youtube-meetings")
# User decision 2026-08-12: test should be a bit harder than train. Ranked by
# youtube-data-pilot/caption-probe.md's EN-words/min + non-VN-shaped rate +
# (inverse) particle rate across all 7 meetings -- val (data.val_meetings,
# README §9) already took the single hardest, `rCd8DSMk3-c`; these two are
# the next-hardest pair.
DEFAULT_TEST_MEETINGS = ["7B24A9GfHAo", "3nuCdzuyqng"]
# A pause of at least this long between two consecutive word starts is a
# genuine gap worth cutting at -- same value as ingest_real_bench.py's
# MIN_SILENCE_SEC, reused here as "a genuine pause" for word timing instead
# of RMS audio energy.
MIN_GAP_SEC = 0.3
# User decision 2026-08-12: trim each meeting to a single middle window of
# this length before segmenting, dropping head/tail on anything longer.
TRIM_TARGET_SEC = 30.0 * 60.0


def is_bracket_label(text: str) -> bool:
    return bool(_BRACKETED.fullmatch(text.strip()))


def drop_bracket_words(words: list[Word]) -> list[Word]:
    return [w for w in words if not is_bracket_label(w.text)]


def middle_window(duration: float, target: float = TRIM_TARGET_SEC) -> tuple[float, float]:
    """The (start, end) of a `target`-second window centered in `duration`.
    Returns the whole (0.0, duration) span untouched if it is already no
    longer than `target`."""
    if duration <= target:
        return 0.0, duration
    start = (duration - target) / 2
    return start, start + target


def window_words(words: list[Word], window_start: float, window_end: float) -> list[Word]:
    """Keep words starting inside [window_start, window_end), rebased to
    window-relative time so downstream cutting is unaffected by where the
    window sits in the original video."""
    return [
        Word(w.text, w.start - window_start, w.end - window_start, w.speaker)
        for w in words if window_start <= w.start < window_end
    ]


def gap_candidates(words: list[Word], min_gap: float = MIN_GAP_SEC) -> list[float]:
    """Midpoint of every inter-word gap at least `min_gap` seconds long,
    measured start-to-start."""
    return [
        (words[i - 1].start + words[i].start) / 2
        for i in range(1, len(words))
        if words[i].start - words[i - 1].start >= min_gap
    ]


def cut_segments(words: list[Word], duration: float) -> list[tuple[float, float, list[Word]]]:
    """Split `words` into (seg_start, seg_end, seg_words) spans, each
    <= MAX_SEGMENT_SEC, cut preferentially at inter-word pauses."""
    cut_times = _choose_splits(duration, gap_candidates(words))
    bounds = [0.0] + cut_times + [duration]
    spans = []
    idx = 0
    for i in range(len(bounds) - 1):
        seg_start, seg_end = bounds[i], bounds[i + 1]
        seg_words = []
        while idx < len(words) and words[idx].start < seg_end:
            seg_words.append(words[idx])
            idx += 1
        spans.append((seg_start, seg_end, seg_words))
    return spans


def ingest_meeting(meeting_id: str, draft_source: str, raw_root: Path, out_root: Path,
                    is_test: bool = False, window: tuple[float, float] | None = None) -> list[dict]:
    raw_dir = raw_root / meeting_id
    prov = json.loads((raw_dir / "provenance.json").read_text(encoding="utf-8"))
    words = drop_bracket_words(load_draft(raw_dir / "captions.json3", draft_source))

    audio, sr = sf.read(str(raw_dir / "audio.wav"), dtype="float32")
    if sr != 16000:
        raise ValueError(f"{raw_dir / 'audio.wav'}: expected 16 kHz, got {sr}")
    full_duration = len(audio) / sr

    window_start, window_end = window if window is not None else middle_window(full_duration)
    words = window_words(words, window_start, window_end)
    audio = audio[int(window_start * sr):int(window_end * sr)]
    duration = window_end - window_start

    audio_dir = out_root / "audio" / meeting_id
    audio_dir.mkdir(parents=True, exist_ok=True)
    for stale in audio_dir.glob("seg_*.wav"):
        stale.unlink()   # a re-run (different trim/cut) writes fewer/renumbered
                          # segments than before -- leftover files from the
                          # prior run are not implicitly valid, clear them first

    records = []
    for i, (seg_start, seg_end, seg_words) in enumerate(cut_segments(words, duration)):
        if not seg_words:
            continue    # genuine silence, or a span that was only bracket labels
        seg_text = transcript(seg_words)
        seg_duration = seg_end - seg_start
        seg_id = f"seg_{i:04d}"
        _check_speech_rate(meeting_id, seg_id, seg_text, seg_duration)

        s_idx, e_idx = int(seg_start * sr), int(seg_end * sr)
        wav_name = f"{seg_id}.wav"
        sf.write(audio_dir / wav_name, audio[s_idx:e_idx], sr)

        records.append({
            "text": seg_text,
            "audio_filepath": f"{meeting_id}/{wav_name}",
            "split": "test" if is_test else "demo",
            "meeting_id": meeting_id,
            "segment_id": seg_id,
            "duration": seg_duration,
            "video_id": prov["video_id"],
            "video_url": prov["video_url"],
            # yt_start/yt_end are positions in the ORIGINAL full video, not
            # window-relative -- add the window offset back.
            "yt_start": window_start + seg_start,
            "yt_end": window_start + seg_end,
            "download_date": prov["download_date"],
            "sha256": prov["sha256"],
            "source": "youtube",
            "label_source": prov["label_source"],
            "asr_draft_model": prov["asr_draft_model"],
            "reviewed_by": "",
            "review_date": "",
            "verified": False,
        })
    return records


def main() -> None:
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")     # Vietnamese output on a cp1252 console
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--draft-source", choices=["json3", "scribe"], default="json3")
    ap.add_argument("--raw-root", type=Path, default=RAW_ROOT)
    ap.add_argument("--out", type=Path, default=OUT_ROOT)
    ap.add_argument("--test-meetings", default=",".join(DEFAULT_TEST_MEETINGS),
                    help="comma-separated meeting_ids to write with split=test")
    ap.add_argument("--window-override", action="append", default=[],
                    dest="window_overrides", metavar="MEETING_ID:START_SEC:END_SEC",
                    help="use this exact [start, end) window instead of middle_window() "
                         "for one meeting; repeatable")
    args = ap.parse_args()

    test_meetings = {m for m in args.test_meetings.split(",") if m}
    window_overrides = {}
    for spec in args.window_overrides:
        meeting_id, start_sec, end_sec = spec.split(":")
        window_overrides[meeting_id] = (float(start_sec), float(end_sec))

    meeting_ids = sorted(p.name for p in args.raw_root.iterdir() if p.is_dir())
    if not meeting_ids:
        raise SystemExit(f"no meetings under {args.raw_root}")

    for meeting_id in meeting_ids:
        records = ingest_meeting(meeting_id, args.draft_source, args.raw_root, args.out,
                                  is_test=meeting_id in test_meetings,
                                  window=window_overrides.get(meeting_id))
        n_long = sum(1 for r in records if r["duration"] > MAX_SEGMENT_SEC)
        if n_long:
            raise RuntimeError(f"{meeting_id}: {n_long} segments exceed {MAX_SEGMENT_SEC}s")

        manifest_path = args.out / f"manifest.{meeting_id}.jsonl"
        with open(manifest_path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{meeting_id}: {len(records)} segments -> {manifest_path}")


if __name__ == "__main__":
    main()
