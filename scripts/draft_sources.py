"""Draft-label sources for the YouTube pilot (youtube-data-pilot/README.md step 2,
SESSIONS.md F1). One intermediate shape -- `Word` -- and one parser per source, so
everything downstream (segment cutting, manifest writing) consumes only
`list[Word]` and swapping labeller becomes a `--draft-source` flag.

Both parsers REQUIRE word-level timing and raise without it. There is
deliberately no proportional-split fallback: that approximation is what
scripts/ingest_real_bench.py has to do for done/ (segment-level timing only),
and it is what produced real_0002/seg_0074 at 475.7 chars/sec.

json3 shape, measured 2026-08-11 on 5 real videos (yt-dlp 2026.07.04), not
assumed:

  {"events": [{"tStartMs": 10920, "dDurationMs": 7840, "wWinId": 1,
               "segs": [{"utf8": "đến", "acAsrConf": 0},
                        {"utf8": " với", "tOffsetMs": 200, "acAsrConf": 0}, ...]}, ...]}

Three properties that shape the parser below:

  1. The first seg of an event carries no `tOffsetMs` -- it is implicitly 0.
     (dGT3YW0AdD8: 5459 segs, 4264 with tOffsetMs, 1195 events -- exactly one
     missing per event.) So "no tOffsetMs anywhere" is the word-level-timing
     test, not "every seg has one".
  2. Roughly half the events are rolling-caption artifacts whose only seg is
     "\n" (dGT3YW0AdD8: 597 of 1195). Blank segs are dropped.
  3. `dDurationMs` is a caption DISPLAY duration, not a speech duration: it
     routinely runs past the next event's start (582 of 597 neighbouring real
     events overlap in time). It is therefore only an upper bound on a word's
     end.

Consequence for word ends: json3 gives word START times only. `Word.end` is
inferred as the next word's start, clipped by the event's display end. Inside
an event there is no other end signal at all, so ends there carry no pause
information -- a downstream segmenter must cut on start-to-start deltas, not on
`next.start - cur.end`.
"""

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Word:
    """One draft-transcript token with its timing, in seconds."""
    text: str
    start: float
    end: float
    speaker: str | None = None


def _finish(starts: list[tuple[str, float, float | None, str | None]]) -> list[Word]:
    """Turn (text, start, display_end, speaker) rows into Words by inferring each
    end from the next start, clipped by that row's display end."""
    words = []
    for i, (text, start, display_end, speaker) in enumerate(starts):
        end = starts[i + 1][1] if i + 1 < len(starts) else None
        if end is None or (display_end is not None and display_end < end):
            end = display_end
        words.append(Word(text, start, max(start, end if end is not None else start), speaker))
    return words


def parse_json3(doc: dict) -> list[Word]:
    """Parse YouTube's json3 auto-caption document (`automatic_captions`, Google
    speech recognition) into words. Raises if it carries no word-level timing."""
    events = doc.get("events") or []
    if not events:
        raise ValueError("json3 document has no events")

    saw_offset = False
    rows: list[tuple[str, float, float | None, str | None]] = []
    for ev in events:
        segs = ev.get("segs") or []
        t_start = ev.get("tStartMs")
        if t_start is None:
            continue
        duration = ev.get("dDurationMs")
        display_end = (t_start + duration) / 1000 if duration is not None else None
        for seg in segs:
            text = (seg.get("utf8") or "").strip()
            if not text:
                continue
            offset = seg.get("tOffsetMs")
            if offset is not None:
                saw_offset = True
            rows.append((text, (t_start + (offset or 0)) / 1000, display_end, None))

    if not rows:
        raise ValueError("json3 document has no non-blank caption segments")
    if not saw_offset:
        raise ValueError(
            "json3 document carries no tOffsetMs on any segment -- this is "
            "segment-level timing, not word-level. Refusing to split text "
            "proportionally (that approximation is the real_0002/seg_0074 bug)."
        )
    return _finish(rows)


def parse_scribe(doc: dict) -> list[Word]:
    """Parse an ElevenLabs Scribe speech-to-text response into words. Raises if any
    word entry lacks `start`/`end`.

    The field names come from Scribe's documented response shape (`words`, each
    with `text`, `start`, `end`, `type`, `speaker_id`) and have NOT been checked
    against a live API call from this repo -- there is no ElevenLabs key here
    (youtube-data-pilot/README.md: "Plan free"). Verify against one real
    response before using this path for a real corpus."""
    entries = doc.get("words")
    if not entries:
        raise ValueError("Scribe response has no `words` list")

    rows: list[tuple[str, float, float | None, str | None]] = []
    for entry in entries:
        if entry.get("type") not in (None, "word"):
            continue                     # spacing / audio_event rows carry no lexical content
        text = (entry.get("text") or "").strip()
        if not text:
            continue
        start, end = entry.get("start"), entry.get("end")
        if start is None or end is None:
            raise ValueError(
                f"Scribe word {text!r} is missing start/end timing -- refusing to "
                "fall back to a proportional text split (the real_0002/seg_0074 bug)."
            )
        rows.append((text, float(start), float(end), entry.get("speaker_id")))

    if not rows:
        raise ValueError("Scribe response has no word-type entries")
    return _finish(rows)


PARSERS = {"json3": parse_json3, "scribe": parse_scribe}


def load(path: str | Path, draft_source: str) -> list[Word]:
    """Read a draft file and parse it with the named source's parser."""
    if draft_source not in PARSERS:
        raise ValueError(f"unknown draft source {draft_source!r}, expected one of {sorted(PARSERS)}")
    with open(path, encoding="utf-8") as f:
        return PARSERS[draft_source](json.load(f))


def transcript(words: list[Word]) -> str:
    """The draft's full text, whitespace-normalised."""
    return " ".join(w.text for w in words)
