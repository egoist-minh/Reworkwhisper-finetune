"""Fetch audio + captions for the youtube-data-pilot meetings (SESSIONS.md F2,
youtube-data-pilot/README.md step 1). Reads youtube-data-pilot/sources.jsonl,
downloads bestaudio with yt-dlp, extracts mono 16 kHz wav via ffmpeg at
extraction time -- `load_audio_16k` (src/data.py:92) raises on stereo, and
YouTube audio is stereo by default -- and saves the exact automatic_captions
json3 track used, since F1 measured the caption endpoint serves a different
recognition revision on every fresh extraction: F3 must parse this stored
file, never re-fetch.

`ffmpeg` is a system dependency, not pip, so its absence is checked once up
front and raises a message naming it, rather than letting yt-dlp's
postprocessor die with an unreadable trace.

Idempotent: a meeting whose three output files already exist and whose
audio.wav sha256 matches the recorded provenance is skipped, not
re-downloaded. A mismatch raises -- the on-disk file changed since the last
fetch and that must be investigated, not silently overwritten.

The raw file kept here is always the full, uncut download for every meeting,
never a trimmed one -- youtube-data-pilot/README.md step 2's "keep the middle,
drop head/tail" is a later editorial choice (step 3's cut), not a fetch-time
one. That is also why the 2 eventual test meetings' uncut file (tier 4b
material) is available for free: it is what every meeting gets here.
"""

import argparse
import json
import shutil
import sys
from datetime import date
from pathlib import Path

from scripts.checksum_dataset import _hash_file
from scripts.probe_youtube_captions import fetch_json3, screen_captions

DEFAULT_SOURCES = Path("youtube-data-pilot/sources.jsonl")
DEFAULT_OUT_ROOT = Path("dataset/youtube-meetings/raw")


def require_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg not found on PATH -- it is a system dependency, not pip "
            "installable (youtube-data-pilot/README.md step 1), needed to "
            "extract mono 16 kHz audio. Install it (e.g. "
            "`winget install Gyan.FFmpeg`) and restart the shell."
        )


def read_sources(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def download_and_draft(url: str, out_dir: Path) -> dict:
    """Download bestaudio -> mono 16 kHz `audio.wav` under `out_dir`, then fetch
    the same session's automatic_captions json3 track. Returns
    {"info": ..., "doc": ..., "track_key": ...}."""
    import yt_dlp

    audio_path = out_dir / "audio.wav"
    opts = {
        "format": "bestaudio/best",
        "outtmpl": str(out_dir / "audio.%(ext)s"),
        "noplaylist": True,   # a supplied URL may carry &list=...&index=N (F1)
        "quiet": True,
        "no_warnings": True,
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "wav"}],
        "postprocessor_args": {"default": ["-ar", "16000", "-ac", "1"]},
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
    if not audio_path.exists():
        raise RuntimeError(
            f"expected {audio_path} after download but it is missing -- "
            "yt-dlp/ffmpeg postprocessing likely failed"
        )

    verdict = screen_captions(info)
    if not verdict["ok"]:
        raise RuntimeError(
            f"{url}: caption screen failed at fetch time (rule {verdict['rule']}: "
            f"{verdict['reason']}) -- this source should already have passed "
            "scripts/probe_youtube_captions.py before being added to sources.jsonl"
        )
    doc = fetch_json3(info["automatic_captions"], verdict["track_key"])
    return {"info": info, "doc": doc, "track_key": verdict["track_key"]}


def fetch_meeting(meeting_id: str, url: str, out_root: Path) -> None:
    out_dir = out_root / meeting_id
    audio_path = out_dir / "audio.wav"
    captions_path = out_dir / "captions.json3"
    provenance_path = out_dir / "provenance.json"

    if audio_path.exists() and captions_path.exists() and provenance_path.exists():
        prov = json.loads(provenance_path.read_text(encoding="utf-8"))
        actual = _hash_file(audio_path)
        if actual == prov["sha256"]:
            print(f"skip {meeting_id}: already fetched, checksum verified")
            return
        raise RuntimeError(
            f"{meeting_id}: {audio_path} sha256 mismatch (expected "
            f"{prov['sha256']}, got {actual}) -- the file changed since the "
            "last fetch; investigate before overwriting"
        )

    print(f"fetching {meeting_id} <- {url}")
    out_dir.mkdir(parents=True, exist_ok=True)
    result = download_and_draft(url, out_dir)

    captions_path.write_text(json.dumps(result["doc"], ensure_ascii=False), encoding="utf-8")
    provenance = {
        "video_id": result["info"].get("id"),
        "video_url": url,
        "yt_start": 0.0,
        "yt_end": float(result["info"].get("duration") or 0.0),
        "download_date": date.today().isoformat(),
        "sha256": _hash_file(audio_path),
        "source": "youtube",
        "label_source": "google_asr",
        "asr_draft_model": result["track_key"],
        "reviewed_by": "",     # filled at README step 4 (human review), not here
        "review_date": "",
    }
    provenance_path.write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  wrote {audio_path}, {captions_path}, {provenance_path}")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")     # Vietnamese output on a cp1252 console
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sources", type=Path, default=DEFAULT_SOURCES)
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    args = ap.parse_args()

    require_ffmpeg()
    sources = read_sources(args.sources)
    if not sources:
        raise SystemExit(f"{args.sources} has no rows")

    for row in sources:
        fetch_meeting(row["meeting_id"], row["video_url"], args.out_root)


if __name__ == "__main__":
    main()
