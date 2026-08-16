"""Review round-trip for the YouTube pilot manifests (SESSIONS.md F4,
youtube-data-pilot/README.md step 4, youtube-data-pilot/style-guide.md).

`--emit MEETING_ID` writes an HTML worksheet with a per-segment `<audio
controls>` element next to a prefilled `<textarea>` -- playback in place is
the slow part of reviewing a real-speech draft, so a plain CSV would force
the reviewer to alt-tab to a file browser per segment. A "Xuất file sửa"
button collects every textarea into one JSON object keyed by `segment_id`
and triggers a browser download -- no server, no JS framework, just a
Blob + <a download>.

Draft text shown in the worksheet is lowercased before display (unless the
segment is already `verified: true`, in which case its stored, reviewer-
approved text is shown as-is) -- matching the casing convention already
measured on `dataset/real-meetings-bench` (real speech: no capitalisation
at all), not `paid-dataset-v2`'s (synthetic: normal sentence casing).
Google ASR's captions capitalise words mid-sentence with no grammatical
rule (measured: `7B24A9GfHAo/seg_0000` capitalises "Là sao Bạn" mid-clause,
no preceding punctuation, no proper noun) -- fixing that by hand on 790
segments would be pure keystroke waste, so it is done once here instead.
Bracket sound labels are stripped from the display text too, defensively --
scripts/ingest_youtube.py already drops whole bracket-label words before
segmenting, so none should remain, but a leftover partial match (a bracket
label glued to a real word) would otherwise reach the reviewer unstripped.

`--apply MEETING_ID --corrections PATH --reviewed-by NAME` reads that JSON
back into the manifest: every segment must have a non-blank entry (a blank
box raises naming the segment -- refusing to silently keep the draft, since
that is exactly the "held-out set independently verified" guarantee this
tool exists to mechanise), and existing records are only ever updated in
place -- segment count and `audio_filepath` never change across an
emit-then-apply round trip.

`--check` raises while any record anywhere under dataset/youtube-meetings/
still has `verified: false` -- train and test both, per
youtube-data-pilot/README.md step 4 ("nhãn train và test đều là nháp ASR,
người soát toàn bộ"), not just the held-out test set.
"""

import argparse
import html
import json
import re
from datetime import date
from pathlib import Path

from scripts.probe_youtube_captions import _BRACKETED

MANIFEST_DIR = Path("dataset/youtube-meetings")
AUDIO_ROOT = MANIFEST_DIR / "audio"
REVIEW_DIR = Path("youtube-data-pilot/review")

_WHITESPACE = re.compile(r"\s+")


def manifest_path(meeting_id: str, manifest_dir: Path = MANIFEST_DIR) -> Path:
    return manifest_dir / f"manifest.{meeting_id}.jsonl"


def load_records(meeting_id: str, manifest_dir: Path = MANIFEST_DIR) -> list[dict]:
    path = manifest_path(meeting_id, manifest_dir)
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_records(meeting_id: str, records: list[dict], manifest_dir: Path = MANIFEST_DIR) -> None:
    path = manifest_path(meeting_id, manifest_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def normalize_for_review(text: str) -> str:
    """Lowercase + strip any residual bracket label -- see module docstring
    for why lowercasing matches the real-speech (not synthetic) casing
    convention already measured in this project."""
    stripped = _BRACKETED.sub("", text.lower())
    return _WHITESPACE.sub(" ", stripped).strip()


def build_html(meeting_id: str, records: list[dict], audio_root: Path = AUDIO_ROOT) -> str:
    rows = []
    for r in records:
        seg_id = r["segment_id"]
        audio_uri = (audio_root / r["audio_filepath"]).resolve().as_uri()
        draft = r["text"] if r.get("verified") else normalize_for_review(r["text"])
        rows.append(f"""
<div class="seg" data-segment-id="{seg_id}">
  <div class="meta">{seg_id} &middot; {r['duration']:.1f}s</div>
  <audio controls src="{audio_uri}"></audio>
  <textarea data-segment-id="{seg_id}">{html.escape(draft)}</textarea>
</div>""")

    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Review {meeting_id}</title>
<style>
body {{ font-family: sans-serif; max-width: 700px; margin: 2em auto; }}
.seg {{ border-bottom: 1px solid #ccc; padding: 0.75em 0; }}
.meta {{ color: #666; font-size: 0.85em; }}
audio {{ display: block; width: 100%; margin: 0.3em 0; }}
textarea {{ width: 100%; height: 3em; font-size: 1em; }}
#export {{ position: sticky; top: 0; background: #fff; padding: 0.5em 0; }}
</style></head>
<body>
<div id="export"><button onclick="exportCorrections()">Xuất file sửa</button></div>
<h1>{meeting_id}</h1>
{"".join(rows)}
<script>
function exportCorrections() {{
  const out = {{}};
  document.querySelectorAll("textarea[data-segment-id]").forEach(function(t) {{
    out[t.getAttribute("data-segment-id")] = t.value;
  }});
  const blob = new Blob([JSON.stringify(out, null, 2)], {{type: "application/json"}});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "corrections.{meeting_id}.json";
  a.click();
}}
</script>
</body></html>"""


def emit_worksheet(meeting_id: str, manifest_dir: Path = MANIFEST_DIR,
                    audio_root: Path = AUDIO_ROOT, out_dir: Path = REVIEW_DIR) -> Path:
    records = load_records(meeting_id, manifest_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"review.{meeting_id}.html"
    out_path.write_text(build_html(meeting_id, records, audio_root), encoding="utf-8")
    return out_path


def apply_corrections(meeting_id: str, corrections_path: Path, reviewed_by: str,
                       manifest_dir: Path = MANIFEST_DIR, review_date: str | None = None) -> int:
    records = load_records(meeting_id, manifest_dir)
    corrections = json.loads(corrections_path.read_text(encoding="utf-8"))
    review_date = review_date or date.today().isoformat()

    updated = []
    for r in records:
        seg_id = r["segment_id"]
        if seg_id not in corrections:
            raise ValueError(
                f"{meeting_id}/{seg_id}: no correction in {corrections_path} -- "
                "every segment must be explicitly reviewed, none skipped"
            )
        text = corrections[seg_id].strip()
        if not text:
            raise ValueError(
                f"{meeting_id}/{seg_id}: blank correction -- refusing to silently "
                "keep the draft text"
            )
        updated.append({
            **r, "text": text, "verified": True, "label_source": "google_asr+human",
            "reviewed_by": reviewed_by, "review_date": review_date,
        })

    write_records(meeting_id, updated, manifest_dir)
    return len(updated)


def check_verified(manifest_dir: Path = MANIFEST_DIR, meeting_ids: list[str] | None = None) -> None:
    files = sorted(manifest_dir.glob("manifest.*.jsonl"))
    if not files:
        raise FileNotFoundError(f"no manifest.*.jsonl under {manifest_dir}")
    if meeting_ids is not None:
        wanted = set(meeting_ids)
        files = [f for f in files if f.stem.split(".", 1)[1] in wanted]

    unverified = []
    for f in files:
        meeting_id = f.stem.split(".", 1)[1]
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                if not r["verified"]:
                    unverified.append(f"{meeting_id}/{r['segment_id']}")

    if unverified:
        raise RuntimeError(
            f"{len(unverified)} segment(s) still verified=false (train and test "
            f"both must be fully reviewed -- README step 4): {unverified[:10]}"
            + (" ..." if len(unverified) > 10 else "")
        )


def main() -> None:
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")     # Vietnamese output on a cp1252 console
    ap = argparse.ArgumentParser(description=__doc__)
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--emit", metavar="MEETING_ID")
    group.add_argument("--apply", metavar="MEETING_ID")
    group.add_argument("--check", action="store_true")
    ap.add_argument("--corrections", type=Path, help="required with --apply")
    ap.add_argument("--reviewed-by", help="required with --apply")
    ap.add_argument("--manifest-dir", type=Path, default=MANIFEST_DIR)
    ap.add_argument("--audio-root", type=Path, default=AUDIO_ROOT)
    ap.add_argument("--out-dir", type=Path, default=REVIEW_DIR)
    ap.add_argument("--meeting-id", action="append", dest="check_meeting_ids",
                    help="with --check: limit to this meeting_id (repeatable); default all")
    args = ap.parse_args()

    if args.emit:
        path = emit_worksheet(args.emit, args.manifest_dir, args.audio_root, args.out_dir)
        print(f"wrote {path}")
    elif args.apply:
        if not args.corrections or not args.reviewed_by:
            ap.error("--apply requires --corrections and --reviewed-by")
        n = apply_corrections(args.apply, args.corrections, args.reviewed_by, args.manifest_dir)
        print(f"{args.apply}: {n} segments marked verified")
    elif args.check:
        check_verified(args.manifest_dir, args.check_meeting_ids)
        print("all segments verified")


if __name__ == "__main__":
    main()
