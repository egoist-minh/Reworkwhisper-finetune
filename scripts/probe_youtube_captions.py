"""Screen a user-supplied list of YouTube URLs before any audio is downloaded
(SESSIONS.md F1, youtube-data-pilot/README.md step 2). No audio download, no
dataset written. No channel enumeration -- candidate URLs come from the user.

Three reject rules, all enforced here rather than discovered after a download:

  1. No Vietnamese entry in `automatic_captions` -> reject.
  2. The Vietnamese entry is a machine TRANSLATION rather than the recognition
     original -> reject, because that pairs Vietnamese text with speech in
     another language. YouTube auto-translates its automatic captions into ~100
     languages, so a `vi` key alone proves nothing.
  3. Words-per-minute collapses across 5-minute buckets -> reject and
     investigate (a transcript does not thin out; a human-written summary does).

How rule 2 is decided, measured 2026-08-11 (yt-dlp 2026.07.04) rather than
guessed from a field shape: `automatic_captions` carries 157 keys, and the
recognition original is a SEPARATE key suffixed `-orig` whose entries' `name`
ends with " (Original)":

    vi-orig -> {"ext": "json3", "name": "Vietnamese (Original)"}   <- recognition
    vi      -> {"ext": "json3", "name": "Vietnamese"}              <- translation target
    ja      -> {"ext": "json3", "name": "Japanese"}                <- translation target

So the spoken language is the language of the "(Original)" key. An
English-spoken video offers `en-orig` plus a translated `vi`, which rule 2
rejects. A video with no "(Original)" key at all is also rejected: nothing in
the response then proves the `vi` entry is not a translation.

WARNING -- the caption endpoint is not deterministic. Measured 2026-08-11: six
fresh extractions of dGT3YW0AdD8's `vi-orig` track returned the plain
recognition revision five times (1196 events, 4866 words, 4 commas) and a
punctuated, visibly better revision once (1714 events, 5218 words, 209 commas,
"embedded" where the plain one has "mãng", "Zero 11" where it has "Zo 11").
Same key, same video, minutes apart. Two consequences:

  - Every number this script reports is a snapshot of ONE fetch, not a property
    of the video. `punctuation per 100 words` identifies which revision was
    captured (~4 plain vs ~14 punctuated) and swings review-effort estimates.
  - Whatever fetches the audio must store the exact json3 it used and segment
    from that stored file, never re-fetch (SESSIONS.md F2 already requires
    this; this is the measured reason).

`--attempts N` re-extracts N times and keeps the richest draft. It defaults to
1 because request volume is limited: probing this hard returns HTTP 429.
"""

import argparse
import json
import re
import sys
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

from scripts.draft_sources import Word, parse_json3, transcript
from scripts.inspect_errors import is_vietnamese_shaped
from src.config import LEXICAL_PARTICLES
from src.metrics import score
from src.normalize import Normalizer

BUCKET_SEC = 300.0
MIN_BUCKET_SEC = 60.0        # a shorter trailing bucket is too noisy to judge wpm on
WPM_TOLERANCE = 0.30         # rule 3: last bucket below (1 - tolerance) x first bucket
ORIGINAL_SUFFIX = " (Original)"
HTTP_TIMEOUT = 30

_BRACKETED = re.compile(r"\[[^\]\n]{1,40}\]")
_PUNCT_MARKS = ".,?!:;"
_STRIP_EDGES = ".,?!:;\"'()[]…-–—"


# ---------------------------------------------------------------- caption dicts

def track_names(caption_dict: dict) -> dict:
    """`{language_key: name of its first entry}` -- the raw `name` strings, which
    are what rule 2 reads."""
    return {k: (v[0].get("name") if v else None) for k, v in caption_dict.items()}


def original_language_keys(caption_dict: dict) -> list[str]:
    """Keys whose entries are marked as the speech-recognition original."""
    return sorted(
        k for k, v in caption_dict.items()
        if any((e.get("name") or "").endswith(ORIGINAL_SUFFIX) for e in v)
    )


def vietnamese_keys(caption_dict: dict) -> list[str]:
    return sorted(k for k in caption_dict if k == "vi" or k.startswith("vi-"))


def screen_captions(info: dict) -> dict:
    """Apply reject rules 1 and 2 to one video's extracted info. Rule 3 needs the
    parsed draft, so it is applied separately by `screen_wpm`."""
    auto = info.get("automatic_captions") or {}
    vi = vietnamese_keys(auto)
    originals = original_language_keys(auto)

    if not vi:
        return {"ok": False, "rule": 1, "track_key": None,
                "reason": "no Vietnamese entry in automatic_captions"}
    if not originals:
        return {"ok": False, "rule": 2, "track_key": None,
                "reason": "no automatic_captions entry is marked "
                          f"'{ORIGINAL_SUFFIX.strip()}', so nothing proves the Vietnamese "
                          "entry is the recognition original rather than a translation"}

    vi_original = [k for k in originals if k in vi]
    if not vi_original:
        return {"ok": False, "rule": 2, "track_key": None,
                "reason": f"recognition original is {originals} (not Vietnamese) -- the "
                          "Vietnamese entry is a machine translation of non-Vietnamese speech"}
    return {"ok": True, "rule": None, "track_key": vi_original[0], "reason": "accepted"}


# ------------------------------------------------------------------ draft stats

def _tokens(words: list[Word]) -> list[str]:
    return [t for w in words for t in w.text.split()]


def _bare(token: str) -> str:
    return token.strip(_STRIP_EDGES).lower()


def wpm_buckets(words: list[Word], duration: float, bucket_sec: float = BUCKET_SEC) -> list[dict]:
    """Words-per-minute per fixed time bucket, bucketed on each word's start time.
    A trailing bucket shorter than MIN_BUCKET_SEC is dropped as too noisy."""
    if duration <= 0:
        return []
    counts: Counter = Counter()
    for w in words:
        counts[int(w.start // bucket_sec)] += len(w.text.split())
    out = []
    for i in range(int(duration // bucket_sec) + 1):
        span = min(bucket_sec, duration - i * bucket_sec)
        if span < MIN_BUCKET_SEC:
            continue
        out.append({"start_sec": i * bucket_sec, "span_sec": span,
                    "words": counts[i], "wpm": counts[i] / (span / 60)})
    return out


def baseline_bucket(buckets: list[dict]) -> dict | None:
    """The first bucket that has any speech in it. Measured need: rCd8DSMk3-c opens
    with 5 empty minutes (a waiting screen), and comparing the last bucket against
    a 0-wpm first bucket disables rule 3 entirely."""
    return next((b for b in buckets if b["wpm"] > 0), None)


def wpm_collapse(buckets: list[dict], tolerance: float = WPM_TOLERANCE) -> bool:
    """Rule 3: the last bucket's word density has fallen below `tolerance` of the
    first bucket with speech in it. A recognition transcript stays flat; a
    human-written summary thins out."""
    first = baseline_bucket(buckets)
    if first is None or len(buckets) < 2 or buckets[-1] is first:
        return False
    return buckets[-1]["wpm"] < first["wpm"] * (1 - tolerance)


def text_stats(words: list[Word], duration: float) -> dict:
    """Everything that decides review effort, measured on the draft as delivered."""
    toks = _tokens(words)
    text = transcript(words)
    n = len(toks)
    bare = [_bare(t) for t in toks]
    english = [t for t in bare
               if len(t) > 1 and t.isalpha() and not is_vietnamese_shaped(t)]
    return {
        "n_words": n,
        "n_word_entries": len(words),
        "multi_token_entries": sum(1 for w in words if len(w.text.split()) > 1),
        "punctuation_per_100_words": 100 * sum(text.count(c) for c in _PUNCT_MARKS) / n if n else 0.0,
        "digit_word_rate": sum(1 for t in toks if any(c.isdigit() for c in t)) / n if n else 0.0,
        "lexical_particle_rate": sum(1 for t in bare if t in LEXICAL_PARTICLES) / n if n else 0.0,
        "english_words_per_min": len(english) / (duration / 60) if duration > 0 else 0.0,
        # as a rate too, because per-minute cannot be compared against a manifest
        "nonvietnamese_word_rate": len(english) / n if n else 0.0,
        "english_examples": [t for t, _ in Counter(english).most_common(15)],
        "bracket_labels": dict(Counter(_BRACKETED.findall(text)).most_common()),
    }


def consensus_cer(auto_text: str, owner_text: str) -> float:
    """CER between the two caption dicts' full transcripts, each concatenated into
    one string (no alignment needed). `subtitles` is the reference and
    `automatic_captions` the hypothesis, so this reads as "how far Google's
    recognition is from what the channel owner typed" -- high agreement means
    review will be light. Read it together with the wpm buckets: a summary-style
    `subtitles` also produces a high CER, for an unrelated reason."""
    norm = Normalizer()
    return score([norm(owner_text)], [norm(auto_text)])["cer"]


# --------------------------------------------------------------------- network

def fetch_info(url: str) -> dict:
    try:
        import yt_dlp
    except ImportError as exc:
        raise RuntimeError(
            "yt-dlp is not installed -- it is in requirements.txt (unpinned on "
            "purpose: YouTube changes server-side and old builds break). "
            "Install it with `pip install yt-dlp`."
        ) from exc
    # noplaylist: a supplied URL often carries &list=...&index=N, and without this
    # yt-dlp walks the whole playlist and dies on the first private member.
    opts = {"skip_download": True, "quiet": True, "no_warnings": True, "noplaylist": True}
    with yt_dlp.YoutubeDL(opts) as ydl:
        return ydl.extract_info(url, download=False)


def fetch_json3(caption_dict: dict, key: str) -> dict:
    entries = [e for e in caption_dict.get(key, []) if e.get("ext") == "json3"]
    if not entries:
        raise ValueError(f"caption track {key!r} offers no json3 format "
                         f"(only {sorted({e.get('ext') for e in caption_dict.get(key, [])})})")
    try:
        with urllib.request.urlopen(entries[0]["url"], timeout=HTTP_TIMEOUT) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        if exc.code == 429:
            raise RuntimeError(
                "HTTP 429 from YouTube's caption endpoint -- rate limited. Wait "
                "before probing again, and keep --attempts low."
            ) from exc
        raise


def owner_transcript(subtitles: dict) -> tuple[str, str] | None:
    """The channel owner's own Vietnamese subtitles as plain text, if any. Returns
    `(key, text)`; falls back to any single language when no Vietnamese key exists."""
    keys = vietnamese_keys(subtitles) or sorted(subtitles)
    for key in keys:
        try:
            words = parse_json3(fetch_json3(subtitles, key))
        except ValueError:
            continue                    # owner-typed tracks often lack word timing
        return key, transcript(words)
    return None


# ----------------------------------------------------------------------- probe

def probe(url: str, bucket_sec: float = BUCKET_SEC, tolerance: float = WPM_TOLERANCE,
          attempts: int = 1) -> dict:
    info = fetch_info(url)
    auto = info.get("automatic_captions") or {}
    subs = info.get("subtitles") or {}
    rec = {
        "url": url,
        "video_id": info.get("id"),
        "title": info.get("title"),
        "duration": float(info.get("duration") or 0.0),
        "auto_keys": sorted(auto),
        "auto_names": track_names(auto),
        "sub_keys": sorted(subs),
        "sub_names": track_names(subs),
        "original_keys": original_language_keys(auto),
    }
    rec["screen"] = screen_captions(info)
    if not rec["screen"]["ok"]:
        return rec

    try:
        doc = fetch_json3(auto, rec["screen"]["track_key"])
        words = parse_json3(doc)
    except (RuntimeError, urllib.error.URLError, ValueError) as exc:
        # Rules 1 and 2 are already decided from the metadata above, and metadata
        # extraction is not what gets rate limited -- so a failed caption fetch
        # loses the measurements, not the verdict.
        rec["draft_error"] = f"{type(exc).__name__}: {exc}"
        return rec
    rec["attempt_word_counts"] = [len(words)]
    for _ in range(attempts - 1):
        # A fresh extraction, not a repeat of the signed URL: which revision the
        # endpoint serves is decided per extraction (see the module docstring).
        retry_info = fetch_info(url)
        retry_verdict = screen_captions(retry_info)
        if not retry_verdict["ok"]:
            continue
        retry_doc = fetch_json3(retry_info["automatic_captions"], retry_verdict["track_key"])
        retry_words = parse_json3(retry_doc)
        rec["attempt_word_counts"].append(len(retry_words))
        if len(retry_words) > len(words):
            doc, words = retry_doc, retry_words

    segs = [s for e in doc.get("events") or [] for s in e.get("segs") or []]
    rec["json3"] = {
        "events": len(doc.get("events") or []),
        "segs": len(segs),
        "segs_with_tOffsetMs": sum(1 for s in segs if "tOffsetMs" in s),
        "word_level_timing": True,      # parse_json3 raises otherwise
    }
    rec["buckets"] = wpm_buckets(words, rec["duration"], bucket_sec)
    rec["stats"] = text_stats(words, rec["duration"])
    rec["sample_text"] = transcript(words)[:300]

    if wpm_collapse(rec["buckets"], tolerance):
        rec["screen"] = {"ok": False, "rule": 3, "track_key": rec["screen"]["track_key"],
                         "reason": f"wpm collapses from {baseline_bucket(rec['buckets'])['wpm']:.0f} "
                                   f"to {rec['buckets'][-1]['wpm']:.0f} across "
                                   f"{bucket_sec / 60:.0f}-minute buckets"}

    owner = owner_transcript(subs)
    if owner:
        rec["consensus"] = {"sub_key": owner[0],
                            "cer": consensus_cer(transcript(words), owner[1])}
    return rec


# ---------------------------------------------------------------------- report

def read_urls(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip().startswith("http")]


def _fmt_record(rec: dict) -> str:
    out = [f"## {rec.get('title') or rec['url']}", ""]
    verdict = "ACCEPT" if rec["screen"]["ok"] else f"REJECT (rule {rec['screen']['rule']})"
    if "draft_error" in rec:
        verdict += " on rules 1–2 only — the draft could not be fetched"
    dur = rec["duration"]
    out += [
        f"- `video_id`: `{rec['video_id']}` · {rec['url']}",
        f"- duration: {dur / 60:.1f} min",
        f"- **{verdict}** — {rec['screen']['reason']}",
        f"- `automatic_captions`: {len(rec['auto_keys'])} keys, "
        f"recognition original {rec['original_keys'] or 'none'}",
        f"- `subtitles`: {len(rec['sub_keys'])} keys {rec['sub_keys'] or ''}",
        "",
    ]
    if "draft_error" in rec:
        out += [f"- draft fetch failed, so rule 3 and every measurement are missing: "
                f"`{rec['draft_error']}`", ""]
    if "consensus" in rec:
        out += [f"- CER `automatic_captions` vs `subtitles[{rec['consensus']['sub_key']}]`: "
                f"{rec['consensus']['cer']:.1%}", ""]
    if "json3" in rec:
        j, s = rec["json3"], rec["stats"]
        out += [
            "| Measurement | Value |",
            "|---|---|",
            f"| json3 events / segs | {j['events']} / {j['segs']} |",
            f"| segs with `tOffsetMs` | {j['segs_with_tOffsetMs']} |",
            f"| words | {s['n_words']} |",
            f"| word entries covering >1 token | {s['multi_token_entries']} |",
            f"| lexical-particle rate | {s['lexical_particle_rate']:.2%} |",
            f"| punctuation per 100 words | {s['punctuation_per_100_words']:.1f} |",
            f"| digit-bearing words | {s['digit_word_rate']:.2%} |",
            f"| non-Vietnamese-shaped words | {s['nonvietnamese_word_rate']:.2%} "
            f"({s['english_words_per_min']:.1f} / min) |",
            f"| bracketed sound labels | {s['bracket_labels'] or 'none'} |",
            f"| words per fetch attempt | {rec.get('attempt_word_counts')} |",
            "",
            "wpm per bucket: " + ", ".join(
                f"{b['start_sec'] / 60:.0f}min={b['wpm']:.0f}" for b in rec["buckets"]) + "",
            "",
            f"non-Vietnamese-shaped examples: {', '.join(s['english_examples']) or 'none'}",
            "",
            f"draft opening: `{rec['sample_text']}`",
            "",
        ]
    out += [
        "<details><summary>raw <code>automatic_captions</code> keys and <code>name</code> fields"
        "</summary>", "", "```",
        " ".join(rec["auto_keys"]),
        "",
        "\n".join(f"{k} = {v!r}" for k, v in sorted(rec["auto_names"].items())),
        "```", "</details>", "",
    ]
    if rec["sub_names"]:
        out += ["<details><summary>raw <code>subtitles</code> <code>name</code> fields</summary>",
                "", "```",
                "\n".join(f"{k} = {v!r}" for k, v in sorted(rec["sub_names"].items())),
                "```", "</details>", ""]
    return "\n".join(out)


def build_report(records: list[dict], failures: list[tuple[str, str]]) -> str:
    accepted = [r for r in records if r["screen"]["ok"]]
    head = [
        "# Caption probe — YouTube pilot",
        "",
        f"`scripts/probe_youtube_captions.py` over {len(records) + len(failures)} supplied "
        f"URLs. No audio downloaded. **{len(accepted)} accepted, "
        f"{len(records) - len(accepted) + len(failures)} rejected.**",
        "",
        "Every number below is a snapshot of one fetch, not a property of the video: "
        "YouTube's caption endpoint serves different recognition revisions for the same "
        "key from one extraction to the next. `punctuation per 100 words` says which one "
        "was captured — around 4 for the plain revision, around 14 for the punctuated "
        "one, which is also the better transcription. See this script's docstring.",
        "",
        "Reference points for the two rates that decide review effort, measured "
        "2026-08-11 over every `manifest.*.jsonl` in the existing corpora: "
        "`dataset/paid-dataset-v2` (synthetic, 74,542 words) has a **7.12%** "
        "lexical-particle rate and 7.23% non-Vietnamese-shaped words; "
        "`dataset/real-meetings-bench` (real speech, human-edited, 7,168 words) has "
        "**2.87%** and 8.75%. The bench figure is itself a floor — its reference is "
        "post-edited PhoWhisper-small output (PROJECT_CORE.md §4).",
        "",
        "| Video | Duration | Verdict | Particle rate | EN words/min | wpm first→last |",
        "|---|---:|---|---:|---:|---|",
    ]
    for r in records:
        v = "accept" if r["screen"]["ok"] else f"reject (rule {r['screen']['rule']})"
        if "draft_error" in r:
            v += ", rule 3 unchecked"
        s, b = r.get("stats"), r.get("buckets") or []
        first = baseline_bucket(b) if b else None
        head.append(
            f"| `{r['video_id']}` | {r['duration'] / 60:.1f} min | {v} | "
            f"{s['lexical_particle_rate']:.2%} | {s['english_words_per_min']:.1f} | "
            f"{first['wpm']:.0f}→{b[-1]['wpm']:.0f} |" if s and first else
            f"| `{r['video_id']}` | {r['duration'] / 60:.1f} min | {v} | — | — | — |")
    for url, err in failures:
        head.append(f"| {url} | — | error | — | — | — |")
    head.append("")
    if failures:
        head += ["Errors:", ""] + [f"- {url}: `{err}`" for url, err in failures] + [""]
    return "\n".join(head) + "\n" + "\n".join(_fmt_record(r) for r in records)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")     # Vietnamese output on a cp1252 console
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("urls", nargs="*", help="YouTube URLs to screen")
    ap.add_argument("--urls-file", type=Path,
                    help="file with one URL per line (non-URL lines ignored)")
    ap.add_argument("--out", type=Path, default=Path("youtube-data-pilot/caption-probe.md"))
    ap.add_argument("--bucket-sec", type=float, default=BUCKET_SEC)
    ap.add_argument("--tolerance", type=float, default=WPM_TOLERANCE)
    ap.add_argument("--attempts", type=int, default=1,
                    help="re-extract this many times per URL and keep the richest draft; "
                         "the caption endpoint serves different revisions per extraction, "
                         "but probing hard returns HTTP 429")
    args = ap.parse_args()

    urls = list(args.urls) + (read_urls(args.urls_file) if args.urls_file else [])
    if not urls:
        ap.error("no URLs given -- pass them positionally or via --urls-file")

    records, failures = [], []
    for url in urls:
        print(f"probing {url}")
        try:
            rec = probe(url, args.bucket_sec, args.tolerance, args.attempts)
        except Exception as exc:
            failures.append((url, f"{type(exc).__name__}: {exc}"))
            print(f"  error: {type(exc).__name__}: {exc}")
            continue
        records.append(rec)
        print(f"  {'ACCEPT' if rec['screen']['ok'] else 'REJECT'}: {rec['screen']['reason']}")

    if not any("stats" in r for r in records) and args.out.exists():
        raise SystemExit(
            f"no URL yielded any measurements this run, so {args.out} was left alone "
            "rather than overwritten with an empty report. Fix the errors above and "
            "re-run (HTTP 429 clears on its own; it can take longer than an hour)."
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(build_report(records, failures), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
