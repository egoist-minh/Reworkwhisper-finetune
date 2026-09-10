"""Decode one model over one or more benchmark suites. Resumable, raw output only.

This is the decode half of the benchmark matrix; `scripts/benchmark_report.py` is
the scoring half. They are separate because scoring is a CPU-only, seconds-long
job whose normalization choices get revisited, while decoding is hours of GPU or
paid API time. Hypotheses are therefore stored **raw and unnormalized** -- every
scoring decision is made afterwards, off the stored files, with no re-decode.

Backends:

  hf          `src/asr.py`'s load + greedy decode path, so numbers land on the same
              scale as every gate number in this repo. No `no_repeat_ngram_size`:
              banning 3-grams costs ~3 CER points on Vietnamese (CLAUDE.md memory
              `ngram-ban-corrupts-vietnamese-decode`), and the 2026-08-27 ElevenLabs
              comparison ran with it on -- which is one reason this harness exists.
  elevenlabs  POST /v1/speech-to-text, one segment per call, on the same segment
              boundaries the GPU backends see. PAID: it prints the audio hours it
              is about to send before the first call.

Resume is by `segment_id` against the output JSONL: a Kaggle session that dies at
80% costs 20%, and a re-run never pays ElevenLabs twice for the same segment.

    python -m scripts.benchmark_run --backend hf --model vinai/PhoWhisper-large \\
        --suites youtube-test,synthetic-test,vivos,cross-domain \\
        --path youtube-test=/kaggle/working/dataset/mixed-noisy-v1 \\
        --path synthetic-test=/kaggle/working/dataset/mixed-noisy-v1 \\
        --path vivos=dataset/vivos \\
        --path cross-domain=/kaggle/input/.../cross-domain-bench \\
        --out Outputs/benchmark-2026-09-10
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

from scripts.benchmark_suites import chunk_audio, load_suite, parse_path_args

ELEVENLABS_URL = "https://api.elevenlabs.io/v1/speech-to-text"


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def out_paths(out_dir: Path, model: str, suite: str) -> tuple[Path, Path]:
    stem = f"{slug(model)}.{suite}"
    return out_dir / f"{stem}.persegment.jsonl", out_dir / f"{stem}.failures.jsonl"


def done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open(encoding="utf-8") as f:
        return {json.loads(line)["segment_id"] for line in f if line.strip()}


class HFBackend:
    """Whisper-family checkpoint on the local GPU. Loaded once per process."""

    def __init__(self, model: str, batch_size: int, language: str, num_beams: int):
        from src import compat

        compat.apply()
        from src.asr import load_for_eval

        self.model, self.processor = load_for_eval(model)
        self.batch_size = batch_size
        self.language = language
        self.num_beams = num_beams

    def transcribe(self, segments: list) -> list[str]:
        """Long segments are chunked to Whisper's 30 s window and the pieces
        re-joined, never truncated -- see benchmark_suites.chunk_audio."""
        from src.asr import transcribe_batch

        pieces, owner = [], []
        for i, seg in enumerate(segments):
            for c in chunk_audio(seg.audio()):
                pieces.append(c)
                owner.append(i)
        hyps = transcribe_batch(self.model, self.processor, pieces,
                                language=self.language, num_beams=self.num_beams,
                                batch_size=self.batch_size, desc=None)
        joined: dict[int, list[str]] = {}
        for i, h in zip(owner, hyps):
            joined.setdefault(i, []).append(h.strip())
        return [" ".join(joined[i]).strip() for i in range(len(segments))]


class ElevenLabsBackend:
    """ElevenLabs speech-to-text. One HTTP call per segment, so the batch is 1."""

    def __init__(self, model: str, language: str, retries: int = 3):
        import requests

        self.requests = requests
        self.model = model
        self.language = language
        self.retries = retries
        self.key = os.environ.get("ELEVENLABS_API_KEY")
        if not self.key:
            raise RuntimeError("ELEVENLABS_API_KEY is not set -- the API backend cannot run")

    def transcribe(self, segments: list) -> list[str]:
        return [self._one(s) for s in segments]

    def _one(self, seg) -> str:
        payload = {"model_id": self.model, "language_code": self.language,
                   "diarize": "false", "tag_audio_events": "false"}
        last = None
        for attempt in range(self.retries):
            resp = self.requests.post(
                ELEVENLABS_URL, headers={"xi-api-key": self.key}, data=payload,
                files={"file": (f"{seg.segment_id}.wav", seg.wav_bytes(), "audio/wav")},
                timeout=180)
            if resp.status_code == 200:
                body = resp.json()
                if "text" not in body:
                    raise RuntimeError(f"{seg.segment_id}: response has no `text` key; "
                                       f"keys={sorted(body)}")
                return body["text"]
            last = f"HTTP {resp.status_code}: {resp.text[:300]}"
            if resp.status_code not in (429, 500, 502, 503, 504):
                break
            time.sleep(2 ** attempt)
        raise RuntimeError(f"{seg.segment_id}: {last}")


def run_suite(backend, model: str, suite: str, segments: list, out_dir: Path,
              batch_size: int, force: bool) -> None:
    from tqdm.auto import tqdm

    jsonl, failures = out_paths(out_dir, model, suite)
    already = set() if force else done_ids(jsonl)
    todo = [s for s in segments if s.segment_id not in already]
    print(f"{model} / {suite}: {len(already)} done, {len(todo)} to decode "
          f"({sum(s.duration for s in todo) / 3600:.2f} h audio)")
    if not todo:
        return

    mode = "w" if force else "a"
    n_failed = 0
    with jsonl.open(mode, encoding="utf-8") as f, failures.open("a", encoding="utf-8") as ff:
        for start in tqdm(range(0, len(todo), batch_size),
                          desc=f"{slug(model)[:20]}/{suite}", unit="batch"):
            batch = todo[start:start + batch_size]
            t0 = time.perf_counter()
            try:
                hyps = backend.transcribe(batch)
            except Exception as e:
                # One bad segment must not throw away hours of decoding. Every
                # failure is written down, and the report excludes those segment_ids
                # from EVERY model so the columns stay comparable.
                n_failed += len(batch)
                for seg in batch:
                    ff.write(json.dumps({"segment_id": seg.segment_id,
                                         "error": f"{type(e).__name__}: {e}"},
                                        ensure_ascii=False) + "\n")
                ff.flush()
                continue
            elapsed = time.perf_counter() - t0
            audio_s = sum(s.duration for s in batch) or 1e-9
            for seg, hyp in zip(batch, hyps):
                f.write(json.dumps({
                    "segment_id": seg.segment_id, "group_id": seg.group_id,
                    "ref": seg.text, "hyp": hyp, "duration": seg.duration,
                    "rtf": elapsed / audio_s,
                }, ensure_ascii=False) + "\n")
            f.flush()
    if n_failed:
        print(f"  {n_failed} segment(s) failed -> {failures}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", required=True, choices=["hf", "elevenlabs"])
    ap.add_argument("--model", required=True,
                    help="HF repo id (hf backend) or ElevenLabs model_id, e.g. scribe_v2")
    ap.add_argument("--suites", required=True,
                    help="comma-separated suite names, see scripts/benchmark_suites.py")
    ap.add_argument("--path", action="append", default=[], metavar="SUITE=DIR",
                    help="manifest directory for a suite; repeat per suite")
    ap.add_argument("--out", required=True, help="output directory for the JSONL files")
    ap.add_argument("--batch-size", type=int, default=8,
                    help="hf: decode batch. elevenlabs: forced to 1 (one call per segment)")
    ap.add_argument("--language", default=None,
                    help="default: vi for hf, vie for elevenlabs (different code sets)")
    ap.add_argument("--num-beams", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None, help="first N segments per suite (smoke run)")
    ap.add_argument("--force", action="store_true", help="ignore existing output and re-decode")
    args = ap.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = parse_path_args(args.path)
    suites = [s.strip() for s in args.suites.split(",") if s.strip()]

    # Load every suite before loading the model: a missing --path should fail in
    # seconds, not after a 3 GB checkpoint download.
    loaded = {s: load_suite(s, paths.get(s), args.limit) for s in suites}
    for s, segs in loaded.items():
        print(f"suite {s}: {len(segs)} segments, {sum(x.duration for x in segs) / 3600:.2f} h")

    if args.backend == "hf":
        backend = HFBackend(args.model, args.batch_size, args.language or "vi", args.num_beams)
        batch_size = args.batch_size
    else:
        pending = sum(s.duration for suite, segs in loaded.items() for s in segs
                      if s.segment_id not in done_ids(out_paths(out_dir, args.model, suite)[0]))
        print(f"PAID BACKEND: about to send {pending / 3600:.2f} h of audio to ElevenLabs "
              f"({args.model}). Already-decoded segments are skipped and not re-billed.")
        backend = ElevenLabsBackend(args.model, args.language or "vie")
        batch_size = 1

    for suite in suites:
        run_suite(backend, args.model, suite, loaded[suite], out_dir, batch_size, args.force)

    print(f"\nwrote {out_dir}. Score with:\n"
          f"  python -m scripts.benchmark_report --dir {out_dir}")


if __name__ == "__main__":
    main()
