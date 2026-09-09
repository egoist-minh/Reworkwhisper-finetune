"""Score a model on the decode path production actually ships, not the gate's.

Every CER in this repo comes from `src/asr.py:transcribe_batch` -- greedy, 30s
hard truncation, fp16, one segment scored in isolation. Module 4 of
`d:\\viet-speech` decodes with a temperature fallback ladder, long-form
`truncation=False`, fp32, and `condition_on_prev_tokens=True`. Those are
different algorithms, not the same algorithm at different settings, so a gate
CER does not bound a production CER and `check_no_regression_vs_production`
comparing two models on the gate path compares them on a path neither runs when
serving users.

This script produces the missing evidence: the same test segments, scored
through module 4 itself. Nothing here re-implements the decode -- it POSTs audio
to `vendor/viet_speech/remote_asr_server.py`, which constructs `PhoWhisperASR`
with its own defaults and (given `gen_params: null`) hands them straight to
`generate()`. `vendor/viet_speech/VENDORED.md` records where that code came from
and what pins it there; `tests/test_module4_profile.py` fails when it drifts.

Torch is never imported: this side joins audio, calls HTTP, and scores with
`src/metrics.py` + `src/normalize.py`. Scoring is this repo's `Normalizer`, not
`viet-speech`'s `normalize_vi`, which has no number convention -- using it would
add a second axis of difference to a measurement whose whole point is isolating
one.

Two splits:

  --split real   the 264 real-bench chunks rejoined into their 196 parent
                 segments (43.2 min) and sent as ONE clip per parent. This is
                 the only way to measure long-form decoding: the ingest script
                 cut those chunks at real silence and contiguously, so
                 concatenating a parent's children reproduces the parent audio.
  --split test   the 654 test segments of the mixed-noisy-v1 run (426 synthetic
                 + 228 youtube, 94.9 min), one segment per call -- the same
                 granularity as module 4's own per-speech-window calls.

Run real first: the production symptom lives there, and `--probe-determinism`
has to be answered before spending the rest of the GPU budget (see Verify below).

    # on the GPU box, having copied this repo across:
    ASR_MODEL_SIZE=winhsss/Reworkwhisper-large-v5 \\
    ASR_MODEL_ID=winhsss/Reworkwhisper-large-v5 \\
    python vendor/viet_speech/remote_asr_server.py

    # from anywhere that can reach it:
    python -m scripts.eval_module4 --split real \\
        --endpoint https://<tunnel>.ngrok-free.dev \\
        --expect-model-id winhsss/Reworkwhisper-large-v5 \\
        --label v5-lambda0.75

`--expect-model-id` is checked against the `model_id` the server reports before
any audio is sent, and is why the server's `ASR_MODEL_ID` should be set to the HF
repo id: that env var is a free-text label, so it only proves which checkpoint is
loaded if whoever starts the server makes it say so.

Responses are appended to `responses_<split>.jsonl` as they arrive and re-read on
the next run, so a dropped tunnel mid-way through a ~2.3 h split resumes instead
of restarting.
"""

import argparse
import base64
import csv
import io
import json
import time
from pathlib import Path

import numpy as np
import requests
import soundfile as sf

from src.config import load as load_config
from src.data import load_audio_16k, load_manifests
from src.gate import (_chunk_suffix, _meeting_to_source, _parent_segment_id,
                      _score_by_meeting, _score_by_source, rejoin_real_chunks,
                      write_predictions)
from src.metrics import (bootstrap_ci, bootstrap_delta_ci, char_counts,
                          english_token_retention, rate, score, verdict)
from src.normalize import Normalizer

TARGET_SR = 16000
OUT_ROOT = Path("Outputs/module4-parity")
REAL_BENCH = Path("dataset/real-meetings-bench")
TEST_MANIFEST = Path("Outputs/v4-mixed-r16/validated_manifest.jsonl")
# mixed-noisy-v1 was built by referencing these two corpora in place rather than
# copying 95 minutes of audio into a third one, and scripts/build_mixed_dataset.py
# guarantees their first-level directory names are disjoint -- so a manifest's
# relative `audio_filepath` resolves under exactly one of them. `_resolve` checks
# that rather than trusting it.
TEST_AUDIO_ROOTS = (Path("dataset/paid-dataset-v2/audio"),
                    Path("dataset/youtube-meetings/audio"))
TIER = {"real": "tier4a_real", "test": "tier1_in_domain"}


def _normalizer(config_path: str) -> Normalizer:
    cfg = load_config(config_path)
    return Normalizer(
        strip_punctuation=cfg.normalization.strip_punctuation,
        lowercase=cfg.normalization.lowercase,
        number_convention=cfg.normalization.number_convention,
        filler_tokens=cfg.normalization.filler_tokens,
    )


def _resolve(rel: str, roots=TEST_AUDIO_ROOTS) -> Path:
    hits = [root / rel for root in roots if (root / rel).exists()]
    if not hits:
        raise FileNotFoundError(f"{rel} is under none of {[str(r) for r in roots]}")
    if len(hits) > 1:
        raise RuntimeError(
            f"{rel} exists under {len(hits)} roots ({[str(h) for h in hits]}) -- the "
            "disjoint-directory assumption these two roots are resolved under is broken")
    return hits[0]


def _wav_bytes(samples: np.ndarray) -> bytes:
    """float32 16 kHz mono -> a PCM_16 WAV buffer. The server reads the sample rate
    back out of this header (`wav_to_pcm16`) and ignores the request's
    `sample_rate` field, so everything must already be at TARGET_SR."""
    buf = io.BytesIO()
    sf.write(buf, samples, TARGET_SR, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def build_real_clips(normalizer: Normalizer) -> list[dict]:
    """One clip per PARENT real-bench segment: sub-chunk audio concatenated in
    chunk order, sub-chunk refs joined by `rejoin_real_chunks` itself.

    Reusing that function rather than re-joining the text here is deliberate --
    it is what produced the `ref` column of every tier4a CSV in Outputs/, so the
    refs below are identical to the gate's byte for byte, and the CER difference
    between the two is attributable to the decode path alone."""
    records = load_manifests(REAL_BENCH)
    refs = {(r["meeting_id"], r["segment_id"]): normalizer(r["text"]) for r in records}
    parents = rejoin_real_chunks(
        [{"segment_id": r["segment_id"], "meeting_id": r["meeting_id"],
          "ref": refs[(r["meeting_id"], r["segment_id"])], "hyp": ""} for r in records])

    chunks: dict[tuple, list[dict]] = {}
    for r in records:
        chunks.setdefault((r["meeting_id"], _parent_segment_id(r["segment_id"])), []).append(r)

    clips = []
    for row in parents:
        key = (row["meeting_id"], row["segment_id"])
        ordered = sorted(chunks[key], key=lambda r: _chunk_suffix(r["segment_id"]))
        audio = np.concatenate([
            load_audio_16k(REAL_BENCH / "audio" / r["audio_filepath"]) for r in ordered])
        clips.append({"segment_id": row["segment_id"], "meeting_id": row["meeting_id"],
                      "ref": row["ref"], "audio": audio,
                      "n_chunks": len(ordered), "duration": len(audio) / TARGET_SR})
    return clips


def build_test_clips(normalizer: Normalizer) -> tuple[list[dict], dict[str, str]]:
    records = [json.loads(l) for l in
               TEST_MANIFEST.read_text(encoding="utf-8").splitlines()]
    test = [r for r in records if r["split"] == "test"]
    clips = []
    for r in test:
        audio = load_audio_16k(_resolve(r["audio_filepath"]))
        clips.append({"segment_id": r["segment_id"], "meeting_id": r["meeting_id"],
                      "ref": normalizer(r["text"]), "audio": audio,
                      "source": r["source"], "duration": len(audio) / TARGET_SR})
    return clips, _meeting_to_source(test)


def post(endpoint: str, audio: np.ndarray, timeout: float) -> dict:
    """One clip, one call -- module 4's own granularity (`transcribe_per_splice`).
    `gen_params: null` is what makes this the production profile: the server
    passes it as `gen_overrides`, and None leaves PhoWhisperASR's constructor
    defaults untouched."""
    resp = requests.post(
        f"{endpoint.rstrip('/')}/transcribe",
        json={"audio_b64": base64.b64encode(_wav_bytes(audio)).decode(),
              "sample_rate": TARGET_SR, "language": "vi", "gen_params": None},
        timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def check_server(endpoint: str, expect_model_id: str, audio: np.ndarray,
                  timeout: float) -> str:
    """Verify gate 1: the server must be serving the checkpoint this run claims to
    measure. Costs one short transcription, before the other ~137 minutes."""
    got = post(endpoint, audio[:TARGET_SR], timeout)["model_id"]
    if got != expect_model_id:
        raise RuntimeError(
            f"server reports model_id {got!r}, expected {expect_model_id!r} -- refusing "
            "to attribute these numbers to a checkpoint that is not loaded")
    print(f"server ok: model_id={got}")
    return got


def probe_determinism(endpoint: str, clip: dict, n: int, timeout: float) -> dict:
    """Verify gate 2. The temperature ladder only engages on a window that trips
    the logprob / compression-ratio thresholds, and above 0.0 generate() samples.
    Whether it engages on THIS audio is a property of the audio, so it has to be
    measured, and a difference here is the finding -- not noise to average away."""
    hyps = [post(endpoint, clip["audio"], timeout)["text"] for _ in range(n)]
    identical = len(set(hyps)) == 1
    print(f"determinism probe on {clip['meeting_id']}/{clip['segment_id']} "
          f"({clip['duration']:.1f}s), {n} calls: "
          f"{'IDENTICAL' if identical else 'DIFFERENT'}")
    for i, h in enumerate(hyps):
        print(f"  [{i}] {h[:160]}")
    if not identical:
        print("  -> the temperature ladder is binding on this audio. Run each split "
              "several times and report the spread; a single pass is not the model's "
              "output, it is one draw from it.")
    return {"segment_id": clip["segment_id"], "meeting_id": clip["meeting_id"],
            "duration": clip["duration"], "n_calls": n, "identical": identical,
            "hyps": hyps}


def transcribe_all(clips: list[dict], endpoint: str, jsonl: Path, timeout: float) -> list[dict]:
    """POST every clip, appending each response to `jsonl` as it lands. Clips whose
    key is already in that file are skipped -- a 2.3 h split that loses its tunnel
    at minute 90 resumes rather than restarting."""
    done = {}
    if jsonl.exists():
        for line in jsonl.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            done[(row["meeting_id"], row["segment_id"])] = row
        print(f"{len(done)} clips already in {jsonl}, resuming")

    jsonl.parent.mkdir(parents=True, exist_ok=True)
    out = []
    with open(jsonl, "a", encoding="utf-8") as f:
        for i, clip in enumerate(clips, 1):
            key = (clip["meeting_id"], clip["segment_id"])
            if key in done:
                out.append(done[key])
                continue
            t0 = time.perf_counter()
            result = post(endpoint, clip["audio"], timeout)
            row = {"segment_id": clip["segment_id"], "meeting_id": clip["meeting_id"],
                   "duration": clip["duration"], "hyp_raw": result["text"],
                   "rtf": result.get("rtf"), "wall_s": time.perf_counter() - t0,
                   "n_segments_returned": len(result.get("segments", []))}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            out.append(row)
            print(f"[{i}/{len(clips)}] {clip['meeting_id']}/{clip['segment_id']} "
                  f"{clip['duration']:.1f}s rtf={row['rtf']:.2f} {len(row['hyp_raw'])} chars")
    return out


def capitalization(hyps_raw: list[str]) -> dict:
    """Tokens the model chose to capitalise, before normalization lowercases them
    away. Needs no reference: it compares one model against another, which is the
    half of the symptom (15 capitalised tokens down to 1 between v4 and v5) that
    every CER in this repo is blind to by construction."""
    tokens = [t for h in hyps_raw for t in h.split()]
    n_cap = sum(1 for t in tokens if t[:1].isupper())
    return {"n_tokens": len(tokens), "n_capitalized": n_cap,
            "rate": n_cap / len(tokens) if tokens else None}


def _read_predictions(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _align_baseline(rows: list[dict], baseline_rows: list[dict]) -> list[dict] | None:
    """Baseline rows in the candidate's order, or None when the two runs did not
    score the same segments. Raises on a shared key whose `ref` differs -- that is
    two different test sets, and a paired bootstrap over it means nothing."""
    index = {(r["meeting_id"], r["segment_id"]): r for r in baseline_rows}
    if index.keys() != {(r["meeting_id"], r["segment_id"]) for r in rows}:
        return None
    aligned = [index[(r["meeting_id"], r["segment_id"])] for r in rows]
    for cand, base in zip(rows, aligned):
        if cand["ref"] != base["ref"]:
            raise RuntimeError(
                f"segment {cand['segment_id']} has a different `ref` in the baseline dir "
                "-- these are not the same test segments")
    return aligned


def score_split(split: str, rows: list[dict], hyps_raw: list[str],
                 meeting_to_source: dict | None, baseline_rows: list[dict] | None) -> dict:
    s = score([r["ref"] for r in rows], [r["hyp"] for r in rows])
    counts = s.pop("_char_counts")
    lo, hi = bootstrap_ci(counts)
    out = {"cer": s["cer"], "wer": s["wer"], "n_segments": s["n_segments"],
           "char_ref_len": s["char_ref_len"], "ci": [lo, hi],
           "retention": english_token_retention([r["ref"] for r in rows],
                                                 [r["hyp"] for r in rows])["retention"],
           "capitalization": capitalization(hyps_raw)}

    if split == "real":
        out["by_meeting"] = _score_by_meeting(rows)
    else:
        # min_improvement_pct/max_retention_regression_pp are left off: those apply
        # tier 1's pass rule against a BASE-MODEL baseline, and the baseline here is
        # another fine-tune serving production. The pass/fail decision on these
        # numbers is scripts/merge_and_push.py's, not this script's.
        out["by_source"] = _score_by_source(rows, meeting_to_source, baseline_rows)

    if baseline_rows is not None:
        aligned = _align_baseline(rows, baseline_rows)
        if aligned is None:
            out["verdict"] = "SKIPPED (baseline dir scored a different segment set)"
        else:
            base_counts = [char_counts(r["ref"], r["hyp"]) for r in aligned]
            out["cer_baseline"] = rate(base_counts)
            out["retention_baseline"] = english_token_retention(
                [r["ref"] for r in aligned], [r["hyp"] for r in aligned])["retention"]
            d_lo, d_hi = bootstrap_delta_ci(base_counts, counts)
            out["delta_ci"] = [d_lo, d_hi]
            out["verdict"] = verdict(d_lo, d_hi)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", required=True, choices=("real", "test"))
    ap.add_argument("--endpoint", required=True, help="base URL of remote_asr_server.py")
    ap.add_argument("--expect-model-id", required=True,
                    help="the model_id the server must report (set ASR_MODEL_ID to the "
                         "HF repo id so this proves which checkpoint is loaded)")
    ap.add_argument("--label", required=True,
                    help=f"output goes to {OUT_ROOT}/<label>/")
    ap.add_argument("--config", default="configs/experiment.yaml",
                    help="read the scoring normalization from here")
    ap.add_argument("--baseline-dir",
                    help=f"another {OUT_ROOT}/<label> dir (production's) to add "
                         "cer_baseline/delta_ci/verdict against")
    ap.add_argument("--probe-determinism", type=int, metavar="N",
                    help="POST the first clip N times, report whether the hypotheses "
                         "agree, and stop without scoring anything")
    ap.add_argument("--limit", type=int, help="first N clips only (smoke test)")
    ap.add_argument("--timeout", type=float, default=900.0,
                    help="per-request seconds; a 212 s clip at fp32 long-form is slow")
    args = ap.parse_args()

    normalizer = _normalizer(args.config)
    meeting_to_source = None
    if args.split == "real":
        clips = build_real_clips(normalizer)
    else:
        clips, meeting_to_source = build_test_clips(normalizer)
    if args.limit:
        clips = clips[:args.limit]
    total_min = sum(c["duration"] for c in clips) / 60
    print(f"{args.split}: {len(clips)} clips, {total_min:.1f} min of audio")

    out_dir = OUT_ROOT / args.label
    out_dir.mkdir(parents=True, exist_ok=True)
    model_id = check_server(args.endpoint, args.expect_model_id, clips[0]["audio"],
                            args.timeout)

    if args.probe_determinism:
        probe = probe_determinism(args.endpoint, clips[0], args.probe_determinism,
                                   args.timeout)
        probe["endpoint"], probe["model_id"] = args.endpoint, model_id
        path = out_dir / f"determinism_probe_{args.split}.json"
        path.write_text(json.dumps(probe, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"wrote {path}")
        return

    responses = transcribe_all(clips, args.endpoint,
                                out_dir / f"responses_{args.split}.jsonl", args.timeout)
    by_key = {(r["meeting_id"], r["segment_id"]): r for r in responses}

    rows, raw_rows, hyps_raw, used = [], [], [], []
    for clip in clips:
        response = by_key[(clip["meeting_id"], clip["segment_id"])]
        raw = response["hyp_raw"]
        hyps_raw.append(raw)
        used.append(response)
        rows.append({"segment_id": clip["segment_id"], "meeting_id": clip["meeting_id"],
                     "ref": clip["ref"], "hyp": normalizer(raw)})
        raw_rows.append({"segment_id": clip["segment_id"], "meeting_id": clip["meeting_id"],
                         "ref": clip["ref"], "hyp": raw})

    tier = TIER[args.split]
    # Exactly write_predictions' four columns, so check_no_regression_vs_production
    # reads these files with no change at all.
    write_predictions(rows, out_dir / f"predictions_{tier}.csv")
    write_predictions(raw_rows, out_dir / f"hyp_raw_{args.split}.csv")

    baseline_rows = None
    if args.baseline_dir:
        baseline_rows = _read_predictions(Path(args.baseline_dir) / f"predictions_{tier}.csv")

    results_path = out_dir / "module4_results.json"
    results = (json.loads(results_path.read_text(encoding="utf-8"))
               if results_path.exists() else {})
    results[tier] = score_split(args.split, rows, hyps_raw, meeting_to_source, baseline_rows)
    results.setdefault("_meta", {})[args.split] = {
        "endpoint": args.endpoint, "model_id": model_id,
        "run_finished": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_clips": len(clips), "audio_minutes": round(total_min, 2),
        "gen_params": None,
        "decode_path": "vendor/viet_speech (module 4), PhoWhisperASR constructor defaults",
        "baseline_dir": args.baseline_dir,
        # Over the clips this run scored, not every line in responses_*.jsonl --
        # a resumed or --limit run would otherwise report a different set's cost.
        "rtf": {"mean": sum(r["rtf"] for r in used) / len(used),
                "max": max(r["rtf"] for r in used)},
    }
    results_path.write_text(json.dumps(results, indent=2, ensure_ascii=False),
                            encoding="utf-8")
    print(json.dumps({k: v for k, v in results[tier].items()
                      if k not in ("by_meeting", "by_source")},
                     indent=2, ensure_ascii=False))
    print(f"wrote {results_path}")


if __name__ == "__main__":
    main()
