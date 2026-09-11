"""Measure what feeding the GPU costs, so a 100 h projection does not assume
audio arrives for free.

A short training probe reads a few hundred wavs (~100 MB) that the OS keeps in
page cache, while a 100 h epoch streams roughly 14 GB across ~53,000 separate
files and resamples most of them from 24 kHz on the CPU. That difference never
shows up in train_runtime, so it gets measured here instead.

Four numbers come out:
  read      raw bytes off disk -- MB/s and files/s. Small-file reads on network
            or overlay storage are IOPS-bound, so files/s matters more than MB/s.
  decode    sf.read + resample_poly through src.data.load_audio_16k.
  mel       WhisperFeatureExtractor on top of that, which the collator runs in
            the same worker process. It pads every segment to a 30 s window, so
            it costs the same for a 4 s clip as a 20 s one and usually dominates
            decode. Measured only with --feature-extractor, since it needs the
            model's preprocessor files.
  feed      per-segment CPU cost times worker count, against the segments/s a
            training step consumes. Below 1.0x the GPU waits on data and a
            faster card buys nothing.

Page cache makes a second run of this script look far faster than the first.
Drop it between runs (`sync && sudo sysctl -w vm.drop_caches=3`) or trust only
the first run against a given set of files.
"""

import argparse
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import load_audio_16k, load_manifests  # noqa: E402


def _pct(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, help="dataset root holding manifest.*.jsonl and audio/")
    ap.add_argument("--files", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--target-hours", type=float, default=100.0,
                    help="corpus size to project the per-epoch read cost for")
    ap.add_argument("--workers", type=int, default=8,
                    help="dataloader_num_workers the real run will use")
    ap.add_argument("--seconds-per-step", type=float, default=None,
                    help="measured step cost, to check whether decode can keep up")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--feature-extractor", default=None,
                    help="model id or path, e.g. vinai/PhoWhisper-large, to also time "
                         "the mel extraction the collator runs in the worker")
    args = ap.parse_args()

    root = Path(args.dataset)
    records = load_manifests(root)
    picked = random.Random(args.seed).sample(records, min(args.files, len(records)))
    paths = [root / "audio" / r["audio_filepath"] for r in picked]
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise SystemExit(f"{len(missing)} of {len(paths)} sampled files missing, first: {missing[0]}")

    # One throwaway pass over the first file per phase. scipy's resampler and
    # WhisperFeatureExtractor's mel filterbank are both built on first call --
    # left in, that one-off lands in p95 and makes the tail look like a stall.
    with open(paths[0], "rb") as f:
        f.read()
    load_audio_16k(paths[0])

    read_times, sizes = [], []
    for p in paths:
        t0 = time.perf_counter()
        with open(p, "rb") as f:
            data = f.read()
        read_times.append(time.perf_counter() - t0)
        sizes.append(len(data))

    decode_times = []
    for p in paths:
        t0 = time.perf_counter()
        load_audio_16k(p)
        decode_times.append(time.perf_counter() - t0)

    mel_times: list[float] = []
    if args.feature_extractor:
        from transformers import WhisperFeatureExtractor

        fe = WhisperFeatureExtractor.from_pretrained(args.feature_extractor)
        fe(load_audio_16k(paths[0]), sampling_rate=16000, return_tensors="np")
        for p in paths:
            audio = load_audio_16k(p)
            t0 = time.perf_counter()
            fe(audio, sampling_rate=16000, return_tensors="np")
            mel_times.append(time.perf_counter() - t0)

    total_mb = sum(sizes) / 1024**2
    read_total = sum(read_times)
    decode_total = sum(decode_times)
    durations = [r["duration"] for r in picked if r.get("duration")]
    hours_sampled = sum(durations) / 3600 if durations else None

    print(f"sampled {len(paths)} files, {total_mb:.1f} MB"
          + (f", {hours_sampled * 60:.1f} min of audio" if hours_sampled else ""))
    print()
    print(f"read    {total_mb / read_total:8.1f} MB/s   {len(paths) / read_total:8.1f} files/s"
          f"   p95 {_pct(read_times, 0.95) * 1000:.1f} ms/file")
    print(f"decode  {len(paths) / decode_total:8.1f} files/s"
          f"   median {_pct(decode_times, 0.5) * 1000:.1f} ms/file"
          f"   p95 {_pct(decode_times, 0.95) * 1000:.1f} ms/file")
    if mel_times:
        print(f"mel     {len(paths) / sum(mel_times):8.1f} files/s"
              f"   median {_pct(mel_times, 0.5) * 1000:.1f} ms/file"
              f"   p95 {_pct(mel_times, 0.95) * 1000:.1f} ms/file")
    else:
        print("mel        (not measured -- pass --feature-extractor to include it; "
              "it usually costs more than decode)")

    if hours_sampled:
        mb_per_hour = total_mb / hours_sampled
        epoch_mb = mb_per_hour * args.target_hours
        print()
        print(f"projection for {args.target_hours:g} h at {mb_per_hour:.0f} MB per audio-hour:")
        print(f"  {epoch_mb / 1024:.1f} GB read per epoch"
              f"  ->  {epoch_mb / (total_mb / read_total) / 60:.1f} min of pure read")

    cpu_total = decode_total + sum(mel_times)
    feed_rate = args.workers * len(paths) / cpu_total
    print()
    label = "decode+mel" if mel_times else "decode only"
    print(f"feed    {feed_rate:.1f} segments/s with {args.workers} workers ({label})")
    if args.seconds_per_step:
        need = args.batch / args.seconds_per_step
        ratio = feed_rate / need
        print(f"        {need:.1f} segments/s consumed at {args.seconds_per_step:g} s/step"
              f" x batch {args.batch}  ->  {ratio:.2f}x headroom")
        if ratio < 1.0:
            print("        BELOW 1.0: the GPU will wait on data. Raise "
                  "dataloader_num_workers or pre-resample the corpus to 16 kHz "
                  "before renting a faster card.")


if __name__ == "__main__":
    main()
