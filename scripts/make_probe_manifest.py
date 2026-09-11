"""Rewrite a run's validated_manifest.jsonl down to a probe-sized train split,
so `--stage train` measures step cost on a chosen sample instead of whichever
segments happen to sit at the top of the file.

`cfg.training.limit` takes the first N records (src/data.py:limit) and
build_mixed_dataset.py concatenates source manifests in sorted filename order,
so the first N of a mixed corpus are all one source. Synthetic segments carry
~67 characters of label against YouTube's ~225, and the collator pads labels to
the longest in the batch, so a probe that lands entirely on synthetic reports
the cheapest steps the run will ever take.

Two modes:
  worst   the longest labels in the split. Every batch pads to the corpus
          maximum, so no batch in the real run can cost more -- the step cost
          this yields is a hard ceiling, and its VRAM is the worst case too.
  random  a seeded sample, for what a typical step actually costs.

val and test rows are left exactly as they are; only train rows are replaced.
The original file is copied to validated_manifest.full.jsonl first, and the
script refuses to run twice without --force so a probe can never eat the real
manifest.
"""

import argparse
import json
import random
import shutil
from pathlib import Path


def load_manifest(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def select(train_rows: list[dict], mode: str, count: int, seed: int) -> list[dict]:
    if count >= len(train_rows):
        return train_rows
    if mode == "worst":
        return sorted(train_rows, key=lambda r: len(r.get("text", "")), reverse=True)[:count]
    return random.Random(seed).sample(train_rows, count)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True,
                    help="outputs/<run_id>, the directory holding validated_manifest.jsonl")
    ap.add_argument("--mode", choices=["worst", "random"], required=True)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--batch", type=int, default=16,
                    help="effective batch: training.batch_size * training.grad_accum_steps")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true",
                    help="rewrite even though a backup already exists")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    manifest = run_dir / "validated_manifest.jsonl"
    backup = run_dir / "validated_manifest.full.jsonl"
    if not manifest.exists():
        raise SystemExit(f"{manifest} missing -- run `--stage baseline` for this run_id first")
    if backup.exists() and not args.force:
        raise SystemExit(
            f"{backup} already exists, so {manifest.name} is already a probe manifest. "
            f"Restore it with `cp {backup} {manifest}` or pass --force."
        )
    if not backup.exists():
        shutil.copy2(manifest, backup)

    rows = load_manifest(backup)
    train_rows = [r for r in rows if r.get("split") == "train"]
    keep = [r for r in rows if r.get("split") != "train"]
    count = args.steps * args.batch
    picked = select(train_rows, args.mode, count, args.seed)

    with open(manifest, "w", encoding="utf-8") as f:
        for r in picked + keep:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    lengths = sorted(len(r.get("text", "")) for r in picked)
    sources = {}
    for r in picked:
        sources[r.get("source", "unknown")] = sources.get(r.get("source", "unknown"), 0) + 1
    print(f"mode={args.mode}  train {len(train_rows)} -> {len(picked)} "
          f"({len(picked) // args.batch} steps at batch {args.batch})")
    print(f"label chars: min={lengths[0]} median={lengths[len(lengths) // 2]} max={lengths[-1]}")
    print(f"sources: {sources}")
    print(f"backup: {backup}")


if __name__ == "__main__":
    main()
