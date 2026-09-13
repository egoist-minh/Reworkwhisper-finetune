"""Raise a corpus's loanword density by dropping train segments that carry none
(docs/v6-curriculum-plan.md §2.4, curriculum phase 2).

`v6-corpus` trains at 2.48% foreign-token density while every measured
evaluation set sits at 5.8-8.7%, and the adapter it produced holds only 48.6%
cross-domain loanword retention against the production adapter's 66.6%. Keeping
just the train segments whose reference carries at least one loanword candidate
leaves 39.76 h at 6.51%, close to the 7.13% density of the corpus that trained
the adapter in production.

Only the TRAIN split is filtered. val and test pass through untouched, so this
corpus's gate numbers stay comparable with the runs that trained on the full
one -- filtering them would change what the numbers mean, not just the model.

"Loanword candidate" is `src.metrics.foreign_token_counts`, the same filter
`english_token_retention` and the gate's cross-domain check score with, applied
to text normalized by the `--config` run's own normalization settings. A
different filter here would make the printed density incomparable with every
retention number in the repo.

    python -m scripts.filter_corpus_density \
        --src dataset/v6-corpus --out dataset/v6-corpus-dense --min-foreign 1

Writes one `manifest.<meeting_id>.jsonl` per meeting with every field of the
kept records untouched (`split` keeps its raw `demo`/`test` value, so
`src.data.resolve_splits` re-derives the same train/val/test), and symlinks
`<out>/audio` at the source's `audio/` -- 14.7 GB stays in one place.
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

from src.config import load as load_config
from src.data import load_manifests, resolve_splits
from src.metrics import foreign_token_counts
from src.normalize import Normalizer


def _split_stats(rows: list[dict], normalizer) -> dict:
    """Segment count, audio hours and foreign-token density for one split."""
    foreign = total = 0
    for r in rows:
        text = normalizer(r["text"])
        foreign += sum(foreign_token_counts(text).values())
        total += len(text.split())
    return {
        "n_segments": len(rows),
        "hours": sum(r["duration"] for r in rows) / 3600,
        "density": foreign / total if total else 0.0,
    }


def filter_dense(records: list[dict], val_meetings: list[str], normalizer,
                  min_foreign: int = 1) -> tuple[list[dict], dict]:
    """Returns (kept records, per-split stats before and after).

    `records` are raw manifest records (`split` in {demo, test}); the resolved
    train/val/test split is derived the same way the pipeline derives it, from
    `val_meetings`, and only resolved-train records are dropped.
    """
    resolved = resolve_splits(records, val_meetings)
    kept, kept_resolved = [], []
    for raw, res in zip(records, resolved):
        if res["split"] != "train" or sum(
                foreign_token_counts(normalizer(raw["text"])).values()) >= min_foreign:
            kept.append(raw)
            kept_resolved.append(res)

    by_split = defaultdict(list)
    for r in resolved:
        by_split[r["split"]].append(r)
    kept_by_split = defaultdict(list)
    for r in kept_resolved:
        kept_by_split[r["split"]].append(r)

    stats = {
        split: {"before": _split_stats(by_split[split], normalizer),
                "after": _split_stats(kept_by_split[split], normalizer)}
        for split in ("train", "val", "test")
    }
    return kept, stats


def _write(kept: list[dict], src: Path, out: Path) -> None:
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"{out} exists and is not empty -- refusing to write over it")
    out.mkdir(parents=True, exist_ok=True)

    by_meeting = defaultdict(list)
    for r in kept:
        by_meeting[r["meeting_id"]].append(r)
    for mid, rows in by_meeting.items():
        with open(out / f"manifest.{mid}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Relative target, so the pair of directories can be moved or mounted
    # anywhere together -- which is what happens on a rented box.
    target = Path(os.path.relpath(src.resolve() / "audio", out.resolve()))
    try:
        (out / "audio").symlink_to(target, target_is_directory=True)
    except OSError as exc:
        raise OSError(
            f"could not symlink {out / 'audio'} -> {target} ({exc}). On Windows this "
            "needs Developer Mode or an elevated shell; the audio is 14.7 GB, so copy "
            "it only if you mean to.") from exc


def _report(stats: dict) -> str:
    lines = []
    for split, s in stats.items():
        before, after = s["before"], s["after"]
        line = (f"{split}: {after['n_segments']} seg / {after['hours']:.2f} h / "
                f"density {100 * after['density']:.2f}%")
        if after["n_segments"] != before["n_segments"]:
            line += (f"   (from {before['n_segments']} seg / {before['hours']:.2f} h / "
                     f"{100 * before['density']:.2f}%)")
        lines.append(line)
    return "\n".join(lines)


def filter_corpus_density(src: Path, out: Path, config_path: str, min_foreign: int,
                           dry_run: bool = False) -> dict:
    cfg = load_config(config_path)
    normalizer = Normalizer(
        strip_punctuation=cfg.normalization.strip_punctuation,
        lowercase=cfg.normalization.lowercase,
        number_convention=cfg.normalization.number_convention,
        filler_tokens=cfg.normalization.filler_tokens,
    )
    records = load_manifests(src)
    kept, stats = filter_dense(records, cfg.data.val_meetings, normalizer, min_foreign)
    print(_report(stats))
    if dry_run:
        return stats
    _write(kept, src, out)
    print(f"wrote {out} ({len(kept)} records, audio symlinked to {src}/audio)")
    return stats


def main() -> None:
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")     # Vietnamese output on a cp1252 console
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--config", default="configs/experiment.yaml",
                    help="source of data.val_meetings and the normalization settings "
                         "the density is measured under")
    ap.add_argument("--min-foreign", type=int, default=1,
                    help="keep a train segment whose reference carries at least this many "
                         "loanword instances (not distinct types)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the split table without writing anything")
    args = ap.parse_args()
    filter_corpus_density(args.src, args.out, args.config, args.min_foreign, args.dry_run)


if __name__ == "__main__":
    main()
