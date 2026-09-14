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
    """Segment count, meeting count, audio hours and foreign-token density for
    one split. The meeting count is what tells the two whole-meeting modes apart
    at a glance: dropping meetings whole moves it, cutting inside them does not."""
    foreign = total = 0
    for r in rows:
        text = normalizer(r["text"])
        foreign += sum(foreign_token_counts(text).values())
        total += len(text.split())
    return {
        "n_segments": len(rows),
        "n_meetings": len({r["meeting_id"] for r in rows}),
        "hours": sum(r["duration"] for r in rows) / 3600,
        "density": foreign / total if total else 0.0,
    }


def _counts(rec: dict, normalizer) -> tuple[int, int]:
    """(foreign-token instances, total tokens) of one record's normalized text."""
    text = normalizer(rec["text"])
    return sum(foreign_token_counts(text).values()), len(text.split())


def _apply(records: list[dict], val_meetings: list[str], normalizer,
            select) -> tuple[list[dict], dict]:
    """Run `select` over the resolved-train records only, and report per-split
    stats before and after. val/test never reach `select`.

    `records` are raw manifest records (`split` in {demo, test}); the resolved
    train/val/test split is derived the same way the pipeline derives it, from
    `val_meetings`.
    """
    resolved = resolve_splits(records, val_meetings)
    train_raw = [raw for raw, res in zip(records, resolved) if res["split"] == "train"]
    keep = {(r["meeting_id"], r["segment_id"]) for r in select(train_raw)}

    kept, kept_resolved = [], []
    for raw, res in zip(records, resolved):
        if res["split"] != "train" or (raw["meeting_id"], raw["segment_id"]) in keep:
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


def _by_meeting(rows: list[dict]) -> dict[str, list[dict]]:
    grouped = defaultdict(list)
    for r in rows:
        grouped[r["meeting_id"]].append(r)
    return grouped


def filter_dense(records: list[dict], val_meetings: list[str], normalizer,
                  min_foreign: int = 1) -> tuple[list[dict], dict]:
    """Per-segment mode: keep a train segment carrying >= `min_foreign` loanword
    instances. Returns (kept records, per-split stats before and after)."""
    def select(train):
        return [r for r in train if _counts(r, normalizer)[0] >= min_foreign]

    return _apply(records, val_meetings, normalizer, select)


def filter_min_meeting_density(records: list[dict], val_meetings: list[str], normalizer,
                                min_density: float) -> tuple[list[dict], dict]:
    """Whole-meeting mode: keep every segment of a train meeting whose own
    loanword density is >= `min_density`, drop the rest of the meetings whole.

    No segment inside a kept meeting is touched, so the kept audio still has the
    Vietnamese/loanword alternation of real speech -- that is the difference from
    `filter_dense`, which reaches the same density by deleting the Vietnamese.
    """
    def select(train):
        kept = []
        for rows in _by_meeting(train).values():
            counts = [_counts(r, normalizer) for r in rows]
            foreign = sum(f for f, _ in counts)
            total = sum(t for _, t in counts)
            if total and foreign / total >= min_density:
                kept.extend(rows)
        return kept

    return _apply(records, val_meetings, normalizer, select)


def filter_meeting_quota(records: list[dict], val_meetings: list[str], normalizer,
                          quota: float, min_meeting_foreign_density: float
                          ) -> tuple[list[dict], dict]:
    """Quota mode: per train meeting, keep every segment that carries a loanword,
    then walk the loanword-free ones longest-first, taking each one that leaves
    the meeting's density >= `quota` and skipping the ones that would breach it
    (a shorter segment later in the walk can still fit). A meeting whose
    loanword-carrying segments alone sit below
    `min_meeting_foreign_density` is dropped whole -- that subset is the densest
    the meeting can get, so it cannot reach the quota without cutting loanwords.

    Longest-first buys the most audio per token of quota spent. Ties break on
    `segment_id`, so a later run rebuilds the same corpus byte for byte.
    """
    def select(train):
        kept = []
        for rows in _by_meeting(train).values():
            counted = [(r, *_counts(r, normalizer)) for r in rows]
            carriers = [(r, f, t) for r, f, t in counted if f > 0]
            if not carriers:
                continue
            foreign = sum(f for _, f, _ in carriers)
            total = sum(t for _, _, t in carriers)
            if not total or foreign / total < min_meeting_foreign_density:
                continue
            meeting_kept = [r for r, _, _ in carriers]
            fillers = sorted(((r, t) for r, f, t in counted if f == 0),
                             key=lambda p: (-p[0]["duration"], p[0]["segment_id"]))
            for r, t in fillers:
                if foreign / (total + t) < quota:
                    continue        # a shorter one may still fit under the quota
                total += t
                meeting_kept.append(r)
            kept.extend(meeting_kept)
        return kept

    return _apply(records, val_meetings, normalizer, select)


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
        line = (f"{split}: {after['n_segments']} seg / {after['n_meetings']} meetings / "
                f"{after['hours']:.2f} h / density {100 * after['density']:.2f}%")
        if after["n_segments"] != before["n_segments"]:
            line += (f"   (from {before['n_segments']} seg / {before['n_meetings']} meetings / "
                     f"{before['hours']:.2f} h / {100 * before['density']:.2f}%)")
        lines.append(line)
    return "\n".join(lines)


def filter_corpus_density(src: Path, out: Path, config_path: str, min_foreign: int | None = None,
                           dry_run: bool = False, min_meeting_density: float | None = None,
                           meeting_quota: float | None = None,
                           min_meeting_foreign_density: float | None = None) -> dict:
    """Dispatch to whichever of the three train-selection modes was asked for."""
    cfg = load_config(config_path)
    normalizer = Normalizer(
        strip_punctuation=cfg.normalization.strip_punctuation,
        lowercase=cfg.normalization.lowercase,
        number_convention=cfg.normalization.number_convention,
        filler_tokens=cfg.normalization.filler_tokens,
    )
    records = load_manifests(src)
    if min_meeting_density is not None:
        kept, stats = filter_min_meeting_density(records, cfg.data.val_meetings, normalizer,
                                                  min_meeting_density)
    elif meeting_quota is not None:
        kept, stats = filter_meeting_quota(records, cfg.data.val_meetings, normalizer,
                                            meeting_quota, min_meeting_foreign_density)
    else:
        kept, stats = filter_dense(records, cfg.data.val_meetings, normalizer,
                                    1 if min_foreign is None else min_foreign)
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
    ap.add_argument("--min-foreign", type=int, default=None,
                    help="per-segment mode (the default, at 1): keep a train segment whose "
                         "reference carries at least this many loanword instances (not "
                         "distinct types)")
    ap.add_argument("--min-meeting-density", type=float, default=None,
                    help="whole-meeting mode: keep every segment of a train meeting whose own "
                         "loanword density is at least this (e.g. 0.03), drop other meetings whole")
    ap.add_argument("--meeting-quota", type=float, default=None,
                    help="quota mode: per train meeting keep all loanword-carrying segments, "
                         "then add loanword-free ones longest-first while density stays at "
                         "least this (e.g. 0.0741)")
    ap.add_argument("--min-meeting-foreign-density", type=float, default=None,
                    help="quota mode only: drop a meeting whose loanword-carrying segments "
                         "alone sit below this density (e.g. 0.045)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the split table without writing anything")
    args = ap.parse_args()

    modes = [args.min_foreign is not None, args.min_meeting_density is not None,
             args.meeting_quota is not None]
    if sum(modes) > 1:
        ap.error("--min-foreign, --min-meeting-density and --meeting-quota are three "
                 "different filters -- pick one")
    if (args.meeting_quota is None) != (args.min_meeting_foreign_density is None):
        ap.error("--meeting-quota and --min-meeting-foreign-density go together")

    filter_corpus_density(args.src, args.out, args.config, args.min_foreign, args.dry_run,
                           args.min_meeting_density, args.meeting_quota,
                           args.min_meeting_foreign_density)


if __name__ == "__main__":
    main()
