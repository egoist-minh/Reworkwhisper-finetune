"""Pick additional meetings to move from train into val, without touching test
or fetching any new data. v6-corpus val is 365 seg / 0.79 h, too small --
ValCER swings 5.91%/7.65%/4.65% between checkpoints in
`Outputs/v6-corpus-r32.run.log`.

Train/val is not baked into the corpus files -- `src.data.resolve_splits`
derives it from `data.val_meetings` at load time (src/data.py). Growing val is
therefore just adding meeting_ids to that list; this script only picks which
ones, it does not write anything.

    python -m scripts.select_val_meetings --dataset dataset/v6-corpus \
        --target-hours 3.0

Selection:
  - Candidates are every meeting currently resolving to train (§ resolve_splits),
    grouped by `source` (synthetic/youtube per manifest field).
  - Target hours to add per source keep val's synthetic/youtube split matching
    train's own ratio, not train's current (skewed 36.8/63.2) ratio -- computed
    from the data, never hardcoded.
  - Within a source, largest meetings first (fewest meetings moved for a given
    number of hours -- less to vet for voice_id overlap).
  - A candidate is skipped if any of its voice_id also occurs in a meeting that
    would remain in train after this selection -- moving it would leave a val
    meeting whose voice the model still trained on, the same leak already
    present in the current 4 val meetings. Skips are reported, not silently
    dropped.
"""

import argparse
from collections import defaultdict

from src.config import load as load_config
from src.data import load_manifests, resolve_splits


def _meeting_info(records: list[dict]) -> dict[str, dict]:
    """One row per meeting_id: duration (h), source, voice_ids. Raises if a
    meeting mixes sources -- resolve_splits already assumes meeting_id is the
    unit of assignment, a mixed meeting would make that unit meaningless."""
    info: dict[str, dict] = {}
    for r in records:
        m = info.setdefault(r["meeting_id"], {"hours": 0.0, "source": r["source"], "voice_ids": set()})
        if m["source"] != r["source"]:
            raise ValueError(f"{r['meeting_id']}: mixed source {m['source']!r}/{r['source']!r}")
        m["hours"] += r["duration"] / 3600
        if r.get("voice_id"):
            m["voice_ids"].add(r["voice_id"])
    return info


def select_additional_val_meetings(records: list[dict], val_meetings: list[str],
                                    target_hours: float) -> tuple[list[str], dict]:
    """Returns (new meeting_ids to add to val_meetings, report)."""
    resolved = resolve_splits(records, val_meetings)
    train_info = _meeting_info([r for r in resolved if r["split"] == "train"])
    val_info = _meeting_info([r for r in resolved if r["split"] == "val"])

    train_hours = {src: sum(m["hours"] for m in train_info.values() if m["source"] == src)
                   for src in ("synthetic", "youtube")}
    val_hours = {src: sum(m["hours"] for m in val_info.values() if m["source"] == src)
                 for src in ("synthetic", "youtube")}
    train_total = sum(train_hours.values())
    train_ratio = {src: (train_hours[src] / train_total if train_total else 0.5) for src in train_hours}

    target_total = sum(val_hours.values()) + target_hours
    need_add = {src: max(0.0, train_ratio[src] * target_total - val_hours[src]) for src in train_hours}

    # voice_id -> set of meeting_ids currently still assigned to train, updated
    # as meetings get picked for val so later picks see the shrunk pool.
    voice_to_meetings: dict[str, set[str]] = defaultdict(set)
    for mid, m in train_info.items():
        for v in m["voice_ids"]:
            voice_to_meetings[v].add(mid)

    added: list[str] = []
    added_hours = {"synthetic": 0.0, "youtube": 0.0}
    skipped_voice_overlap: list[str] = []

    for src in ("synthetic", "youtube"):
        candidates = sorted(
            (mid for mid, m in train_info.items() if m["source"] == src and mid not in added),
            key=lambda mid: train_info[mid]["hours"], reverse=True,
        )
        for mid in candidates:
            if added_hours[src] >= need_add[src]:
                break
            m = train_info[mid]
            overlap = any(voice_to_meetings[v] - {mid} for v in m["voice_ids"])
            if overlap:
                skipped_voice_overlap.append(mid)
                continue
            added.append(mid)
            added_hours[src] += m["hours"]
            for v in m["voice_ids"]:
                voice_to_meetings[v].discard(mid)

    shortfall = {src: max(0.0, need_add[src] - added_hours[src]) for src in need_add}
    report = {
        "before": {"val_hours": dict(val_hours), "val_meetings": len(val_info)},
        "target_hours_requested": target_hours,
        "need_add_hours": need_add,
        "added_meetings": added,
        "added_hours": added_hours,
        "skipped_voice_overlap": skipped_voice_overlap,
        "shortfall_hours": shortfall,
        "new_val_meetings": sorted(set(val_meetings) | set(added)),
    }
    return added, report


def _print_report(report: dict) -> None:
    print(f"current val: {report['before']['val_meetings']} meetings, "
          f"{sum(report['before']['val_hours'].values()):.2f} h "
          f"(synthetic {report['before']['val_hours']['synthetic']:.2f} h, "
          f"youtube {report['before']['val_hours']['youtube']:.2f} h)")
    print(f"requested +{report['target_hours_requested']:.2f} h -> "
          f"need synthetic +{report['need_add_hours']['synthetic']:.2f} h, "
          f"youtube +{report['need_add_hours']['youtube']:.2f} h")
    print(f"picked {len(report['added_meetings'])} meetings: "
          f"synthetic +{report['added_hours']['synthetic']:.2f} h, "
          f"youtube +{report['added_hours']['youtube']:.2f} h")
    if report["skipped_voice_overlap"]:
        print(f"skipped {len(report['skipped_voice_overlap'])} meetings for voice_id overlap "
              f"with train: {report['skipped_voice_overlap']}")
    for src, h in report["shortfall_hours"].items():
        if h > 0:
            print(f"WARNING: {src} short by {h:.2f} h -- not enough voice-disjoint "
                  "candidate meetings left in train to hit the target")
    flow_list = "[" + ",".join(report["new_val_meetings"]) + "]"
    print(f"\ndata.val_meetings ({len(report['new_val_meetings'])} meetings):")
    print(flow_list)
    print(f"\n--override data.val_meetings={flow_list}")


def main() -> None:
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True, help="corpus dir, e.g. dataset/v6-corpus")
    ap.add_argument("--config", default="configs/experiment.yaml",
                     help="source of the current data.val_meetings")
    ap.add_argument("--target-hours", type=float, default=3.0,
                     help="additional val hours to reach on top of the current val")
    args = ap.parse_args()

    cfg = load_config(args.config)
    records = load_manifests(args.dataset)
    _, report = select_additional_val_meetings(records, cfg.data.val_meetings, args.target_hours)
    _print_report(report)


if __name__ == "__main__":
    main()
