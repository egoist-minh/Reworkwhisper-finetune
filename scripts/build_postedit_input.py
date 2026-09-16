"""Build the input for an LLM post-edit pass over one system's decode output.

The question this measures: can a language model, reading only what the ASR model
produced, repair enough of it to move CER and loanword retention -- no retraining, no
audio. A post-editor that helps here is deployable tomorrow; one that does not is a dead
end worth knowing about before the next GPU rental.

Two leaks are shut off here, both of which would turn the measurement into a
self-congratulation:

  * The reference never enters the input. Obvious, and still worth stating, because the
    neighbouring segments' references would smuggle it in at one remove -- so the
    context either side is the SAME system's hypothesis for those segments, which is
    what a deployed post-editor would actually have.
  * Only one system's reading goes in. Handing the post-editor four systems' output
    measures an ensemble, not a post-editor, and an ensemble costs four decodes at
    inference time (one of them a paid API). Pass --ensemble to measure that instead,
    deliberately and under its own name.

Writes:

  Outputs/<system>-postedit-input.jsonl   -- give this to the post-editing session
  Outputs/<system>-postedit-key.json      -- the reference, for the scoring step only
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

SYSTEMS = {
    "v6_l075": "vinai_phowhisper_large_tmp_step800_lam0_75",
    "v6_l100": "vinai_phowhisper_large_outputs_v6_100h_steps_checkpoints_step_800",
    "v5": "winhsss_reworkwhisper_large_v5",
    "base": "vinai_phowhisper_large",
    "scribe": "scribe_v2",
}
CONTEXT = 2


def read_decode(bench_dir: Path, system: str, suite: str) -> list[dict]:
    f = bench_dir / f"{SYSTEMS[system]}.{suite}.persegment.jsonl"
    if not f.exists():
        raise FileNotFoundError(f"{f} -- no decode output for {system}")
    return [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--system", default="v6_l075", choices=sorted(SYSTEMS))
    ap.add_argument("--decodes", default="Outputs/bench-merged-v6-100h")
    ap.add_argument("--suite", default="cross-domain")
    ap.add_argument("--ensemble", action="store_true",
                    help="also include the other systems' readings -- measures an "
                         "ensemble, not a post-editor; name it as such in any report")
    ap.add_argument("--out", default="Outputs")
    args = ap.parse_args()

    bench = Path(args.decodes)
    rows = read_decode(bench, args.system, args.suite)
    others = {}
    if args.ensemble:
        for key in SYSTEMS:
            if key != args.system:
                try:
                    others[key] = {r["segment_id"]: r["hyp"] for r in read_decode(bench, key, args.suite)}
                except FileNotFoundError:
                    pass

    by_meeting: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_meeting[r["group_id"]].append(r)
    for segs in by_meeting.values():
        segs.sort(key=lambda r: r["segment_id"])

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.system + ("-ensemble" if args.ensemble else "")
    input_path = out_dir / f"{stem}-postedit-input.jsonl"
    key_path = out_dir / f"{stem}-postedit-key.json"

    key = {}
    with open(input_path, "w", encoding="utf-8") as fh:
        for meeting_id, segs in sorted(by_meeting.items()):
            for i, r in enumerate(segs):
                sid = r["segment_id"]
                rec = {
                    "segment_id": sid,
                    "meeting_id": meeting_id,
                    "position": f"{i + 1}/{len(segs)}",
                    "truoc": [s["hyp"] for s in segs[max(0, i - CONTEXT):i]],
                    "ban_may": r["hyp"],
                    "sau": [s["hyp"] for s in segs[i + 1:i + 1 + CONTEXT]],
                }
                if others:
                    rec["he_khac"] = {k: v.get(sid, "") for k, v in others.items()}
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                key[sid] = {"ref": r["ref"], "hyp_goc": r["hyp"]}

    key_path.write_text(json.dumps(key, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(key)} segments -> {input_path}")
    print(f"key (withhold from the post-editor) -> {key_path}")


if __name__ == "__main__":
    main()
