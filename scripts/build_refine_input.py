"""Build the blind input for a reference-refinement pass over `cross-domain-bench`.

The refiner must not see the reference of the segment it is judging -- an existing
transcript anchors a reader hard, and the errors worth finding are exactly the ones a
previous reader already accepted. So the segment's own `text` is withheld mechanically
here rather than by asking the refiner not to look at it: the input file simply does not
contain it. What the refiner gets instead is the two segments either side (their
references, as conversational context) and the five systems' hypotheses for the segment
itself, relabelled A-E per segment so no system's reputation travels with its output.

Writes two files:

  Outputs/cross-domain-refine-input.jsonl  -- give this to the refiner
  Outputs/cross-domain-refine-key.json     -- the letter->system map and the true
                                              reference; keep it back for the diff step
"""

import argparse
import hashlib
import json
import zipfile
from collections import defaultdict
from pathlib import Path

# One reading per model. The run's two v6 columns are the same checkpoint at two
# lambdas, so carrying both would hand one model two correlated votes and blunt the
# signal the refiner works from -- where the readings genuinely diverge. Lambda 0.75
# is the one kept: it is the better of the two on this bench (8.81% CER against
# 9.67%, docs/v6-100h-steps-plan.md and the task ledger).
SYSTEMS = {
    "v5": "winhsss_reworkwhisper_large_v5",
    "base": "vinai_phowhisper_large",
    "scribe": "scribe_v2",
    "v6_l075": "vinai_phowhisper_large_tmp_step800_lam0_75",
}
CONTEXT = 2  # segments either side


def load_bench(path: Path) -> list[dict]:
    """Manifests from an extracted bench directory, or from the zip if that is all
    this machine has."""
    if path.is_dir():
        rows = []
        for f in sorted(path.glob("manifest.*.jsonl")):
            rows += [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
        return rows
    with zipfile.ZipFile(path) as z:
        rows = []
        for name in z.namelist():
            if name.startswith("manifest.") and name.endswith(".jsonl"):
                text = z.read(name).decode("utf-8")
                rows += [json.loads(l) for l in text.splitlines() if l.strip()]
        return rows


def load_hyps(bench_dir: Path, suite: str) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = defaultdict(dict)
    for key, stem in SYSTEMS.items():
        f = bench_dir / f"{stem}.{suite}.persegment.jsonl"
        if not f.exists():
            raise FileNotFoundError(f"{f} -- decode output for {key} is missing")
        for line in f.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            out[r["segment_id"]][key] = r["hyp"]
    return out


def letters(segment_id: str) -> list[str]:
    """A per-segment permutation of the five systems, stable across runs so a rebuilt
    input file diffs cleanly, and different per segment so the refiner cannot learn
    which letter is which system."""
    keys = sorted(SYSTEMS)
    digest = hashlib.sha256(segment_id.encode("utf-8")).digest()
    order = sorted(keys, key=lambda k: digest[keys.index(k)])
    return order


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bench", default="dataset/cross-domain-bench",
                    help="bench directory, or the zip when it is not extracted")
    ap.add_argument("--decodes", default="Outputs/bench-merged-v6-100h")
    ap.add_argument("--suite", default="cross-domain")
    ap.add_argument("--out", default="Outputs")
    args = ap.parse_args()

    bench = Path(args.bench)
    if not bench.exists():
        bench = bench.with_suffix(".zip")
    rows = load_bench(bench)
    hyps = load_hyps(Path(args.decodes), args.suite)

    by_meeting: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_meeting[r["meeting_id"]].append(r)
    for meeting in by_meeting.values():
        meeting.sort(key=lambda r: r["segment_id"])

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    input_path = out_dir / "cross-domain-refine-input.jsonl"
    key_path = out_dir / "cross-domain-refine-key.json"

    key: dict[str, dict] = {}
    n = 0
    with open(input_path, "w", encoding="utf-8") as fh:
        for meeting_id, segs in sorted(by_meeting.items()):
            for i, r in enumerate(segs):
                sid = r["segment_id"]
                full_id = f"{meeting_id}/{sid}"
                if full_id not in hyps:
                    raise KeyError(f"no decode rows for {full_id}")
                order = letters(full_id)
                labelled = {chr(65 + j): hyps[full_id][k] for j, k in enumerate(order)}
                before = [s["text"] for s in segs[max(0, i - CONTEXT):i]]
                after = [s["text"] for s in segs[i + 1:i + 1 + CONTEXT]]
                fh.write(json.dumps({
                    "segment_id": full_id,
                    "meeting_id": meeting_id,
                    "position": f"{i + 1}/{len(segs)}",
                    "duration_s": round(r["duration"], 1),
                    "truoc": before,
                    "sau": after,
                    "ban_doc": labelled,
                }, ensure_ascii=False) + "\n")
                key[full_id] = {
                    "ref": r["text"],
                    "systems": {chr(65 + j): k for j, k in enumerate(order)},
                }
                n += 1

    key_path.write_text(json.dumps(key, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{n} segments -> {input_path}")
    print(f"key (withhold from the refiner) -> {key_path}")


if __name__ == "__main__":
    main()
