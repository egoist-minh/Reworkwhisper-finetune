"""Eval gate. PROJECT_CORE.md §6 Stage 4.

Scope cut for the 2026-08-02 deadline (see handoff): only tiers 1 (in-domain),
2 (OOD), and 4a (real audio, segmented) run. Tier 3 (RTF) is dropped and
tier 4b (long-form) is deferred -- do not re-expand scope here without
updating configs/experiment.yaml + PROJECT_CORE.md first.

Any tier FAIL halts the pipeline: no adapter is marked pass, nothing pushed
to HF (§0 problem 4, "Fail Loudly").

Tier 4a statistical rigor (added 2026-08-02, PROJECT_CORE.md §4 "statistical
power" + "six properties that limit what tier 4 can claim"): with only ~264
real-bench segments from 2 recordings, a bare point-estimate CER comparison
against a zero-tolerance threshold cannot distinguish a real regression from
sampling noise, and pools two different rooms/speaker-sets into one number.
Three additions address this, all using data already produced by the
baseline/gate stages -- no new real audio needed:
  - `by_meeting`: CER broken out per real-bench recording, so a regression
    concentrated in one room isn't hidden by averaging with the other.
  - `delta_ci` / `verdict`: a PAIRED bootstrap (`bootstrap_delta_ci`) between
    baseline and candidate on the identical segments, tighter than treating
    the two CERs as independent draws, plus the already-written but
    previously-unused `verdict()` classifying the result as
    IMPROVED/REGRESSED/INCONCLUSIVE instead of a threshold-only pass/fail.
    Requires `baseline_real_csv` (baseline stage's
    `audit/predictions_baseline_real.csv`) so both sides score the same
    segments in the same order.
  - `normalization_check`: rescoring under the opposite number convention
    (word-to-digit vs as-written) to surface whether a result is
    normalization-dominated rather than model-dominated (§6 "convention-
    sensitivity check", previously spec'd but not built). Diagnostic only --
    does not affect `pass`/`overall_pass`. Gated on `cfg.normalization.
    audit_conversions` (was declared in `src/config.py` but never read
    anywhere until 2026-08-04 -- also added the same check to tier1_in_domain,
    since without it a tier1 CER gain can't be attributed between digit-
    normalization and actual acoustic learning).

Rejoin-before-scoring (found 2026-08-04 reading v3-r16's tier4a audit):
`scripts/ingest_real_bench.py` splits any real-bench segment over 30s into
sub-chunks, cutting audio at real silence but dividing the *text* only
proportionally by each chunk's share of duration -- not real alignment, per
that script's own docstring. When speech rate is uneven this mis-attributes
words across chunk boundaries, so scoring a sub-chunk's hyp against its own
ref inflates CER for both baseline and candidate alike (confirmed on v3-r16:
pooled CER dropped from 45.8%/48.0% to 31.6%/35.1% baseline/candidate once
rejoined). `rejoin_real_chunks`/`score_real` below concatenate sub-chunks
back to their parent segment (exact, by the ingest script's own invariant)
before computing CER/CI/by_meeting/delta_ci -- changes the absolute CER
number but not the tier's pass/fail semantics.
"""

import re
from pathlib import Path

from src.asr import transcribe_batch
from src.metrics import score, char_counts, rate, bootstrap_ci, bootstrap_delta_ci, verdict
from src.normalize import Normalizer


def _eval_split(model, processor, dataset, normalizer, eval_cfg, desc: str | None = None) -> dict:
    """Returns the usual score() dict plus `_predictions`: one row per segment
    (segment_id, meeting_id, ref, hyp) so a run's evidence includes what the
    model actually said, not just the aggregate CER (§0 "Evidence-Based").
    `desc` labels the transcribe_batch progress bar."""
    n = min(len(dataset), eval_cfg.limit) if eval_cfg.limit else len(dataset)
    items = [dataset[i] for i in range(n)]
    audios = [it["audio"] for it in items]
    refs = [normalizer(it["text"]) for it in items]
    hyps_raw = transcribe_batch(model, processor, audios, language=eval_cfg.language,
                                 num_beams=eval_cfg.num_beams, batch_size=eval_cfg.batch_size,
                                 desc=desc)
    hyps = [normalizer(h) for h in hyps_raw]
    result = score(refs, hyps)
    result["_predictions"] = [
        {"segment_id": it.get("segment_id"), "meeting_id": it.get("meeting_id"),
         "ref": r, "hyp": h}
        for it, r, h in zip(items, refs, hyps)
    ]
    return result


def _score_by_meeting(predictions: list[dict]) -> dict:
    """Per-meeting_id CER -- real-bench pools segments from multiple recordings
    (different rooms/speaker sets); one pooled CER can hide a regression that's
    actually concentrated in a single recording."""
    by_meeting: dict[str, list[dict]] = {}
    for row in predictions:
        by_meeting.setdefault(row["meeting_id"], []).append(row)
    result = {}
    for mid, rows in by_meeting.items():
        s = score([r["ref"] for r in rows], [r["hyp"] for r in rows])
        result[mid] = {"cer": s["cer"], "wer": s["wer"], "n_segments": s["n_segments"]}
    return result


def _parent_segment_id(segment_id: str) -> str:
    """`seg_0000` or `seg_0000_3` (scripts/ingest_real_bench.py naming) -> `seg_0000`."""
    m = re.match(r"(seg_\d{4})(?:_\d+)?$", segment_id)
    return m.group(1) if m else segment_id


def _chunk_suffix(segment_id: str) -> int:
    m = re.match(r"seg_\d{4}(?:_(\d+))?$", segment_id)
    return int(m.group(1)) if m and m.group(1) else 0


def rejoin_real_chunks(predictions: list[dict]) -> list[dict]:
    """Concatenate ingest sub-chunks (module docstring, "Rejoin-before-scoring")
    back to their parent segment, in chunk order, per meeting_id. Concatenation
    reproduces the original pre-split ref exactly -- ingest script's own
    invariant -- so this only undoes the sub-chunk text/audio mis-attribution,
    it does not lose or alter any text."""
    groups: dict[tuple, list[dict]] = {}
    for row in predictions:
        key = (row["meeting_id"], _parent_segment_id(row["segment_id"]))
        groups.setdefault(key, []).append(row)
    out = []
    for (meeting_id, parent), rows in groups.items():
        rows = sorted(rows, key=lambda r: _chunk_suffix(r["segment_id"]))
        out.append({
            "segment_id": parent, "meeting_id": meeting_id,
            "ref": " ".join(r["ref"] for r in rows),
            "hyp": " ".join(r["hyp"] for r in rows),
        })
    return out


def score_real(predictions: list[dict]) -> dict:
    """CER for a real-bench split at the parent-segment level (see
    `rejoin_real_chunks`) instead of raw ingest sub-chunks. `_char_counts` is
    per-parent-segment, for `bootstrap_ci`/`bootstrap_delta_ci`; `_rejoined`
    is the rejoined rows, for `_score_by_meeting`."""
    rejoined = rejoin_real_chunks(predictions)
    counts = [char_counts(r["ref"], r["hyp"]) for r in rejoined]
    return {"cer": rate(counts), "_char_counts": counts, "_rejoined": rejoined}


def _load_char_counts_from_predictions(csv_path: Path) -> list:
    """Reconstruct per-segment Counts from a predictions_*.csv (ref/hyp already
    normalized the same way `score()` would have seen them at write time)."""
    import csv as csv_module

    with open(csv_path, encoding="utf-8") as f:
        rows = list(csv_module.DictReader(f))
    return [char_counts(r["ref"], r["hyp"]) for r in rows]


def run_gate(cfg, model, processor, normalizer, test_ds, ood_ds, real_ds, baseline: dict,
             baseline_real_csv: str | Path | None = None) -> dict:
    """cfg: src.config.Config. `baseline` is metrics/baseline.json's parsed dict
    (cer_test, cer_ood, cer_real -- all computed once at Stage 1, never
    recomputed here -- PROJECT_CORE.md §2.1 invariant 3). `results["_predictions"]`
    holds per-tier prediction rows for `write_predictions` -- pop it before
    treating `results` as pure tier/pass data (e.g. `overall_pass`).
    `baseline_real_csv`: path to baseline stage's `audit/predictions_baseline_real.csv`,
    used for the paired tier-4a comparison (module docstring) -- optional, skipped
    with a note if not given or the segment count doesn't match."""
    results = {}
    predictions = {}

    test_metrics = _eval_split(model, processor, test_ds, normalizer, cfg.eval, desc="gate:tier1_in_domain")
    predictions["tier1_in_domain"] = test_metrics.pop("_predictions")
    tier1_bound = (1 - cfg.gates.min_improvement_pct / 100) * baseline["cer_test"]
    results["tier1_in_domain"] = {
        "cer": test_metrics["cer"], "bound": tier1_bound,
        "pass": test_metrics["cer"] <= tier1_bound,
    }

    if cfg.normalization.audit_conversions:
        tier1_alt_convention = ("as_written" if cfg.normalization.number_convention == "word_to_digit"
                                 else "word_to_digit")
        tier1_alt_normalizer = Normalizer(
            strip_punctuation=cfg.normalization.strip_punctuation,
            lowercase=cfg.normalization.lowercase,
            number_convention=tier1_alt_convention,
            filler_tokens=cfg.normalization.filler_tokens,
        )
        tier1_alt_metrics = _eval_split(model, processor, test_ds, tier1_alt_normalizer, cfg.eval,
                                         desc="gate:tier1_in_domain_normcheck")
        tier1_alt_metrics.pop("_predictions")
        results["tier1_in_domain"]["normalization_check"] = {
            cfg.normalization.number_convention: test_metrics["cer"],
            tier1_alt_convention: tier1_alt_metrics["cer"],
            "delta_pp": round(abs(test_metrics["cer"] - tier1_alt_metrics["cer"]) * 100, 3),
        }

    ood_metrics = _eval_split(model, processor, ood_ds, normalizer, cfg.eval, desc="gate:tier2_ood")
    predictions["tier2_ood"] = ood_metrics.pop("_predictions")
    tier2_bound = baseline["cer_ood"] + cfg.sweep.ood_cer_budget
    results["tier2_ood"] = {
        "cer": ood_metrics["cer"], "bound": tier2_bound,
        "pass": ood_metrics["cer"] <= tier2_bound,
    }

    if real_ds is not None:
        real_metrics = _eval_split(model, processor, real_ds, normalizer, cfg.eval, desc="gate:tier4a_real")
        real_predictions = real_metrics.pop("_predictions")
        predictions["tier4a_real"] = real_predictions
        real_scored = score_real(real_predictions)
        lo, hi = bootstrap_ci(real_scored["_char_counts"])
        tier4a_bound = baseline["cer_real"] + cfg.gates.real_cer_regression_pp
        results["tier4a_real"] = {
            "cer": real_scored["cer"], "ci": [lo, hi], "bound": tier4a_bound,
            "pass": real_scored["cer"] <= tier4a_bound,
            "by_meeting": _score_by_meeting(real_scored["_rejoined"]),
        }

        if baseline_real_csv is not None and Path(baseline_real_csv).exists():
            import csv as _csv_module
            with open(baseline_real_csv, encoding="utf-8") as f:
                base_rows = list(_csv_module.DictReader(f))
            base_counts = score_real(base_rows)["_char_counts"]
            cand_counts = real_scored["_char_counts"]
            if len(base_counts) == len(cand_counts):
                d_lo, d_hi = bootstrap_delta_ci(base_counts, cand_counts)
                results["tier4a_real"]["delta_ci"] = [d_lo, d_hi]
                results["tier4a_real"]["verdict"] = verdict(d_lo, d_hi)
            else:
                results["tier4a_real"]["verdict"] = (
                    f"SKIPPED (baseline has {len(base_counts)} segments, "
                    f"candidate has {len(cand_counts)} -- not the same segments)"
                )

        if cfg.normalization.audit_conversions:
            alt_convention = ("as_written" if cfg.normalization.number_convention == "word_to_digit"
                               else "word_to_digit")
            alt_normalizer = Normalizer(
                strip_punctuation=cfg.normalization.strip_punctuation,
                lowercase=cfg.normalization.lowercase,
                number_convention=alt_convention,
                filler_tokens=cfg.normalization.filler_tokens,
            )
            alt_metrics = _eval_split(model, processor, real_ds, alt_normalizer, cfg.eval,
                                       desc="gate:tier4a_real_normcheck")
            alt_scored = score_real(alt_metrics.pop("_predictions"))
            results["tier4a_real"]["normalization_check"] = {
                cfg.normalization.number_convention: real_scored["cer"],
                alt_convention: alt_scored["cer"],
                "delta_pp": round(abs(real_scored["cer"] - alt_scored["cer"]) * 100, 3),
            }
    else:
        results["tier4a_real"] = {"pass": None, "note": "real_bench_path not configured"}

    results["overall_pass"] = all(
        t.get("pass") is not False for t in results.values() if isinstance(t, dict)
    )
    results["_predictions"] = predictions
    return results


def write_gate_results(results: dict, out_dir: str | Path) -> Path:
    import json

    out_dir = Path(out_dir)
    predictions = results.pop("_predictions", {})

    out = out_dir / "metrics" / "gate_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    for tier, rows in predictions.items():
        write_predictions(rows, out_dir / "audit" / f"predictions_{tier}.csv")
    return out


def write_predictions(rows: list[dict], out_path: str | Path) -> Path:
    import csv

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["segment_id", "meeting_id", "ref", "hyp"])
        writer.writeheader()
        writer.writerows(rows)
    return out_path
