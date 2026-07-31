"""CER / WER and confidence intervals. No jiwer -- one less version to fight.

CER is the headline metric for Vietnamese ASR. Always CORPUS-level
(sum of edits / sum of reference lengths), never the mean of per-segment CERs: the
latter lets a 3-character segment outweigh a 300-character one.
"""

import random
from dataclasses import dataclass


def levenshtein(a, b) -> int:
    """Edit distance over any two sequences. Two rows of memory, O(len(a)*len(b)) time."""
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1,          # deletion
                           cur[j - 1] + 1,       # insertion
                           prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


@dataclass
class Counts:
    """Per-segment edit counts, kept unaggregated so bootstrap can resample them."""
    edits: int
    ref_len: int


def char_counts(ref: str, hyp: str) -> Counts:
    return Counts(levenshtein(ref, hyp), len(ref))


def word_counts(ref: str, hyp: str) -> Counts:
    r = ref.split()
    return Counts(levenshtein(r, hyp.split()), len(r))


def rate(counts: list[Counts]) -> float:
    """Corpus-level error rate. Empty reference corpus is a caller bug, not a 0.0."""
    denom = sum(c.ref_len for c in counts)
    if denom == 0:
        raise ValueError("reference corpus has zero length -- nothing to score")
    return sum(c.edits for c in counts) / denom


def score(refs: list[str], hyps: list[str]) -> dict:
    """CER and WER over aligned reference/hypothesis lists, plus the raw per-segment
    counts a gate needs for its interval."""
    if len(refs) != len(hyps):
        raise ValueError(f"length mismatch: {len(refs)} refs vs {len(hyps)} hyps")
    cc = [char_counts(r, h) for r, h in zip(refs, hyps)]
    wc = [word_counts(r, h) for r, h in zip(refs, hyps)]
    return {
        "n_segments": len(refs),
        "cer": rate(cc),
        "wer": rate(wc),
        "char_edits": sum(c.edits for c in cc),
        "char_ref_len": sum(c.ref_len for c in cc),
        "_char_counts": cc,
    }


def bootstrap_ci(counts: list[Counts], n_resamples: int = 1000, seed: int = 42,
                 alpha: float = 0.05) -> tuple[float, float]:
    """Segment-level bootstrap CI for one corpus rate."""
    rng = random.Random(seed)
    k = len(counts)
    vals = []
    for _ in range(n_resamples):
        pick = [counts[rng.randrange(k)] for _ in range(k)]
        denom = sum(c.ref_len for c in pick)
        if denom:
            vals.append(sum(c.edits for c in pick) / denom)
    vals.sort()
    lo = vals[int(alpha / 2 * len(vals))]
    hi = vals[min(len(vals) - 1, int((1 - alpha / 2) * len(vals)))]
    return lo, hi


def bootstrap_delta_ci(base: list[Counts], cand: list[Counts], n_resamples: int = 1000,
                       seed: int = 42, alpha: float = 0.05) -> tuple[float, float]:
    """PAIRED bootstrap CI for (base_cer - cand_cer): positive means cand is better.

    Both lists must be the same segments in the same order -- resampling them
    independently would discard the pairing and inflate the interval.
    """
    if len(base) != len(cand):
        raise ValueError("paired bootstrap needs the same segments on both sides")
    rng = random.Random(seed)
    k = len(base)
    vals = []
    for _ in range(n_resamples):
        idx = [rng.randrange(k) for _ in range(k)]
        bd = sum(base[i].ref_len for i in idx)
        cd = sum(cand[i].ref_len for i in idx)
        if bd and cd:
            vals.append(sum(base[i].edits for i in idx) / bd
                        - sum(cand[i].edits for i in idx) / cd)
    vals.sort()
    lo = vals[int(alpha / 2 * len(vals))]
    hi = vals[min(len(vals) - 1, int((1 - alpha / 2) * len(vals)))]
    return lo, hi


def verdict(lo: float, hi: float) -> str:
    """A CI straddling zero is INCONCLUSIVE, not a pass. With 196 real-bench segments
    the interval is ~±1.5pp, so smaller differences genuinely cannot be resolved."""
    if lo > 0:
        return "IMPROVED"
    if hi < 0:
        return "REGRESSED"
    return "INCONCLUSIVE"
