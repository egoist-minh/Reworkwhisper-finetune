"""Tests for src/metrics.py, including bootstrap_delta_ci/verdict -- written
earlier but never exercised by a test until now (2026-08-02, wired into
gate.py's tier 4a paired comparison)."""

from src.metrics import Counts, char_counts, score, bootstrap_ci, bootstrap_delta_ci, verdict


def test_char_counts_identical_is_zero_edits():
    c = char_counts("hello", "hello")
    assert c.edits == 0
    assert c.ref_len == 5


def test_score_corpus_level_not_mean_of_segment_cers():
    # A 3-char segment with 100% error should not outweigh a 300-char segment
    # with 0% error when averaged naively -- corpus-level CER must be
    # sum(edits)/sum(ref_len), not mean(per-segment CER).
    refs = ["abc", "x" * 300]
    hyps = ["zzz", "x" * 300]
    result = score(refs, hyps)
    assert result["cer"] == 3 / 303


def test_bootstrap_ci_contains_point_estimate_range():
    counts = [Counts(edits=1, ref_len=10) for _ in range(50)]
    lo, hi = bootstrap_ci(counts)
    assert 0 <= lo <= hi <= 1


def test_bootstrap_delta_ci_positive_when_candidate_better():
    # base has more edits per ref char than candidate -> candidate is better ->
    # (base_cer - cand_cer) > 0 across (almost) every resample.
    base = [Counts(edits=5, ref_len=10) for _ in range(30)]
    cand = [Counts(edits=1, ref_len=10) for _ in range(30)]
    lo, hi = bootstrap_delta_ci(base, cand)
    assert lo > 0
    assert verdict(lo, hi) == "IMPROVED"


def test_bootstrap_delta_ci_negative_when_candidate_worse():
    base = [Counts(edits=1, ref_len=10) for _ in range(30)]
    cand = [Counts(edits=5, ref_len=10) for _ in range(30)]
    lo, hi = bootstrap_delta_ci(base, cand)
    assert hi < 0
    assert verdict(lo, hi) == "REGRESSED"


def test_bootstrap_delta_ci_rejects_mismatched_lengths():
    import pytest
    with pytest.raises(ValueError):
        bootstrap_delta_ci([Counts(1, 10)], [Counts(1, 10), Counts(1, 10)])


def test_verdict_inconclusive_when_ci_straddles_zero():
    assert verdict(-0.5, 0.5) == "INCONCLUSIVE"


def test_verdict_boundary_is_inconclusive_not_a_pass():
    # lo == 0 or hi == 0 does not strictly satisfy "lo > 0" / "hi < 0" --
    # exactly-zero-touching is treated as inconclusive, not a pass.
    assert verdict(0.0, 0.5) == "INCONCLUSIVE"
    assert verdict(-0.5, 0.0) == "INCONCLUSIVE"
