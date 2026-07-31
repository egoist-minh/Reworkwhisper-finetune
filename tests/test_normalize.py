"""Unit tests for src/normalize.py, PROJECT_CORE.md §6 Stage 4 contract."""

import pytest

from src.normalize import words_to_digits, Normalizer


@pytest.mark.parametrize("text,expect", [
    # Digit-string mode: no multiplier in the run -> read as a code, not summed.
    ("không chín tám bảy", "0987"),
    # Arithmetic mode.
    ("một trăm hai mươi ba", "123"),
    ("mười lăm", "15"),
    ("năm trăm", "500"),
    ("một trăm linh năm", "105"),
    ("hai mươi tư", "24"),
    ("hai ba nghìn", "23000"),  # consecutive bare digits accumulate: 23 * 1000
])
def test_words_to_digits_arith(text, expect):
    got, n = words_to_digits(text)
    assert got == expect
    assert n == 1


def test_lone_khong_left_alone():
    """A standalone 'không' is negation, not a spoken zero."""
    got, n = words_to_digits("tôi không đi")
    assert got == "tôi không đi"
    assert n == 0


def test_symmetric_normalizer_number_convention():
    norm = Normalizer(filler_tokens=["ừm", "ờm", "ehm", "uhm", "hmm"])
    hyp = norm("Ừm, mười lăm phút nữa.")
    ref = norm("mười lăm phút nữa.")
    assert hyp == ref
