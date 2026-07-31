"""Scoring normalization. PROJECT_CORE.md §6 Stage 4.

The only normalization function in the repo. Applied IDENTICALLY to hypothesis and
reference -- that symmetry is what makes the ambiguous cases safe: `một` is both "one"
and the indefinite article, `năm` is both "five" and "year", but if both sides convert
the same way the edit distance is unaffected. Never apply this to training targets.
"""

import re
import unicodedata
from dataclasses import dataclass, field

# Punctuation to drop. Keeps intra-word hyphen/apostrophe out of scope on purpose --
# Vietnamese does not use them and English tokens are kept verbatim.
_PUNCT = re.compile(r"[.,!?;:\"'`()\[\]{}<>/\\|~^*_=+&%$#@…“”‘’–—]")
_WS = re.compile(r"\s+")

_DIGITS = {
    "không": 0, "một": 1, "hai": 2, "ba": 3, "bốn": 4, "năm": 5,
    "sáu": 6, "bảy": 7, "tám": 8, "chín": 9,
    "tư": 4,    # positional variant: hai mươi tư = 24
    "lăm": 5,   # positional variant: mười lăm = 15
    "bẩy": 7,   # northern variant
}
_SCALES = {"nghìn": 1000, "ngàn": 1000, "triệu": 10**6, "tỷ": 10**9, "tỉ": 10**9}
_ZERO_FILLER = {"linh", "lẻ"}
_MULTIPLIERS = set(_SCALES) | _ZERO_FILLER | {"mười", "mươi", "trăm"}
_NUMWORDS = set(_DIGITS) | _MULTIPLIERS


def _parse_run(tokens: list[str]) -> str:
    """Vietnamese number words -> digits.

    A run with no multiplier is read as a DIGIT STRING, the way phone numbers and codes
    are spoken: "không chín tám bảy" -> "0987", not 0+9+8+7. Otherwise arithmetic, where
    `mười` is standalone ten, `mươi` a tens multiplier, and `linh`/`lẻ` a zero
    placeholder (một trăm linh năm = 105).
    """
    if len(tokens) > 1 and not any(t in _MULTIPLIERS for t in tokens):
        return "".join(str(_DIGITS[t]) for t in tokens)
    return str(_arith(tokens))


def _arith(tokens: list[str]) -> int:
    total = 0     # groups already multiplied by a scale (nghìn and up)
    cur = 0       # current group, < 1000
    pending = 0   # bare digit(s) awaiting a multiplier
    prev_digit = False
    for t in tokens:
        if t == "mười":
            cur += 10
            pending = 0
        elif t == "mươi":
            cur += pending * 10
            pending = 0
        elif t == "trăm":
            cur += pending * 100
            pending = 0
        elif t in _SCALES:
            total += (cur + pending) * _SCALES[t]
            cur = pending = 0
        elif t in _ZERO_FILLER:
            continue
        else:
            # Consecutive bare digits accumulate as a written number, not as a
            # replacement: "hai ba nghìn" is 23 nghìn, not 3 nghìn.
            pending = pending * 10 + _DIGITS[t] if prev_digit else _DIGITS[t]
        prev_digit = t not in _MULTIPLIERS
    return total + cur + pending


def words_to_digits(text: str) -> tuple[str, int]:
    """Collapse maximal runs of number words into digits. Returns (text, n_conversions).

    A lone `không` is left alone -- it is overwhelmingly the negation particle, and a
    spoken zero only ever appears inside a longer run (phone numbers, codes).
    """
    toks = text.split()
    out, n, i = [], 0, 0
    while i < len(toks):
        if toks[i] in _NUMWORDS:
            j = i
            while j < len(toks) and toks[j] in _NUMWORDS:
                j += 1
            run = toks[i:j]
            # strip trailing filler-only words so "năm linh" does not eat the linh
            while run and run[-1] in _ZERO_FILLER:
                j -= 1
                run = run[:-1]
            skip = len(run) == 1 and run[0] in _ZERO_FILLER | {"không"}
            if run and not skip:
                out.append(_parse_run(run))
                n += 1
            else:
                out.extend(run)
            i = j
        else:
            out.append(toks[i])
            i += 1
    return " ".join(out), n


@dataclass
class NormStats:
    """Audit counters. `conversions` gating on 0 catches a silently broken parser."""
    conversions: int = 0
    fillers_removed: int = 0


@dataclass
class Normalizer:
    strip_punctuation: bool = True
    lowercase: bool = True
    number_convention: str = "word_to_digit"   # or "as_written"
    filler_tokens: list[str] = field(default_factory=list)
    stats: NormStats = field(default_factory=NormStats)

    def __post_init__(self):
        if self.number_convention not in ("word_to_digit", "as_written"):
            raise ValueError(f"unknown number_convention: {self.number_convention}")
        self._fillers = {unicodedata.normalize("NFC", f.lower()) for f in self.filler_tokens}

    def __call__(self, text: str) -> str:
        t = unicodedata.normalize("NFC", text)
        if self.lowercase:
            t = t.lower()
        if self.strip_punctuation:
            t = _PUNCT.sub(" ", t)
        t = _WS.sub(" ", t).strip()
        if self._fillers:
            toks = [x for x in t.split() if x not in self._fillers]
            self.stats.fillers_removed += len(t.split()) - len(toks)
            t = " ".join(toks)
        if self.number_convention == "word_to_digit":
            t, n = words_to_digits(t)
            self.stats.conversions += n
        return t
