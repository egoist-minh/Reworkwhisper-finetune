"""Tests for src/asr.py's decode-path kwargs. No torch, no transformers, no GPU --
only the feature test that decides whether the anti-loop kwargs are passed."""

from src.asr import COMPRESSION_RATIO_THRESHOLD, FALLBACK_TEMPERATURES, fallback_kwargs


class _Modern:
    def generate(self, input_features=None, language=None, task=None, num_beams=1,
                 temperature=None, compression_ratio_threshold=None, **kwargs):
        ...


class _Old:
    """A build predating temperature fallback: it takes **kwargs but would raise
    from _validate_model_kwargs on anything it does not consume."""

    def generate(self, input_features=None, language=None, task=None, num_beams=1, **kwargs):
        ...


def test_passes_the_anti_loop_kwargs_when_generate_names_them():
    assert fallback_kwargs(_Modern(), FALLBACK_TEMPERATURES, COMPRESSION_RATIO_THRESHOLD) == {
        "temperature": FALLBACK_TEMPERATURES,
        "compression_ratio_threshold": COMPRESSION_RATIO_THRESHOLD,
    }


def test_passes_nothing_when_generate_only_has_kwargs_to_swallow_them():
    assert fallback_kwargs(_Old(), FALLBACK_TEMPERATURES, COMPRESSION_RATIO_THRESHOLD) == {}


def test_temperature_ladder_starts_greedy_so_a_clean_decode_is_unchanged():
    # Only a hypothesis that trips the compression ratio is ever re-decoded at a
    # sampling temperature; everything else stays the greedy decode every past
    # CER in this repo was measured with.
    assert FALLBACK_TEMPERATURES[0] == 0.0
    assert list(FALLBACK_TEMPERATURES) == sorted(FALLBACK_TEMPERATURES)
