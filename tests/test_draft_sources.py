"""Tests for scripts/draft_sources.py. No network, no model, no GPU.

The json3 fixtures mirror the shape measured on real videos 2026-08-11 (see the
module docstring there): first seg of an event carries no `tOffsetMs`, rolling
"\\n"-only events exist, and `dDurationMs` is a display duration that can run
past the next event's start."""

import pytest

from scripts.draft_sources import Word, load, parse_json3, parse_scribe, transcript

JSON3 = {
    "wireMagic": "pb3",
    "events": [
        {"tStartMs": 900, "dDurationMs": 100, "aAppend": 1, "segs": [{"utf8": "\n"}]},
        {"tStartMs": 1000, "dDurationMs": 2000, "segs": [
            {"utf8": "chào", "acAsrConf": 0},
            {"utf8": " anh", "tOffsetMs": 500, "acAsrConf": 0},
        ]},
        {"tStartMs": 4000, "dDurationMs": 1000, "segs": [
            {"utf8": "ạ", "acAsrConf": 0},
            {"utf8": " vâng", "tOffsetMs": 200, "acAsrConf": 0},
        ]},
    ],
}

SCRIBE = {
    "language_code": "vi",
    "words": [
        {"text": "chào", "start": 1.0, "end": 1.4, "type": "word", "speaker_id": "speaker_0"},
        {"text": " ", "start": 1.4, "end": 1.5, "type": "spacing"},
        {"text": "anh", "start": 1.5, "end": 2.0, "type": "word", "speaker_id": "speaker_0"},
    ],
}


def test_parse_json3_words_and_inferred_ends():
    words = parse_json3(JSON3)
    assert [w.text for w in words] == ["chào", "anh", "ạ", "vâng"]
    # first seg of an event has no tOffsetMs -> implicitly the event's own start
    assert words[0].start == 1.0
    assert words[1].start == 1.5
    # end = next word's start ...
    assert words[0].end == 1.5
    # ... clipped by the event's display end, which is what makes a pause visible
    assert words[1].end == 3.0
    # last word of the draft has no next start, so it takes the display end
    assert words[3].end == 5.0
    assert all(w.speaker is None for w in words)


def test_parse_json3_drops_blank_rolling_events():
    assert len(parse_json3(JSON3)) == 4          # the "\n" event contributes nothing


def test_parse_json3_raises_without_word_timing():
    doc = {"events": [{"tStartMs": 0, "dDurationMs": 5000, "segs": [{"utf8": "chào anh ạ"}]}]}
    with pytest.raises(ValueError, match="no tOffsetMs"):
        parse_json3(doc)


def test_parse_json3_raises_on_empty_document():
    with pytest.raises(ValueError, match="no events"):
        parse_json3({"events": []})
    with pytest.raises(ValueError, match="no non-blank"):
        parse_json3({"events": [{"tStartMs": 0, "segs": [{"utf8": " \n "}]}]})


def test_parse_scribe_keeps_words_drops_spacing():
    words = parse_scribe(SCRIBE)
    assert [w.text for w in words] == ["chào", "anh"]
    assert words[0].start == 1.0 and words[0].end == 1.4
    assert words[0].speaker == "speaker_0"


def test_parse_scribe_raises_without_word_timing():
    doc = {"words": [{"text": "chào", "type": "word"}]}
    with pytest.raises(ValueError, match="missing start/end"):
        parse_scribe(doc)
    with pytest.raises(ValueError, match="no `words` list"):
        parse_scribe({"text": "chào anh"})


def test_both_parsers_return_the_same_shape():
    """What makes --draft-source a flag: downstream sees only list[Word]."""
    for words in (parse_json3(JSON3), parse_scribe(SCRIBE)):
        assert all(isinstance(w, Word) and w.end >= w.start for w in words)
    assert transcript(parse_scribe(SCRIBE)) == "chào anh"


def test_load_dispatches_and_rejects_unknown_source(tmp_path):
    import json
    path = tmp_path / "captions.json3"
    path.write_text(json.dumps(JSON3), encoding="utf-8")
    assert transcript(load(path, "json3")) == "chào anh ạ vâng"
    with pytest.raises(ValueError, match="unknown draft source"):
        load(path, "whisper")
