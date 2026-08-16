"""Tests for scripts/probe_youtube_captions.py. Offline: every function under
test takes an already-extracted info dict or an already-parsed word list, so the
three reject rules are exercised without touching the network.

The `name`-field shapes below are copied from real yt-dlp output measured
2026-08-11, not invented -- that is the whole basis of reject rule 2."""

from scripts.draft_sources import Word
from scripts.probe_youtube_captions import (
    baseline_bucket, build_report, consensus_cer, original_language_keys, read_urls,
    screen_captions, text_stats, track_names, vietnamese_keys, wpm_buckets, wpm_collapse,
)


def _tracks(name: str) -> list[dict]:
    return [{"ext": ext, "name": name, "url": f"https://example/{ext}"}
            for ext in ("json3", "vtt")]


VI_SPOKEN = {"automatic_captions": {"vi-orig": _tracks("Vietnamese (Original)"),
                                    "vi": _tracks("Vietnamese"),
                                    "ja": _tracks("Japanese")},
             "subtitles": {}}
EN_SPOKEN = {"automatic_captions": {"en-orig": _tracks("English (Original)"),
                                    "vi": _tracks("Vietnamese")},
             "subtitles": {}}


def test_original_language_keys_reads_the_name_suffix():
    auto = VI_SPOKEN["automatic_captions"]
    assert original_language_keys(auto) == ["vi-orig"]
    assert vietnamese_keys(auto) == ["vi", "vi-orig"]
    assert track_names(auto)["vi"] == "Vietnamese"


def test_rule_2_accepts_a_vietnamese_recognition_original():
    verdict = screen_captions(VI_SPOKEN)
    assert verdict["ok"] and verdict["track_key"] == "vi-orig"


def test_rule_2_rejects_a_translated_vietnamese_track():
    verdict = screen_captions(EN_SPOKEN)
    assert not verdict["ok"] and verdict["rule"] == 2
    assert "machine translation" in verdict["reason"]


def test_rule_2_rejects_when_no_track_is_marked_original():
    verdict = screen_captions({"automatic_captions": {"vi": _tracks("Vietnamese")}})
    assert not verdict["ok"] and verdict["rule"] == 2


def test_rule_1_rejects_a_video_without_any_vietnamese_track():
    verdict = screen_captions({"automatic_captions": {"en-orig": _tracks("English (Original)")}})
    assert not verdict["ok"] and verdict["rule"] == 1


def _words(per_bucket: list[int], bucket_sec: float = 300.0) -> list[Word]:
    out = []
    for i, n in enumerate(per_bucket):
        for j in range(n):
            t = i * bucket_sec + j * (bucket_sec / max(n, 1))
            out.append(Word("từ", t, t + 0.2))
    return out


def test_wpm_buckets_measure_each_bucket_separately():
    buckets = wpm_buckets(_words([500, 250]), duration=600.0)
    assert [b["words"] for b in buckets] == [500, 250]
    assert buckets[0]["wpm"] == 100.0 and buckets[1]["wpm"] == 50.0


def test_wpm_buckets_drop_a_too_short_trailing_bucket():
    # 620 s = two full buckets plus a 20 s tail, too short to judge density on
    buckets = wpm_buckets(_words([500, 500, 3]), duration=620.0)
    assert len(buckets) == 2


def test_rule_3_flags_a_collapse_and_passes_a_flat_transcript():
    # the 169 -> 32 wpm collapse viet-speech recorded on an owner-typed track
    assert wpm_collapse(wpm_buckets(_words([845, 160]), duration=600.0))
    # speech recognition does not summarise: flat within tolerance
    assert not wpm_collapse(wpm_buckets(_words([845, 800]), duration=600.0))


def test_rule_3_baseline_skips_leading_empty_buckets():
    """rCd8DSMk3-c opens with 5 empty minutes (a waiting screen). Comparing the
    last bucket against a 0-wpm first bucket would disable rule 3."""
    buckets = wpm_buckets(_words([0, 845, 160]), duration=900.0)
    assert baseline_bucket(buckets)["wpm"] > 0
    assert wpm_collapse(buckets)


def test_text_stats_measures_review_effort_signals():
    words = [Word(t, i * 1.0, i * 1.0 + 0.5) for i, t in enumerate(
        ["dạ", "vâng", "em", "dùng", "Kubernetes", "tầm", "95%", "ạ.", "[Âm", "nhạc]"])]
    stats = text_stats(words, duration=60.0)
    assert stats["n_words"] == 10
    assert stats["lexical_particle_rate"] == 0.3          # dạ, vâng, ạ
    assert stats["digit_word_rate"] == 0.1                # 95%
    assert "kubernetes" in stats["english_examples"]
    assert stats["english_words_per_min"] == 1.0
    assert stats["bracket_labels"] == {"[Âm nhạc]": 1}


def test_text_stats_counts_multi_token_entries():
    stats = text_stats([Word("chào anh", 0.0, 1.0), Word("ạ", 1.0, 1.2)], duration=60.0)
    assert stats["n_words"] == 3 and stats["multi_token_entries"] == 1


def test_consensus_cer_is_zero_on_identical_transcripts():
    assert consensus_cer("chào anh ạ", "chào anh ạ") == 0.0
    assert consensus_cer("chào anh ạ", "chào em ạ") > 0.0


def test_report_survives_a_rate_limited_draft_fetch():
    """HTTP 429 on the caption endpoint loses the measurements, not the rules-1-2
    verdict -- metadata extraction is not what gets rate limited."""
    record = {
        "url": "https://youtu.be/x", "video_id": "x", "title": "họp thử",
        "duration": 1200.0, "auto_keys": ["vi", "vi-orig"],
        "auto_names": {"vi": "Vietnamese", "vi-orig": "Vietnamese (Original)"},
        "sub_keys": [], "sub_names": {}, "original_keys": ["vi-orig"],
        "screen": {"ok": True, "rule": None, "track_key": "vi-orig", "reason": "accepted"},
        "draft_error": "RuntimeError: HTTP 429 from YouTube's caption endpoint",
    }
    report = build_report([record], failures=[])
    assert "rule 3 unchecked" in report
    assert "HTTP 429" in report
    assert "vi-orig = 'Vietnamese (Original)'" in report


def test_read_urls_ignores_title_and_blank_lines(tmp_path):
    path = tmp_path / "urls.txt"
    path.write_text("\nTư vấn 1-1 với bạn Khương:\n"
                    "https://www.youtube.com/watch?v=dGT3YW0AdD8\n\n"
                    "Webinar: Con đường\nhttps://www.youtube.com/watch?v=rCd8DSMk3-c\n",
                    encoding="utf-8")
    assert read_urls(path) == ["https://www.youtube.com/watch?v=dGT3YW0AdD8",
                              "https://www.youtube.com/watch?v=rCd8DSMk3-c"]
