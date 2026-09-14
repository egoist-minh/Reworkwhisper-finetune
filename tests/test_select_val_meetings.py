from scripts.select_val_meetings import select_additional_val_meetings


def _meeting(meeting_id, source, hours, voice_id=None, split="demo", n_segments=4):
    """`n_segments` records whose durations sum to `hours`, sharing `voice_id`."""
    per = hours * 3600 / n_segments
    return [
        {"meeting_id": meeting_id, "segment_id": f"seg_{i}", "split": split,
         "source": source, "duration": per, "voice_id": voice_id}
        for i in range(n_segments)
    ]


def test_picks_meetings_to_hit_target_hours_by_source_ratio():
    records = (
        _meeting("val_syn", "synthetic", 0.3, voice_id="v_val_syn")
        + _meeting("val_yt", "youtube", 0.5, voice_id="v_val_yt")
        + _meeting("tr_syn_a", "synthetic", 1.0, voice_id="v_a")
        + _meeting("tr_syn_b", "synthetic", 1.0, voice_id="v_b")
        + _meeting("tr_yt_a", "youtube", 1.0, voice_id="v_c")
        + _meeting("tr_yt_b", "youtube", 1.0, voice_id="v_d")
    )
    val_meetings = ["val_syn", "val_yt"]
    added, report = select_additional_val_meetings(records, val_meetings, target_hours=2.0)

    assert set(added) <= {"tr_syn_a", "tr_syn_b", "tr_yt_a", "tr_yt_b"}
    assert report["shortfall_hours"]["synthetic"] == 0
    assert report["shortfall_hours"]["youtube"] == 0
    # train pool here is an even 50/50 synthetic/youtube ratio -- picks should be too
    assert report["added_hours"]["synthetic"] > 0
    assert report["added_hours"]["youtube"] > 0


def test_skips_candidate_whose_voice_id_overlaps_remaining_train():
    records = (
        _meeting("val_syn", "synthetic", 0.1, voice_id="v_val")
        # tr_a and tr_b share a voice_id -- picking one for val would still
        # leave that voice in train, so neither should be picked.
        + _meeting("tr_a", "synthetic", 1.0, voice_id="shared")
        + _meeting("tr_b", "synthetic", 1.0, voice_id="shared")
        # tr_c is voice-exclusive and should be picked instead.
        + _meeting("tr_c", "synthetic", 1.0, voice_id="v_c")
    )
    added, report = select_additional_val_meetings(records, ["val_syn"], target_hours=1.0)

    assert "tr_a" not in added
    assert "tr_b" not in added
    assert "tr_c" in added
    assert "tr_a" in report["skipped_voice_overlap"]
    assert "tr_b" in report["skipped_voice_overlap"]


def test_reports_shortfall_instead_of_silently_underdelivering():
    records = (
        _meeting("val_syn", "synthetic", 0.1, voice_id="v_val")
        + _meeting("tr_a", "synthetic", 0.5, voice_id="v_a")
    )
    added, report = select_additional_val_meetings(records, ["val_syn"], target_hours=5.0)

    assert added == ["tr_a"]
    assert report["shortfall_hours"]["synthetic"] > 0


def test_mixed_source_within_one_meeting_raises():
    records = [
        {"meeting_id": "m1", "segment_id": "s0", "split": "demo",
         "source": "synthetic", "duration": 1.0, "voice_id": "v1"},
        {"meeting_id": "m1", "segment_id": "s1", "split": "demo",
         "source": "youtube", "duration": 1.0, "voice_id": "v1"},
    ]
    try:
        select_additional_val_meetings(records, [], target_hours=1.0)
        assert False, "expected ValueError"
    except ValueError:
        pass
