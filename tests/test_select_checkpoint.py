from scripts.select_checkpoint import pick_best


def test_highest_retention_wins_even_at_worse_cer():
    # The whole point of the plan this serves: val CER and in-domain CER cannot see
    # retention, so a candidate is not allowed to win on CER alone.
    rows = [{"step": 200, "lambda": 1.0, "retention": 0.70, "cer": 0.09},
            {"step": 400, "lambda": 1.0, "retention": 0.62, "cer": 0.07}]
    assert pick_best(rows)["step"] == 200


def test_ties_break_on_cer_then_on_earlier_step():
    rows = [{"step": 800, "lambda": 1.0, "retention": 0.66, "cer": 0.08},
            {"step": 600, "lambda": 1.0, "retention": 0.66, "cer": 0.07},
            {"step": 400, "lambda": 1.0, "retention": 0.66, "cer": 0.07}]
    assert pick_best(rows)["step"] == 400
