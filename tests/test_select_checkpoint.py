from scripts.select_checkpoint import pick_best


def test_lowest_cer_wins_even_at_worse_retention():
    rows = [{"step": 200, "lambda": 1.0, "retention": 0.70, "cer": 0.09},
            {"step": 400, "lambda": 1.0, "retention": 0.62, "cer": 0.07}]
    assert pick_best(rows)["step"] == 400


def test_ties_break_on_retention_then_on_earlier_step():
    rows = [{"step": 800, "lambda": 1.0, "retention": 0.60, "cer": 0.07},
            {"step": 600, "lambda": 1.0, "retention": 0.66, "cer": 0.07},
            {"step": 400, "lambda": 1.0, "retention": 0.66, "cer": 0.07}]
    assert pick_best(rows)["step"] == 400
