# v3-r16 at λ=0.5 — production reference predictions

What `winhsss/Reworkwhisper-large-v4` actually serves, scored on the **654
segments of v4-mixed-r16's mixed test split** (228 youtube + 426 synthetic), so
it joins 1:1 with a candidate's `audit/predictions_tier1_in_domain.csv`. Pass it
to `scripts/merge_and_push.py --production-predictions`.

Produced by `scripts/eval_v4_mixed_at_lambda.py` on 2026-08-18: v3-r16's
weights, v4-mixed-r16's manifest.

`Outputs/v3-r16/audit/predictions_tier1_in_domain.csv` is **not** usable for
this. It was scored at λ=1.0 (that run gated under the old "largest λ within
budget" rule; 0.5 was a later human pick), over v3-r16's own 426-synthetic
split, which has no youtube half at all — the slice where a mixed-data candidate
gains the most.

`summary.json` carries the per-source numbers: youtube CER 0.1205 (retention
0.7746), synthetic CER 0.0196, aggregate 0.0805. The youtube figure reproduces
the earlier H6 measurement of this same model at this same λ on these same 228
segments.

Keep this file. Recomputing it costs a GPU session.
