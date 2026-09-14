#!/usr/bin/env bash
# Fresh LoRA on the full v6 corpus -- no init_adapter, so this starts from
# vinai/PhoWhisper-large and uses cfg.lora (rank 16 / alpha 32, the same shape v5
# trained at). It is not a rung of the density ladder and does not compare with
# Outputs/ladder-phaseA/ (those are rank 32 and all warm-started from v6-corpus-r32).
#
# What it answers: v6-corpus-r32 saw the 101 h exactly once at LR 2e-4 and its train
# loss floored at 0.20 by step 200 of 2186. This run gives the same data five passes
# at a quarter of the step size, so "the 1-epoch run was under-trained" stops being an
# open question either way.
#
# What it does NOT address: the corpus is still 2.48% foreign-token density against a
# 7.41% benchmark. That is the measured cause of v6's retention collapse
# (48.6% vs v5's 66.6%) and no hyperparameter here touches it. Read the
# cross_domain_bench block of gate_results.json before reading anything else.
set -u
export PYTHONPATH=. TRANSFORMERS_AUTO_CONVERSION=0

# Val is 365 segments / 0.79 h as the corpus ships -- ValCER swings several points
# between rounds at that size, and this run has 10 rounds to read a trend from.
# Run this first, then paste its data.val_meetings line into OV below.
#   PYTHONPATH=. python scripts/select_val_meetings.py --dataset dataset/v6-corpus
VAL_MEETINGS=""   # e.g. --override data.val_meetings=[id1,id2,...]

R=v6-full-e5
OV="--override run_id=$R
    --override data.dataset_path=dataset/v6-corpus
    --override data.real_bench_path=null
    --override data.cross_domain_path=dataset/cross-domain-bench
    $VAL_MEETINGS
    --override training.epochs=5
    --override training.learning_rate=5.0e-5
    --override training.batch_size=16 --override training.grad_accum_steps=1
    --override training.gradient_checkpointing=false
    --override training.eval_steps=1100
    --override eval.batch_size=64"

echo "=== $R :: baseline ==="
python -m src.pipeline --stage baseline   $OV || { echo "BASELINE FAILED"; exit 1; }
echo "=== $R :: train ==="
python -m src.pipeline --stage train      $OV || { echo "TRAIN FAILED"; exit 1; }
echo "=== $R :: sweep-gate ==="
python -m src.pipeline --stage sweep-gate $OV || { echo "SWEEP-GATE FAILED"; exit 1; }
echo "=== $R done ==="
