#!/usr/bin/env bash
# Measure what a training step costs on this box, so a corpus of any size can be
# projected from it. Step cost does not depend on corpus size -- the model and
# batch are fixed and Whisper pads every clip to a 30 s window -- so only the
# step count changes with the corpus. See docs/runbook.md B5b.
#
# Two training probes (worst-case batch, then a random sample) plus an I/O probe,
# under ~20 min of GPU on top of one baseline pass.
#
#   RUN_ID=v6-probe DATASET=dataset/mixed-noisy-v1 bash scripts/probe_timing.sh
#
# Env: PY (python to use), RUN_ID, DATASET, RANK, BATCH, STEPS, BASE_MODEL.
# Optional: HF_REPO + HF_PROBE_FILE to also measure how fast the corpus would
# land on this box, and CORPUS_GB to project that onto a full transfer.
set -euo pipefail

PY="${PY:-python}"
RUN_ID="${RUN_ID:-v6-probe}"
DATASET="${DATASET:-dataset/mixed-noisy-v1}"
RANK="${RANK:-32}"
BATCH="${BATCH:-16}"
STEPS="${STEPS:-40}"
BASE_MODEL="${BASE_MODEL:-vinai/PhoWhisper-large}"
OUT="outputs/$RUN_ID"
HF_REPO="${HF_REPO:-}"
HF_PROBE_FILE="${HF_PROBE_FILE:-}"
CORPUS_GB="${CORPUS_GB:-14.4}"
CONFIG="${CONFIG:-configs/experiment.yaml}"

export PYTHONPATH="${PYTHONPATH:-.}"
export TRANSFORMERS_AUTO_CONVERSION=0

# real_bench_path=null, never =None: apply_override runs the value through
# yaml.safe_load, which reads None as the *string* 'None' -- truthy, so the
# pipeline would go looking for a directory called None.
OV=(--override "run_id=$RUN_ID"
    --override "base_model=$BASE_MODEL"
    --override "data.dataset_path=$DATASET"
    --override "lora.rank=$RANK" --override "lora.alpha=$((RANK * 2))"
    --override "training.batch_size=$BATCH" --override training.grad_accum_steps=1
    --override training.gradient_checkpointing=false
    --override training.val_limit=32
    --override training.epochs=1
    --override data.real_bench_path=null)

echo "== gpu =="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
echo "cpus: $(nproc)"

echo
echo "== baseline (eval.limit=40, just to write validated_manifest.jsonl) =="
"$PY" -m src.pipeline --config "$CONFIG" --stage baseline "${OV[@]}" --override eval.limit=40

for MODE in worst random; do
    echo
    echo "== probe: $MODE =="
    # Both probes select from the untouched backup, so running random after
    # worst still samples the full split.
    if [ -f "$OUT/validated_manifest.full.jsonl" ]; then
        cp "$OUT/validated_manifest.full.jsonl" "$OUT/validated_manifest.jsonl"
    fi
    "$PY" scripts/make_probe_manifest.py --run-dir "$OUT" --mode "$MODE" \
          --steps "$STEPS" --batch "$BATCH" --force
    rm -rf "$OUT/checkpoints"
    "$PY" -m src.pipeline --config "$CONFIG" --stage train "${OV[@]}"
    cp "$OUT/metrics/timing.json" "$OUT/metrics/timing.$MODE.json"
done

# Leave the run with its real manifest, or a later stage silently trains on the
# few hundred segments the probe left behind.
cp "$OUT/validated_manifest.full.jsonl" "$OUT/validated_manifest.jsonl"

SPS=$("$PY" -c "import json;print(json.load(open('$OUT/metrics/timing.random.json'))['steady_state']['cycle_seconds']['median'])")

echo
echo "== probe: io =="
sync && sysctl -w vm.drop_caches=3 2>/dev/null || echo "(no root: page cache not dropped, read numbers will be optimistic)"
"$PY" scripts/probe_io.py --dataset "$DATASET" --files 300 --target-hours 100 \
      --workers 8 --batch "$BATCH" --seconds-per-step "$SPS" \
      --feature-extractor "$BASE_MODEL"

echo
echo "== transfer: disk write =="
# 14 GB has to be written somewhere. A slow disk, not the network, is then the
# thing that decides how long getting the corpus onto this box takes.
dd if=/dev/zero of=/tmp/_writeprobe.bin bs=1M count=2000 oflag=direct 2>&1 | tail -1 ||   dd if=/dev/zero of=/tmp/_writeprobe.bin bs=1M count=2000 conv=fdatasync 2>&1 | tail -1
rm -f /tmp/_writeprobe.bin

if [ -n "$HF_REPO" ] && [ -n "$HF_PROBE_FILE" ]; then
    echo
    echo "== transfer: hub download =="
    # Measured with the python API rather than the CLI, whose name changed
    # between huggingface_hub versions.
    HF_REPO="$HF_REPO" HF_PROBE_FILE="$HF_PROBE_FILE" CORPUS_GB="$CORPUS_GB" "$PY" - <<'PYEOF'
import os
import time
from pathlib import Path

from huggingface_hub import hf_hub_download

repo, name = os.environ["HF_REPO"], os.environ["HF_PROBE_FILE"]
corpus_gb = float(os.environ["CORPUS_GB"])
print(f"hf_transfer: {os.environ.get('HF_HUB_ENABLE_HF_TRANSFER', 'unset')}")
t0 = time.perf_counter()
path = hf_hub_download(repo_id=repo, filename=name, repo_type="dataset",
                       local_dir="/tmp/_hfprobe")
elapsed = time.perf_counter() - t0
mb = Path(path).stat().st_size / 1024**2
print(f"{mb:.0f} MB in {elapsed:.1f} s  ->  {mb / elapsed:.1f} MB/s")
print(f"{corpus_gb:g} GB corpus would land in "
      f"{corpus_gb * 1024 / (mb / elapsed) / 60:.1f} min at that rate")
PYEOF
else
    echo
    echo "== transfer: hub download skipped (set HF_REPO and HF_PROBE_FILE) =="
fi

echo
echo "== summary =="
"$PY" - "$OUT" <<'PYEOF'
import json
import sys

out = sys.argv[1]
for mode in ("worst", "random"):
    t = json.load(open(f"{out}/metrics/timing.{mode}.json"))
    s = t["steady_state"]
    print(f"\n[{mode}]  {t['gpu']}")
    print(f"  cycle   median {s['cycle_seconds']['median']:.3f} s/step   "
          f"p95 {s['cycle_seconds']['p95']:.3f}   ({s['steps_measured']} steps measured)")
    print(f"  compute median {s['compute_seconds']['median']:.3f} s   "
          f"dataloader gap median {s['dataloader_gap_seconds']['median']:.3f} s")
    print(f"  peak VRAM {t['peak_vram_reserved_gb']:.1f} GB reserved")
    print(f"  one-off: model load {t['model_load_seconds']:.0f} s, "
          f"train overhead {t['train_overhead_seconds']:.0f} s, dead {t['dead_seconds']:.0f} s")
    print(f"  eval {t['eval_seconds']:.0f} s for this probe's capped val set")
    # 10 steps go to warmup. Too few left and the median is noise: on the local
    # run that first exercised this, 5 measured steps put the worst-case ceiling
    # *below* the typical step, which cannot happen by construction.
    if s["steps_measured"] < 20:
        total = s["steps_measured"] + s["warmup_steps_excluded"]
        print(f"  WARNING: only {s['steps_measured']} steps measured out of {total}. "
              f"Raise STEPS to at least 40 and re-run; this median is noise.")

worst = json.load(open(f"{out}/metrics/timing.worst.json"))["steady_state"]["cycle_seconds"]["median"]
typical = json.load(open(f"{out}/metrics/timing.random.json"))["steady_state"]["cycle_seconds"]["median"]
print(f"\nceiling is {worst / typical:.2f}x the typical step")
if worst < typical:
    print("  WARNING: a ceiling below the typical step means the probes were "
          "too short to separate. Raise STEPS and re-run before using either "
          "number.")
print("\nhours per epoch = steps_per_epoch * cycle_seconds / 3600")
for steps in (1000, 2000, 2730, 4000):
    print(f"  {steps:>5} steps:  typical {steps * typical / 3600:5.2f} h   "
          f"ceiling {steps * worst / 3600:5.2f} h")
PYEOF
