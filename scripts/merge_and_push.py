"""Merge a run's selected LoRA adapter into the base weights and push the
result to HF Hub as a standalone model (no `peft` needed to load it).

Same recipe as `notebooks/reeval_lambda.ipynb`'s publish cell -- load base +
adapter, `merge_and_unload()`, save model + processor, push -- with two
differences, both deliberate:

  * The merge runs in **fp32 on CPU**, not the fp16/bf16 `src.asr.load_for_eval`
    uses. Eval dtype is an eval speed decision; a published checkpoint is
    written once and loaded many times, so the delta is added at full precision
    and stored at full precision. Callers who want fp16 pass
    `torch_dtype=torch.float16` at load time.
  * Every claim this script depends on is checked before the push, not assumed:
    the gate must have passed, the adapter must belong to this run's base model
    and rank, its baked lambda must be a real row in `lambda_sweep.csv` whose
    OOD CER matches the gate's tier-2 number, and the merged model must produce
    the same logits as the unmerged PeftModel. Any mismatch raises.

The HF token is read from the environment (HF_TOKEN), never from config, never
written into the model card.

    python -m scripts.merge_and_push \
        --run-dir outputs/v4-mixed-r16 \
        --adapter winhsss/Reworkwhisper-large-v5 \
        --repo-id winhsss/Reworkwhisper-large-v5 \
        --out /kaggle/working/merged-v4-mixed-r16 \
        --delete-remote-adapter --confirm
"""

import argparse
import csv
import json
import os
import subprocess
from pathlib import Path

ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors")


def _load_run(run_dir: Path) -> tuple[dict, dict, list[dict]]:
    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    gate = json.loads((run_dir / "metrics" / "gate_results.json").read_text(encoding="utf-8"))
    with open(run_dir / "metrics" / "lambda_sweep.csv", encoding="utf-8") as f:
        sweep = [{k: float(v) for k, v in row.items()} for row in csv.DictReader(f)]
    return cfg, gate, sweep


def check_provenance(cfg: dict, gate: dict, sweep: list[dict], adapter: str) -> float:
    """Everything that must be true before this adapter is allowed near the Hub.
    Returns the lambda baked into the adapter."""
    from peft import PeftConfig

    if gate.get("overall_pass") is not True:
        raise RuntimeError(
            f"gate_results.json overall_pass is {gate.get('overall_pass')!r}, not True -- "
            "this adapter was never cleared for publication.")

    acfg = PeftConfig.from_pretrained(adapter)
    if acfg.base_model_name_or_path != cfg["base_model"]:
        raise RuntimeError(
            f"adapter was trained on {acfg.base_model_name_or_path!r} but this run's "
            f"base_model is {cfg['base_model']!r} -- merging them is meaningless.")
    if acfg.r != cfg["lora"]["rank"]:
        raise RuntimeError(f"adapter rank {acfg.r} != config lora.rank {cfg['lora']['rank']}")

    # save_with_lambda ships lambda as `lora_alpha <- lambda * alpha` (src/lora.py),
    # so lambda is recoverable from the adapter alone. Cross-check it against the
    # sweep: the row must exist, and its OOD CER must be the number the gate scored.
    lam = acfg.lora_alpha / cfg["lora"]["alpha"]
    row = next((r for r in sweep if abs(r["lambda"] - lam) < 1e-9), None)
    if row is None:
        raise RuntimeError(
            f"adapter's lora_alpha {acfg.lora_alpha} implies lambda={lam}, which is not a "
            f"row in lambda_sweep.csv ({sorted(r['lambda'] for r in sweep)})")
    tier2 = gate["tier2_ood"]["cer"]
    if abs(row["ood_cer"] - tier2) > 1e-9:
        raise RuntimeError(
            f"lambda={lam} sweep row has ood_cer {row['ood_cer']}, but gate tier2_ood scored "
            f"{tier2} -- the adapter on disk is not the one the gate measured.")
    print(f"provenance ok: base={cfg['base_model']} rank={acfg.r} lambda={lam} "
          f"tier2_ood_cer={tier2:.6f}")
    return lam


def merge(base_model: str, adapter: str, tol: float):
    """fp32 CPU merge, verified against the unmerged PeftModel's logits."""
    import torch
    from peft import PeftModel
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    model = WhisperForConditionalGeneration.from_pretrained(
        base_model, torch_dtype=torch.float32, use_safetensors=False)
    processor = WhisperProcessor.from_pretrained(base_model)
    peft_model = PeftModel.from_pretrained(model, adapter).eval()

    from peft.tuners.lora import LoraLayer
    n_layers = sum(isinstance(m, LoraLayer) for m in peft_model.modules())
    if n_layers == 0:
        raise RuntimeError(f"no LoRA layers injected from {adapter} -- nothing to merge")

    dummy = torch.zeros(1, model.config.num_mel_bins, 3000, dtype=torch.float32)
    decoder_ids = torch.tensor([[model.config.decoder_start_token_id]])
    with torch.no_grad():
        before = peft_model(input_features=dummy, decoder_input_ids=decoder_ids).logits

    merged = peft_model.merge_and_unload().eval()
    with torch.no_grad():
        after = merged(input_features=dummy, decoder_input_ids=decoder_ids).logits

    max_diff = (before - after).abs().max().item()
    print(f"merged {n_layers} LoRA layers; max logit diff vs unmerged {max_diff:.2e}")
    if max_diff > tol:
        raise RuntimeError(f"merge changed behavior: max logit diff {max_diff:.2e} > tol {tol:.0e}")
    return merged, processor


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def model_card(repo_id: str, cfg: dict, gate: dict, lam: float) -> str:
    rows = "\n".join(
        f"| {tier} | {row['cer']:.4f} | {row['bound']:.4f} | {row['pass']} |"
        for tier, row in gate.items() if isinstance(row, dict) and "cer" in row)
    return f"""# {repo_id}

`{cfg['base_model']}` with the run `{cfg['run_id']}` LoRA adapter **merged into the
base weights**. Load it as a plain Whisper checkpoint -- no `peft`, no adapter:

```python
from transformers import WhisperForConditionalGeneration, WhisperProcessor
model = WhisperForConditionalGeneration.from_pretrained("{repo_id}")
processor = WhisperProcessor.from_pretrained("{repo_id}")
```

Weights are stored fp32 (the merge was computed fp32); pass
`torch_dtype=torch.float16` at load time for fp16 inference.

- LoRA rank {cfg['lora']['rank']}, alpha {cfg['lora']['alpha']}, targets `{', '.join(cfg['lora']['target_modules'])}`
- Shipped at lambda = {lam} (adapter delta scaled by {lam} before merging)
- Pipeline commit `{_git_commit()}`, run `{cfg['run_id']}`

## Gate results (lambda = {lam})

| Tier | CER | Bound | Pass |
|---|---|---|---|
{rows}

Overall: **{'PASS' if gate.get('overall_pass') else 'FAIL'}**

Tier 1 is in-domain meeting speech (synthetic TTS + reviewed YouTube audio),
tier 2 is VIVOS held out as an out-of-domain regression guard, tier 4a is real
recorded meetings. CER is the headline metric for Vietnamese ASR, not WER.
Absolute CER on the synthetic slice does not transfer to real speech -- compare
against the same tier's baseline, not across tiers.
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True, help="run artifacts dir (config.json + metrics/)")
    ap.add_argument("--adapter", required=True, help="adapter dir or HF repo id holding it")
    ap.add_argument("--repo-id", required=True, help="HF repo to push the merged model to")
    ap.add_argument("--out", required=True, help="local dir for the merged model")
    ap.add_argument("--public", action="store_true", help="push public (default: private)")
    ap.add_argument("--tol", type=float, default=1e-3, help="max logit diff allowed by the merge check")
    ap.add_argument("--delete-remote-adapter", action="store_true",
                    help=f"after a successful upload, delete {' + '.join(ADAPTER_FILES)} from the repo")
    ap.add_argument("--confirm", action="store_true", help="required: without it, nothing is pushed")
    args = ap.parse_args()

    run_dir, out = Path(args.run_dir), Path(args.out)
    # Checked up front, not just before the upload: reading `--adapter` from a
    # private repo needs the same token, and that read happens minutes earlier.
    if args.confirm and not os.environ.get("HF_TOKEN"):
        raise RuntimeError("HF_TOKEN is not set in the environment -- cannot push.")
    cfg, gate, sweep = _load_run(run_dir)

    from src import compat
    compat.apply()

    lam = check_provenance(cfg, gate, sweep, args.adapter)
    merged, processor = merge(cfg["base_model"], args.adapter, args.tol)

    out.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(out)
    processor.save_pretrained(out)
    missing = [f for f in ("config.json", "generation_config.json", "preprocessor_config.json",
                           "tokenizer_config.json") if not (out / f).exists()]
    if missing:
        raise RuntimeError(f"merged model saved without {missing} -- it will not load standalone")
    (out / "README.md").write_text(model_card(args.repo_id, cfg, gate, lam), encoding="utf-8")
    print(f"merged model written to {out}")

    if not args.confirm:
        print("--confirm not set: stopping before the push. Nothing was uploaded.")
        return

    from src.hub import push_adapter

    # gate_results=None: push_adapter would overwrite README.md with the bare gate
    # table, and this repo ships a full model card written above.
    url = push_adapter(out, args.repo_id, private=not args.public, gate_results=None)
    print(f"pushed merged model to {url}")

    if args.delete_remote_adapter:
        from huggingface_hub import HfApi

        # Ask what is in the repo instead of catching a not-found exception: which
        # error class `delete_file` raises for a missing path has moved between
        # huggingface_hub versions, and requirements.txt pins none of them.
        api = HfApi()
        present = set(api.list_repo_files(args.repo_id))
        for name in ADAPTER_FILES:
            if name not in present:
                print(f"{name} not present in {args.repo_id}, nothing to delete")
                continue
            api.delete_file(path_in_repo=name, repo_id=args.repo_id,
                            commit_message=f"remove {name}: repo now ships the merged model")
            print(f"deleted {name} from {args.repo_id}")


if __name__ == "__main__":
    main()
