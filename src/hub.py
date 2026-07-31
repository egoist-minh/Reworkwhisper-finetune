"""Push the selected adapter to HF Hub. PROJECT_CORE.md §2.1 Stage 5.

Push is a pure function of gate_results.json's status -- pipeline.py only
calls this when `results["overall_pass"]` is true and `cfg.hub.push` is set.
No credential is ever written into an artifact: the HF token is read from
the environment (HF_TOKEN), never from config or written into the model card.
"""

from pathlib import Path


def push_adapter(adapter_dir: str | Path, repo_id: str, private: bool = True,
                  gate_results: dict | None = None) -> str:
    """Uploads `adapter_dir` (already contains adapter_config.json + weights,
    written by src.lora.save_with_lambda) to `repo_id`. Returns the repo URL."""
    from huggingface_hub import HfApi

    adapter_dir = Path(adapter_dir)
    api = HfApi()
    api.create_repo(repo_id, private=private, exist_ok=True)

    if gate_results is not None:
        (adapter_dir / "README.md").write_text(
            _model_card(repo_id, gate_results), encoding="utf-8")

    api.upload_folder(folder_path=str(adapter_dir), repo_id=repo_id)
    return f"https://huggingface.co/{repo_id}"


def _model_card(repo_id: str, gate_results: dict) -> str:
    lines = [f"# {repo_id}", "", "## Gate results", "", "| Tier | CER | Bound | Pass |",
             "|---|---|---|---|"]
    for tier, row in gate_results.items():
        if tier == "overall_pass" or not isinstance(row, dict):
            continue
        cer = row.get("cer")
        bound = row.get("bound")
        cer_s = f"{cer:.4f}" if cer is not None else "-"
        bound_s = f"{bound:.4f}" if bound is not None else "-"
        lines.append(f"| {tier} | {cer_s} | {bound_s} | {row.get('pass')} |")
    lines.append("")
    lines.append(f"Overall: **{'PASS' if gate_results.get('overall_pass') else 'FAIL'}**")
    return "\n".join(lines)
