import glob, os, shutil
from pathlib import Path

SRC = Path("dataset/v6-corpus")
ADD = Path("dataset/v6-corpus-addon")
MIX = Path("dataset/mixed-noisy-v1")
OUT = Path("dataset/v6-phase2")
(OUT / "audio").mkdir(parents=True, exist_ok=True)

def ids(root):
    return sorted(Path(f).name[len("manifest."):-len(".jsonl")]
                  for f in glob.glob(str(root / "manifest.*.jsonl")))

wanted = [(m, SRC) for m in ids(MIX)] + [(m, ADD) for m in ids(ADD)]

copied, missing = 0, []
for m, root in wanted:
    man = root / f"manifest.{m}.jsonl"
    if not man.exists():
        missing.append(m)
        continue
    shutil.copy(man, OUT / man.name)
    link = OUT / "audio" / m
    if not link.exists():
        os.symlink((root / "audio" / m).resolve(), link, target_is_directory=True)
    copied += 1
print("copied", copied, "| missing", len(missing), missing)

from src.config import load
from src.data import load_manifests, resolve_splits, split_stats
cfg = load("configs/experiment.yaml",
           overrides=["data.dataset_path=dataset/v6-phase2"])
print(split_stats(resolve_splits(load_manifests(cfg.data.dataset_path),
                                 cfg.data.val_meetings)))
