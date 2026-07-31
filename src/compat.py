"""Environment fixes that must run before any peft/transformers import path is used.

Every entry here exists because it was observed failing on a real Kaggle image,
not because it might fail in theory. Feature-detect, never branch on version strings
(Kaggle patches images in place, so version numbers lie).
"""

import sys


def neutralize_broken_peft_probes() -> list[str]:
    """peft probes optional quantization backends via is_*_available() during LoRA
    injection. On Kaggle, torchao 0.10 < the 0.16 peft wants, and the probe RAISES
    instead of returning False -- which aborts injection for every target layer.

    Replaces any probe that raises with a False stub. Must also patch modules that did
    `from peft.import_utils import is_x_available`, since rebinding the origin module
    alone leaves those copies intact.

    Returns the names it had to neutralize (empty list = environment is clean).
    """
    import peft.import_utils as piu

    broken = []
    for name in [n for n in dir(piu) if n.startswith("is_") and n.endswith("_available")]:
        fn = getattr(piu, name)
        if not callable(fn):
            continue
        try:
            fn()
        except TypeError:
            continue  # takes arguments; not a zero-arg probe
        except Exception:
            broken.append(name)

    def stub():
        return False

    for name in broken:
        setattr(piu, name, stub)
        for mod in list(sys.modules.values()):
            if mod is None:
                continue
            if getattr(mod, "__name__", "").startswith("peft") and hasattr(mod, name):
                setattr(mod, name, stub)
    return broken


def version_table() -> dict[str, str]:
    """Versions of everything that can break us. Dumped on any unhandled exception so
    the cause of a failed GPU run is visible without a second run."""
    out = {"python": sys.version.split()[0]}
    for name in ("torch", "transformers", "peft", "numpy", "scipy", "soundfile",
                 "yaml", "accelerate", "huggingface_hub", "pyarrow"):
        try:
            out[name] = __import__(name).__version__
        except Exception as e:
            out[name] = f"<{type(e).__name__}>"
    try:
        import torch
        out["cuda"] = torch.version.cuda or "cpu"
        out["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
    except Exception:
        pass
    return out


def apply() -> dict:
    """Single entry point. Call once at pipeline start, before building any model."""
    return {"neutralized": neutralize_broken_peft_probes(), "versions": version_table()}
