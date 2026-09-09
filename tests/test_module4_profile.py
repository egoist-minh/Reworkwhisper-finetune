"""Guard the vendored module-4 decode path against silent drift.

`scripts/eval_module4.py`'s numbers are only "what production does" for as long
as `vendor/viet_speech/` still is what production runs. Two ways that stops being
true, and one test each:

  * `viet-speech` changes a decode parameter and nobody re-copies. Caught by
    hashing every vendored file against its `d:\\viet-speech` counterpart --
    skipped when that repo is not on this machine, which is the normal case on a
    GPU box.
  * Someone edits the copy here to "fix" something. Caught by the same hash test
    where the source repo exists, and by the explicit default/kwargs assertions
    below everywhere else -- those need no source repo, only the profile written
    down in `vendor/viet_speech/VENDORED.md`.

Torch-free: importing PhoWhisperASR pulls in numpy and stdlib only, and
`_gen_kwargs` is a pure dict build, so this runs on any machine.

Set VIET_SPEECH_REPO to point the hash test at a checkout elsewhere.
"""

import hashlib
import inspect
import os
import sys
from pathlib import Path

import pytest

VENDOR = Path(__file__).resolve().parents[1] / "vendor" / "viet_speech"
SOURCE = Path(os.environ.get("VIET_SPEECH_REPO", r"D:\viet-speech"))

# Path here -> path in the source repo. Only `remote_asr_server.py` differs: it
# sits at the vendor root so running it puts `backend/` on sys.path[0] without a
# launcher (vendor/viet_speech/VENDORED.md, "Files"). Contents are identical.
VENDORED_FILES = {
    "backend/__init__.py": "backend/__init__.py",
    "backend/core/__init__.py": "backend/core/__init__.py",
    "backend/adapters/__init__.py": "backend/adapters/__init__.py",
    "backend/adapters/asr/__init__.py": "backend/adapters/asr/__init__.py",
    "backend/core/types.py": "backend/core/types.py",
    "backend/core/audio.py": "backend/core/audio.py",
    "backend/core/interfaces.py": "backend/core/interfaces.py",
    "backend/adapters/asr/phowhisper.py": "backend/adapters/asr/phowhisper.py",
    "remote_asr_server.py": "scripts/remote_asr_server.py",
}

# vendor/viet_speech/VENDORED.md, "The decode profile these files carry".
PROFILE_DEFAULTS = {
    "no_repeat_ngram_size": 0,
    "condition_on_prev_tokens": True,
    "no_speech_threshold": 0.6,
    "logprob_threshold": -1.0,
    "compression_ratio_threshold": 2.4,
    "temperature": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
}

PROFILE_GEN_KWARGS = {
    "task": "transcribe",
    "language": "vi",
    "return_timestamps": True,
    "condition_on_prev_tokens": True,
    "no_speech_threshold": 0.6,
    "logprob_threshold": -1.0,
    "compression_ratio_threshold": 2.4,
    "temperature": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
}


def _phowhisper_class():
    if str(VENDOR) not in sys.path:
        sys.path.insert(0, str(VENDOR))
    from backend.adapters.asr.phowhisper import PhoWhisperASR

    return PhoWhisperASR


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.skipif(not SOURCE.is_dir(),
                    reason=f"{SOURCE} not on this machine -- nothing to compare against")
@pytest.mark.parametrize("here,there", sorted(VENDORED_FILES.items()))
def test_vendored_file_matches_the_source_repo(here, there):
    """Compares the source repo's WORKING TREE, not a commit: the GPU box serves
    whatever was in the uploaded zip, and at copy time that zip was byte-identical
    to the working tree (VENDORED.md, "Source")."""
    src = SOURCE / there
    if not src.exists():
        pytest.fail(f"{src} no longer exists -- module 4 moved, and vendor/viet_speech/"
                    f"{here} is now a copy of something that is gone")
    assert _sha256(VENDOR / here) == _sha256(src), (
        f"vendor/viet_speech/{here} differs from {src}. Either module 4 changed and the "
        "copy is stale, or the copy was edited here. Re-copy from the source and update "
        "the hash table in vendor/viet_speech/VENDORED.md -- never edit the copy.")


def test_constructor_defaults_are_the_documented_profile():
    """Runs with no source repo: this is what protects the GPU box, where only the
    vendored copy exists and a hash comparison has nothing to compare to."""
    params = inspect.signature(_phowhisper_class().__init__).parameters
    for name, expected in PROFILE_DEFAULTS.items():
        assert name in params, f"PhoWhisperASR.__init__ no longer takes {name}"
        assert params[name].default == expected, (
            f"PhoWhisperASR.__init__ default for {name} is {params[name].default!r}, "
            f"not the {expected!r} recorded in vendor/viet_speech/VENDORED.md -- every "
            "module-4 CER measured under the old value is now describing a decode path "
            "that no longer runs")


def test_gen_kwargs_with_no_overrides_is_the_production_profile():
    """The defaults above only matter through this dict, which is what reaches
    `generate()`. `gen_params: null` from scripts/eval_module4.py arrives here as
    `gen_overrides=None`, so this is exactly the call production makes."""
    cls = _phowhisper_class()
    asr = cls.__new__(cls)  # no __init__: constructing one would need transformers
    for name, default in PROFILE_DEFAULTS.items():
        setattr(asr, name, default)

    assert asr._gen_kwargs("vi", None) == PROFILE_GEN_KWARGS
    # no_repeat_ngram_size=0 must reach generate() as an ABSENT key, not a zero --
    # a zero is not a valid n-gram ban and transformers does not treat it as "off".
    assert "no_repeat_ngram_size" not in asr._gen_kwargs("vi", None)


def test_gen_kwargs_honours_the_models_yaml_override():
    """`config/models.yaml`'s `reworkwhisper-large-v5-remote` entry pins
    no_repeat_ngram_size/condition_on_prev_tokens per request, to override a GPU box
    still running an older zip. Those two values equal today's defaults, so the
    override must be a no-op -- if this ever fails, the yaml and the adapter have
    diverged and production's real profile is whichever the deployed server uses."""
    cls = _phowhisper_class()
    asr = cls.__new__(cls)
    for name, default in PROFILE_DEFAULTS.items():
        setattr(asr, name, default)

    override = {"no_repeat_ngram_size": 0, "condition_on_prev_tokens": True}
    assert asr._gen_kwargs("vi", override) == asr._gen_kwargs("vi", None)


def test_vendored_hashes_match_the_table_in_vendored_md():
    """VENDORED.md's hash table is the record a reader checks the copy against, so
    it has to stay true even when the source repo is absent."""
    doc = (VENDOR / "VENDORED.md").read_text(encoding="utf-8")
    for here in VENDORED_FILES:
        digest = _sha256(VENDOR / here)
        assert f"| `{here}` | `{digest}` |" in doc, (
            f"VENDORED.md has no row matching {here} at sha256 {digest} -- update the "
            "table (and the commit it names) whenever the copies are refreshed")
