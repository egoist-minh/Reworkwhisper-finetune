# Vendored module 4 (ASR) from `d:\viet-speech`

Byte-identical copies of the decode path production serves. **Do not edit any file
listed below.** They exist here so the GPU box can run production's own decoder
without cloning `viet-speech` (that repo is private and cannot be pulled from
Kaggle/Colab). Every fix belongs upstream in `viet-speech`, then gets re-copied here.

## Why copies and not a re-implementation

`scripts/eval_module4.py` measures how far this repo's eval gate
(`src/asr.py:transcribe_batch`) sits from the decode path that actually ships. That
measurement is only worth anything if the shipping path is the real one: a
hand-written copy of PhoWhisperASR's six `generate()` kwargs in this repo would be a
second definition free to drift from the first, and the drift would be invisible in
exactly the number the measurement exists to produce.

`tests/test_module4_profile.py` enforces this: it hashes each file below against its
`d:\viet-speech` counterpart when that repo is present, and asserts the constructor
defaults independently when it is not.

## Source

| | |
|---|---|
| Repo | `d:\viet-speech` |
| Commit | `9795707a3df44017056fd0b375845f11fdf4d4b0` |
| Copied | 2026-08-19 |

Two files were copied from the **working tree**, not from that commit:
`backend/adapters/asr/phowhisper.py` and `remote_asr_server.py` had uncommitted
changes at copy time. The working tree is the right side to copy, and this is
checkable rather than assumed: both files are byte-identical to their entries in
`d:\viet-speech\viet-speech-remote.zip`, the archive uploaded to the GPU box that
serves module 4 (`config/models.yaml` notes the server "runs whatever code was in
the uploaded zip"). The uncommitted delta versus the commit adds `transcribe_batch`
and refactors `transcribe` into `_gen_kwargs`/`_features`/`_to_transcript`/`_samples`;
it changes no decode parameter, and single-clip `transcribe` behaves identically
under both.

## Files

Paths are as laid out here. Only one differs from upstream: upstream's
`scripts/remote_asr_server.py` sits at the root here, so `python
vendor/viet_speech/remote_asr_server.py` puts `backend/` on `sys.path[0]` with no
launcher and no `sys.path` edit inside a copied file. Contents are unchanged.

| Path here | sha256 |
|---|---|
| `backend/__init__.py` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `backend/core/__init__.py` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `backend/adapters/__init__.py` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `backend/adapters/asr/__init__.py` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `backend/core/types.py` | `164caabf9d89f50b32081b88b4d7facddda7dc481a18cab94b0a9aca6211f20f` |
| `backend/core/audio.py` | `7d60b0120ce96d13160d566be69a350d1b4eb73eb906778ff8e0eeb0fbf8b637` |
| `backend/core/interfaces.py` | `8f746d76d35ae63c19866ed8711723cd37490f0ae573b4ef2162aef5014852f5` |
| `backend/adapters/asr/phowhisper.py` | `d9f07e5b2cf29b240fe862d17080928dd1889bd8190857834637ba1142c7337b` |
| `remote_asr_server.py` | `4fb11fc0e6e579b83f526e530d55066b53fe4944354785403ca56467c5349f34` |

`backend/core/audio.py` and `backend/core/interfaces.py` are here only because
`phowhisper.py` imports `wav_to_pcm16` and the `ASREngine` base class from them.
They were copied whole rather than trimmed to those two symbols so the hash check
above stays a plain file comparison.

## The decode profile these files carry

`PhoWhisperASR.__init__` defaults, which `_gen_kwargs` turns into the `generate()`
call for every production transcription:

```python
no_repeat_ngram_size = 0            # dropped from gen_kwargs entirely when 0
condition_on_prev_tokens = True
no_speech_threshold = 0.6
logprob_threshold = -1.0
compression_ratio_threshold = 2.4
temperature = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
```

plus `task="transcribe"`, `language=<lang>`, `return_timestamps=True`. Weights load
`torch_dtype=torch.float32`; features use `truncation=False`, `padding="longest"`,
`return_attention_mask=True`.

`config/models.yaml` in `viet-speech` sets `gen_params: {no_repeat_ngram_size: 0,
condition_on_prev_tokens: true}` on the `reworkwhisper-large-v5-remote` entry. Those
two values equal the defaults above, so that override is a no-op today — it is stated
there to protect against a GPU box still running an older zip, per its own comment.
`scripts/eval_module4.py` sends `gen_params: null` and therefore gets the defaults.

## Running the server

On the GPU box, with `fastapi`, `uvicorn`, `transformers`, `torch` and `HF_TOKEN`
present:

```bash
# ASR_MODEL_ID is a free-text label the server echoes back as `model_id`. Setting it
# to the checkpoint's repo id is what lets eval_module4.py's --expect-model-id prove
# which weights are loaded.
ASR_MODEL_SIZE=winhsss/Reworkwhisper-large-v5 \
ASR_MODEL_ID=winhsss/Reworkwhisper-large-v5 \
python vendor/viet_speech/remote_asr_server.py   # POST /transcribe on :8002
```

Nothing in this repo outside `scripts/eval_module4.py` and
`tests/test_module4_profile.py` reads these files, and neither `src/` nor the
pipeline imports them.
