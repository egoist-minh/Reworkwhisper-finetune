"""Colab/Kaggle GPU server for ASR (module 4).

Run this on a GPU notebook, then tunnel it (ngrok/cloudflared) and register
the public URL against `phowhisper-large-remote` (or `whisper-large-remote`)
via POST /admin/register-backend on your local speech-api instance.

Usage (in a Colab/Kaggle cell):
    !pip install -e ".[asr-torch]"
    !python scripts/remote_asr_server.py
    # in another cell: expose port 8002 with ngrok/cloudflared and copy the URL

Env vars:
    ASR_MODEL_SIZE   PhoWhisper size (small/medium/large) or any HF repo id,
                      e.g. winhsss/Reworkwhisper-large-v4 (default: large)
    ASR_MODEL_ID     label attached to responses, should match the remote
                      model id registered via /admin/register-backend
                      (default: phowhisper-large-remote)
    HF_TOKEN         only needed for gated/private checkpoints

Re-running the cell is safe: an earlier instance of this script still holding
port 8002 is terminated before the model loads (see _free_port).
"""

from __future__ import annotations

import base64
import logging
import os
import signal
import socket
import time
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from backend.adapters.asr.phowhisper import PhoWhisperASR
from backend.core.types import Device

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PORT = 8002


def _port_free(port: int) -> bool:
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind(("0.0.0.0", port))
        except OSError:
            return False
    return True


def _free_port(port: int, timeout: float = 20.0) -> None:
    """Stop an earlier run of this same script that is still holding `port`.

    Re-running the notebook cell leaves the previous process bound, and uvicorn
    only discovers that AFTER this module's eager model load -- so a restart used
    to burn minutes of weight loading and then die on
    "[Errno 98] address already in use". Called before that load for this reason.

    Only processes whose /proc cmdline names this script are signalled, so
    nothing else on the box is touched. No-ops off Linux (no /proc).
    """
    if _port_free(port):
        return
    if not Path("/proc").is_dir():
        logger.warning("Port %d busy and no /proc to find the owner; bind will fail.", port)
        return

    me = os.getpid()
    name = Path(__file__).name
    stale = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == me:
            continue
        try:
            cmdline = (entry / "cmdline").read_bytes().decode("utf-8", "ignore")
        except OSError:
            continue
        if name in cmdline:
            stale.append(int(entry.name))
    if not stale:
        logger.warning("Port %d busy but no earlier %s found; bind will fail.", port, name)
        return

    logger.info("Port %d held by %s; terminating pid(s) %s.", port, name, stale)
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for pid in stale:
            try:
                os.kill(pid, sig)
            except OSError:
                pass
        deadline = time.monotonic() + timeout / 2
        while time.monotonic() < deadline:
            if _port_free(port):
                logger.info("Port %d released.", port)
                return
            time.sleep(0.5)
    logger.warning("Port %d still held after SIGKILL; bind will fail.", port)


# Called before the model load below, not after -- see _free_port's docstring.
_free_port(PORT)

app = FastAPI()
asr = PhoWhisperASR(
    os.getenv("ASR_MODEL_ID", "phowhisper-large-remote"),
    device=Device.CUDA,
    model_size=os.getenv("ASR_MODEL_SIZE", "large"),
)
# Load eagerly at process start, not on the first request — a cold model load
# (weights download + peft adapter init) can take minutes and blow past the
# client's request timeout if it happens inside the first /transcribe call.
asr.load()


class TranscribeRequest(BaseModel):
    audio_b64: str
    sample_rate: int
    language: str | None = None
    # Per-request decoding kwargs. The model is loaded once at process start, so
    # this is what lets a caller sweep decoding settings (or a `-remote` config
    # entry set them at all) without restarting the notebook.
    gen_params: dict | None = None


class TranscribeBatchRequest(BaseModel):
    clips_b64: list[str]
    sample_rate: int
    language: str | None = None
    gen_params: dict | None = None


@app.post("/transcribe")
def transcribe(req: TranscribeRequest):
    audio = base64.b64decode(req.audio_b64)
    result = asr.transcribe(
        audio, req.sample_rate, req.language, gen_overrides=req.gen_params
    )
    return result.to_dict()


@app.post("/transcribe_batch")
def transcribe_batch(req: TranscribeBatchRequest):
    """Several clips in one batched generate() -- module 4's per-speech-window
    calls, sent together. Whisper's encoder always runs a 30s window, so short
    clips one at a time leave the GPU mostly idle; the client batches them
    (backend/core/meeting.py:transcribe_per_splice) and this returns one result
    per clip, in the order they were sent."""
    clips = [base64.b64decode(c) for c in req.clips_b64]
    results = asr.transcribe_batch(
        clips, req.sample_rate, req.language, gen_overrides=req.gen_params
    )
    return {"results": [r.to_dict() for r in results]}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
