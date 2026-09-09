from __future__ import annotations

import io
import wave
from collections.abc import AsyncIterator

import numpy as np


def pcm16_to_wav(pcm: bytes, sample_rate: int, channels: int = 1) -> bytes:
    """Wrap raw 16-bit PCM in a WAV container."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm)
    return buf.getvalue()


def wav_to_pcm16(wav_bytes: bytes) -> tuple[bytes, int]:
    """Extract raw PCM and sample rate from a WAV buffer."""
    with wave.open(io.BytesIO(wav_bytes), "rb") as w:
        return w.readframes(w.getnframes()), w.getframerate()


TARGET_SAMPLE_RATE = 16000


def to_wav_16k_mono(wav_bytes: bytes) -> bytes:
    """Normalize any WAV to 16 kHz mono 16-bit PCM.

    ASR and diarization both assume 16 kHz mono, but browser/phone uploads are
    often 44.1/48 kHz stereo — feeding those in raw makes MFCC frames overrun
    the FFT size and degrades speaker embeddings. Uses stdlib `audioop` only, so
    it stays on the torch-free default path.
    """
    import audioop

    with wave.open(io.BytesIO(wav_bytes), "rb") as w:
        channels = w.getnchannels()
        width = w.getsampwidth()
        rate = w.getframerate()
        pcm = w.readframes(w.getnframes())

    if channels == 1 and width == 2 and rate == TARGET_SAMPLE_RATE:
        return wav_bytes

    if width != 2:
        pcm = audioop.lin2lin(pcm, width, 2)
        width = 2
    if channels == 2:
        pcm = audioop.tomono(pcm, width, 0.5, 0.5)
    if rate != TARGET_SAMPLE_RATE:
        pcm, _ = audioop.ratecv(pcm, width, 1, rate, TARGET_SAMPLE_RATE, None)

    return pcm16_to_wav(pcm, TARGET_SAMPLE_RATE, channels=1)


def decode_upload(data: bytes) -> bytes:
    """Any uploaded audio container -> 16 kHz mono 16-bit PCM WAV.

    WAV takes the stdlib path (to_wav_16k_mono), so the common case needs no
    new dependency. Anything else -- m4a/aac, mp3, ogg/opus, flac -- goes
    through PyAV's bundled FFmpeg, fully in memory (no system ffmpeg, no temp
    file). Raises ValueError with a user-facing message on failure.
    """
    if data[:4] == b"RIFF" and data[8:12] == b"WAVE":
        try:
            return to_wav_16k_mono(data)
        except wave.Error:
            pass  # non-PCM RIFF (float32/GSM/RF64) -- fall through to PyAV

    return _decode_with_av(data)


def _decode_with_av(data: bytes) -> bytes:
    try:
        import av
    except ModuleNotFoundError as e:
        raise ValueError(
            "Uploaded file is not WAV. Install the audio extra to accept "
            "mp3/m4a/ogg/flac: pip install -e '.[audio]'"
        ) from e

    try:
        pcm = bytearray()
        with av.open(io.BytesIO(data)) as container:
            stream = container.streams.audio[0]
            resampler = av.audio.resampler.AudioResampler(
                format="s16", layout="mono", rate=TARGET_SAMPLE_RATE
            )
            for frame in container.decode(stream):
                for out in resampler.resample(frame):
                    pcm.extend(out.to_ndarray().tobytes())
            for out in resampler.resample(None):  # flush trailing samples
                pcm.extend(out.to_ndarray().tobytes())
    except (av.FFmpegError, IndexError) as e:
        raise ValueError(
            "Could not decode the uploaded audio file (unsupported format or "
            "corrupt data)."
        ) from e

    return pcm16_to_wav(bytes(pcm), TARGET_SAMPLE_RATE)


# Crude on purpose: the pipeline strips silence before embedding, so what
# matters here is "how many seconds of signal did we actually capture", not
# frame-accurate VAD.
VOICED_THRESHOLD = 1e-3


def audio_stats(wav_bytes: bytes) -> dict:
    """Duration, speech seconds, peak and clipping for one 16-bit PCM WAV take.

    These four numbers decide whether a take is usable, and the only moment
    they are cheap to act on is right after it's recorded.
    """
    try:
        with wave.open(io.BytesIO(wav_bytes), "rb") as w:
            rate = w.getframerate()
            channels = w.getnchannels()
            width = w.getsampwidth()
            frames = w.readframes(w.getnframes())
    except Exception as exc:  # noqa: BLE001 — any broken-upload error becomes one ValueError
        raise ValueError(f"không đọc được file WAV: {exc}") from exc

    if width != 2:
        raise ValueError(f"expected 16-bit PCM WAV, got {width * 8}-bit")

    samples = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    if not rate or not samples.size:
        raise ValueError("empty or malformed WAV")

    return {
        "sample_rate": rate,
        "channels": channels,
        "duration": round(len(samples) / rate, 2),
        "voiced_seconds": round(float((np.abs(samples) > VOICED_THRESHOLD).sum()) / rate, 2),
        "peak": round(float(np.max(np.abs(samples))), 3),
        "clipped_ratio": round(float((np.abs(samples) >= 0.999).mean()), 5),
    }


async def segment_on_silence(
    chunks: AsyncIterator[bytes],
    sample_rate: int,
    silence_ms: int = 600,
    threshold: int = 500,
) -> AsyncIterator[bytes]:
    """Group a stream of raw PCM16 chunks into utterances split on silence.

    Lightweight energy-based VAD: emit a WAV buffer once `silence_ms` of
    below-threshold audio follows some speech. Good enough for near-real-time
    on CPU; swap for webrtcvad/silero later behind this same signature.
    """
    import audioop

    buffer = bytearray()
    silent_run = 0
    bytes_per_ms = int(sample_rate * 2 / 1000)  # 16-bit mono
    silence_bytes = silence_ms * bytes_per_ms

    async for chunk in chunks:
        buffer.extend(chunk)
        if audioop.rms(chunk, 2) < threshold:
            silent_run += len(chunk)
        else:
            silent_run = 0
        if silent_run >= silence_bytes and len(buffer) > silence_bytes:
            yield pcm16_to_wav(bytes(buffer), sample_rate)
            buffer.clear()
            silent_run = 0

    if buffer:
        yield pcm16_to_wav(bytes(buffer), sample_rate)
