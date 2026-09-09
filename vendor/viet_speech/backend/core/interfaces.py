from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

import numpy as np

from .audio import segment_on_silence
from .types import (
    AudioResult,
    Device,
    OverlapLabel,
    SpeakerTrack,
    SpeakerTurn,
    SpeechSegment,
    Transcript,
)

if TYPE_CHECKING:
    from .roster import MeetingRoster


class ASREngine(ABC):
    """One implementation per ASR model. Constructed by the registry from config.

    Subclasses set model_id/device via __init__ and must implement load() and
    transcribe(). stream() has a working default (buffer on silence) so every
    model gets near-real-time for free.
    """

    model_id: str
    device: Device
    supported_languages: list[str] = ["vi"]

    @abstractmethod
    def load(self) -> None:
        """Lazy-load weights. Called once on first use, not at import time."""

    @abstractmethod
    def transcribe(
        self, audio: bytes, sample_rate: int, language: str | None = None
    ) -> Transcript:
        """Batch transcription of a complete WAV-encoded audio buffer."""

    async def stream(
        self,
        chunks: AsyncIterator[bytes],
        sample_rate: int,
        language: str | None = None,
    ) -> AsyncIterator[Transcript]:
        """Near-real-time: yield partial transcripts as audio arrives.

        Default impl accumulates PCM chunks and flushes through transcribe()
        whenever a silence gap is detected. Streaming-native models override.
        """
        async for utterance in segment_on_silence(chunks, sample_rate):
            yield self.transcribe(utterance, sample_rate, language)

    def unload(self) -> None:
        """Free memory. Useful when swapping models on a constrained machine."""


class DiarizationEngine(ABC):
    """One implementation per diarizer. Answers *who spoke when* — no ASR.

    Language-independent: the same instance serves English and Vietnamese.
    Subclasses set model_id/device via __init__ and implement load()/diarize().
    """

    model_id: str
    device: Device

    @abstractmethod
    def load(self) -> None:
        """Lazy-load weights. Called once on first use, not at import time."""

    @abstractmethod
    def diarize(self, audio: bytes, sample_rate: int) -> list[SpeakerTurn]:
        """Return speaker turns for a complete WAV-encoded audio buffer."""

    def unload(self) -> None:
        """Free memory. Useful when swapping models on a constrained machine."""


class VADEngine(ABC):
    """Module 1. Answers *when* someone spoke — no identity, no text."""

    model_id: str
    device: Device

    @abstractmethod
    def load(self) -> None:
        """Lazy-load weights. Called once on first use, not at import time."""

    @abstractmethod
    def detect(self, audio: bytes, sample_rate: int) -> list[SpeechSegment]:
        """Return the stretches of a complete WAV-encoded buffer that contain speech."""

    def unload(self) -> None:
        """Free memory. Useful when swapping models on a constrained machine."""


class OverlapDetector(ABC):
    """Module 2. For each speech segment (module 1), single- or multi-speaker?

    Independent of diarization/identification — must run before both, since it
    decides whether a segment needs Source Separation (module 3) at all.
    """

    model_id: str
    device: Device

    @abstractmethod
    def load(self) -> None:
        """Lazy-load weights. Called once on first use, not at import time."""

    @abstractmethod
    def detect(
        self, audio: bytes, sample_rate: int, segments: list[SpeechSegment]
    ) -> list[OverlapLabel]:
        """Label each input segment as overlapping or not."""

    def unload(self) -> None:
        """Free memory. Useful when swapping models on a constrained machine."""


class SourceSeparator(ABC):
    """Module 3. Splits overlapping segments into per-speaker streams, then
    splices those streams together with the untouched single-speaker segments
    into one continuous track per placeholder speaker for the whole meeting.

    Grouping "which stream belongs to which placeholder speaker" is NOT free —
    it needs its own embedding/clustering pass across the whole meeting (see
    docs/bao-cao-tien-do-va-ke-hoach-2026-08-03.md, "open problem"). Implementers
    should resolve that internally before returning tracks.
    """

    model_id: str
    device: Device

    @abstractmethod
    def load(self) -> None:
        """Lazy-load weights. Called once on first use, not at import time."""

    @abstractmethod
    def separate(
        self, audio: bytes, sample_rate: int, labels: list[OverlapLabel]
    ) -> list[SpeakerTrack]:
        """Return one SpeakerTrack per placeholder speaker found in the meeting."""

    def set_roster(self, roster: MeetingRoster | None) -> dict | None:
        """Optional adapter capability: pin whatever internal speaker-
        clustering decision this separator makes to a fixed set of
        enrollment centroids for who's actually in *this* meeting, instead of
        clustering from scratch. Base no-op — most separators have no such
        step to patch; the pipeline calls this the same way it calls the
        optional `transcribe_words()` on ASREngine, treating a None return as
        "nothing to report" for the UI.

        Implementers that DO patch something must also undo it when called
        with `roster=None` (or a roster with nobody usable): the model these
        adapters patch is typically a process-wide shared cache, so without
        this a meeting run with a roster would leak its centroids into the
        next run that has none.
        """
        return None

    def unload(self) -> None:
        """Free memory. Useful when swapping models on a constrained machine."""


class SeparationEngine(ABC):
    """One implementation per source-separation model. Splits mixed-speaker
    audio into per-speaker streams, upstream of diarization/ASR.

    Subclasses set model_id/device via __init__ and implement load()/separate().
    """

    model_id: str
    device: Device

    @abstractmethod
    def load(self) -> None:
        """Lazy-load weights. Called once on first use, not at import time."""

    @abstractmethod
    def separate(self, audio: bytes, sample_rate: int) -> list[bytes]:
        """Split a mixed WAV-encoded audio buffer into per-speaker WAV buffers."""

    def unload(self) -> None:
        """Free memory. Useful when swapping models on a constrained machine."""


class SpeakerEmbeddingEngine(ABC):
    """One implementation per speaker-embedding model (module 5). Turns a
    voice clip into one fixed-size feature vector for cosine-similarity
    matching downstream (module 6's SpeakerStore, backend/core/speaker_id.py).

    Subclasses set model_id/device via __init__ and implement load()/embed().
    """

    model_id: str
    device: Device

    @abstractmethod
    def load(self) -> None:
        """Lazy-load weights. Called once on first use, not at import time."""

    @abstractmethod
    def embed(self, audio: bytes, sample_rate: int) -> np.ndarray | None:
        """One feature vector for a WAV-encoded voice clip, or None if the
        clip is empty/too short to embed."""

    def unload(self) -> None:
        """Free memory. Useful when swapping models on a constrained machine."""


class TextEmbeddingEngine(ABC):
    """One implementation per text-embedding model. Turns a chunk of
    transcript text into one fixed-size vector for pgvector similarity
    search (Giai đoạn 4 indexing, backend/core/indexing.py) -- the semantic-
    search counterpart to SpeakerEmbeddingEngine, which embeds audio instead.

    Subclasses set model_id/device via __init__ and implement load()/embed().
    """

    model_id: str
    device: Device

    @abstractmethod
    def load(self) -> None:
        """Lazy-load/validate credentials. Called once on first use."""

    @abstractmethod
    def embed(self, text: str) -> list[float] | None:
        """One feature vector for a chunk of text, or None if the text is
        empty."""

    def unload(self) -> None:
        """Free memory. Useful when swapping models on a constrained machine."""


class LLMEngine(ABC):
    """One implementation per LLM backend (local or remote API). Module 8 glue:
    MeetingAI (core/meeting_ai.py) prompts this to summarize/extract/answer over
    a transcript's text -- the engine itself knows nothing about meetings.

    Subclasses set model_id/device via __init__ and implement load()/complete().
    """

    model_id: str
    device: Device

    @abstractmethod
    def load(self) -> None:
        """Lazy-load/validate credentials. Called once on first use."""

    @abstractmethod
    def complete(self, prompt: str, system: str | None = None) -> str:
        """Return the model's text completion for prompt (+ optional system prompt)."""

    def unload(self) -> None:
        """Free memory. Useful when swapping models on a constrained machine."""


class TTSEngine(ABC):
    model_id: str
    device: Device
    voices: list[str] = []
    supported_languages: list[str] = ["vi"]

    @abstractmethod
    def load(self) -> None:
        ...

    @abstractmethod
    def synthesize(
        self, text: str, voice: str | None = None, language: str | None = None
    ) -> AudioResult:
        ...

    @property
    def supports_cloning(self) -> bool:
        return False

    def unload(self) -> None:
        ...
