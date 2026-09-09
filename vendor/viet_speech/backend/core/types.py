from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Device(str, Enum):
    CPU = "cpu"
    CUDA = "cuda"
    REMOTE = "remote"


@dataclass
class Segment:
    text: str
    start: float  # seconds
    end: float
    confidence: float | None = None


@dataclass
class Transcript:
    text: str
    language: str = "vi"
    segments: list[Segment] = field(default_factory=list)
    model_id: str = ""
    rtf: float | None = None  # real-time factor = proc_time / audio_duration
    confidence: float | None = None  # average log probability from model

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "language": self.language,
            "model_id": self.model_id,
            "rtf": self.rtf,
            "confidence": self.confidence,
            "segments": [
                {"text": s.text, "start": s.start, "end": s.end, "confidence": s.confidence}
                for s in self.segments
            ],
        }


@dataclass
class AudioResult:
    audio: bytes  # encoded WAV bytes
    sample_rate: int
    model_id: str = ""
    rtf: float | None = None


@dataclass
class SpeakerTurn:
    """A stretch of audio attributed to one speaker by the diarizer.

    Language-independent — carries no text, only who-spoke-when.
    """

    start: float  # seconds
    end: float
    speaker: str


@dataclass
class SpeakerSegment:
    """ASR text merged onto a diarization turn: speaker-labelled, time-coded."""

    text: str
    start: float  # seconds
    end: float
    speaker: str


@dataclass
class SpeechSegment:
    """Module 1 (VAD) output: one stretch of audio that contains speech."""

    start: float  # seconds
    end: float


@dataclass
class OverlapLabel:
    """Module 2 (Overlap Detection) output: is this stretch single- or multi-speaker?"""

    start: float  # seconds
    end: float
    overlapping: bool


@dataclass
class TimeMapping:
    """One splice in a per-speaker concatenated track (module 3's output).

    `local_start`/`local_end` are positions inside the speaker's own WAV file;
    `source_start`/`source_end` are the true timestamps in the original meeting
    recording. Module 7 needs this to convert an ASR transcript's in-file times
    back to real meeting time before merging speakers together.
    """

    local_start: float
    local_end: float
    source_start: float
    source_end: float


@dataclass
class SpeakerTrack:
    """Module 3 output: one placeholder speaker's audio, spliced from single-
    speaker segments and separated overlap streams, spanning the whole meeting."""

    label: str  # placeholder id, e.g. "Speaker 1" — not yet a real name
    audio: bytes  # encoded WAV bytes
    sample_rate: int
    mapping: list[TimeMapping] = field(default_factory=list)


@dataclass
class SpeakerEmbedding:
    """Module 5 output: one vector representing a SpeakerTrack's voice."""

    label: str  # matches SpeakerTrack.label
    vector: list[float]
    model_id: str = ""


@dataclass
class IdentifiedSpeaker:
    """Module 6 output: a SpeakerTrack matched against the enrollment DB."""

    label: str  # placeholder id from module 3, e.g. "Speaker 1"
    name: str  # real name from the enrollment DB, or "Unknown"
    similarity: float = 0.0


@dataclass
class MeetingTranscript:
    segments: list[SpeakerSegment] = field(default_factory=list)
    speakers: list[str] = field(default_factory=list)
    language: str = "vi"
    asr_model_id: str = ""
    diar_model_id: str = ""
    separator_model_id: str | None = None
    rtf: float | None = None  # real-time factor = proc_time / audio_duration

    def to_dict(self) -> dict:
        return {
            "language": self.language,
            "asr_model_id": self.asr_model_id,
            "diar_model_id": self.diar_model_id,
            "separator_model_id": self.separator_model_id,
            "rtf": self.rtf,
            "speakers": self.speakers,
            "segments": [
                {
                    "speaker": s.speaker,
                    "text": s.text,
                    "start": s.start,
                    "end": s.end,
                }
                for s in self.segments
            ],
        }

    @staticmethod
    def from_dict(data: dict) -> "MeetingTranscript":
        """Rebuild from the exact shape MeetingTranscript.to_dict() / the
        /meeting/transcribe response produce -- keeps module 8 in lockstep
        with whatever module 7 actually returns."""
        return MeetingTranscript(
            segments=[
                SpeakerSegment(
                    text=s["text"], start=s["start"], end=s["end"], speaker=s["speaker"]
                )
                for s in data.get("segments", [])
            ],
            speakers=data.get("speakers", []),
            language=data.get("language", "vi"),
            asr_model_id=data.get("asr_model_id", ""),
            diar_model_id=data.get("diar_model_id", ""),
            separator_model_id=data.get("separator_model_id"),
            rtf=data.get("rtf"),
        )


@dataclass
class ActionItem:
    task: str
    owner: str | None = None
    deadline: str | None = None

    def to_dict(self) -> dict:
        return {"task": self.task, "owner": self.owner, "deadline": self.deadline}


@dataclass
class Decision:
    text: str

    def to_dict(self) -> dict:
        return {"text": self.text}
