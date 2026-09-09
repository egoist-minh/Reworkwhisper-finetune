from __future__ import annotations

import logging
import os
import time

from backend.core.audio import wav_to_pcm16
from backend.core.interfaces import ASREngine
from backend.core.types import Device, Segment, Transcript

logger = logging.getLogger(__name__)

_MODEL_MAP = {
    "small": "vinai/PhoWhisper-small",
    "medium": "vinai/PhoWhisper-medium",
    "large": "vinai/PhoWhisper-large",
}


class _DropDuplicateProcessorWarning(logging.Filter):
    """Drop transformers' benign "custom logits processor ... also created"
    notice. Whisper builds SuppressTokens processors both from generation_config
    and inside generate(); the duplicate is applied correctly, only noisy. This
    hides the message without touching model behaviour."""

    def filter(self, record: logging.LogRecord) -> bool:
        return "custom logits processor" not in record.getMessage()


_warning_filter_installed = False


def _silence_duplicate_processor_warning() -> None:
    global _warning_filter_installed
    if _warning_filter_installed:
        return
    logging.getLogger("transformers.generation.utils").addFilter(
        _DropDuplicateProcessorWarning()
    )
    _warning_filter_installed = True


class PhoWhisperASR(ASREngine):
    """VinAI PhoWhisper — Whisper fine-tuned on Vietnamese.

    Calls WhisperProcessor + WhisperForConditionalGeneration directly rather
    than transformers.pipeline(): the pipeline imports torchcodec during
    preprocess regardless of input form, and torchcodec's DLLs fail to load
    on this machine (RuntimeError: Could not load libtorchcodec), even though
    decoding is never actually needed for raw ndarray input. Expected accuracy
    leader for Vietnamese; run the 'small' size on CPU, larger sizes on the
    remote GPU.
    """

    def __init__(
        self,
        model_id: str,
        device: Device = Device.CPU,
        model_size: str = "small",
        token: str | None = None,
        no_repeat_ngram_size: int = 0,
        condition_on_prev_tokens: bool = True,
        no_speech_threshold: float = 0.6,
        logprob_threshold: float = -1.0,
        compression_ratio_threshold: float = 2.4,
        temperature: tuple[float, ...] | float = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    ):
        self.model_id = model_id
        self.device = device
        # Off by default. Set >0 to forbid re-emitting any n-gram of that length,
        # which pushes a looping decoder to EOS instead of letting it run to the
        # cap. This used to default to 3 and was measured to be the single largest
        # error source on Vietnamese meeting speech: conversational Vietnamese
        # repeats filler 3-grams constantly ("nói chung là", "cũng như là"), and
        # every time the ban bound, the decoder swerved off the correct token onto
        # a near-homophone — "thì" became "thị"/"thi", and "google chat" came back
        # as "google track"/"traff"/"trash" purely to avoid a repeat. Lifting it
        # took garbled tone variants from 27 to 0 over three meeting tracks.
        # Re-measure with scripts/sweep_decoding.py before changing this back.
        self.no_repeat_ngram_size = no_repeat_ngram_size
        # Anti-hallucination on silence/non-speech — Whisper's standard long-form
        # thresholds, which this transformers path otherwise leaves off (decoding
        # greedy at temp 0 with no gate, so a silent 30s window gets forced into a
        # made-up phrase). A window whose no-speech prob exceeds
        # no_speech_threshold AND whose avg logprob is below logprob_threshold is
        # dropped instead of decoded; temperature is the fallback ladder used when
        # a window fails the compression-ratio / logprob checks. All tunable via
        # config `params`.
        # condition_on_prev_tokens carries the previous window's text into the next
        # one. It costs some isolation — a bad window can prime its successor — but
        # it is what keeps repetition in check now that the n-gram ban is off:
        # with the ban lifted and this False, the loop it was guarding against
        # returns ("tình tình tình" x10). With both as set here, the worst repeat
        # over the same audio is a legitimate filler phrase at 7 occurrences.
        self.condition_on_prev_tokens = condition_on_prev_tokens
        self.no_speech_threshold = no_speech_threshold
        self.logprob_threshold = logprob_threshold
        self.compression_ratio_threshold = compression_ratio_threshold
        self.temperature = temperature
        # A size key ("small"/"medium"/"large") maps to a VinAI repo; anything
        # else (e.g. a fine-tune like "user/my-phowhisper") is used verbatim.
        self.hf_name = _MODEL_MAP.get(model_size, model_size)
        # Private repos need an HF token; fall back to the standard env vars.
        self.token = token or os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
        self._model = None
        self._processor = None
        self._torch_device = "cuda" if device == Device.CUDA else "cpu"

    def load(self) -> None:
        if self._model is not None:
            return
        logger.info("Loading PhoWhisper '%s' on %s...", self.hf_name, self.device.value)
        _silence_duplicate_processor_warning()
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        # clean_up_tokenization_spaces off: PhoWhisper is BPE, for which it
        # strips spaces before punctuation (destructive) — transformers
        # already ignores it, this just silences the warning.
        self._processor = WhisperProcessor.from_pretrained(
            self.hf_name,
            token=self.token,
            clean_up_tokenization_spaces=False,
        )
        import torch

        # Explicit fp32, everywhere, deliberately: some hub checkpoints (e.g.
        # winhsss/Reworkwhisper-large-v4) are stored in fp16, and recent
        # transformers versions preserve the checkpoint's saved dtype by default
        # instead of upcasting. fp16 weights would run ~2x faster on a GPU's
        # tensor cores, but this project spends its GPU budget on accuracy, not
        # latency (docs/module4-latency-optimization.md, section 2.4) -- module
        # 4's speedups all come from doing fewer, fuller decodes instead.
        self._model = WhisperForConditionalGeneration.from_pretrained(
            self.hf_name, token=self.token, torch_dtype=torch.float32
        )
        self._model.to(self._torch_device)
        self._model.eval()
        logger.info("PhoWhisper '%s' ready.", self.hf_name)

    def _gen_kwargs(self, lang: str, gen_overrides: dict | None) -> dict:
        gen_kwargs = {
            "task": "transcribe",
            "language": lang,
            "return_timestamps": True,
            "condition_on_prev_tokens": self.condition_on_prev_tokens,
            "no_speech_threshold": self.no_speech_threshold,
            "logprob_threshold": self.logprob_threshold,
            "compression_ratio_threshold": self.compression_ratio_threshold,
            "temperature": self.temperature,
        }
        if self.no_repeat_ngram_size:
            gen_kwargs["no_repeat_ngram_size"] = self.no_repeat_ngram_size
        if gen_overrides:
            gen_kwargs.update(gen_overrides)
            logger.info("Decoding overrides for this call: %s", gen_overrides)
        # An override of 0/null means "no n-gram ban", which generate() expresses
        # by the key being absent rather than zero.
        if not gen_kwargs.get("no_repeat_ngram_size"):
            gen_kwargs.pop("no_repeat_ngram_size", None)
        return gen_kwargs

    def _features(self, samples: list, sample_rate: int):
        """Log-mel features for one or more clips, ready for generate().

        truncation=False + padding="longest" + attention_mask: the pattern
        transformers documents for long-form generation without the pipeline
        (WhisperForConditionalGeneration.generate's docstring). Passing a fixed
        30s-padded batch instead would silently truncate any meeting turn longer
        than 30s. With more than one clip, "longest" pads the batch to its
        longest member and the attention mask marks the padding -- which is why
        module 4 packs its windows to a near-uniform length before batching
        (track_trim.trim_silence's pack_gap): a batch of one 28s and one 2s clip
        pays 28s of encoder for both.
        """
        inputs = self._processor(
            samples,
            sampling_rate=sample_rate,
            return_tensors="pt",
            truncation=False,
            padding="longest",
            return_attention_mask=True,
        )
        inputs = inputs.to(self._torch_device)
        # The feature extractor always emits fp32. So does load() -- this keeps
        # the two in step if a caller ever loads the weights in another dtype.
        inputs["input_features"] = inputs["input_features"].to(self._model.dtype)
        return inputs

    def _to_transcript(
        self, token_ids, duration: float, lang: str, rtf: float | None
    ) -> Transcript:
        decoded = self._processor.tokenizer.decode(
            token_ids, skip_special_tokens=True, output_offsets=True
        )
        text = decoded["text"].strip()
        segs = [
            Segment(
                text=c["text"].strip(),
                start=(c["timestamp"][0] or 0.0),
                end=(c["timestamp"][1] or 0.0),
            )
            for c in decoded["offsets"]
        ]
        if not segs and text:
            # Whisper decoded real text but emitted no well-formed timestamp
            # token pair, so output_offsets=True yielded an empty offset list.
            # Callers that consume only `segments` -- module 4 of the pipeline
            # does -- would silently lose the whole utterance, so fall back to
            # one segment spanning the clip. Measured on a real run: 9 of 11
            # such clips carried real speech, two of them whole meeting turns.
            logger.warning(
                "No timestamp offsets for %.1fs of audio; falling back to one "
                "segment over the whole clip (%d chars).",
                duration,
                len(text),
            )
            segs = [Segment(text=text, start=0.0, end=duration)]
        return Transcript(
            text=text,
            language=lang,
            segments=segs,
            model_id=self.model_id,
            rtf=rtf,
        )

    def _samples(self, audio: bytes) -> tuple[list, int, float]:
        import numpy as np

        pcm, sr = wav_to_pcm16(audio)
        samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        return samples, sr, len(pcm) / (sr * 2)

    def transcribe(
        self,
        audio: bytes,
        sample_rate: int,
        language: str | None = None,
        gen_overrides: dict | None = None,
    ) -> Transcript:
        """`gen_overrides` patches this call's generate() kwargs only, leaving the
        instance's configured defaults alone -- one loaded model can therefore be
        swept across decoding settings without a reload (see
        scripts/sweep_decoding.py). Optional keyword, so the interface's
        three-argument call still works."""
        self.load()
        samples, sr, duration = self._samples(audio)

        # Force Vietnamese transcription (PhoWhisper is a Vietnamese fine-tune):
        # skip Whisper's language-autodetect, and use task/language instead of
        # the deprecated forced_decoder_ids path.
        lang = language or "vi"

        logger.info("Transcribing %.1fs of audio (lang=%s)...", duration, lang)
        gen_kwargs = self._gen_kwargs(lang, gen_overrides)
        inputs = self._features(samples, sr)

        import torch

        t0 = time.perf_counter()
        with torch.no_grad():
            generated_ids = self._model.generate(**inputs, **gen_kwargs)
        elapsed = time.perf_counter() - t0

        rtf = (elapsed / duration) if duration else None
        result = self._to_transcript(generated_ids[0], duration, lang, rtf)
        logger.info(
            "Transcribed %.1fs in %.1fs (RTF %.2f), %d chars.",
            duration,
            elapsed,
            rtf or 0.0,
            len(result.text),
        )
        return result

    def transcribe_batch(
        self,
        clips: list[bytes],
        sample_rate: int,
        language: str | None = None,
        gen_overrides: dict | None = None,
    ) -> list[Transcript]:
        """Decode several clips in ONE generate() call, one Transcript each.

        Why this exists: Whisper's encoder is fixed at a 30s window, so a call on
        a 3s clip costs the same encoder pass as one on a 28s clip, and module 4
        makes one call per speech window (~50-70 per meeting on a 3-speaker
        recording). Batching puts those windows through the GPU concurrently
        instead of one after another. Nothing about any single clip's decode
        changes: batch members share no state, so each row sees the same
        features and the same generate() kwargs it would alone.

        Batch members are padded to the longest one, so callers should group
        clips of similar length (module 4's packed windows already are).
        """
        self.load()
        if not clips:
            return []
        decoded = [self._samples(c) for c in clips]
        samples = [d[0] for d in decoded]
        sr = decoded[0][1]
        durations = [d[2] for d in decoded]
        lang = language or "vi"

        logger.info(
            "Transcribing a batch of %d clips, %.1fs total (lang=%s)...",
            len(clips),
            sum(durations),
            lang,
        )
        gen_kwargs = self._gen_kwargs(lang, gen_overrides)
        inputs = self._features(samples, sr)

        import torch

        t0 = time.perf_counter()
        with torch.no_grad():
            generated_ids = self._model.generate(**inputs, **gen_kwargs)
        elapsed = time.perf_counter() - t0

        total = sum(durations)
        # Per-clip RTF isn't measurable inside a batch -- one generate() covers
        # all of them -- so every row carries the batch's RTF.
        rtf = (elapsed / total) if total else None
        results = [
            self._to_transcript(ids, dur, lang, rtf)
            for ids, dur in zip(generated_ids, durations)
        ]
        logger.info(
            "Transcribed %d clips (%.1fs) in %.1fs (RTF %.2f).",
            len(clips),
            total,
            elapsed,
            rtf or 0.0,
        )
        return results

    def unload(self) -> None:
        self._model = None
        self._processor = None
