"""
Audio source adapters.

Two kinds of source exist, and they behave differently on purpose:

  PULL sources  (WavFileSource)  - the server loops calling .read(n)
  PUSH sources  (MicSource)      - audio arrives over the WebSocket and is
                                   handed straight to Session.push_audio()

Everything downstream expects float32 mono @ 16 kHz, so conversion happens here.
"""

import logging

import numpy as np

from realtime.resample import TARGET_SR, resample, to_mono

logger = logging.getLogger(__name__)


class SourceAdapter:
    """Base adapter. `caller` is what the dashboard shows as the call's identity."""

    def __init__(self, caller="unknown"):
        self.caller = caller
        self.sample_rate = TARGET_SR

    def read(self, size):
        """Pull `size` samples. Return None at end of stream."""
        return None

    def close(self):
        pass


class WavFileSource(SourceAdapter):
    """
    Reads a .wav/.flac/.mp3 from disk, downmixed to mono and resampled to 16 kHz.

    Used by /api/score-file and by replay mode. Decodes lazily on first read so
    constructing one is cheap.
    """

    def __init__(self, path, caller=None):
        super().__init__(caller or f"file:{path}")
        self.path = str(path)
        self._samples = None
        self._pos = 0
        self._eof = False

    def _load(self):
        if self._samples is not None:
            return
        try:
            import soundfile as sf
            data, sr = sf.read(self.path, dtype="float32", always_2d=True)
            mono = data.mean(axis=1).astype(np.float32)
        except Exception:
            # soundfile can't do mp3 everywhere; fall back to the stdlib for wav.
            import wave
            with wave.open(self.path, "rb") as w:
                sr = w.getframerate()
                ch = w.getnchannels()
                raw = w.readframes(w.getnframes())
            mono = to_mono(np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0, ch)

        self._samples = resample(mono, sr, TARGET_SR) if sr != TARGET_SR else mono
        self.source_sample_rate = sr
        logger.info(
            "WavFileSource %s: %d samples @ %d Hz -> %d @ %d Hz (%.1fs)",
            self.path, mono.size, sr, self._samples.size, TARGET_SR,
            self._samples.size / TARGET_SR,
        )

    @property
    def duration_sec(self):
        self._load()
        return float(self._samples.size / TARGET_SR)

    def read(self, size):
        """Return up to `size` float32 samples, or None once exhausted."""
        self._load()
        if self._eof or self._pos >= self._samples.size:
            self._eof = True
            return None
        chunk = self._samples[self._pos: self._pos + int(size)]
        self._pos += chunk.size
        return chunk.astype(np.float32, copy=False)

    def reset(self):
        self._pos = 0
        self._eof = False


class MicSource(SourceAdapter):
    """
    Push source for browser microphone audio.

    The WebSocket handler decodes each binary frame and calls
    Session.push_audio() directly, so read() is never used; this class exists to
    carry the caller identity and the incoming sample rate.
    """

    def __init__(self, caller="browser-mic", sample_rate=TARGET_SR):
        super().__init__(caller)
        self.sample_rate = int(sample_rate)
        self.bytes_received = 0

    def read(self, size):
        return None
