"""
Voice Activity Detection - the silence gate.

Why this exists: the model was trained on clean studio speech. A window that is
mostly silence looks like nothing it has ever seen, and it tends to call that
"fake". Dropping near-silent windows before they reach the scorer removes a large
share of our false alarms, and it is cheap.

Method (no external deps, runs in microseconds):
  - split the window into 25 ms frames
  - a frame is speech if its RMS clears an energy floor AND its zero-crossing
    rate is not so high that it looks like hiss/fricative noise rather than voice
  - the window passes if enough of its frames are speech

`last_stats` is left populated after every call so the dashboard can plot what
the gate actually saw, rather than asking the user to trust it.
"""

import numpy as np

SAMPLE_RATE = 16000
FRAME_MS = 25


class VAD:
    def __init__(
        self,
        threshold_energy=0.01,
        threshold_zcr=0.1,
        zcr_ceiling=0.35,
        min_speech_ratio=0.20,
        frame_samples=None,
    ):
        self.threshold_energy = float(threshold_energy)
        self.threshold_zcr = float(threshold_zcr)
        self.zcr_ceiling = float(zcr_ceiling)
        self.min_speech_ratio = float(min_speech_ratio)
        self.frame_samples = int(frame_samples or SAMPLE_RATE * FRAME_MS / 1000)

        self.last_stats = {}
        self.windows_seen = 0
        self.windows_passed = 0

    @staticmethod
    def _zcr(frame):
        if frame.size < 2:
            return 0.0
        return float(np.mean(np.abs(np.diff(np.signbit(frame).astype(np.int8)))))

    def is_speech(self, window):
        """True if the window carries enough voice to be worth scoring."""
        w = np.asarray(window, dtype=np.float32).reshape(-1)
        self.windows_seen += 1

        if w.size == 0:
            self.last_stats = {"rms": 0.0, "zcr": 0.0, "speech_ratio": 0.0, "passed": False}
            return False

        n = self.frame_samples
        n_frames = max(1, w.size // n)
        frames = w[: n_frames * n].reshape(n_frames, n)

        rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))
        zcr = np.array([self._zcr(f) for f in frames])

        speech = (rms > self.threshold_energy) & (zcr < self.zcr_ceiling)
        ratio = float(np.mean(speech))
        passed = ratio >= self.min_speech_ratio

        self.last_stats = {
            "rms": float(np.sqrt(np.mean(w.astype(np.float64) ** 2))),
            "peak": float(np.max(np.abs(w))),
            "zcr": float(np.mean(zcr)),
            "speech_ratio": ratio,
            "passed": bool(passed),
        }
        if passed:
            self.windows_passed += 1
        return passed

    def stats(self):
        return {
            "windows_seen": self.windows_seen,
            "windows_passed": self.windows_passed,
            "windows_rejected": self.windows_seen - self.windows_passed,
            "pass_rate": (self.windows_passed / self.windows_seen) if self.windows_seen else 0.0,
        }
