"""
Sliding-window ring buffer for SONIX live detection.

Emits fixed-length windows at a fixed hop, so a continuous audio stream becomes
the discrete 4-second windows the scoring pipeline expects.

    16 kHz * 4.0 s = 64000 samples per window
    16 kHz * 0.5 s =  8000 samples per hop  (87.5% overlap)

The first window is emitted only once a FULL window of audio has arrived - we
never zero-pad up to length, because a half-silent window is exactly the input
that produces a false "fake" verdict.
"""

import numpy as np

SAMPLE_RATE = 16000
WINDOW_SEC = 4.0
HOP_SEC = 0.5


class RingBuffer:
    """Accumulates float32 mono audio and emits overlapping fixed-length windows."""

    def __init__(self, window_samples=None, hop_samples=None, capacity=None):
        # `capacity` is accepted for backwards compatibility with the old stub.
        self.window = int(window_samples or SAMPLE_RATE * WINDOW_SEC)
        self.hop = int(hop_samples or SAMPLE_RATE * HOP_SEC)

        self._buf = np.zeros(0, dtype=np.float32)
        self._pending = []

        self.total_pushed = 0        # absolute sample count ever received
        self.windows_emitted = 0
        self._next_emit_at = self.window   # absolute index where window 0 ends

    @property
    def capacity(self):
        return self.window

    def push(self, samples):
        """Append a chunk of float32 mono @16kHz and emit any newly complete windows."""
        if samples is None:
            return
        s = np.asarray(samples, dtype=np.float32).reshape(-1)
        if s.size == 0:
            return

        self._buf = np.concatenate([self._buf, s])
        self.total_pushed += s.size

        # Absolute index of self._buf[0]
        base = self.total_pushed - self._buf.size

        while self._next_emit_at <= self.total_pushed:
            end_local = self._next_emit_at - base
            start_local = end_local - self.window
            if start_local < 0:
                # Data already trimmed away - should not happen, skip forward.
                self._next_emit_at += self.hop
                continue
            self._pending.append(self._buf[start_local:end_local].copy())
            self.windows_emitted += 1
            self._next_emit_at += self.hop

        # Keep only what the next window can still need.
        keep = self.window + self.hop
        if self._buf.size > keep:
            self._buf = self._buf[-keep:]

    def get_emitted_windows(self):
        """Return windows completed since the last call, then clear them."""
        out = self._pending
        self._pending = []
        return out

    def reset(self):
        self._buf = np.zeros(0, dtype=np.float32)
        self._pending = []
        self.total_pushed = 0
        self.windows_emitted = 0
        self._next_emit_at = self.window

    def stats(self):
        return {
            "total_pushed": int(self.total_pushed),
            "seconds_buffered": float(self.total_pushed / SAMPLE_RATE),
            "windows_emitted": int(self.windows_emitted),
            "window_samples": self.window,
            "hop_samples": self.hop,
        }
