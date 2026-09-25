"""Checks for the live mic path that do NOT need a microphone or a model.

Run: python demo/test_live.py

The window/resample arithmetic is the part worth pinning down. It has to turn
whatever odd rate the device hands us into exactly 64000 samples at 16 kHz --
get it wrong and the front-end silently scores a window that is the wrong
length, which does not raise, it just returns a wrong number.
"""

import sys
from collections import deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from live import SR, WIN, WIN_SECONDS, LiveMic     # noqa: E402


def _fill(mic, seconds, rate):
    """Pretend the PortAudio callback delivered `seconds` of tone at `rate`."""
    n = int(rate * seconds)
    t = np.arange(n, dtype=np.float32) / rate
    mic.native_sr = rate
    mic.audio = deque(np.sin(2 * np.pi * 220 * t).astype(np.float32),
                      maxlen=int(rate * 12))


def test_window_is_exact_at_every_device_rate():
    for rate in (16000, 44100, 48000, 22050, 8000):
        mic = LiveMic()
        _fill(mic, 6.0, rate)
        win = mic._window()
        assert win is not None, f"{rate}: expected a window from 6 s of audio"
        assert win.shape == (WIN,), f"{rate}: got {win.shape}, want ({WIN},)"
        assert win.dtype == np.float32, f"{rate}: dtype {win.dtype}"
        assert np.isfinite(win).all(), f"{rate}: non-finite samples"


def test_no_window_until_four_seconds():
    mic = LiveMic()
    _fill(mic, WIN_SECONDS - 0.5, 44100)
    assert mic._window() is None, "scored a partial window instead of waiting"
    assert 0.0 < mic.buffering() < 1.0
    _fill(mic, WIN_SECONDS + 0.1, 44100)
    assert mic._window() is not None
    assert mic.buffering() == 1.0


def test_swapping_head_keeps_history():
    """The whole point of the live section: changing model must not reset the
    series. ckpt/model_key are plain attributes the scorer reads per window."""
    mic = LiveMic()
    mic.ckpt, mic.model_key = "a.pt", "a"
    mic.scores.extend([(0.5, 0.1, "a"), (1.0, 0.2, "a")])
    before = list(mic.scores)

    mic.ckpt, mic.model_key = "b.pt", "b"       # what the picker does
    mic.scores.append((1.5, 0.9, "b"))

    t, s, k = mic.series()
    assert list(zip(t, s, k))[:2] == before, "history was lost on model swap"
    assert k == ["a", "a", "b"], f"per-point model labels wrong: {k}"


def test_history_is_bounded():
    mic = LiveMic(history=10)
    for i in range(50):
        mic.scores.append((i * 0.5, 0.5, "a"))
    assert len(mic.series()[0]) == 10, "score history grew past its bound"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("\nall live-path checks passed")
