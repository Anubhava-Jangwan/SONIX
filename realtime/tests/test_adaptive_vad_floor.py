"""The adaptive silence-gate floor must never sit above the clip's own speech.

Regression test for the upload path reporting "0 / N windows scored" on quiet
recordings. `_adaptive_vad_floor` clamped its lower bound to 0.0006, which for a
clip below roughly -65 dBFS lands ABOVE the measured speech level -- every frame
then fails the energy test and nothing is scored.
"""

import numpy as np
import pytest

from realtime.server import SonicServer
from realtime.vad import VAD

WIN = 64000
floor_of = SonicServer._adaptive_vad_floor


def _speech_like(n, rms, seed=0):
    """Band-limited noise at a target RMS -- ZCR low enough to read as voice."""
    rng = np.random.RandomState(seed)
    x = np.convolve(rng.randn(n + 64), np.ones(64) / 64, mode="valid")[:n]
    return (x / np.sqrt(np.mean(x ** 2)) * rms).astype(np.float32)


@pytest.mark.parametrize("rms", [0.2, 0.05, 0.01, 0.002, 0.0005, 0.0001])
def test_floor_never_exceeds_clip_speech_level(rms):
    floor, speech, _ = floor_of(_speech_like(WIN, rms))
    assert floor <= speech, (
        f"floor {floor:.7f} above speech level {speech:.7f} -- every window "
        f"would be gated out"
    )


@pytest.mark.parametrize("rms", [0.2, 0.01, 0.002, 0.0005, 0.0001])
def test_quiet_speech_still_passes_the_gate(rms):
    w = _speech_like(WIN, rms)
    floor, _, _ = floor_of(w)
    assert VAD(threshold_energy=floor).is_speech(w), f"quiet clip at rms={rms} rejected"


@pytest.mark.parametrize("name,w", [
    ("digital silence", np.zeros(WIN, np.float32)),
    ("dither", (np.random.RandomState(1).randn(WIN) * 3e-5).astype(np.float32)),
    ("room tone", (np.random.RandomState(2).randn(WIN) * 3e-4).astype(np.float32)),
])
def test_dead_audio_is_still_rejected(name, w):
    """Lowering the energy floor must not let silence through. The ZCR ceiling
    catches it independently -- that is why capping the floor is safe."""
    floor, _, _ = floor_of(w)
    assert not VAD(threshold_energy=floor).is_speech(w), f"{name} passed the gate"


def test_loud_clip_floor_is_unchanged():
    """The fix must not move the gate for normal-level audio."""
    floor, speech, _ = floor_of(_speech_like(WIN, 0.2))
    assert floor == pytest.approx(0.01), "studio-level ceiling should still apply"
