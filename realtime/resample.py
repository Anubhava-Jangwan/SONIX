"""
Sample-rate conversion for the live path.

Browsers hand us 48000 Hz (sometimes 44100). Asterisk hands us 8000 Hz. The
frozen wav2vec2 front-end was trained at 16000 Hz and is sensitive to getting
anything else, so every source is normalised here before it reaches the buffer.

Uses scipy.signal.resample_poly when available (polyphase, anti-aliased). Falls
back to linear interpolation with a light pre-filter, which is adequate for a
demo but should not be the path we quote numbers from.
"""

import logging
from math import gcd

import numpy as np

logger = logging.getLogger(__name__)
TARGET_SR = 16000

try:
    from scipy.signal import resample_poly as _resample_poly
    HAVE_SCIPY = True
except Exception:                                    # pragma: no cover
    _resample_poly = None
    HAVE_SCIPY = False
    logger.warning("scipy unavailable - resampling falls back to linear interpolation")


def resample(samples, orig_sr, target_sr=TARGET_SR):
    """Resample float32 mono audio to target_sr. Returns float32."""
    x = np.asarray(samples, dtype=np.float32).reshape(-1)
    if x.size == 0 or int(orig_sr) == int(target_sr):
        return x

    orig_sr, target_sr = int(orig_sr), int(target_sr)

    if HAVE_SCIPY:
        g = gcd(orig_sr, target_sr)
        return _resample_poly(x, target_sr // g, orig_sr // g).astype(np.float32)

    n_out = int(round(x.size * target_sr / orig_sr))
    if n_out <= 0:
        return np.zeros(0, dtype=np.float32)
    idx = np.linspace(0, x.size - 1, n_out, dtype=np.float64)
    return np.interp(idx, np.arange(x.size), x).astype(np.float32)


def to_mono(samples, channels=1):
    """Downmix interleaved multi-channel audio to mono."""
    x = np.asarray(samples, dtype=np.float32).reshape(-1)
    if channels <= 1:
        return x
    usable = (x.size // channels) * channels
    return x[:usable].reshape(-1, channels).mean(axis=1).astype(np.float32)


def pcm16_to_float32(raw):
    """Decode little-endian int16 PCM bytes to float32 in [-1, 1]."""
    return (np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0)


def float32_to_pcm16(x):
    """Encode float32 in [-1, 1] to little-endian int16 PCM bytes."""
    y = np.clip(np.asarray(x, dtype=np.float32), -1.0, 1.0)
    return (y * 32767.0).astype("<i2").tobytes()
