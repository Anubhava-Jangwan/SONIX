from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import BinaryIO, Iterator

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

TARGET_SR = 16000

def load_audio_mono_16k(source: str | Path | BytesIO | BinaryIO) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(source, dtype="float32", always_2d=False)
    audio = np.asarray(audio, dtype=np.float32)

    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    elif audio.ndim != 1:
        raise ValueError(f"Expected mono or multi-channel waveform, got shape {audio.shape}")

    if sr <= 0:
        raise ValueError(f"Invalid sample rate: {sr}")

    if sr != TARGET_SR:
        import math
        g = math.gcd(int(sr), TARGET_SR)
        audio = resample_poly(
            audio,
            TARGET_SR // g,
            int(sr) // g,
        ).astype(np.float32, copy=False)
        sr = TARGET_SR

    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
    return audio, TARGET_SR

def make_windows(
    audio: np.ndarray,
    sr: int = TARGET_SR,
    win_s: float = 4.0,
    hop_s: float = 0.5,
) -> Iterator[tuple[float, np.ndarray]]:
    if sr <= 0:
        raise ValueError("sr must be positive")
    if win_s <= 0 or hop_s <= 0:
        raise ValueError("win_s and hop_s must be positive")
    if hop_s > win_s:
        raise ValueError("hop_s should not exceed win_s")

    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    win_samples = int(round(win_s * sr))
    hop_samples = int(round(hop_s * sr))

    if audio.size == 0:
        yield 0.0, np.zeros(win_samples, dtype=np.float32)
        return

    start = 0
    while start + win_samples <= audio.size:
        yield start / sr, audio[start:start+win_samples].astype(np.float32, copy=False)
        start += hop_samples

    if start < audio.size:
        padded = np.zeros(win_samples, dtype=np.float32)
        chunk = audio[start:]
        padded[:chunk.size] = chunk
        yield start / sr, padded

def audio_duration_seconds(audio: np.ndarray, sr: int) -> float:
    return float(len(audio) / sr) if sr else 0.0
