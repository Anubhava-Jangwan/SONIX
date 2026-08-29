from __future__ import annotations

from collections import deque
from typing import Iterable, Iterator

import numpy as np

from windowing import make_windows


def iter_windows(audio: np.ndarray, sr: int = 16000) -> Iterator[tuple[int, float, np.ndarray]]:
    """Yield window index, start time, and 4-second audio window."""
    for idx, (start_s, window) in enumerate(make_windows(audio, sr=sr, win_s=4.0, hop_s=0.5)):
        yield idx, start_s, window


def demo_score_stream(num_windows: int, filename: str, step_delay_s: float = 0.50) -> Iterator[tuple[int, float]]:
    """Deterministic mock stream: yields ONE score at a time, not a precomputed list."""
    import time

    rng = np.random.default_rng(590017752)
    name = filename.lower()

    for idx in range(num_windows):
        start_s = idx * 0.5
        if "clone" in name or "cloned" in name:
            score = float(np.clip(0.78 + 0.12 * rng.random(), 0, 1))
        elif "switch" in name:
            if start_s >= 15.0:
                score = float(np.clip(0.72 + 0.18 * rng.random(), 0, 1))
            else:
                score = float(np.clip(0.15 * rng.random(), 0, 1))
        else:
            score = float(np.clip(0.60 * rng.beta(2.0, 8.0), 0, 1))

        # Mock only: real model must yield when its inference result is ready.
        time.sleep(step_delay_s)
        yield idx, score
