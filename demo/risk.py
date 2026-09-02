from __future__ import annotations

from collections import deque
from typing import Iterable

import numpy as np

# Presentation only - the band colours come from the shared palette so the
# demo, the live dashboard and the extension cannot drift apart. theme imports
# nothing, so this stays a leaf dependency.
import theme

BAND_INFO = {
    "GREEN": {
        "name": "GREEN — LOW",
        "action": "Proceed normally",
        "border": theme.GOOD,
    },
    "AMBER": {
        "name": "AMBER — ELEVATED",
        "action": "Call back on a number you already have",
        "border": theme.WARN,
    },
    "RED": {
        "name": "RED — HIGH",
        "action": "Second-level approval required",
        "border": theme.CRIT,
    },
}

def band_from_score(score: float, amber_threshold: float, red_threshold: float) -> str:
    if red_threshold <= amber_threshold:
        raise ValueError("red_threshold must be greater than amber_threshold")
    score = float(score)
    if score >= red_threshold:
        return "RED"
    if score >= amber_threshold:
        return "AMBER"
    return "GREEN"

def moving_average(scores: Iterable[float], window: int = 5) -> np.ndarray:
    if window <= 0:
        raise ValueError("window must be positive")
    arr = np.asarray(list(scores), dtype=float)
    if arr.size == 0:
        return np.array([], dtype=float)
    out = np.empty_like(arr)
    for i in range(len(arr)):
        start = max(0, i-window+1)
        out[i] = np.mean(arr[start:i+1])
    return out

def hysteresis_bands(
    scores: Iterable[float],
    amber_threshold: float,
    red_threshold: float,
    agree_count: int = 3,
    history_size: int = 5,
    initial_band: str = "GREEN",
    warmup_windows: int = 5,
) -> list[str]:
    if history_size <= 0 or agree_count <= 0 or agree_count > history_size:
        raise ValueError("Need 0 < agree_count <= history_size")
    if warmup_windows < 0:
        raise ValueError("warmup_windows cannot be negative")
    if initial_band not in BAND_INFO:
        raise ValueError(f"Unknown initial band: {initial_band}")

    candidates = deque(maxlen=history_size)
    current = initial_band
    output: list[str] = []

    for score in scores:
        candidate = band_from_score(score, amber_threshold, red_threshold)
        candidates.append(candidate)
        count = sum(x == candidate for x in candidates)

        if len(output) + 1 > warmup_windows and candidate != current and count >= agree_count:
            current = candidate

        output.append(current)
    return output

def process_scores(
    raw_scores: Iterable[float],
    amber_threshold: float,
    red_threshold: float,
) -> tuple[np.ndarray, list[str]]:
    arr = np.asarray(list(raw_scores), dtype=float)
    if arr.size == 0:
        return np.array([], dtype=float), []
    smoothed = moving_average(arr, 5)
    bands = hysteresis_bands(
        smoothed,
        amber_threshold,
        red_threshold,
        agree_count=3,
        history_size=5,
        warmup_windows=5,
    )
    return smoothed, bands
