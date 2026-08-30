from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np


def yugal_score_stream(wav_path: str | Path) -> Iterator[tuple[int, float]]:
    """Adapter for Yugal's continuous scoring interface.

    Preferred Yugal interface:

        def score_stream(wav_path):
            yield score_for_window_0
            yield score_for_window_1
            ...

    Each yielded value is one score for one 4-second window, in order.
    We attach window indices here so the UI never needs to know model internals.

    For a window-level implementation, the adapter also supports:

        def score_window(window_16k_float32):
            return score

    The real model should stay loaded in memory; do not reload it for every window.
    """
    try:
        from model import score_stream  # type: ignore
    except ImportError:
        score_stream = None

    if score_stream is not None:
        for idx, score in enumerate(score_stream(str(wav_path))):
            yield idx, float(score)
        return

    raise ImportError(
        "model.py was found without score_stream(wav_path). "
        "Add Yugal's continuous generator interface before enabling real-model mode."
    )
