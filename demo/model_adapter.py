from __future__ import annotations

from pathlib import Path
from typing import Iterator


def yugal_score_stream(wav_path, ckpt_path=None) -> Iterator[tuple[int, float]]:
    """Adapter for Yugal's continuous scoring interface.

    Preferred Yugal interface:

        def score_stream(wav_path, ckpt_path=None):
            yield score_for_window_0
            yield score_for_window_1
            ...

    Each yielded value is one score for one 4-second window, in order. We attach
    window indices here so the UI never needs to know model internals. `ckpt_path`
    selects which trained head to use (baseline head.pt vs augmented head_aug.pt);
    if the model's score_stream predates that argument we fall back to calling it
    without one.

    The real model should stay loaded in memory; do not reload it for every window.
    """
    try:
        from model import score_stream  # type: ignore
    except ImportError:
        score_stream = None

    if score_stream is not None:
        try:
            gen = score_stream(str(wav_path), ckpt_path=ckpt_path)
        except TypeError:
            gen = score_stream(str(wav_path))
        for idx, score in enumerate(gen):
            yield idx, float(score)
        return

    raise ImportError(
        "model.py was found without score_stream(wav_path). "
        "Add Yugal's continuous generator interface before enabling real-model mode."
    )
