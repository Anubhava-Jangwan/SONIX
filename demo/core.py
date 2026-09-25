"""Scoring state and helpers for the single-page app.

A scored clip is cached in session_state by (clip, model). Moving a threshold
re-derives the bands from those cached scores instantly -- it does not re-run
the 300M-param front-end.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import streamlit as st

import ui
from risk import hysteresis_bands, moving_average

REPO = Path(__file__).resolve().parents[1]
DEMO_CLIPS = REPO / "demo_clips"

# One registry for every surface. This module used to keep its own three-head
# dict, so the demo could not reach v3 (the server's DEFAULT_KEY) or robust_v2
# -- the only two heads with a matched-pair number against them.
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from realtime.models import DEFAULT_KEY, REGISTRY as MODELS   # noqa: E402


def available_models():
    """Registered heads whose checkpoint is actually on disk, in registry order."""
    return {k: v for k, v in MODELS.items() if (REPO / v[1]).exists()}


def default_index(models):
    """Where the picker should open: the registry default if its checkpoint is
    present, else the first head that is."""
    keys = list(models)
    return keys.index(DEFAULT_KEY) if DEFAULT_KEY in keys else 0


def defaults():
    st.session_state.setdefault("amber", 0.10)
    st.session_state.setdefault("red", 0.90)
    st.session_state.setdefault("scores", {})   # (clip, model) -> [raw scores]
    st.session_state.setdefault("clip", None)   # (name, path)


def bands_from(raw, amber, red):
    """Raw per-window scores -> (smoothed, bands). Pure; no model involved."""
    arr = np.asarray(raw, dtype=float)
    if not arr.size:
        return np.array([]), []
    smoothed = moving_average(arr, window=5)
    bands = hysteresis_bands(smoothed, amber_threshold=amber, red_threshold=red,
                             agree_count=3, history_size=5,
                             initial_band="GREEN", warmup_windows=5)
    return smoothed, bands


def threshold_controls(key_prefix: str = ""):
    """Inline, on-page. There is no sidebar to hide these in, and they are the
    controls a jury actually asks about."""
    c1, c2 = st.columns(2)
    amber = c1.slider("Amber threshold", 0.05, 0.90,
                      float(st.session_state["amber"]), 0.01,
                      key=f"{key_prefix}amber_s",
                      help="Above this, a window counts toward ELEVATED.")
    red = c2.slider("Red threshold", float(min(1.0, amber + 0.05)), 0.99,
                    float(max(st.session_state["red"], amber + 0.05)), 0.01,
                    key=f"{key_prefix}red_s",
                    help="Above this, a window counts toward HIGH.")
    st.session_state["amber"], st.session_state["red"] = amber, red
    return amber, red


def clip_picker(key_prefix: str = ""):
    """Upload, or pick one that ships with the repo."""
    local = sorted(p for p in DEMO_CLIPS.glob("*")
                   if p.suffix.lower() in (".wav", ".flac", ".mp3", ".ogg"))
    c1, c2 = st.columns([3, 2])
    up = c1.file_uploader("Upload a call recording", type=["wav", "flac", "mp3", "ogg"],
                          key=f"{key_prefix}up")
    names = ["--"] + [p.name for p in local]
    pick = c2.selectbox("or use a bundled clip", names, key=f"{key_prefix}pick")

    if up is not None:
        import tempfile
        tmp = Path(tempfile.gettempdir()) / f"sonix_{up.name}"
        tmp.write_bytes(up.getbuffer())
        return up.name, tmp
    if pick != "--":
        return pick, DEMO_CLIPS / pick
    return None, None


def clip_facts(path: Path):
    """Duration and window count, read from the header -- no decode.

    Window count MUST be derived at 16 kHz, not at the file's native rate. The
    scorer resamples first, so counting 44.1 kHz frames against a 64000-sample
    window reports ~2.7x too many (a 36.8 s clip showed 195 instead of 66).
    """
    import soundfile as sf
    info = sf.info(str(path))
    dur = info.frames / info.samplerate
    frames_16k = int(dur * 16000)
    n_win = max(1, (frames_16k - 64000) // 8000 + 1) if frames_16k >= 64000 else 1
    return dur, n_win, info.samplerate


def short_clip_warning(dur, n_win):
    if dur < 4.0:
        st.warning(
            f"This clip is {dur:.2f} s -- shorter than the 4.0 s window, so it "
            f"yields a single window and the timeline will be one point. "
            f"It is repeat-padded to fill the window, which shifts the score.",
            icon=":material/warning:")
        return True
    return False


def band_legend():
    st.markdown(ui.legend(), unsafe_allow_html=True)
