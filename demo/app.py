from __future__ import annotations

import io
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from risk import hysteresis_bands, moving_average
from streaming import demo_score_stream, iter_windows
from windowing import load_audio_mono_16k

st.set_page_config(page_title="SONIX", page_icon="🛡️", layout="wide")

GREEN = "#16a34a"
AMBER = "#d97706"
RED = "#dc2626"
BG = "#0d1117"


def band_info(band: str):
    if band == "RED":
        return "RED — HIGH", "Second-level approval required", RED
    if band == "AMBER":
        return "AMBER — ELEVATED", "Call back on a number you already have", AMBER
    return "GREEN — LOW", "Proceed normally", GREEN


def fake_score_stream(num_windows: int, filename: str):
    yield from demo_score_stream(num_windows, filename, step_delay_s=0.50)


def plot_timeline(times, raw_scores, amber, red, current_time=None, switch_time=None):
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.set_facecolor("#111827")
    fig.patch.set_facecolor(BG)

    times = np.asarray(times, dtype=float)
    scores = np.asarray(raw_scores, dtype=float)
    if scores.size:
        smoothed = moving_average(scores, window=5)
    else:
        smoothed = np.array([])

    if scores.size:
        ax.scatter(times, scores, s=20, alpha=0.35, label="Raw score")
        ax.plot(times, smoothed, linewidth=3, label="5-window smoothed")

    ax.axhline(amber, linestyle="--", linewidth=1.5, label=f"Amber {amber:.2f}")
    ax.axhline(red, linestyle="--", linewidth=1.5, label=f"Red {red:.2f}")

    if switch_time is not None:
        ax.axvline(switch_time, linestyle=":", linewidth=2, label="15 s switch")

    if current_time is not None:
        ax.axvline(current_time, linewidth=2, alpha=0.6, label="Analysis position")

    ax.set_ylim(0, 1)
    max_time = max(float(current_time or 0), float(times[-1] if times.size else 0), 1.0)
    if switch_time is not None:
        max_time = max(max_time, switch_time)
    ax.set_xlim(0, max(5.0, max_time + 1.0))
    ax.set_xlabel("Time (seconds)", color="#e5e7eb", fontsize=12, fontweight="bold")
    ax.set_ylabel("Spoof score (higher = more likely fake)", color="#e5e7eb", fontsize=11)
    ax.tick_params(axis="both", colors="#cbd5e1", labelsize=10)
    for _s in ax.spines.values():
        _s.set_color("#334155")
    # clear, visible time ticks along the whole clip
    _mt = float(ax.get_xlim()[1])
    _step = 5 if _mt <= 60 else (10 if _mt <= 180 else 30)
    ax.set_xticks(np.arange(0, _mt + 0.1, _step))
    ax.grid(alpha=0.18, color="#94a3b8")
    _leg = ax.legend(loc="upper left", ncols=2, facecolor="#1f2937", edgecolor="#334155", fontsize=9)
    for _t in _leg.get_texts():
        _t.set_color("#e5e7eb")
    fig.tight_layout()
    return fig


st.markdown("# SONIX")
st.caption("Real-time windowed voice-cloning risk detector")
st.divider()

with st.sidebar:
    st.header("Detection settings")
    amber = st.slider("Amber threshold", 0.10, 0.90, 0.45, 0.01)
    red = st.slider("Red threshold", float(min(1.0, amber + 0.05)), 1.00, max(0.70, min(0.99, amber + 0.25)), 0.01)
    st.divider()
    mode = st.radio("Scoring mode", ["Mock streaming", "Yugal live score stream"], index=0)
    st.caption("Mock mode emits one score every 0.5 s. Real mode consumes Yugal's score_stream() generator.")

uploaded = st.file_uploader("Choose a .wav or .flac call recording", type=["wav", "flac"])
if uploaded is None:
    st.info("Upload an audio file to start. The graph will be empty until scores arrive.")
    st.stop()

name = uploaded.name
raw_bytes = uploaded.getvalue()

try:
    audio, sr = load_audio_mono_16k(io.BytesIO(raw_bytes))
except Exception as exc:
    st.error(f"Could not read the audio: {exc}")
    st.stop()

st.audio(raw_bytes, format="audio/flac" if name.lower().endswith(".flac") else "audio/wav")
duration_s = len(audio) / sr
windows_count = sum(1 for _ in iter_windows(audio, sr))

m1, m2, m3 = st.columns(3)
m1.metric("Audio duration", f"{duration_s:.2f} s")
m2.metric("4-second windows", str(windows_count))
m3.metric("Hop", "0.5 s")

st.warning(
    "REAL-TIME ARCHITECTURE: the model score is consumed one result at a time. "
    "Nothing is plotted until a score arrives."
    if mode == "Yugal live score stream"
    else "DEMO ARCHITECTURE: fake scores are generated and delivered one at a time to simulate a live model stream."
)

start = st.button("▶ Start analysis", type="primary", use_container_width=True)

if not start:
    st.info("Press Start analysis. The timeline starts empty and grows only as scores arrive.")
    st.stop()

# Prepare model input file only for Yugal's current/expected path-based adapter.
tmp_path = None
if mode == "Yugal live score stream":
    suffix = Path(name).suffix.lower() or ".wav"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(raw_bytes)
    tmp.close()
    tmp_path = tmp.name

status = st.empty()
metrics = st.empty()
risk_box = st.empty()
chart = st.empty()

# State held only for this run. Every yielded score updates the UI before the next score is consumed.
times: list[float] = []
raw_scores: list[float] = []

try:
    if mode == "Mock streaming":
        score_iterator = fake_score_stream(windows_count, name)
    else:
        from model_adapter import yugal_score_stream
        score_iterator = yugal_score_stream(tmp_path)

    for idx, score in score_iterator:
        # The score index corresponds to the 4-s window start: 0, 0.5, 1.0, ...
        t = idx * 0.5
        if idx >= windows_count:
            break

        times.append(t)
        raw_scores.append(float(np.clip(score, 0.0, 1.0)))

        smoothed = moving_average(raw_scores, window=5)
        bands = hysteresis_bands(
            smoothed,
            amber_threshold=amber,
            red_threshold=red,
            agree_count=3,
            history_size=5,
            initial_band="GREEN",
            warmup_windows=5,
        )
        current_band = bands[-1]
        band_name, band_action, border = band_info(current_band)

        risk_box.markdown(
            f'''<div style="border:4px solid {border}; border-radius:20px; padding:24px; text-align:center; background:#111827;">
            <div style="font-size:44px; font-weight:900; color:{border};">{band_name}</div>
            <div style="font-size:19px; font-weight:700; color:#f3f4f6;">{band_action}</div>
            </div>''',
            unsafe_allow_html=True,
        )

        status.info(
            f"Receiving model scores · window {idx + 1} / {windows_count} · "
            f"window start {t:.1f} s · latest score {raw_scores[-1]:.3f}"
        )
        metrics.write({
            "scores_received": len(raw_scores),
            "windows_expected": windows_count,
            "analysis_time": f"{t:.1f} s",
            "latest_score": round(raw_scores[-1], 4),
            "current_band": current_band,
        })

        fig = plot_timeline(
            times,
            raw_scores,
            amber,
            red,
            current_time=t,
            switch_time=15.0 if "switch" in name.lower() and duration_s >= 15 else None,
        )
        chart.pyplot(fig, clear_figure=True)
        plt.close(fig)

    status.success("Model score stream ended. No new points will be added.")
finally:
    if tmp_path:
        try:
            Path(tmp_path).unlink(missing_ok=True)
        except Exception:
            pass

st.caption(
    "Important: in Yugal live-score mode this UI consumes a score generator. "
    "Yugal's model must yield one score per 4-second window in order. "
    "The UI never preloads the score list."
)
