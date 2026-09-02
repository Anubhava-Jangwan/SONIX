from __future__ import annotations

import io
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

import theme
from risk import hysteresis_bands, moving_average
from streaming import demo_score_stream, iter_windows
from windowing import load_audio_mono_16k

st.set_page_config(page_title="SONIX", page_icon="SONIX", layout="wide")
st.markdown(theme.page_css(), unsafe_allow_html=True)

TRAIN_AUG_CMD = (
    "python src/train.py --emb-root outputs/embeddings "
    "--extra-emb-root outputs/embeddings_g711 --out outputs/models/head_aug.pt"
)

MODEL_DESCRIPTIONS = {
    "head": (
        "**Baseline head** - trained on clean ASVspoof-2019 LA embeddings. "
        "In-domain eval EER ~= **1.49%** on unseen attacks."
    ),
    "head_aug": (
        "**Augmented head** - trained on clean **+ G.711 phone-codec** embeddings, "
        "so it holds up on compressed real-call audio."
    ),
    "head_robust": (
        "**Robust head** - the newly added robust checkpoint. Run the same clip "
        "through this tab to compare it against the other heads."
    ),
}


def band_info(band: str):
    if band == "RED":
        return "RED - HIGH", "Second-level approval required", theme.CRIT
    if band == "AMBER":
        return "AMBER - ELEVATED", "Call back on a number you already have", theme.WARN
    return "GREEN - LOW", "Proceed normally", theme.GOOD


def fake_score_stream(num_windows: int, filename: str):
    yield from demo_score_stream(num_windows, filename, step_delay_s=0.50)


def real_score_stream(tmp_path: str, ckpt_path: str):
    """Live scores from the real model for a specific checkpoint."""
    from model_adapter import yugal_score_stream
    yield from yugal_score_stream(tmp_path, ckpt_path=ckpt_path)


@st.cache_resource(show_spinner=False)
def _ensure_model(ckpt_path: str) -> str:
    """Load the shared front-end and this head once across Streamlit reruns."""
    from score_file import _load_head
    return _load_head(ckpt_path)


def plot_timeline(times, raw_scores, amber, red, current_time=None, switch_time=None):
    fig, ax = plt.subplots(figsize=(12, 4.8))

    times = np.asarray(times, dtype=float)
    scores = np.asarray(raw_scores, dtype=float)
    smoothed = moving_average(scores, window=5) if scores.size else np.array([])

    if scores.size:
        # Raw sits behind, small and muted: it is the evidence, not the verdict.
        ax.scatter(times, scores, s=13, alpha=0.45, color=theme.SERIES_MUTED,
                   linewidths=0, label="Raw score", zorder=2)
        ax.plot(times, smoothed, linewidth=2.2, color=theme.SERIES,
                label="5-window smoothed", zorder=3)

    # Thresholds are quiet dashed rules, not bands.
    ax.axhline(amber, linestyle="--", linewidth=1.0, color=theme.WARN,
               alpha=0.75, label=f"Amber {amber:.2f}", zorder=1)
    ax.axhline(red, linestyle="--", linewidth=1.0, color=theme.CRIT,
               alpha=0.75, label=f"Red {red:.2f}", zorder=1)

    if switch_time is not None:
        ax.axvline(switch_time, linestyle=":", linewidth=1.2, color=theme.INK_3,
                   alpha=0.8, label="15 s switch", zorder=1)

    if current_time is not None:
        ax.axvline(current_time, linewidth=1.2, color=theme.ACCENT, alpha=0.5,
                   label="Analysis position", zorder=1)

    ax.set_ylim(0, 1)
    max_time = max(float(current_time or 0), float(times[-1] if times.size else 0), 1.0)
    if switch_time is not None:
        max_time = max(max_time, switch_time)
    ax.set_xlim(0, max(5.0, max_time + 1.0))
    theme.style_axes(
        fig, ax,
        xlabel="Time (seconds)",
        ylabel="Spoof score (higher = more likely fake)",
    )

    max_x = float(ax.get_xlim()[1])
    step = 5 if max_x <= 60 else (10 if max_x <= 180 else 30)
    ax.set_xticks(np.arange(0, max_x + 0.1, step))
    theme.style_legend(ax)
    fig.tight_layout()
    return fig


def _readout(pairs):
    """A compact label/value strip - same fields the old dict dump carried."""
    cells = "".join(
        f'<div style="min-width:112px;">'
        f'<div style="font-size:{theme.FS_CAPTION};letter-spacing:.08em;'
        f'text-transform:uppercase;color:{theme.INK_3};">{k}</div>'
        f'<div style="font-size:{theme.FS_BODY};color:{theme.INK};'
        f'font-variant-numeric:tabular-nums;margin-top:2px;">{v}</div></div>'
        for k, v in pairs
    )
    return (
        f'<div style="display:flex;flex-wrap:wrap;gap:24px;border:{theme.BORDER};'
        f'border-radius:{theme.RADIUS_SM};padding:{theme.PAD_SM} {theme.PAD};'
        f'background:{theme.SURFACE};">{cells}</div>'
    )


def run_analysis(*, mode, ckpt_path, model_label, name, raw_bytes, duration_s,
                 windows_count, amber, red):
    """Stream scores one window at a time into the active Streamlit tab."""
    tmp_path = None
    if mode == "real":
        suffix = Path(name).suffix.lower() or ".wav"
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.write(raw_bytes)
        tmp.close()
        tmp_path = tmp.name

    status = st.empty()
    metrics = st.empty()
    risk_box = st.empty()
    chart = st.empty()

    if mode == "real":
        status.info(f"Loading the {model_label.lower()} model - the first run takes a few seconds...")
        try:
            with st.spinner(f"Loading {model_label.lower()} model (cached after this)..."):
                _ensure_model(ckpt_path)
        except Exception as exc:
            status.error(f"{model_label} model failed to load: {exc}")
            if tmp_path:
                Path(tmp_path).unlink(missing_ok=True)
            return
        status.success(f"{model_label} model ready - streaming scores...")

    plot_every = 1 if windows_count <= 40 else (2 if windows_count <= 120 else 3)
    times: list[float] = []
    raw_scores: list[float] = []

    try:
        score_iterator = (
            real_score_stream(tmp_path, ckpt_path)
            if mode == "real"
            else fake_score_stream(windows_count, name)
        )

        for idx, score in score_iterator:
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
                theme.risk_card(
                    band_name, band_action, border,
                    eyebrow=f"{model_label} model",
                ),
                unsafe_allow_html=True,
            )

            status.info(
                f"{model_label} model | window {idx + 1} / {windows_count} | "
                f"window start {t:.1f} s | latest score {raw_scores[-1]:.3f}"
            )
            metrics.markdown(
                _readout([
                    ("Model", model_label),
                    ("Scores received", f"{len(raw_scores)} / {windows_count}"),
                    ("Analysis time", f"{t:.1f} s"),
                    ("Latest score", f"{raw_scores[-1]:.3f}"),
                    ("Current band", current_band),
                ]),
                unsafe_allow_html=True,
            )

            if idx % plot_every == 0 or (idx + 1) >= windows_count:
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

        status.success(f"{model_label} model: score stream ended. No new points will be added.")
    except Exception as exc:
        status.error(f"{model_label} model failed: {exc}")
    finally:
        if tmp_path:
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except Exception:
                pass


st.markdown("# SONIX")
st.caption("Real-time windowed voice-cloning risk detector - compare every available model head")
st.divider()

try:
    from score_file import discover_checkpoints
    available_models = discover_checkpoints()
except Exception:
    available_models = []

with st.sidebar:
    st.header("Detection settings")
    amber = st.slider("Amber threshold", 0.05, 0.90, 0.10, 0.01)
    red = st.slider("Red threshold", float(min(1.0, amber + 0.05)), 1.00, 0.90, 0.01)
    st.caption("Defaults are the data-driven thresholds from the eval score distribution.")
    st.divider()
    mode_label = st.radio(
        "Scoring mode",
        ["Real model (live stream)", "Mock streaming"],
        index=0 if available_models else 1,
    )
    mode = "real" if mode_label.startswith("Real") else "mock"
    st.caption("Real mode consumes score_stream() from the trained model. Mock mode emits fake scores.")
    st.divider()
    st.markdown("**Checkpoints**")
    if available_models:
        for model in available_models:
            st.markdown(f"- {model['label']} `{model['filename']}`: found")
    else:
        st.markdown("- No `.pt` checkpoints found in `models/` or `outputs/models/`")

uploaded = st.file_uploader("Choose a .wav or .flac call recording", type=["wav", "flac"])
if uploaded is None:
    st.info("Upload an audio file to start. Then run the same clip through each model tab.")
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

st.info(
    "Upload once, then run the same clip through each model tab and compare the "
    "timelines. Nothing is plotted until scores arrive - the model is consumed "
    "one window at a time, exactly like a live call."
)

if mode == "real" and not available_models:
    st.warning("No model checkpoints were found. Add `.pt` files to `models/` or `outputs/models/`.")
    st.stop()

tab_models = available_models or [
    {"id": "mock", "label": "Mock", "path": "", "filename": "mock-stream"}
]
tabs = st.tabs([f"{model['label']} model" for model in tab_models])

for tab, model in zip(tabs, tab_models):
    with tab:
        st.markdown(
            MODEL_DESCRIPTIONS.get(
                model["id"],
                f"**{model['label']} head** - checkpoint `{model['filename']}`."
            )
        )
        if model["id"] == "head_aug" and mode == "real":
            st.caption("Retrain command, if you need to refresh this head:")
            st.code(TRAIN_AUG_CMD, language="bash")
        if mode == "real":
            st.caption(f"Checkpoint: `{model['path']}`")

        if st.button(f"Run {model['label'].lower()} analysis", type="primary",
                     use_container_width=True, key=f"run_{model['id']}"):
            run_analysis(
                mode=mode,
                ckpt_path=model["path"],
                model_label=model["label"],
                name=name,
                raw_bytes=raw_bytes,
                duration_s=duration_s,
                windows_count=windows_count,
                amber=amber,
                red=red,
            )
        else:
            st.caption(f"Press Run to stream this clip through the {model['label'].lower()} model.")

st.divider()
st.caption(
    "Real-time architecture: this UI consumes a score generator and never preloads "
    "the score list. The frozen wav2vec2 front-end is loaded once and shared; each "
    "tab only swaps in its own trained head, so the available models run back-to-back."
)
