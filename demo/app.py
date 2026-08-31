from __future__ import annotations

import io
import json
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from risk import hysteresis_bands, moving_average
from streaming import demo_score_stream, iter_windows, replay_score_stream
from windowing import load_audio_mono_16k


st.set_page_config(
    page_title="SONIX",
    page_icon="🛡️",
    layout="wide",
)

GREEN = "#16a34a"
AMBER = "#d97706"
RED = "#dc2626"
BG = "#0d1117"

# Repository root:
# D:\SIH\
REPO_ROOT = Path(__file__).resolve().parents[1]

# Model checkpoints
BASELINE_CKPT = "outputs/models/head.pt"
AUGMENTED_CKPT = "outputs/models/head_aug.pt"

TRAIN_AUG_CMD = (
    "python src/train.py --emb-root outputs/embeddings "
    "--extra-emb-root outputs/embeddings_g711 "
    "--out outputs/models/head_aug.pt"
)

# Validated demo/replay assets
DEMO_CLIPS_DIR = REPO_ROOT / "demo_clips"
SCORES_DIR = REPO_ROOT / "scores"


def get_replay_clips():
    """Return replay clips that have matching audio and score JSON.

    Expected structure:

        demo_clips/
            genuine1.flac
            clone1.flac

        scores/
            genuine1.json
            clone1.json

    Each JSON must contain:

        {
            "clip": "genuine1.flac",
            "scores": [0.02, 0.03, 0.05]
        }

    Matching is performed using the filename stem.
    """

    audio_files = {}

    if not DEMO_CLIPS_DIR.exists():
        return []

    if not SCORES_DIR.exists():
        return []

    # Find supported audio files.
    for path in DEMO_CLIPS_DIR.glob("*"):
        if (
            path.is_file()
            and path.suffix.lower() in {".wav", ".flac"}
        ):
            audio_files[path.stem] = path

    replay_clips = []

    # Find score JSONs and match them to audio files.
    for score_file in sorted(SCORES_DIR.glob("*.json")):
        try:
            with open(
                score_file,
                "r",
                encoding="utf-8",
            ) as f:
                payload = json.load(f)

            clip_name = payload.get("clip")
            scores = payload.get("scores")

            # Validate required JSON fields.
            if not isinstance(clip_name, str):
                continue

            if not isinstance(scores, list) or not scores:
                continue

            # Match audio using filename stem.
            clip_stem = Path(clip_name).stem
            audio_path = audio_files.get(clip_stem)

            if audio_path is None:
                continue

            replay_clips.append(
                {
                    "name": clip_stem,
                    "audio_path": audio_path,
                    "score_path": score_file,
                    "clip_name": clip_name,
                    "score_count": len(scores),
                }
            )

        except Exception:
            # Ignore malformed JSON files rather than
            # crashing the entire demo.
            continue

    return replay_clips


def band_info(band: str):
    if band == "RED":
        return (
            "RED — HIGH",
            "Second-level approval required",
            RED,
        )

    if band == "AMBER":
        return (
            "AMBER — ELEVATED",
            "Call back on a number you already have",
            AMBER,
        )

    return (
        "GREEN — LOW",
        "Proceed normally",
        GREEN,
    )


def fake_score_stream(
    num_windows: int,
    filename: str,
):
    yield from demo_score_stream(
        num_windows,
        filename,
        step_delay_s=0.50,
    )


def real_score_stream(
    tmp_path: str,
    ckpt_path: str,
):
    """Live scores from the real model."""
    from model_adapter import yugal_score_stream

    yield from yugal_score_stream(
        tmp_path,
        ckpt_path=ckpt_path,
    )


@st.cache_resource(show_spinner=False)
def _ensure_model(
    ckpt_path: str,
) -> str:
    """Load the front-end + head once and keep it warm."""
    from score_file import _load_head

    return _load_head(ckpt_path)


def plot_timeline(
    times,
    raw_scores,
    amber,
    red,
    current_time=None,
    switch_time=None,
):
    fig, ax = plt.subplots(
        figsize=(12, 5.5)
    )

    ax.set_facecolor("#111827")
    fig.patch.set_facecolor(BG)

    times = np.asarray(
        times,
        dtype=float,
    )

    scores = np.asarray(
        raw_scores,
        dtype=float,
    )

    if scores.size:
        smoothed = moving_average(
            scores,
            window=5,
        )
    else:
        smoothed = np.array([])

    if scores.size:
        ax.scatter(
            times,
            scores,
            s=20,
            alpha=0.35,
            label="Raw score",
        )

        ax.plot(
            times,
            smoothed,
            linewidth=3,
            label="5-window smoothed",
        )

    ax.axhline(
        amber,
        linestyle="--",
        linewidth=1.5,
        label=f"Amber {amber:.2f}",
    )

    ax.axhline(
        red,
        linestyle="--",
        linewidth=1.5,
        label=f"Red {red:.2f}",
    )

    if switch_time is not None:
        ax.axvline(
            switch_time,
            linestyle=":",
            linewidth=2,
            label="15 s switch",
        )

    if current_time is not None:
        ax.axvline(
            current_time,
            linewidth=2,
            alpha=0.6,
            label="Analysis position",
        )

    ax.set_ylim(
        0,
        1,
    )

    max_time = max(
        float(current_time or 0),
        float(
            times[-1]
            if times.size
            else 0
        ),
        1.0,
    )

    if switch_time is not None:
        max_time = max(
            max_time,
            switch_time,
        )

    ax.set_xlim(
        0,
        max(
            5.0,
            max_time + 1.0,
        ),
    )

    ax.set_xlabel(
        "Time (seconds)",
        color="#e5e7eb",
        fontsize=12,
        fontweight="bold",
    )

    ax.set_ylabel(
        "Spoof score (higher = more likely fake)",
        color="#e5e7eb",
        fontsize=11,
    )

    ax.tick_params(
        axis="both",
        colors="#cbd5e1",
        labelsize=10,
    )

    for spine in ax.spines.values():
        spine.set_color("#334155")

    max_time_for_ticks = float(
        ax.get_xlim()[1]
    )

    step = (
        5
        if max_time_for_ticks <= 60
        else 10
        if max_time_for_ticks <= 180
        else 30
    )

    ax.set_xticks(
        np.arange(
            0,
            max_time_for_ticks + 0.1,
            step,
        )
    )

    ax.grid(
        alpha=0.18,
        color="#94a3b8",
    )

    legend = ax.legend(
        loc="upper left",
        ncols=2,
        facecolor="#1f2937",
        edgecolor="#334155",
        fontsize=9,
    )

    for text in legend.get_texts():
        text.set_color("#e5e7eb")

    fig.tight_layout()

    return fig


def run_analysis(
    *,
    mode,
    ckpt_path,
    model_label,
    name,
    raw_bytes,
    duration_s,
    windows_count,
    amber,
    red,
    replay_file=None,
):
    """Stream scores and render timeline + risk band.

    mode:
        real   -> trained model
        replay -> precomputed score JSON
        mock   -> deterministic mock stream
    """

    tmp_path = None

    if mode == "real":
        suffix = (
            Path(name).suffix.lower()
            or ".wav"
        )

        tmp = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix,
        )

        tmp.write(raw_bytes)
        tmp.close()

        tmp_path = tmp.name

    status = st.empty()
    metrics = st.empty()
    risk_box = st.empty()
    chart = st.empty()

    # Load the real model only when needed.
    if mode == "real":
        status.info(
            f"Loading the {model_label.lower()} model — "
            "the first run takes a few seconds…"
        )

        try:
            with st.spinner(
                f"Loading {model_label.lower()} model "
                "(cached after this)…"
            ):
                _ensure_model(
                    ckpt_path
                )

        except Exception as exc:
            status.error(
                f"{model_label} model failed to load: {exc}"
            )

            if tmp_path:
                Path(tmp_path).unlink(
                    missing_ok=True
                )

            return

        status.success(
            f"{model_label} model ready — streaming scores…"
        )

    # Replay validation.
    if mode == "replay":
        if replay_file is None:
            st.error(
                "No replay score file selected."
            )
            return

        try:
            with open(
                replay_file,
                "r",
                encoding="utf-8",
            ) as f:
                replay_payload = json.load(f)

            replay_scores = replay_payload.get(
                "scores"
            )

            if not isinstance(
                replay_scores,
                list,
            ):
                raise ValueError(
                    "Replay JSON must contain a 'scores' list."
                )

            if not replay_scores:
                raise ValueError(
                    "Replay JSON contains no scores."
                )

            replay_windows_count = len(
                replay_scores
            )

        except Exception as exc:
            st.error(
                f"Could not read replay score file: {exc}"
            )
            return

        # Replay length comes from the precomputed score list.
        windows_count = replay_windows_count

    # Reduce expensive matplotlib redraws on long clips.
    plot_every = (
        1
        if windows_count <= 40
        else 2
        if windows_count <= 120
        else 3
    )

    times: list[float] = []
    raw_scores: list[float] = []

    try:
        if mode == "real":
            score_iterator = real_score_stream(
                tmp_path,
                ckpt_path,
            )

        elif mode == "replay":
            score_iterator = replay_score_stream(
                str(replay_file)
            )

        else:
            score_iterator = fake_score_stream(
                windows_count,
                name,
            )

        for idx, score in score_iterator:

            # One score every 0.5 seconds.
            t = idx * 0.5

            if idx >= windows_count:
                break

            times.append(t)

            raw_scores.append(
                float(
                    np.clip(
                        score,
                        0.0,
                        1.0,
                    )
                )
            )

            # Existing SONIX smoothing logic.
            smoothed = moving_average(
                raw_scores,
                window=5,
            )

            # Existing SONIX hysteresis logic.
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

            band_name, band_action, border = band_info(
                current_band
            )

            # Risk display.
            risk_box.markdown(
                f"""
<div style="
border:4px solid {border};
border-radius:20px;
padding:24px;
text-align:center;
background:#111827;
">
<div style="
font-size:14px;
font-weight:700;
letter-spacing:2px;
color:#9ca3af;
">{model_label.upper()} MODEL</div>
<div style="
font-size:44px;
font-weight:900;
color:{border};
">{band_name}</div>
<div style="
font-size:19px;
font-weight:700;
color:#f3f4f6;
">{band_action}</div>
</div>
""",
                unsafe_allow_html=True,
            )

            status.info(
                f"{model_label} model · "
                f"window {idx + 1} / {windows_count} · "
                f"window start {t:.1f} s · "
                f"latest score {raw_scores[-1]:.3f}"
            )

            metrics.write(
                {
                    "model": model_label,
                    "mode": mode,
                    "scores_received": len(
                        raw_scores
                    ),
                    "windows_expected": windows_count,
                    "analysis_time": f"{t:.1f} s",
                    "latest_score": round(
                        raw_scores[-1],
                        4,
                    ),
                    "current_band": current_band,
                }
            )

            if (
                idx % plot_every == 0
                or (idx + 1) >= windows_count
            ):
                fig = plot_timeline(
                    times,
                    raw_scores,
                    amber,
                    red,
                    current_time=t,
                    switch_time=(
                        15.0
                        if (
                            "switch"
                            in name.lower()
                            and duration_s >= 15
                        )
                        else None
                    ),
                )

                chart.pyplot(
                    fig,
                    clear_figure=True,
                )

                plt.close(fig)

        status.success(
            f"{model_label} model: "
            "score stream ended. "
            "No new points will be added."
        )

    except Exception as exc:
        status.error(
            f"{model_label} model failed: {exc}"
        )

    finally:
        if tmp_path:
            try:
                Path(tmp_path).unlink(
                    missing_ok=True
                )
            except Exception:
                pass


# ---------------------------------------------------------------------
# SONIX UI
# ---------------------------------------------------------------------

st.markdown("# SONIX")

st.caption(
    "Real-time windowed voice-cloning risk detector — "
    "baseline vs codec-robust model"
)

st.divider()


# ---------------------------------------------------------------------
# Checkpoint availability
# ---------------------------------------------------------------------

try:
    from score_file import checkpoint_available

    baseline_ok = checkpoint_available(
        BASELINE_CKPT
    )

    augmented_ok = checkpoint_available(
        AUGMENTED_CKPT
    )

except Exception:
    baseline_ok = False
    augmented_ok = False


# ---------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------

with st.sidebar:
    st.header("Detection settings")

    amber = st.slider(
        "Amber threshold",
        0.05,
        0.90,
        0.10,
        0.01,
    )

    red = st.slider(
        "Red threshold",
        float(
            min(
                1.0,
                amber + 0.05,
            )
        ),
        1.00,
        0.90,
        0.01,
    )

    st.caption(
        "Defaults are the data-driven thresholds from "
        "the eval score distribution "
        "(amber 0.10, red 0.90)."
    )

    st.divider()

    mode_label = st.radio(
        "Scoring mode",
        [
            "Real model (live stream)",
            "Replay (precomputed)",
            "Mock streaming",
        ],
        index=(
            0
            if baseline_ok
            else 1
        ),
    )

    if mode_label.startswith("Real"):
        mode = "real"

    elif mode_label.startswith("Replay"):
        mode = "replay"

    else:
        mode = "mock"

    st.caption(
        "Real mode uses the trained model. "
        "Replay mode uses validated precomputed scores. "
        "Mock mode is only for UI testing."
    )

    st.divider()

    st.markdown("**Checkpoints**")

    st.markdown(
        "- Baseline `head.pt`: "
        f"{'✅ found' if baseline_ok else '❌ missing'}"
    )

    st.markdown(
        "- Augmented `head_aug.pt`: "
        f"{'✅ found' if augmented_ok else '❌ not trained yet'}"
    )


# ---------------------------------------------------------------------
# Audio / replay clip selection
# ---------------------------------------------------------------------

replay_file = None
selected_clip = None

if mode == "replay":

    replay_clips = get_replay_clips()

    if not replay_clips:
        st.warning(
            "No validated replay clips found. "
            "Add a matching audio file to demo_clips/ "
            "and score JSON to scores/."
        )

        st.stop()

    selected_clip = st.selectbox(
        "Demo clip",
        replay_clips,
        format_func=lambda clip: clip["name"],
    )

    # Automatically pair audio + score JSON.
    replay_file = selected_clip["score_path"]

    audio_path = selected_clip["audio_path"]

    name = audio_path.name
    raw_bytes = audio_path.read_bytes()

    st.caption(
        f"Validated clip: `{selected_clip['clip_name']}` · "
        f"{selected_clip['score_count']} precomputed scores"
    )

else:

    uploaded = st.file_uploader(
        "Choose a .wav or .flac call recording",
        type=["wav", "flac"],
    )

    if uploaded is None:
        st.info(
            "Upload an audio file to start. "
            "Then run the same clip through each model tab."
        )

        st.stop()

    name = uploaded.name
    raw_bytes = uploaded.getvalue()


# ---------------------------------------------------------------------
# Audio information
# ---------------------------------------------------------------------

try:
    audio, sr = load_audio_mono_16k(
        io.BytesIO(raw_bytes)
    )

except Exception as exc:
    st.error(
        f"Could not read the audio: {exc}"
    )

    st.stop()


st.audio(
    raw_bytes,
    format=(
        "audio/flac"
        if name.lower().endswith(".flac")
        else "audio/wav"
    ),
)


duration_s = len(audio) / sr

uploaded_windows_count = sum(
    1
    for _ in iter_windows(
        audio,
        sr,
    )
)


m1, m2, m3 = st.columns(3)

m1.metric(
    "Audio duration",
    f"{duration_s:.2f} s",
)

m2.metric(
    "4-second windows",
    str(uploaded_windows_count),
)

m3.metric(
    "Hop",
    "0.5 s",
)


if mode == "replay":
    st.info(
        "Validated demo clip selected. "
        "Its precomputed scores are replayed "
        "one window at a time at the same "
        "0.5 s cadence as the streaming pipeline."
    )

else:
    st.info(
        "Upload once, then run the same clip through "
        "each model tab and compare the timelines. "
        "Nothing is plotted until scores arrive — "
        "the model is consumed one window at a time, "
        "exactly like a live call."
    )


# ---------------------------------------------------------------------
# Model tabs
# ---------------------------------------------------------------------

tab_base, tab_aug = st.tabs(
    [
        "🛡️ Baseline model",
        "🧪 Augmented (codec-robust)",
    ]
)


with tab_base:

    st.markdown(
        "**Baseline head** — trained on clean "
        "ASVspoof-2019 LA embeddings. "
        "In-domain eval EER ≈ **1.49%** "
        "on unseen attacks."
    )

    if mode == "real" and not baseline_ok:

        st.warning(
            f"Baseline checkpoint not found. "
            f"Expected `{BASELINE_CKPT}`."
        )

    else:

        if st.button(
            "▶ Run baseline analysis",
            type="primary",
            use_container_width=True,
            key="run_base",
        ):

            run_analysis(
                mode=mode,
                ckpt_path=BASELINE_CKPT,
                model_label="Baseline",
                name=name,
                raw_bytes=raw_bytes,
                duration_s=duration_s,
                windows_count=uploaded_windows_count,
                amber=amber,
                red=red,
                replay_file=replay_file,
            )

        else:

            if mode == "replay":
                st.caption(
                    "Press Run to replay the selected "
                    "validated demo clip."
                )
            else:
                st.caption(
                    "Press Run to stream this clip "
                    "through the baseline model."
                )


with tab_aug:

    st.markdown(
        "**Augmented head** — trained on clean "
        "**+ G.711 phone-codec** embeddings, "
        "so it holds up on the compressed audio "
        "you get over real phone calls."
    )

    if mode == "real" and not augmented_ok:

        st.warning(
            "The augmented model `head_aug.pt` "
            "isn't trained yet."
        )

        st.markdown(
            "Train it — fast, head-only, "
            "on the cached codec embeddings:"
        )

        st.code(
            TRAIN_AUG_CMD,
            language="bash",
        )

        st.caption(
            "This tab lights up automatically once "
            "the checkpoint exists — just rerun the app."
        )

    else:

        if st.button(
            "▶ Run augmented analysis",
            type="primary",
            use_container_width=True,
            key="run_aug",
        ):

            run_analysis(
                mode=mode,
                ckpt_path=AUGMENTED_CKPT,
                model_label="Augmented",
                name=name,
                raw_bytes=raw_bytes,
                duration_s=duration_s,
                windows_count=uploaded_windows_count,
                amber=amber,
                red=red,
                replay_file=replay_file,
            )

        else:

            if mode == "replay":
                st.caption(
                    "Press Run to replay the selected "
                    "validated demo clip."
                )
            else:
                st.caption(
                    "Press Run to stream this clip "
                    "through the augmented model."
                )


# ---------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------

st.divider()

st.caption(
    "Real-time architecture: this UI consumes "
    "a score generator and never preloads the "
    "score list. The frozen wav2vec2 front-end "
    "is loaded once and shared; each tab only "
    "swaps in its own trained head, so baseline "
    "and augmented run back-to-back."
)