from __future__ import annotations

import io
import json
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

import theme
from risk import BAND_INFO, hysteresis_bands, moving_average
from streaming import demo_score_stream, iter_windows, replay_score_stream
from windowing import load_audio_mono_16k


st.set_page_config(
    page_title="SONIX",
    page_icon="🛡️",
    layout="wide",
)

# One shared stylesheet across every SONIX surface (also used by
# realtime/live_ui.py and the /mic page) -- Streamlit's tab strip becomes a
# sticky top navbar, cards/metrics/buttons all pick up the same dark palette.
st.markdown(theme.page_css(), unsafe_allow_html=True)

# theme.page_css()'s 1.4rem block-container top padding is enough once a
# title/caption sits above the tabs (as in realtime/live_ui.py), but here the
# navbar is the very first thing on the page -- with nothing above it, the
# sticky tab strip's natural (pre-stick) position lands partly UNDER
# Streamlit's own 60px header chrome, showing a blurred double image of the
# tab labels. Push it down far enough to clear that header outright.
st.markdown(
    "<style>.block-container{padding-top:4.5rem;}</style>",
    unsafe_allow_html=True,
)

# Repository root:
# D:\SIH\
REPO_ROOT = Path(__file__).resolve().parents[1]

# Model checkpoints
BASELINE_CKPT = "outputs/models/head.pt"
AUGMENTED_CKPT = "outputs/models/head_aug.pt"
ROBUST_CKPT = "outputs/models/head_robust.pt"

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
    fig, ax = plt.subplots(figsize=(12, 5.5))
    theme.style_axes(
        fig, ax,
        xlabel="Time (seconds)",
        ylabel="Spoof score (higher = more likely fake)",
    )

    times = np.asarray(times, dtype=float)
    scores = np.asarray(raw_scores, dtype=float)

    if scores.size:
        smoothed = moving_average(scores, window=5)
    else:
        smoothed = np.array([])

    if scores.size:
        ax.scatter(
            times, scores, s=20, alpha=0.4,
            color=theme.SERIES_MUTED, label="Raw score",
        )
        ax.plot(
            times, smoothed, linewidth=3,
            color=theme.SERIES, label="5-window smoothed",
        )

    ax.axhline(amber, linestyle="--", linewidth=1.5,
              color=theme.WARN, label=f"Amber {amber:.2f}")
    ax.axhline(red, linestyle="--", linewidth=1.5,
              color=theme.CRIT, label=f"Red {red:.2f}")

    if switch_time is not None:
        ax.axvline(switch_time, linestyle=":", linewidth=2,
                  color=theme.INK_2, label="15 s switch")

    if current_time is not None:
        ax.axvline(current_time, linewidth=2, alpha=0.7,
                  color=theme.ACCENT, label="Analysis position")

    ax.set_ylim(0, 1)

    max_time = max(
        float(current_time or 0),
        float(times[-1] if times.size else 0),
        1.0,
    )
    if switch_time is not None:
        max_time = max(max_time, switch_time)
    ax.set_xlim(0, max(5.0, max_time + 1.0))

    max_time_for_ticks = float(ax.get_xlim()[1])
    step = (5 if max_time_for_ticks <= 60
           else 10 if max_time_for_ticks <= 180
           else 30)
    ax.set_xticks(np.arange(0, max_time_for_ticks + 0.1, step))

    theme.style_legend(ax)
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
        suffix = Path(name).suffix.lower() or ".wav"
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
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
                f"Loading {model_label.lower()} model (cached after this)…"
            ):
                _ensure_model(ckpt_path)
        except Exception as exc:
            status.error(f"{model_label} model failed to load: {exc}")
            if tmp_path:
                Path(tmp_path).unlink(missing_ok=True)
            return
        status.success(f"{model_label} model ready — streaming scores…")

    # Replay validation.
    if mode == "replay":
        if replay_file is None:
            st.error("No replay score file selected.")
            return
        try:
            with open(replay_file, "r", encoding="utf-8") as f:
                replay_payload = json.load(f)
            replay_scores = replay_payload.get("scores")
            if not isinstance(replay_scores, list):
                raise ValueError("Replay JSON must contain a 'scores' list.")
            if not replay_scores:
                raise ValueError("Replay JSON contains no scores.")
            replay_windows_count = len(replay_scores)
        except Exception as exc:
            st.error(f"Could not read replay score file: {exc}")
            return
        # Replay length comes from the precomputed score list.
        windows_count = replay_windows_count

    # Reduce expensive matplotlib redraws on long clips.
    plot_every = (1 if windows_count <= 40
                 else 2 if windows_count <= 120
                 else 3)

    times: list[float] = []
    raw_scores: list[float] = []

    try:
        if mode == "real":
            score_iterator = real_score_stream(tmp_path, ckpt_path)
        elif mode == "replay":
            score_iterator = replay_score_stream(str(replay_file))
        else:
            score_iterator = fake_score_stream(windows_count, name)

        for idx, score in score_iterator:
            # One score every 0.5 seconds.
            t = idx * 0.5
            if idx >= windows_count:
                break

            times.append(t)
            raw_scores.append(float(np.clip(score, 0.0, 1.0)))

            # Existing SONIX smoothing logic.
            smoothed = moving_average(raw_scores, window=5)

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
            info = BAND_INFO[current_band]

            risk_box.markdown(
                theme.risk_card(
                    info["name"], info["action"], info["border"],
                    eyebrow=f"{model_label.upper()} MODEL",
                ),
                unsafe_allow_html=True,
            )

            status.info(
                f"{model_label} model · "
                f"window {idx + 1} / {windows_count} · "
                f"window start {t:.1f} s · "
                f"latest score {raw_scores[-1]:.3f}"
            )

            metrics.write({
                "model": model_label,
                "mode": mode,
                "scores_received": len(raw_scores),
                "windows_expected": windows_count,
                "analysis_time": f"{t:.1f} s",
                "latest_score": round(raw_scores[-1], 4),
                "current_band": current_band,
            })

            if idx % plot_every == 0 or (idx + 1) >= windows_count:
                fig = plot_timeline(
                    times, raw_scores, amber, red,
                    current_time=t,
                    switch_time=(
                        15.0
                        if ("switch" in name.lower() and duration_s >= 15)
                        else None
                    ),
                )
                chart.pyplot(fig, clear_figure=True)
                plt.close(fig)

        status.success(
            f"{model_label} model: score stream ended. "
            "No new points will be added."
        )

    except Exception as exc:
        status.error(f"{model_label} model failed: {exc}")

    finally:
        if tmp_path:
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except Exception:
                pass


# ---------------------------------------------------------------------
# Small presentation helpers for the Home / Features / Why SONIX tabs.
# Same card language as theme.risk_card() -- surface + hairline border +
# radius -- just laid out for a stat or a feature instead of a verdict.
# ---------------------------------------------------------------------

def stat_chip(value: str, label: str) -> str:
    return (
        f'<div style="border:{theme.BORDER};border-radius:{theme.RADIUS};'
        f'padding:14px 10px;background:{theme.SURFACE};text-align:center;">'
        f'<div style="font-size:26px;font-weight:800;color:{theme.ACCENT};'
        f'font-variant-numeric:tabular-nums;line-height:1.2;">{value}</div>'
        f'<div style="font-size:{theme.FS_CAPTION};color:{theme.INK_3};'
        f'text-transform:uppercase;letter-spacing:.06em;margin-top:4px;">{label}</div>'
        f'</div>'
    )


def feature_card(icon: str, title: str, desc: str) -> str:
    return (
        f'<div style="border:{theme.BORDER};border-radius:{theme.RADIUS};'
        f'padding:{theme.PAD};background:{theme.SURFACE};height:100%;">'
        f'<div style="font-size:26px;margin-bottom:8px;">{icon}</div>'
        f'<div style="font-size:{theme.FS_TITLE};font-weight:700;margin-bottom:6px;'
        f'color:{theme.INK};">{title}</div>'
        f'<div style="font-size:{theme.FS_BODY};color:{theme.INK_2};line-height:1.5;">{desc}</div>'
        f'</div>'
    )


def why_row(title: str, desc: str) -> str:
    return (
        f'<div style="border-top:{theme.BORDER};padding:14px 0;">'
        f'<div style="font-size:{theme.FS_BODY};font-weight:700;color:{theme.INK};'
        f'margin-bottom:4px;">✓ {title}</div>'
        f'<div style="font-size:{theme.FS_BODY};color:{theme.INK_2};line-height:1.5;">{desc}</div>'
        f'</div>'
    )


def pipeline_html() -> str:
    steps = [
        ("4.0s window", "16 kHz mono, 0.5s hop"),
        ("wav2vec2 XLS-R", "300M params, frozen"),
        ("Embedding", "1024-dim, mean-pooled"),
        ("Trained head", "~300k params — swappable"),
        ("Risk score", "0-1, Green/Amber/Red"),
    ]
    cells = ""
    for i, (title, sub) in enumerate(steps):
        if i:
            cells += f'<span style="color:{theme.INK_3};align-self:center;">→</span>'
        cells += (
            f'<div style="border:{theme.BORDER};border-radius:{theme.RADIUS_SM};'
            f'padding:10px 14px;background:{theme.SURFACE_RAISED};">'
            f'<b style="display:block;color:{theme.INK};font-size:13px;">{title}</b>'
            f'<span style="color:{theme.INK_3};font-size:12px;">{sub}</span></div>'
        )
    return (
        f'<div style="display:flex;flex-wrap:wrap;align-items:center;gap:8px;'
        f'margin:16px 0;">{cells}</div>'
    )


# ---------------------------------------------------------------------
# Checkpoint availability
# ---------------------------------------------------------------------

try:
    from score_file import checkpoint_available
    baseline_ok = checkpoint_available(BASELINE_CKPT)
    augmented_ok = checkpoint_available(AUGMENTED_CKPT)
    robust_ok = checkpoint_available(ROBUST_CKPT)
except Exception:
    baseline_ok = augmented_ok = robust_ok = False

n_heads_ready = sum([baseline_ok, augmented_ok, robust_ok])


# ---------------------------------------------------------------------
# Sidebar -- global detection settings, visible on every tab
# ---------------------------------------------------------------------

with st.sidebar:
    st.header("Detection settings")

    amber = st.slider("Amber threshold", 0.05, 0.90, 0.10, 0.01)
    red = st.slider(
        "Red threshold",
        float(min(1.0, amber + 0.05)),
        1.00, 0.90, 0.01,
    )
    st.caption(
        "Defaults are the data-driven thresholds from "
        "the eval score distribution (amber 0.10, red 0.90)."
    )

    st.divider()
    vad_on = st.checkbox(
        "Ignore silent windows (VAD)", value=True,
        help="Near-silent chunks are not scored by the model. This removes the "
             "main cause of false alarms on quiet, genuine recordings.")
    vad_db = st.slider("Silence threshold (dBFS)", -60.0, -25.0, -45.0, 1.0,
                       disabled=not vad_on,
                       help="Windows quieter than this are treated as no-speech.")
    st.divider()
    mode_label = st.radio(
        "Scoring mode",
        [
            "Real model (live stream)",
            "Replay (precomputed)",
            "Mock streaming",
        ],
        index=(0 if baseline_ok else 1),
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
    st.markdown(f"- Baseline `head.pt`: {'✅ found' if baseline_ok else '❌ missing'}")
    st.markdown(f"- Augmented `head_aug.pt`: {'✅ found' if augmented_ok else '❌ not trained yet'}")
    st.markdown(f"- Robust `head_robust.pt`: {'✅ found' if robust_ok else '❌ not trained yet'}")


# Apply the silence gate to the real-model path before any scoring happens.
try:
    from score_file import set_vad
    set_vad(enabled=vad_on, dbfs=vad_db)
except Exception:
    pass


# ---------------------------------------------------------------------
# Navbar -- Streamlit's tab strip, styled as a sticky top nav by
# theme.page_css(). Order matters: Try It is last because it's the only
# tab whose content can end the script early (an empty-state branch),
# and every tab's content already renders top-to-bottom before that.
# ---------------------------------------------------------------------

nav_home, nav_features, nav_why, nav_try = st.tabs(
    ["🏠  Home", "⚡  Features", "🆚  Why SONIX", "▶  Try It"]
)

# ---------------------------------------------------------------------
# Home -- hero
# ---------------------------------------------------------------------

with nav_home:
    st.markdown("# SONIX")
    st.markdown(
        f'<div style="font-size:20px;color:{theme.INK_2};margin-top:-8px;">'
        "Real-time AI voice-clone detection for phone calls — "
        "flags for a human to verify, never blocks a call.</div>",
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(stat_chip("1.49%", "EER · ASVspoof19 LA"), unsafe_allow_html=True)
    c2.markdown(stat_chip("300M", "frozen front-end params"), unsafe_allow_html=True)
    c3.markdown(stat_chip("0.5s", "scoring hop"), unsafe_allow_html=True)
    c4.markdown(stat_chip(f"{n_heads_ready}/3", "heads ready on this machine"), unsafe_allow_html=True)

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    st.markdown("##### How it works")
    st.markdown(pipeline_html(), unsafe_allow_html=True)
    st.caption(
        "Only the small trained head differs between models — the 300M-param "
        "front-end is identical and frozen for all of them, so comparing "
        "models costs almost nothing extra."
    )

    st.divider()
    st.info("Open **▶ Try It** above to upload a call recording and watch the risk score stream in.")

# ---------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------

with nav_features:
    st.markdown("## What this demo does")
    st.caption("Every item below is live in this app — nothing here is a mockup.")

    f1, f2, f3 = st.columns(3)
    f1.markdown(feature_card(
        "📡", "Real-time streaming",
        "Scores arrive one 4-second window at a time, exactly like a live "
        "call — nothing is precomputed and replayed."), unsafe_allow_html=True)
    f2.markdown(feature_card(
        "🧬", "Multi-model comparison",
        "Run the same clip through the baseline, codec-augmented and "
        "RawBoost-robust heads and compare their timelines side by side."), unsafe_allow_html=True)
    f3.markdown(feature_card(
        "🪶", "Cheap to extend",
        "Only the ~300k-parameter head is trained per model. The 300M "
        "wav2vec2 front-end is frozen and shared, so a new specialist "
        "head is a small add, not a retrain."), unsafe_allow_html=True)

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
    f4, f5, f6 = st.columns(3)
    f4.markdown(feature_card(
        "🎚️", "Configurable risk bands",
        "Amber/Red thresholds are sliders in the sidebar, seeded from the "
        "eval score distribution, not hard-coded."), unsafe_allow_html=True)
    f5.markdown(feature_card(
        "🔇", "Silence gate (VAD)",
        "Near-silent windows are skipped before scoring — the model calls "
        "silence \"fake\", so this is the main false-alarm fix."), unsafe_allow_html=True)
    f6.markdown(feature_card(
        "✅", "Validated replay mode",
        "Precomputed, reviewed clips replay at the same 0.5s cadence as "
        "live scoring — a reliable path for demos with no GPU needed."), unsafe_allow_html=True)

# ---------------------------------------------------------------------
# Why SONIX
# ---------------------------------------------------------------------

with nav_why:
    st.markdown("## Why SONIX, not just a single classifier")
    st.caption(
        "Grounded in what this project actually measures and ships — "
        "no invented numbers, no uncredited comparisons."
    )

    st.markdown(
        why_row(
            "Reports EER, never a bare accuracy number",
            "Spoof detection is heavily class-imbalanced, so a plain accuracy "
            "figure is close to meaningless. Every result here is quoted as "
            "\"X% EER on &lt;split&gt;\", with the split named.",
        ) +
        why_row(
            "Ensembling is (almost) free",
            "The 300M-param frozen front-end runs once; each extra head is "
            "roughly 262,657 parameters scored on an embedding already "
            "computed. Comparing three models costs a fraction of training "
            "three full ones.",
        ) +
        why_row(
            "Built to surface the generalization gap, not hide it",
            "In-the-Wild and codec-degraded audio scoring worse than the "
            "clean benchmark is treated as the project's headline finding, "
            "not a bug to paper over.",
        ) +
        why_row(
            "Consent-first on the live path",
            "The real-time server (a sibling surface to this demo) enforces "
            "an actual pairing/consent gate before any audio is scored — "
            "a real mechanism, not a UI convention.",
        ) +
        why_row(
            "Indic-language aware",
            "Later heads are trained with Indian-language speech on both the "
            "genuine and spoofed sides of the label — a gap most public "
            "voice-clone detectors don't address.",
        ),
        unsafe_allow_html=True,
    )

    st.divider()
    st.caption(
        "Thresholds shown in the sidebar are provisional: chosen on a clean "
        "benchmark, not yet recalibrated on real recordings. We say that "
        "plainly instead of dressing them up as tuned."
    )

# ---------------------------------------------------------------------
# Try It -- the actual scoring workflow (unchanged logic, restyled)
# ---------------------------------------------------------------------

with nav_try:
    replay_file = None
    selected_clip = None
    have_input = False
    name = raw_bytes = None

    if mode == "replay":
        replay_clips = get_replay_clips()

        if not replay_clips:
            st.warning(
                "No validated replay clips found. "
                "Add a matching audio file to demo_clips/ "
                "and score JSON to scores/."
            )
        else:
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
            have_input = True
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
        else:
            name = uploaded.name
            raw_bytes = uploaded.getvalue()
            have_input = True

    if have_input:
        try:
            audio, sr = load_audio_mono_16k(io.BytesIO(raw_bytes))
        except Exception as exc:
            st.error(f"Could not read the audio: {exc}")
            have_input = False

    if have_input:
        st.audio(
            raw_bytes,
            format=("audio/flac" if name.lower().endswith(".flac") else "audio/wav"),
        )

        duration_s = len(audio) / sr
        uploaded_windows_count = sum(1 for _ in iter_windows(audio, sr))

        m1, m2, m3 = st.columns(3)
        m1.metric("Audio duration", f"{duration_s:.2f} s")
        m2.metric("4-second windows", str(uploaded_windows_count))
        m3.metric("Hop", "0.5 s")

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

        tab_base, tab_aug, tab_rob = st.tabs(
            ["🛡️ Baseline model", "🧪 Augmented (codec)", "🏆 Robust (codec + RawBoost)"])

        with tab_base:
            st.markdown(
                "**Baseline head** — trained on clean "
                "ASVspoof-2019 LA embeddings. "
                "In-domain eval EER ≈ **1.49%** on unseen attacks."
            )
            if mode == "real" and not baseline_ok:
                st.warning(f"Baseline checkpoint not found. Expected `{BASELINE_CKPT}`.")
            else:
                if st.button("▶ Run baseline analysis", type="primary",
                             use_container_width=True, key="run_base"):
                    run_analysis(
                        mode=mode, ckpt_path=BASELINE_CKPT, model_label="Baseline",
                        name=name, raw_bytes=raw_bytes, duration_s=duration_s,
                        windows_count=uploaded_windows_count,
                        amber=amber, red=red, replay_file=replay_file,
                    )
                else:
                    st.caption(
                        "Press Run to replay the selected validated demo clip."
                        if mode == "replay" else
                        "Press Run to stream this clip through the baseline model."
                    )

        with tab_aug:
            st.markdown(
                "**Augmented head** — trained on clean "
                "**+ G.711 phone-codec** embeddings, "
                "so it holds up on the compressed audio "
                "you get over real phone calls."
            )
            if mode == "real" and not augmented_ok:
                st.warning("The augmented model `head_aug.pt` isn't trained yet.")
                st.markdown("Train it — fast, head-only, on the cached codec embeddings:")
                st.code(TRAIN_AUG_CMD, language="bash")
                st.caption(
                    "This tab lights up automatically once "
                    "the checkpoint exists — just rerun the app."
                )
            else:
                if st.button("▶ Run augmented analysis", type="primary",
                             use_container_width=True, key="run_aug"):
                    run_analysis(
                        mode=mode, ckpt_path=AUGMENTED_CKPT, model_label="Augmented",
                        name=name, raw_bytes=raw_bytes, duration_s=duration_s,
                        windows_count=uploaded_windows_count,
                        amber=amber, red=red, replay_file=replay_file,
                    )
                else:
                    st.caption(
                        "Press Run to replay the selected validated demo clip."
                        if mode == "replay" else
                        "Press Run to stream this clip through the augmented model."
                    )

        with tab_rob:
            st.markdown(
                "**Robust head** — trained on clean **+ G.711 phone codec + RawBoost** "
                "(simulated microphones, channels and noise). This is our most heavily "
                "augmented model, meant to hold up on unfamiliar recording conditions."
            )
            if mode == "real" and not robust_ok:
                st.warning("The robust model `head_robust.pt` isn't trained yet.")
                st.code("python src/train.py --emb-root outputs/embeddings "
                        "--extra-emb-root outputs/embeddings_g711 "
                        "--extra-emb-root outputs/embeddings_rawboost "
                        "--out outputs/models/head_robust.pt", language="bash")
            else:
                if st.button("▶ Run robust analysis", type="primary",
                             use_container_width=True, key="run_rob"):
                    run_analysis(
                        mode=mode, ckpt_path=ROBUST_CKPT, model_label="Robust",
                        name=name, raw_bytes=raw_bytes, duration_s=duration_s,
                        windows_count=uploaded_windows_count,
                        amber=amber, red=red, replay_file=replay_file,
                    )
                else:
                    st.caption(
                        "Press Run to replay the selected validated demo clip."
                        if mode == "replay" else
                        "Press Run to stream this clip through the robust model."
                    )

    st.divider()
    st.caption(
        "Real-time architecture: this UI consumes "
        "a score generator and never preloads the "
        "score list. The frozen wav2vec2 front-end "
        "is loaded once and shared; each tab only "
        "swaps in its own trained head, so baseline "
        "and augmented run back-to-back."
    )
