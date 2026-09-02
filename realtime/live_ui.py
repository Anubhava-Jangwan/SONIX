"""SONIX live detection dashboard - Streamlit UI.

Layout, in one paragraph: the live microphone panel IS the /mic page, inlined
flush with the tab (no card, no second frame), and it owns the live verdict and
the live chart because it gets scores over the WebSocket the instant they are
produced. Everything Streamlit draws around it - the call detail, the audio-path
chart, history - is polled telemetry and is therefore a couple of seconds behind;
those two clocks must never plot the same series, or the page shows one number
twice with a visible lag between them.

The poll runs inside an st.fragment. A full-page rerun would tear down and
remount the capture iframe, which ends the recording; a fragment reruns its own
block only, so capture survives every refresh.
"""

import json
import os
import sys
from collections import deque
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
import streamlit.components.v1 as components

# The smoothing and band rules the file-scoring flow already uses. demo/ is not a
# package, so it goes on the path rather than being copied in here: a live band
# and an uploaded-clip band must be the same decision made the same way.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "demo"))
# `streamlit run realtime/live_ui.py` puts realtime/ on the path, not the repo
# root, so `realtime.miccapture` would not import without this.
sys.path.insert(0, str(ROOT))
import theme  # noqa: E402
from risk import BAND_INFO, process_scores  # noqa: E402

from realtime.miccapture import EMBED_HEIGHT  # noqa: E402

st.set_page_config(
    page_title="SONIX Live",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(theme.page_css(), unsafe_allow_html=True)

# Env override so a second dashboard can be pointed at a second server without
# editing the sidebar by hand: SONIX_SERVER=http://localhost:8010 streamlit run ...
HTTP_URL = os.environ.get("SONIX_SERVER", "http://localhost:8000")
WS_URL = HTTP_URL.replace("http://", "ws://").replace("https://", "wss://")

# --- Palette -------------------------------------------------------------
# Status colours are reserved for the risk band and never reused as a series
# colour. One palette for the whole product: demo/theme.py.
GOOD, WARNING, CRITICAL = theme.GOOD, theme.WARN, theme.CRIT
SERIES_1, MUTED = theme.SERIES, theme.SERIES_MUTED

# Provisional thresholds. These came from the CLEAN benchmark and have NOT been
# recalibrated on real recordings yet - Lane 1 owns that. Labelled on screen so
# nobody mistakes them for tuned values. realtime/miccapture.py holds the same
# two numbers for the in-page band; they must not drift apart.
AMBER_AT, RED_AT = 0.35, 0.65

# How much of the timeline stays on screen while a call is live.
ROLLING_SEC = 60.0

# Toggle options are compared by identity, not by their leading character, so
# the labels can be reworded without breaking the branch below.
SRC_MIC = "Live microphone"
SRC_UPLOAD = "Upload a recording"

BAND_COLOUR = {"GREEN": GOOD, "AMBER": WARNING, "RED": CRITICAL}


with st.sidebar:
    st.markdown("### Connection")
    server_url = st.text_input("Server URL", HTTP_URL)
    ws_url = st.text_input("WebSocket URL", WS_URL)
    st.markdown("### Refresh")
    auto_refresh = st.checkbox("Auto refresh", value=True)
    refresh_interval = st.slider("Every (seconds)", 1, 30, 2)
    st.caption(
        "Only the call detail polls. The microphone panel is pushed over the "
        "WebSocket and updates the moment a window is scored."
    )


# --- Data ----------------------------------------------------------------
def get_server_status():
    try:
        resp = requests.get(f"{server_url}/api/status", timeout=2)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None


def get_telemetry(limit=240):
    """Per-call window telemetry: ring buffer, VAD decisions, scores."""
    try:
        resp = requests.get(f"{server_url}/api/telemetry", params={"limit": limit}, timeout=2)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None


def post_json(path, payload):
    try:
        return requests.post(f"{server_url}{path}", json=payload, timeout=3)
    except Exception as e:
        st.error(f"Request failed: {e}")
        return None


def ordered_scores(scores):
    """Raw per-window scores in window order (JSON gives us string keys)."""
    return [float(scores[k]["score"]) for k in sorted(scores, key=int)]


def smoothed_verdict(scores):
    """(smoothed series, band, recommended action) from the raw window scores.

    Runs demo/risk.py's moving average + hysteresis - the same path a clip
    uploaded through the demo goes down - so the band shown on a live call is
    not a second, quietly different rule.
    """
    raw = ordered_scores(scores)
    if not raw:
        return [], None, None
    smoothed, bands = process_scores(raw, AMBER_AT, RED_AT)
    band = bands[-1]
    return list(smoothed), band, BAND_INFO[band]["action"]


# --- Chrome --------------------------------------------------------------
def section(title, hint=""):
    """A section heading: one line of type, no box. Panels are made by the
    content sitting on the page ground, not by drawing frames around it."""
    st.markdown(
        f'<div style="margin:26px 0 10px;">'
        f'<span style="font-size:{theme.FS_CAPTION};font-weight:600;'
        f'letter-spacing:.09em;text-transform:uppercase;color:{theme.INK_3};">'
        f'{title}</span>'
        + (f'<span style="font-size:{theme.FS_CAPTION};color:{theme.INK_3};'
           f'margin-left:10px;letter-spacing:0;text-transform:none;">{hint}</span>'
           if hint else "")
        + "</div>",
        unsafe_allow_html=True,
    )


def base_layout(fig, height=300, y_title=""):
    """Single chart chrome, shared with the matplotlib timeline in demo/app.py."""
    return theme.plotly_layout(fig, height=height, y_title=y_title)


def live_badge():
    """A metric-styled card with a pulsing dot, in place of a plain 'Live' text
    metric - the one number on the page that is actually live now looks it."""
    return (
        f'<div style="border:{theme.BORDER};border-radius:{theme.RADIUS};'
        f'padding:{theme.PAD_SM} {theme.PAD};background:{theme.SURFACE};">'
        f'<div style="font-size:{theme.FS_CAPTION};letter-spacing:.08em;'
        f'text-transform:uppercase;color:{theme.INK_3};">Connection</div>'
        f'<div style="display:flex;align-items:center;gap:7px;margin-top:3px;'
        f'font-size:22px;font-weight:650;color:{theme.INK};">'
        f'<span style="width:8px;height:8px;border-radius:50%;background:{theme.GOOD};'
        f'display:inline-block;animation:sonix-pulse 1.8s {theme.EASE} infinite;"></span>'
        f'Live</div></div>'
    )


def risk_chart(windows, scores, t0, rolling_sec=ROLLING_SEC):
    """P(AI voice) over time for a NON-microphone call (a phone leg).

    Microphone calls do not use this: their chart lives in the capture panel,
    where it is pushed over the socket instead of polled, so it does not lag.
    """
    idx_to_t = {w["window_idx"]: w["t"] - t0 for w in windows if w["window_idx"] is not None}
    pts = sorted(
        ((idx_to_t.get(int(k), float(v.get("timestamp", t0)) - t0), float(v["score"]))
         for k, v in scores.items()),
        key=lambda p: p[0],
    )

    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    fig = go.Figure()
    # Thresholds are quiet dashed rules. Full-height tinted bands coloured 100%
    # of the plot to say one thing, and made every real signal harder to read.
    for y, label, colour in ((AMBER_AT, "Amber", WARNING), (RED_AT, "Red", CRITICAL)):
        fig.add_hline(y=y, line=dict(color=colour, width=1, dash="dot"),
                      annotation_text=label, annotation_position="right",
                      annotation_font=dict(color=colour, size=11))

    if pts:
        smoothed, _bands = process_scores(ys, AMBER_AT, RED_AT)
        # Raw sits behind, thin and muted: it is the evidence, not the verdict.
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines+markers", name="Raw window score",
            line=dict(color=MUTED, width=1),
            marker=dict(size=4, color=MUTED),
            hovertemplate="%{x:.1f}s &nbsp; raw %{y:.1%}<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=list(xs), y=list(smoothed), mode="lines", name="Smoothed (5-window)",
            line=dict(color=SERIES_1, width=2.2),
            hovertemplate="%{x:.1f}s &nbsp; smoothed %{y:.1%}<extra></extra>",
        ))

    # Scroll once the call is longer than the visible span; before that, hold a
    # full-width empty frame so the axes do not rescale under the operator.
    t_end = max(xs[-1] if xs else 0.0, rolling_sec)
    fig.update_xaxes(range=[max(0.0, t_end - rolling_sec), t_end])
    fig.update_yaxes(range=[0, 1], tickformat=".0%")
    return base_layout(fig, 320, "P(AI voice)")


def audio_path_chart(windows, t0):
    """What the silence gate actually saw: speech ratio per window, and what it
    dropped. This is the one thing the capture panel does NOT show, which is why
    it is worth polling for."""
    if not windows:
        return None

    xs = [w["t"] - t0 for w in windows]
    ys = [w.get("speech_ratio", 0.0) for w in windows]
    rej = [(w["t"] - t0, w.get("speech_ratio", 0.0)) for w in windows if not w.get("vad_passed")]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="lines", name="Speech ratio",
        line=dict(color=SERIES_1, width=2),
        hovertemplate="%{x:.1f}s &nbsp; %{y:.0%} speech<extra></extra>",
    ))
    if rej:
        fig.add_trace(go.Scatter(
            x=[r[0] for r in rej], y=[r[1] for r in rej], mode="markers",
            name=f"Dropped as silence ({len(rej)})",
            marker=dict(size=8, color=MUTED, symbol="x",
                        line=dict(color=theme.INK_2, width=1.5)),
            hovertemplate="%{x:.1f}s &nbsp; dropped<extra></extra>",
        ))
    fig.update_yaxes(range=[0, 1], tickformat=".0%")
    return base_layout(fig, 240, "Speech in window")


# --- Panels --------------------------------------------------------------
def render_upload_panel(key: str):
    """Post-call scoring for a recorded clip. Takes a widget-key prefix because
    it renders in two places and Streamlit needs distinct widget ids."""
    # Checked upfront rather than only after scoring, so a clip is never sent
    # to a server that would only return mock numbers for it - the button
    # itself says why before anyone wastes the upload.
    up_tel = get_telemetry(limit=1)
    up_scoring = bool(up_tel and up_tel.get("scoring_available"))
    up_synth = bool(up_tel and up_tel.get("scoring_synthetic"))
    if up_tel is None:
        st.error("Cannot reach the detection server.")
    elif not up_scoring:
        st.warning(
            "**Scoring is switched off — no trained head is loaded.** The decode, "
            "windowing and silence-gate path still runs, but the server would only "
            "return mock numbers, so scoring is disabled here. Start the server "
            "with `--ckpt outputs/models/head.pt` to enable it."
        )
    elif up_synth:
        st.warning("Untrained dev checkpoint loaded — every score below is noise.")

    uploaded_file = st.file_uploader("WAV or MP3 recording", type=["wav", "mp3"],
                                     key=f"{key}_uploader")
    if not uploaded_file:
        st.caption("Post-call scoring: the whole clip is windowed and scored server-side.")
        return

    st.audio(uploaded_file)
    if not st.button("Score this file", type="primary", key=f"{key}_score",
                     disabled=not up_scoring):
        return

    with st.spinner("Embedding and scoring..."):
        try:
            files = {"file": uploaded_file.getvalue()}
            response = requests.post(f"{server_url}/api/score-file", files=files, timeout=120)
        except Exception as e:
            st.error(f"Error: {e}")
            return

    if response.status_code != 200:
        st.error(f"Server error {response.status_code}: {response.text}")
        return

    result = response.json()
    scores_dict = result.get("scores", {})

    section("Result", result["call_id"])
    if scores_dict:
        smoothed, band, action = smoothed_verdict(scores_dict)
        st.markdown(
            theme.risk_card(f"{band.title()} · {smoothed[-1]:.0%}", action,
                            BAND_COLOUR[band], eyebrow="Risk band"),
            unsafe_allow_html=True,
        )
        # History tab reads this - a session-scoped ring of the last 5 scored
        # uploads, newest first. Not a database: this is a demo dashboard, and
        # a browser-session list is what "recently scored" needs to mean here.
        history = st.session_state.setdefault("upload_history", deque(maxlen=5))
        history.append({
            "filename": uploaded_file.name,
            "call_id": result["call_id"],
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "mean_score": float(smoothed[-1]),
            "band": band,
            "windows_scored": len(scores_dict),
        })

    c1, c2, c3 = st.columns(3)
    c1.metric("Mean score", f"{(result['summary']['mean_score'] or 0):.1%}")
    c2.metric("Max score", f"{(result['summary']['max_score'] or 0):.1%}")
    c3.metric("Min score", f"{(result['summary']['min_score'] or 0):.1%}")
    if not scores_dict:
        st.caption(
            "No windows scored — the clip is shorter than the 4-second "
            "analysis window, or the whole thing was dropped as silence."
        )

    if scores_dict:
        section("Score timeline")
        rows = sorted(({"window": int(k), "score": v["score"]} for k, v in scores_dict.items()),
                      key=lambda r: r["window"])
        fig = px.line(pd.DataFrame(rows), x="window", y="score", markers=True)
        fig.update_traces(line=dict(color=SERIES_1, width=2.2),
                          marker=dict(size=4, color=SERIES_1))
        for y, colour in ((AMBER_AT, WARNING), (RED_AT, CRITICAL)):
            fig.add_hline(y=y, line=dict(color=colour, width=1, dash="dot"))
        base_layout(fig, 340, "P(AI voice)")
        fig.update_xaxes(title_text="Window index")
        fig.update_yaxes(range=[0, 1], tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True, key=f"{key}_timeline")

        st.download_button("Download JSON", data=json.dumps(result, indent=2),
                           file_name=f"{result['call_id']}_scores.json",
                           mime="application/json", key=f"{key}_dl")


def render_call_detail(call_id, call, scoring_available):
    """Everything the capture panel does not already show for one call."""
    state = call.get("state", "unknown")
    windows = call.get("windows", [])
    scores = call.get("scores", {})
    vad = call.get("vad", {})
    rb = call.get("ringbuffer", {})
    t0 = windows[0]["t"] if windows else 0
    is_mic = call_id.startswith("mic_")

    if state == "consent_pending":
        p1, p2 = st.columns([3, 1])
        p1.markdown(
            f"<div style='font-size:{theme.FS_CAPTION};letter-spacing:.09em;"
            f"text-transform:uppercase;color:{theme.INK_3};'>Pairing code</div>"
            f"<div style='font-family:{theme.MONO_STACK};font-size:1.7rem;"
            f"font-weight:700;letter-spacing:.16em;color:{theme.INK};'>"
            f"{call.get('pairing_code', '------')}</div>"
            f"<div style='font-size:{theme.FS_CAPTION};color:{theme.INK_3};'>"
            f"expires in {call.get('pairing_expires_in', 0)}s &mdash; audio is "
            f"buffered but not scored until you approve</div>",
            unsafe_allow_html=True,
        )
        p2.write("")
        if p2.button("Approve", key=f"ok_{call_id}", type="primary",
                     use_container_width=True):
            if post_json("/api/approve", {"call_id": call_id}):
                st.rerun()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Duration", f"{call.get('duration', 0):.0f}s")
    m2.metric("Windows emitted", rb.get("windows_emitted", 0))
    m3.metric("Passed silence gate",
              f"{vad.get('windows_passed', 0)} / {vad.get('windows_seen', 0)}")

    smoothed, band_key, band_action = smoothed_verdict(scores)
    if scoring_available and band_key:
        m4.markdown(
            theme.risk_card(f"{band_key.title()} · {smoothed[-1]:.0%}", band_action,
                            BAND_COLOUR[band_key], eyebrow="Risk band", compact=True),
            unsafe_allow_html=True,
        )
    else:
        m4.metric("Risk band", "—")

    dropped = call.get("backlog", {}).get("dropped", 0)
    if dropped:
        st.warning(
            f"Scoring fell behind real time — {dropped} oldest window(s) dropped "
            "so the timeline keeps up with the call."
        )

    # A microphone call already has a live, socket-pushed risk chart in the
    # panel above. Drawing a polled copy here would put the same series on
    # screen twice, seconds apart, which reads as a bug.
    if scoring_available and not is_mic:
        section("Risk timeline", f"last {ROLLING_SEC:.0f}s")
        st.plotly_chart(risk_chart(windows, scores, t0), use_container_width=True,
                        key=f"risk_{call_id}")

    section("Audio path", "what the silence gate saw")
    fig = audio_path_chart(windows, t0)
    if fig:
        st.plotly_chart(fig, use_container_width=True, key=f"audio_{call_id}")
        st.caption(
            f"{rb.get('windows_emitted', 0)} windows of 4.0s at a 0.5s hop from "
            f"{rb.get('seconds_buffered', 0):.1f}s of audio. "
            f"{vad.get('windows_rejected', 0)} dropped as silence "
            f"({vad.get('pass_rate', 0):.0%} pass rate)."
        )
    else:
        st.caption("Waiting for the first 4-second window to fill.")

    if st.button("End call", key=f"end_{call_id}"):
        if post_json("/api/end-call", {"call_id": call_id}):
            st.rerun()


# --- Page ----------------------------------------------------------------
st.markdown("# SONIX")
st.caption("Real-time AI voice-clone detection · frozen wav2vec2 XLS-R front-end + trained MLP head")

tab_live, tab_upload, tab_history, tab_status, tab_extension = st.tabs(
    ["🎙️ Live", "📤 Upload", "🕘 History", "⚙️ System", "🧩 Extension"]
)

with tab_live:
    source = st.radio("Audio source", [SRC_MIC, SRC_UPLOAD],
                      horizontal=True, label_visibility="collapsed", key="live_source")

    if source == SRC_MIC:
        # OUTSIDE the auto-refreshing fragment on purpose. The iframe holds the
        # open microphone, the AudioContext and the WebSocket; a rerun that
        # remounts it silently ends the recording. Unchanged args keep the same
        # DOM node, so it survives every fragment refresh below.
        components.iframe(f"{server_url}/mic?embed=1", height=EMBED_HEIGHT)
        st.caption(
            f"Microphone access needs a secure context — use localhost. "
            f"[Open the capture page in its own tab]({server_url}/mic) if the "
            f"panel is blocked."
        )
    else:
        render_upload_panel("home")

    @st.fragment(run_every=refresh_interval if auto_refresh else None)
    def live_detail():
        """Polled half of the tab. A fragment, not a full rerun: a full rerun
        would tear down the capture iframe above and end the recording."""
        telemetry = get_telemetry()

        if telemetry is None:
            st.error("Cannot reach the detection server.")
            st.code("python -m realtime.server --mock --ws-port 8000", language="bash")
            return

        scoring_available = telemetry.get("scoring_available", False)
        calls = telemetry.get("calls", {})
        engine = telemetry.get("engine_stats", {})

        section("Server")
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(live_badge(), unsafe_allow_html=True)
        c2.metric("Active calls",
                  f"{telemetry.get('active_calls', 0)} / {telemetry.get('max_calls', 0)}")
        c3.metric("Windows scored", engine.get("total_windows_scored", 0))
        c4.metric("Scoring", "Live model" if scoring_available else "Off")

        if not scoring_available:
            st.warning(
                "**Risk scoring is switched off — no trained head is loaded.** "
                "Capture, buffering and the silence gate below are live and real. "
                "The Green/Amber/Red band stays hidden until `head.pt` is available "
                "and passed with `--ckpt`, so no mock number is ever shown as a verdict."
            )

        if not calls:
            st.caption("No active calls. Press Start capture above to open one.")
            return

        for call_id, call in calls.items():
            section(f"{call.get('caller', 'unknown')} · {call_id}",
                    call.get("state", "unknown").upper())
            render_call_detail(call_id, call, scoring_available)

    live_detail()

with tab_upload:
    render_upload_panel("tab")

with tab_history:
    section("Upload history", "last 5 scored files, this browser session")
    history = list(st.session_state.get("upload_history", []))
    if not history:
        st.caption(
            "Nothing scored yet. Score a file from the Upload tab and it "
            "shows up here — most recent first."
        )
    else:
        for entry in reversed(history):
            c1, c2 = st.columns([3, 1])
            with c1:
                st.markdown(
                    f'<div style="padding-top:10px;">'
                    f'<div style="font-weight:600;color:{theme.INK};">{entry["filename"]}</div>'
                    f'<div style="font-size:{theme.FS_CAPTION};color:{theme.INK_3};'
                    f'margin-top:2px;">{entry["call_id"]} · {entry["timestamp"]} · '
                    f'{entry["windows_scored"]} windows scored</div></div>',
                    unsafe_allow_html=True,
                )
            with c2:
                st.markdown(
                    theme.risk_card(
                        f'{entry["band"].title()} · {entry["mean_score"]:.0%}',
                        "", BAND_COLOUR[entry["band"]], compact=True,
                    ),
                    unsafe_allow_html=True,
                )
            st.divider()

with tab_status:
    section("System status")
    status = get_server_status()
    if not status:
        st.error("Cannot connect to server.")
        st.code("python -m realtime.server --mock --ws-port 8000", language="bash")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Mode", status["mode"])
        c2.metric("Active calls", f"{status['active_calls']} / {status['max_calls']}")
        c3.metric("WebSocket clients", status["ws_clients"])
        c1, c2, c3 = st.columns(3)
        c1.metric("Batches", status["engine_stats"]["total_batches"])
        c2.metric("Windows scored", status["engine_stats"]["total_windows_scored"])
        c3.metric("Server time", str(status["timestamp"])[11:19])
        section("Endpoints")
        st.code(f"HTTP      {server_url}\nWebSocket {ws_url}\nCapture   {server_url}/mic")

with tab_extension:
    section("Chrome extension", "live risk band, overlaid on a Google Meet call")
    st.markdown(
        "Captures the **other participants' audio** in a Google Meet tab — "
        "never your own microphone — and streams it to this same detection "
        "server, showing a live Green / Amber / Red band in the toolbar popup "
        "and as an overlay on the Meet page itself."
    )
    st.warning(
        "**Chrome only.** Manifest V3, requires Chrome 116 or newer. There is "
        "no Firefox or Safari build."
    )

    section("Install")
    st.markdown(
        f"""
1. Open `chrome://extensions` in Chrome
2. Turn on **Developer mode** (top right)
3. Click **Load unpacked**
4. Select the `extension/` folder inside `{ROOT.name}`

The SONIX icon appears in the toolbar — pin it so it stays visible during a call.
"""
    )

    section("Before you start")
    st.markdown(
        "The extension is a client only — this server does the scoring, so "
        "it must already be running (the same one this dashboard talks to):"
    )
    st.code(
        "python -m realtime.server --ckpt outputs/models/head.pt "
        "--ws-port 8000 --mode webrtc",
        language="bash",
    )
    st.caption(
        "Then join a Meet call, open the SONIX popup, and approve the pairing "
        "code from the Live tab here — same consent flow as the microphone panel."
    )
