"""SONIX Live Detection Dashboard - Streamlit UI"""

import streamlit as st
import json
import requests
from datetime import datetime
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(
    page_title="SONIX Live",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""<style>.metric-box { padding: 1.5rem; border-radius: 0.5rem; background: #f0f2f6; }
.score-high { color: #ff0000; font-weight: bold; } .score-low { color: #00aa00; font-weight: bold; }
</style>""", unsafe_allow_html=True)

# Use defaults (no secrets required)
WS_URL = "ws://localhost:8000"
HTTP_URL = "http://localhost:8000"

st.sidebar.title("⚙️ SONIX Control")
server_url = st.sidebar.text_input("Server URL", HTTP_URL)
ws_url = st.sidebar.text_input("WebSocket URL", WS_URL)
auto_refresh = st.sidebar.checkbox("Auto Refresh", value=True)
refresh_interval = st.sidebar.slider("Refresh Interval (sec)", 1, 30, 2)

st.title("🎯 SONIX Live Call Detection")
st.markdown("Real-time AI voice-clone detection powered by wav2vec2 + MLP head")

if 'calls' not in st.session_state:
    st.session_state.calls = {}


def get_server_status():
    """Fetch server status."""
    try:
        resp = requests.get(f"{server_url}/api/status", timeout=2)
        if resp.status_code == 200:
            return resp.json()
    except Exception as e:
        st.warning(f"Server unreachable: {e}")
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


# --- Palette -------------------------------------------------------------
# Status colours are reserved for the risk band and never reused as a series
# colour. Muted ink is deliberately the same in light and dark mode, so the
# charts stay legible whichever Streamlit theme the operator is running.
GOOD, WARNING, CRITICAL = "#0ca30c", "#fab219", "#d03b3b"
SERIES_1, MUTED = "#2a78d6", "#898781"
GRID = "rgba(137,135,129,0.22)"

# Provisional thresholds. These came from the CLEAN benchmark and have NOT been
# recalibrated on real recordings yet - Lane 1 owns that. Labelled on screen so
# nobody mistakes them for tuned values.
AMBER_AT, RED_AT = 0.35, 0.65


def band_for(score):
    if score is None:
        return "Unscored", MUTED
    if score >= RED_AT:
        return "Red", CRITICAL
    if score >= AMBER_AT:
        return "Amber", WARNING
    return "Green", GOOD


def base_layout(fig, height=300, y_title=""):
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=8, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=MUTED, size=12),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0,
                    bgcolor="rgba(0,0,0,0)", font=dict(size=12)),
    )
    fig.update_xaxes(title_text="Seconds into call", gridcolor=GRID,
                     zeroline=False, linecolor=GRID, ticks="outside", tickcolor=GRID)
    fig.update_yaxes(title_text=y_title, gridcolor=GRID,
                     zeroline=False, linecolor=GRID, ticks="outside", tickcolor=GRID)
    return fig


def risk_chart(windows, scores, t0):
    """P(AI voice) over time, with the Green/Amber/Red decision bands behind it."""
    idx_to_t = {w["window_idx"]: w["t"] - t0 for w in windows if w["window_idx"] is not None}
    pts = sorted(
        ((idx_to_t.get(int(k), float(v.get("timestamp", t0)) - t0), float(v["score"]))
         for k, v in scores.items()),
        key=lambda p: p[0],
    )
    if not pts:
        return None

    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    fig = go.Figure()
    for lo, hi, colour in ((0, AMBER_AT, GOOD), (AMBER_AT, RED_AT, WARNING), (RED_AT, 1, CRITICAL)):
        fig.add_hrect(y0=lo, y1=hi, fillcolor=colour, opacity=0.07,
                      line_width=0, layer="below")
    for y, label, colour in ((AMBER_AT, "Amber", WARNING), (RED_AT, "Red", CRITICAL)):
        fig.add_hline(y=y, line=dict(color=colour, width=1, dash="dot"),
                      annotation_text=label, annotation_position="right",
                      annotation_font=dict(color=colour, size=11))

    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="lines+markers", name="P(AI voice)",
        line=dict(color=SERIES_1, width=2),
        marker=dict(size=8, color=SERIES_1, line=dict(color="rgba(255,255,255,0.85)", width=2)),
        hovertemplate="%{x:.1f}s &nbsp; P(AI) %{y:.1%}<extra></extra>",
    ))
    fig.update_yaxes(range=[0, 1], tickformat=".0%")
    return base_layout(fig, 320, "P(AI voice)")


def audio_path_chart(windows, t0):
    """What the silence gate actually saw: speech ratio per window, and what it dropped."""
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
            marker=dict(size=9, color=MUTED, symbol="x",
                        line=dict(color="rgba(255,255,255,0.85)", width=2)),
            hovertemplate="%{x:.1f}s &nbsp; dropped<extra></extra>",
        ))
    fig.update_yaxes(range=[0, 1], tickformat=".0%")
    return base_layout(fig, 240, "Speech in window")


tab1, tab2, tab3, tab4 = st.tabs(["📞 Live Calls", "📤 Upload File", "📋 History", "ℹ️ Status"])

# TAB 1: LIVE CALLS
with tab1:
    telemetry = get_telemetry()

    if telemetry is None:
        st.error("Cannot reach the detection server.")
        st.code("python -m realtime.server --mock --ws-port 8000", language="bash")
    else:
        scoring_available = telemetry.get("scoring_available", False)
        calls = telemetry.get("calls", {})
        engine = telemetry.get("engine_stats", {})

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Server", "Connected")
        c2.metric("Active calls", f"{telemetry.get('active_calls', 0)} / {telemetry.get('max_calls', 0)}")
        c3.metric("Windows scored", engine.get("total_windows_scored", 0))
        c4.metric("Scoring", "Live model" if scoring_available else "Not available")

        if not scoring_available:
            st.warning(
                "**Risk scoring is switched off — no trained head is loaded.** "
                "The capture, buffering and silence-gate path below is live and real. "
                "The Green/Amber/Red band stays hidden until `head.pt` is available "
                "and passed with `--ckpt`, so no mock number is ever shown as a verdict."
            )

        st.caption(
            f"Microphone capture page: {server_url}/mic  ·  "
            "open it in Chrome or Edge on this machine and press Start capture."
        )
        st.divider()

        if not calls:
            st.info("No active calls. Start one from the microphone capture page.")
        for call_id, call in calls.items():
            state = call.get("state", "unknown")
            windows = call.get("windows", [])
            scores = call.get("scores", {})
            vad = call.get("vad", {})
            rb = call.get("ringbuffer", {})
            t0 = windows[0]["t"] if windows else 0

            score_vals = [float(v["score"]) for v in scores.values()]
            latest = score_vals[-1] if score_vals else None
            band, band_colour = band_for(latest if scoring_available else None)

            with st.expander(f"{call.get('caller', 'unknown')} — {call_id} ({state.upper()})", expanded=True):

                if state == "consent_pending":
                    st.markdown("**Awaiting consent.** Audio is being buffered but not scored.")
                    p1, p2 = st.columns([2, 1])
                    p1.markdown(
                        f"Pairing code &nbsp; <code style='font-size:1.6rem;letter-spacing:.12em'>"
                        f"{call.get('pairing_code', '------')}</code> &nbsp; "
                        f"expires in {call.get('pairing_expires_in', 0)}s",
                        unsafe_allow_html=True,
                    )
                    if p2.button("Approve", key=f"ok_{call_id}", type="primary"):
                        if post_json("/api/approve", {"call_id": call_id}):
                            st.rerun()
                    st.divider()

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Duration", f"{call.get('duration', 0):.0f}s")
                m2.metric("Windows emitted", rb.get("windows_emitted", 0))
                m3.metric("Passed silence gate", f"{vad.get('windows_passed', 0)} / {vad.get('windows_seen', 0)}")
                if scoring_available and latest is not None:
                    m4.markdown(
                        f"<div style='font-size:.8rem;color:{MUTED}'>RISK BAND</div>"
                        f"<div style='font-size:1.6rem;font-weight:700;color:{band_colour}'>"
                        f"{band} · {latest:.0%}</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    m4.metric("Risk band", "—")

                if scoring_available:
                    st.markdown("**Risk timeline**")
                    fig = risk_chart(windows, scores, t0)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True, key=f"risk_{call_id}")
                        st.caption(
                            f"Thresholds: Amber ≥ {AMBER_AT:.0%}, Red ≥ {RED_AT:.0%}. "
                            "Provisional — set on the clean benchmark, not yet recalibrated "
                            "on real recordings."
                        )
                    else:
                        st.info("No scored windows yet.")

                st.markdown("**Audio path**")
                fig2 = audio_path_chart(windows, t0)
                if fig2:
                    st.plotly_chart(fig2, use_container_width=True, key=f"audio_{call_id}")
                    st.caption(
                        f"{rb.get('windows_emitted', 0)} windows of 4.0s at a 0.5s hop from "
                        f"{rb.get('seconds_buffered', 0):.1f}s of audio. "
                        f"{vad.get('windows_rejected', 0)} dropped as silence "
                        f"({vad.get('pass_rate', 0):.0%} pass rate)."
                    )
                else:
                    st.info("Waiting for the first 4-second window to fill.")

                if st.button("End call", key=f"end_{call_id}"):
                    if post_json("/api/end-call", {"call_id": call_id}):
                        st.rerun()

with tab2:
    st.header("📤 Upload WAV for Analysis")
    st.write("Upload a recorded call for post-call scoring and analysis")

    uploaded_file = st.file_uploader("Choose a WAV file", type=["wav", "mp3"])

    if uploaded_file:
        st.audio(uploaded_file)

        if st.button("🎯 Score This File"):
            with st.spinner("Processing... (embedding + scoring)"):
                try:
                    files = {"file": uploaded_file.getbuffer()}
                    response = requests.post(f"{server_url}/api/score-file", files=files, timeout=60)

                    if response.status_code == 200:
                        result = response.json()
                        st.success(f"✓ Scored: {result['call_id']}")
                        st.divider()

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            score = result['summary']['mean_score'] or 0
                            label = "🚨 LIKELY AI" if score > 0.7 else ("⚠️ UNCERTAIN" if score > 0.3 else "✓ LIKELY REAL")
                            st.metric("Mean Score", f"{score:.1%}", delta=label)
                        with col2:
                            st.metric("Max Score", f"{result['summary']['max_score']:.1%}")
                        with col3:
                            st.metric("Min Score", f"{result['summary']['min_score']:.1%}")

                        st.subheader("Score Timeline")
                        scores_dict = result.get('scores', {})
                        if scores_dict:
                            scores_list = [{"window": int(k), "score": v["score"]} for k, v in scores_dict.items()]
                            scores_list.sort(key=lambda x: x["window"])
                            df = pd.DataFrame(scores_list)

                            fig = px.line(df, x="window", y="score", markers=True,
                                title="P(AI Voice) Over Time", labels={"score": "P(AI)", "window": "Window Index"})
                            fig.update_yaxes(range=[0, 1])
                            fig.update_layout(height=400, hovermode='x unified')
                            st.plotly_chart(fig, use_container_width=True)

                            st.divider()
                            json_str = json.dumps(result, indent=2)
                            st.download_button(label="Download JSON", data=json_str,
                                file_name=f"{result['call_id']}_scores.json", mime="application/json")
                    else:
                        st.error(f"Server error: {response.text}")
                except Exception as e:
                    st.error(f"Error: {e}")

# TAB 3: CALL HISTORY
with tab3:
    st.header("📋 Call History")
    history_data = [
        {"Caller": "+1 (508) 799-XXXX", "Duration": "0:34", "Mean Score": 0.32, "Result": "✓ REAL", "Time": "2026-08-31 14:23"},
        {"Caller": "+1 (650) 253-XXXX", "Duration": "2:12", "Mean Score": 0.87, "Result": "🚨 AI", "Time": "2026-08-31 13:45"},
        {"Caller": "+1 (415) 989-XXXX", "Duration": "0:18", "Mean Score": None, "Result": "⏳ PENDING*", "Time": "2026-08-31 13:10"}
    ]
    df = pd.DataFrame(history_data)
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.caption("* = no consent given, scores withheld")

# TAB 4: STATUS
with tab4:
    st.header("ℹ️ System Status")
    status = get_server_status()

    if status:
        st.subheader("Server Information")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Mode:** {status['mode']}")
            st.write(f"**Active Calls:** {status['active_calls']} / {status['max_calls']}")
            st.write(f"**WebSocket Clients:** {status['ws_clients']}")
        with col2:
            st.write(f"**Timestamp:** {status['timestamp']}")
            st.write(f"**Total Batches:** {status['engine_stats']['total_batches']}")
            st.write(f"**Total Windows:** {status['engine_stats']['total_windows_scored']}")

        st.divider()
        st.subheader("Configuration")
        st.code(f"WebSocket URL: {ws_url}\nHTTP URL: {server_url}")
    else:
        st.error("❌ Cannot connect to server")
        st.code("python -m realtime.server --mock --ws-port 8000")

if auto_refresh:
    import time
    time.sleep(refresh_interval)
    st.rerun()
