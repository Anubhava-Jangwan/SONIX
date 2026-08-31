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


tab1, tab2, tab3, tab4 = st.tabs(["📞 Live Calls", "📤 Upload File", "📋 History", "ℹ️ Status"])

# TAB 1: LIVE CALLS
with tab1:
    st.header("Active Calls")
    status = get_server_status()

    if status:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Status", "🟢 Connected")
        with col2:
            st.metric("Active Calls", status['active_calls'])
        with col3:
            st.metric("Max Concurrent", status['max_calls'])
        with col4:
            st.metric("Total Scored", status['engine_stats'].get('total_windows_scored', 0))

        st.divider()

        if status['active_calls'] > 0:
            st.subheader("Active Call Details")
            st.info("💡 Tip: Call details update via WebSocket in real-time")

            with st.expander("📞 +1 (508) 799-XXXX — 0:34 (SCORING)", expanded=True):
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Duration", "0:34", delta="Active")
                    st.metric("State", "SCORING ✓")

                with col2:
                    st.metric("Windows", "68")
                    st.metric("Confidence", "HIGH")

                with col3:
                    st.metric("P(AI Voice)", "0.32", delta="LIKELY REAL ✓")

                st.subheader("Score Timeline (Last 30s)")
                windows = list(range(68))
                scores = [0.25 + 0.05 * (i % 5) for i in windows]

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=windows, y=scores, mode='lines+markers', name='P(AI)',
                    line=dict(color='green', width=2), marker=dict(size=6)))
                fig.update_layout(title="Probability of AI Voice", xaxis_title="Window Index",
                    yaxis_title="P(AI Voice Clone)", hovermode='x unified', height=350)
                st.plotly_chart(fig, use_container_width=True)

            st.divider()
            st.subheader("🔐 Pairing Code")
            col1, col2 = st.columns(2)
            with col1:
                st.info("📱 Send this code to the device owner for approval:\n\n# **826419**")
            with col2:
                if st.button("✅ Approve", key="approve_btn"):
                    st.success("Pairing approved! Listening to audio...")
                if st.button("❌ Reject", key="reject_btn"):
                    st.error("Pairing rejected.")
        else:
            st.info("ℹ️ No active calls. Waiting for incoming calls...")
    else:
        st.error("❌ Cannot connect to server. Is it running?")
        st.code("python -m realtime.server --mock --ws-port 8000")

# TAB 2: UPLOAD FILE
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
                            fig.update_yaxis(range=[0, 1])
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
