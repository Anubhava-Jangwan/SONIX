"""SONIX demo -- one page, scrolled.

    streamlit run demo/app.py

Sections are anchors on this page, not separate st.Pages. Theme lives in
.streamlit/config.toml; demo/ui.py adds only what config cannot express.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import streamlit as st

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import ui                                                      # noqa: E402
from core import (available_models, bands_from, clip_facts,    # noqa: E402
                  clip_picker, default_index, defaults, REPO,
                  short_clip_warning, threshold_controls)

st.set_page_config(page_title="SONIX", page_icon=":material/graphic_eq:",
                   layout="wide", initial_sidebar_state="collapsed")
st.markdown(ui.page_css(), unsafe_allow_html=True)
st.markdown(ui.nav(), unsafe_allow_html=True)

defaults()
models = available_models()


def score_clip(path, model_key, n_win, label=""):
    """Run one clip through one head, streaming progress. Returns raw scores."""
    from model_adapter import yugal_score_stream
    import score_file as S

    S.set_vad(enabled=True)
    raw, prog = [], st.progress(0.0, "loading front-end...")
    try:
        for _i, score in yugal_score_stream(str(path),
                                            ckpt_path=str(REPO / models[model_key][1])):
            raw.append(float(score))
            prog.progress(min(1.0, len(raw) / max(1, n_win)),
                          f"{label}window {len(raw)} / {n_win}")
    except Exception as exc:
        prog.empty(); st.exception(exc); return None
    prog.empty()
    return raw


def render_result(raw, path, model_label, key_prefix=""):
    """Verdict + painted waveform + stats + thresholds. Shared by the mic and
    upload sections so a recording and a file are read exactly the same way."""
    import score_file as S

    amber, red = st.session_state["amber"], st.session_state["red"]
    smoothed, bands = bands_from(raw, amber, red)
    worst = max(bands, key=lambda b: {"GREEN": 0, "AMBER": 1, "RED": 2}[b])
    flagged = sum(b != "GREEN" for b in bands)

    st.markdown(ui.verdict(worst, float(max(smoothed)), model_label,
                           flagged / len(bands)), unsafe_allow_html=True)
    st.write("")

    wav = S._load_audio(str(path))
    st.markdown('<div class="sx-eyebrow">Waveform, painted by band</div>',
                unsafe_allow_html=True)
    st.plotly_chart(ui.waveform(wav, bands, len(wav) / 16000),
                    use_container_width=True, config={"displayModeBar": False},
                    key=f"{key_prefix}wave")
    st.markdown(ui.legend(), unsafe_allow_html=True)

    m = st.columns(4)
    m[0].markdown(ui.stat("Windows", str(len(raw)), "4.0 s each, 0.5 s hop"),
                  unsafe_allow_html=True)
    m[1].markdown(ui.stat("Peak", f"{max(raw):.1%}", "highest single window"),
                  unsafe_allow_html=True)
    m[2].markdown(ui.stat("Mean", f"{smoothed.mean():.1%}", "smoothed"),
                  unsafe_allow_html=True)
    m[3].markdown(ui.stat("Flagged", f"{flagged}/{len(bands)}", "above amber",
                          ui.BAND[worst] if flagged else ui.GREEN),
                  unsafe_allow_html=True)

    with st.expander("Thresholds and the score timeline"):
        st.caption("These re-derive the bands from scores already computed. "
                   "Nothing is re-scored, so the waveform repaints immediately.")
        a2, r2 = threshold_controls(key_prefix)
        sm2, _ = bands_from(raw, a2, r2)
        st.plotly_chart(ui.timeline([i * 0.5 for i in range(len(raw))],
                                    raw, sm2, a2, r2),
                        use_container_width=True, key=f"{key_prefix}tl")


# ======================================================================
# 1 — OVERVIEW
# ======================================================================
st.markdown('<div id="overview" class="sx-anchor"></div>'
            '<div class="sx-eyebrow">SIH26104 &middot; Team SONIX</div>'
            '<h1 class="sx-display">A cloned voice<br>'
            '<span class="sx-grad">leaves a trace.</span></h1>',
            unsafe_allow_html=True)
st.markdown('<div class="sx-lede">SONIX scores a call continuously and raises a '
            'flag for a human to verify. It never blocks the call.</div>',
            unsafe_allow_html=True)
st.write("")

cols = st.columns(4)
for col, (l, v, s, tone) in zip(cols, [
        ("Eval EER", "1.49%", "ASVspoof-2019 LA, in-domain", ""),
        ("Front-end", "300M", "wav2vec2 XLS-R, frozen", ""),
        ("Latency", "0.5 s", "hop, on a 4.0 s window", ""),
        ("Heads", str(len(models)), "swappable, shared front-end",
         ui.GREEN if models else ui.AMBER)]):
    col.markdown(ui.stat(l, v, s, tone), unsafe_allow_html=True)

st.write("")
st.markdown("##### The signal chain")
for col, (n, title, body) in zip(st.columns(5), [
        ("01", "Capture", "16 kHz mono &middot; 4.0 s window &middot; 0.5 s hop"),
        ("02", "Front-end", "wav2vec2 XLS-R &middot; 300M &middot; frozen"),
        ("03", "Embedding", "1024-dim vector per window"),
        ("04", "Head", "~300k params &middot; trained &middot; swappable"),
        ("05", "Band", "smoothed, then 3-of-5 hysteresis")]):
    col.markdown(ui.card(
        f'<div style="font-family:JetBrains Mono,monospace;font-size:11px;'
        f'color:{ui.ACCENT};letter-spacing:.1em">{n}</div>'
        f'<div style="font-size:15px;font-weight:600;color:{ui.INK};'
        f'margin-top:8px">{title}</div>'
        f'<div style="font-size:12.5px;color:{ui.INK_3};line-height:1.55;'
        f'margin-top:5px">{body}</div>', pad="16px 18px"), unsafe_allow_html=True)

st.markdown('<hr class="sx-rule">', unsafe_allow_html=True)

# ======================================================================
# 2 — LIVE MIC
# ======================================================================
st.markdown(ui.anchor("live", "Speak into the mic", "Live capture"),
            unsafe_allow_html=True)
st.markdown('<div class="sx-lede">Record straight from the browser at 16 kHz '
            'and score it through the same path a call takes.</div>',
            unsafe_allow_html=True)
st.write("")

if not models:
    st.error("No trained head under outputs/models/ — scoring is unavailable.",
             icon=":material/error:")
else:
    lc1, lc2 = st.columns([3, 2])
    with lc1:
        rec = st.audio_input("Record a few seconds of speech", key="mic")
    with lc2:
        live_key = st.selectbox("Head", list(models), key="mic_head",
                                index=default_index(models),
                                format_func=lambda k: models[k][0])
        st.caption("Speak for at least 4 seconds — that is one window. "
                   "Shorter clips get repeat-padded, which shifts the score.")

    if rec is not None:
        import tempfile
        mic_path = Path(tempfile.gettempdir()) / "sonix_mic.wav"
        mic_path.write_bytes(rec.getbuffer())
        dur, n_win, _ = clip_facts(mic_path)
        short_clip_warning(dur, n_win)
        sig = f"mic::{len(rec.getbuffer())}::{live_key}"
        if st.button("Score this recording", type="primary",
                     use_container_width=True, key="mic_run"):
            got = score_clip(mic_path, live_key, n_win, "recording: ")
            if got:
                st.session_state["scores"][sig] = got
        if st.session_state["scores"].get(sig):
            render_result(st.session_state["scores"][sig], mic_path,
                          models[live_key][0], "mic_")

with st.expander("Continuous streaming over WebSocket (separate service)"):
    st.markdown(
        "`st.audio_input` above records then scores. The **continuous** path — "
        "a live call scored window-by-window as it arrives, with the consent "
        "gate and audit trail — runs as its own aiohttp service, not inside "
        "Streamlit:")
    st.code("python -m realtime.server --ws-port 8000 --mode webrtc\n"
            "# then open http://localhost:8000/mic", language="bash")
    try:
        import requests
        r = requests.get("http://localhost:8000/api/status", timeout=1.5)
        st.success(f"Server is up: {r.json()}", icon=":material/check_circle:")
    except Exception:
        st.caption("Not running right now — start it with the command above. "
                   "Its own dashboard is `streamlit run realtime/live_ui.py`.")

st.markdown('<hr class="sx-rule">', unsafe_allow_html=True)

# ======================================================================
# 3 - ANALYZE A FILE
# ======================================================================
st.markdown(ui.anchor("analyze", "Where does the voice stop being real?",
                      "Analyze a recording"), unsafe_allow_html=True)
st.markdown('<div class="sx-lede">Every 4-second window is scored on its own. '
            'The waveform is painted with the band at each moment, so the '
            'evidence and the verdict are the same object.</div>',
            unsafe_allow_html=True)
st.write("")

if models:
    name, path = clip_picker()
    if path is None:
        st.info("Upload a recording, or pick one bundled with the repo.",
                icon=":material/upload:")
    else:
        dur, n_win, sr = clip_facts(path)
        st.audio(str(path))
        key = st.selectbox("Model head", list(models), key="file_head",
                           index=default_index(models),
                           format_func=lambda k: models[k][0],
                           help="The 300M front-end is frozen and shared; only "
                                "this ~300k-param head changes.")
        st.caption(models[key][2])
        short_clip_warning(dur, n_win)

        cache_key = f"{name}::{key}"
        if st.button(f"Run {models[key][0].lower()} analysis", type="primary",
                     use_container_width=True, key="file_run"):
            got = score_clip(path, key, n_win)
            if got:
                st.session_state["scores"][cache_key] = got
        if st.session_state["scores"].get(cache_key):
            render_result(st.session_state["scores"][cache_key], path,
                          models[key][0], "file_")

st.markdown('<hr class="sx-rule">', unsafe_allow_html=True)

# ======================================================================
# 4 - COMPARE HEADS
# ======================================================================
st.markdown(ui.anchor("compare", "Same clip, different head", "Compare"),
            unsafe_allow_html=True)
st.markdown('<div class="sx-lede">The front-end is frozen and shared, so the '
            'only thing that differs between these is the ~300k-param head. '
            'That is what makes comparing detectors cheap.</div>',
            unsafe_allow_html=True)
st.write("")

_RANK = {"GREEN": 0, "AMBER": 1, "RED": 2}

if len(models) < 2:
    st.warning(f"Only {len(models)} head available - nothing to compare.",
               icon=":material/info:")
else:
    cname, cpath = clip_picker("cmp_")
    if cpath is None:
        st.info("Pick a recording to compare heads on.", icon=":material/upload:")
    else:
        cdur, cwin, _ = clip_facts(cpath)
        short_clip_warning(cdur, cwin)
        # The front-end is shared but not cached across heads: every head
        # selected here costs one more full pass of the 300M model. With nine
        # registered heads "run everything" is a several-minute button, so the
        # set is chosen explicitly and defaults to the three worth contrasting.
        pick = st.multiselect(
            "Heads to compare", list(models), key="cmp_heads",
            default=[k for k in ("v3", "robust_v2", "baseline") if k in models],
            format_func=lambda k: models[k][0],
            help="Each head adds a full pass of the frozen front-end.")

        if st.button(f"Run {len(pick)} head(s) on this clip", type="primary",
                     use_container_width=True, key="cmp_run",
                     disabled=not pick):
            for mk in pick:
                got = score_clip(cpath, mk, cwin, f"{models[mk][0]}: ")
                if got:
                    st.session_state["scores"][f"{cname}::{mk}"] = got

        have = {k: st.session_state["scores"].get(f"{cname}::{k}") for k in pick}
        have = {k: v for k, v in have.items() if v}
        if have:
            amber, red = st.session_state["amber"], st.session_state["red"]
            rows = list(have.items())
            for start in range(0, len(rows), 4):      # 4 per row, not 9 abreast
                for col, (mk, raw) in zip(st.columns(4), rows[start:start + 4]):
                    sm, bands = bands_from(raw, amber, red)
                    flagged = sum(b != "GREEN" for b in bands)
                    worst = max(bands, key=lambda b: _RANK[b])
                    col.markdown(ui.stat(models[mk][0], worst,
                                         f"peak {max(raw):.0%} &middot; "
                                         f"{flagged}/{len(bands)} flagged",
                                         ui.BAND[worst]), unsafe_allow_html=True)
            if len(have) > 1:
                import plotly.graph_objects as go
                fig = ui.timeline([], [], [], amber, red, height=320)
                pal = ui.SERIES
                for n, (mk, raw) in enumerate(have.items()):
                    sm, _ = bands_from(raw, amber, red)
                    fig.add_trace(go.Scatter(
                        x=[i * 0.5 for i in range(len(raw))], y=list(sm),
                        mode="lines", name=models[mk][0],
                        line=dict(color=pal[n % len(pal)], width=2.5,
                                  shape="spline"),
                        hovertemplate="%{x:.1f}s &nbsp; <b>%{y:.0%}</b>"
                                      "<extra></extra>"))
                widest = max(len(v) for v in have.values()) * 0.5 + 1
                fig.update_xaxes(range=[0, max(5.0, widest)])
                st.plotly_chart(fig, use_container_width=True, key="cmp_tl")
                st.caption("Smoothed series only - several sets of raw points "
                           "on one axis is noise, not evidence.")

st.markdown('<hr class="sx-rule">', unsafe_allow_html=True)

# ======================================================================
# 5 - METHOD
# ======================================================================
st.markdown(ui.anchor("method", "Method, and what it cannot do yet", "Method"),
            unsafe_allow_html=True)
_a, _b = st.columns(2)
_a.markdown(ui.card(
    f'<div style="font-size:15px;font-weight:600;color:{ui.INK}">'
    f'Why a frozen front-end</div>'
    f'<div style="font-size:14px;color:{ui.INK_2};line-height:1.65;margin-top:6px">'
    f'wav2vec2 XLS-R is 300M parameters and stays frozen. Only a ~300k-param '
    f'head is trained, so swapping a detector costs a few hundred kilobytes '
    f'and no GPU time.</div>'), unsafe_allow_html=True)
_b.markdown(ui.card(
    f'<div style="font-size:15px;font-weight:600;color:{ui.INK}">'
    f'Why bands, not a verdict</div>'
    f'<div style="font-size:14px;color:{ui.INK_2};line-height:1.65;margin-top:6px">'
    f'A single window is noisy. Scores are smoothed over 5 windows, then a '
    f'3-of-5 hysteresis rule gates the band so it cannot flicker mid-call.</div>'),
    unsafe_allow_html=True)

st.write("")
st.markdown("##### Known limits")
st.caption("Measured, not assumed.")
for _title, _tone, _body in [
    ("Domain transfer", ui.RED,
     "A spoof probe trained on English ASVspoof scores AUC 0.9992 in-domain "
     "and 0.5196 on Indic audio. 0.50 is chance."),
    ("What the embedding encodes", ui.AMBER,
     "The same vectors decode language at 99.33% (12-way, chance 8.33%) and "
     "corpus at 100%. Mean-pooling keeps the confounds and averages the "
     "transient vocoder artifacts away."),
    ("Held-out evidence", ui.AMBER,
     "sonix_real/ is 9 true matched real/clone pairs - a smoke test, not an "
     "evaluation set."),
]:
    st.markdown(ui.card(
        f'<div style="display:flex;gap:12px;align-items:flex-start">'
        f'<span style="width:7px;height:7px;border-radius:2px;background:{_tone};'
        f'margin-top:7px;flex:none"></span><div>'
        f'<div style="font-size:14.5px;font-weight:600;color:{ui.INK}">{_title}</div>'
        f'<div style="font-size:13.5px;color:{ui.INK_2};line-height:1.6;'
        f'margin-top:2px">{_body}</div></div></div>', pad="14px 18px"),
        unsafe_allow_html=True)
    st.write("")
