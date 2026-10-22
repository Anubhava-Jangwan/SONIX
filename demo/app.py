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
from core import (available_models, band_legend, bands_from,   # noqa: E402
                  clip_facts, clip_picker, default_index, defaults, REPO,
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
st.markdown('<div class="sx-lede">Continuous capture at 16 kHz, scored window '
            'by window as you speak — on this page, on this port. Change the '
            'head mid-sentence and the graph keeps going: the 300M front-end '
            'is frozen and shared, so a swap only reloads a ~300k-param MLP.'
            '</div>', unsafe_allow_html=True)
st.write("")

if not models:
    st.error("No trained head under outputs/models/ — scoring is unavailable.",
             icon=":material/error:")
else:
    @st.cache_resource
    def _mic():
        """One capture object for the whole server process. cache_resource
        (not session_state) because the threads must outlive any single
        rerun — session_state is rebuilt far too eagerly for that."""
        from live import LiveMic
        return LiveMic()

    mic = _mic()

    # The controls live OUTSIDE the auto-rerunning fragment, deliberately.
    # A fragment with run_every replaces its DOM roughly once a second, and a
    # real mouse click is not instantaneous -- press and release land either
    # side of a redraw, the button node is swapped in between, and the click
    # is never delivered. It looks exactly like a dead button, and it is only
    # intermittent, which is worse. Nothing here needs the timer: the mic
    # object lives in cache_resource, so a full rerun from one of these
    # widgets re-reads the same running capture and the graph carries on.
    c1, c2, c3 = st.columns([2, 3, 3])

    with c1:
        if mic.running:
            if st.button("Stop", use_container_width=True, key="live_stop"):
                mic.stop()
                st.rerun()
        else:
            if st.button("Start listening", type="primary",
                         use_container_width=True, key="live_start"):
                k = (st.session_state.get("live_head")
                     or list(models)[default_index(models)])
                mic.start(str(REPO / models[k][1]), k,
                          device=st.session_state.get("live_dev"))
                st.rerun()

    with c2:
        # Selecting a head only writes an attribute the scorer reads on its
        # next window. No restart, no cleared history.
        k = st.selectbox("Head", list(models), key="live_head",
                         index=default_index(models),
                         format_func=lambda x: models[x][0],
                         label_visibility="collapsed")
        if mic.running and k != mic.model_key:
            mic.ckpt, mic.model_key = str(REPO / models[k][1]), k

    with c3:
        if not mic.running:
            devs = mic.input_devices()
            if devs:
                names = dict(devs)
                st.selectbox("Input device", [i for i, _ in devs],
                             key="live_dev", format_func=lambda i: names[i],
                             label_visibility="collapsed",
                             help="A microphone records your own voice. A "
                                  "loopback device (Stereo Mix, What U Hear) "
                                  "records whoever is on the call — that is "
                                  "the side you want to screen for a clone.")
            else:
                st.warning("No recording device found.",
                           icon=":material/mic_off:")

    # Bands, outside the fragment for the same reason the buttons are: a
    # slider you drag while the DOM is being replaced under you is unusable.
    # They only re-derive bands from scores already computed, so moving one
    # repaints instantly and re-scores nothing.
    with st.expander("Band thresholds", expanded=False):
        st.caption("Where AMBER and RED begin. These re-read scores that are "
                   "already computed — nothing is sent through the model "
                   "again, and the graph repaints immediately.")
        threshold_controls("live_")

    # Only the readout is on the timer. Everything it shows already exists in
    # the deques before it runs, so a slow rerun delays the picture by a frame
    # -- it never leaves a hole in the series.
    @st.fragment(run_every=1.0)
    def live_panel():
        # --- is the mic actually hearing anything? -----------------------
        if mic.running:
            rms, _peak = mic.level()
            hearing = rms > 0.003            # the server's silence floor
            oc1, oc2 = st.columns([1, 2])
            with oc1:
                st.markdown(
                    f'<div style="display:flex;justify-content:center">'
                    f'{ui.siri_orb(mic.envelope(), rms, hearing)}</div>',
                    unsafe_allow_html=True)
            with oc2:
                st.markdown(
                    f'<div class="sx-eyebrow">Microphone</div>'
                    f'<div style="font-size:19px;font-weight:600;margin-top:4px;'
                    f'color:{ui.GREEN if hearing else ui.INK_3}">'
                    + ("Hearing you" if hearing else "Silent")
                    + '</div>'
                    f'<div style="font-size:12.5px;color:{ui.INK_2};'
                    f'margin-top:4px;line-height:1.5">'
                    + ("Audio is above the gate — these windows reach the model."
                       if hearing else
                       "Below the silence gate. Windows this quiet are skipped "
                       "before the model sees them, which is why the graph can "
                       "sit still while the mic is open.")
                    + '</div>' + ui.level_bar(rms),
                    unsafe_allow_html=True)
            st.write("")
        if mic.error:
            st.error(mic.error, icon=":material/error:")
        elif not mic.running:
            st.caption("Idle — nothing is being recorded.")
        elif mic.warming:
            st.caption("Loading the frozen 300M front-end — first start only, "
                       "then heads swap instantly.")
        elif mic.buffering() < 1.0:
            st.caption(f"Filling the first 4 s window… {mic.buffering():.0%}")
        else:
            st.caption(f"{mic.windows_scored} windows scored"
                       + (f" · {mic.windows_dropped} dropped to stay live"
                          if mic.windows_dropped else ""))

        times, raw, keys = mic.series()
        if not raw:
            st.info("Press **Start listening**, then speak for four seconds — "
                    "that is one window.", icon=":material/mic:")
            return

        amber, red = st.session_state["amber"], st.session_state["red"]
        smoothed, bands = bands_from(raw, amber, red)
        st.markdown(ui.verdict(bands[-1], float(smoothed[-1]),
                               models[keys[-1]][0] if keys[-1] in models else "—",
                               sum(b != "GREEN" for b in bands) / len(bands)),
                    unsafe_allow_html=True)
        st.write("")
        st.plotly_chart(ui.timeline(times, raw, smoothed, amber, red),
                        use_container_width=True, key="live_tl")
        band_legend()

    live_panel()

with st.expander("Record a fixed clip instead"):
    lc1, lc2 = st.columns([3, 2])
    with lc1:
        rec = st.audio_input("Record a few seconds of speech", key="mic")
    with lc2:
        live_key = st.selectbox("Head", list(models) if models else [],
                                key="mic_head",
                                index=default_index(models) if models else 0,
                                format_func=lambda k: models[k][0])
        st.caption("Speak for at least 4 seconds — that is one window. "
                   "Shorter clips get repeat-padded, which shifts the score.")

    if models and rec is not None:
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

st.markdown('<hr class="sx-rule">', unsafe_allow_html=True)

# ======================================================================
# 6 - CHROME EXTENSION
# ======================================================================
# Steps mirror extension/README.md. If that file changes, change this too --
# a wrong install step reads as a broken extension.
st.markdown(ui.anchor("extension", "Run it inside a Google Meet call",
                      "Chrome extension"), unsafe_allow_html=True)
st.markdown('<div class="sx-lede">The extension streams the Meet tab\'s audio '
            'to the local SONIX server and paints the same live graph over '
            'the call, in a side panel. It captures the <b>other</b> '
            'participants — the people calling you — not your microphone.'
            '</div>', unsafe_allow_html=True)
st.write("")

_ext_dir = REPO / "extension"

SERVER = "http://localhost:8000"


@st.cache_resource
def _http():
    """One pooled session for every poll on this page.

    A fresh requests.get() per poll opens a new TCP connection each time; at
    one poll every couple of seconds that is pure latency added to a render
    the user is waiting on."""
    import requests
    s = requests.Session()
    s.headers["Connection"] = "keep-alive"
    return s


def _get(path: str, timeout: float = 0.6):
    """GET from the detection server, or None.

    The timeout is deliberately short. This runs inside an auto-rerunning
    fragment, and Streamlit dims the whole page while the script is executing
    -- so a 2 s timeout against a server that is down meant the page spent
    every other second greyed out and unusable. Localhost answers in
    milliseconds or it is not answering at all.
    """
    try:
        return _http().get(f"{SERVER}{path}", timeout=timeout).json()
    except Exception:
        return None


@st.cache_data(ttl=3.0, show_spinner=False)
def _server_status():
    """Cached: this sits at page level and would otherwise re-probe the server
    on every single rerun, including every tick of the live-mic fragment."""
    return _get("/api/status", timeout=0.6)


_j = _server_status()
_srv_up = _j is not None
_srv_detail = (
    f"mode {_j.get('mode', '?')} · "
    f"scoring {'available' if _j.get('scoring_available') else 'OFF'} · "
    f"{_j.get('active_calls', 0)} active call(s)"
    if _srv_up else "Not reachable on http://localhost:8000"
)

_c1, _c2 = st.columns(2)
with _c1:
    st.markdown(ui.stat("Extension folder",
                        "present" if _ext_dir.exists() else "missing",
                        str(_ext_dir),
                        ui.GREEN if _ext_dir.exists() else ui.RED),
                unsafe_allow_html=True)
with _c2:
    st.markdown(ui.stat("Detection server",
                        "running" if _srv_up else "not running",
                        _srv_detail, ui.GREEN if _srv_up else ui.AMBER),
                unsafe_allow_html=True)
st.write("")

if not _srv_up:
    st.warning("The extension is only a client — with no server on port 8000 "
               "it captures audio and scores nothing. Start it first.",
               icon=":material/warning:")

st.markdown("**1 — Start the detection server** (separate terminal, kept running)")
st.code(f"cd {REPO}\n"
        "python -m realtime.server --ckpt outputs/models/head_v3.pt "
        "--ws-port 8000 --mode webrtc", language="bash")
st.caption("Swap `--ckpt` for any head under `outputs/models/`. Without a "
           "`--ckpt` the server runs but reports `scoring_available: false`, "
           "and the panel says so instead of inventing a number.")

st.markdown("**2 — Approve the call, right here.** The server refuses to "
            "buffer or score a single sample until the call's pairing code is "
            "approved; that gate is enforced server-side, not by the "
            "extension. This panel polls the server every two seconds, so a "
            "call shows up here the moment the extension starts one.")


def _approve(call_id: str) -> tuple[bool, str]:
    try:
        r = _http().post(f"{SERVER}/api/approve",
                         json={"call_id": call_id}, timeout=2)
        if r.status_code == 200:
            return True, ""
        return False, f"HTTP {r.status_code}: {r.text[:120]}"
    except Exception as exc:
        return False, str(exc)


@st.fragment(run_every=3.0)
def pending_calls():
    """Live list of calls waiting on consent, with an Approve button each.

    Replaces the operator dashboard that used to live in realtime/live_ui.py.
    Polled rather than pushed because this page has no socket to the server.
    Three seconds is well inside the 120 s a pairing code lives for, and every
    tick costs a render -- Streamlit greys the page out while the script runs,
    so polling faster makes the page feel worse, not more live.
    """
    data = _get("/api/telemetry", timeout=0.6)
    if data is None:
        st.warning("Detection server is not reachable on port 8000 — nothing "
                   "can be approved until it is running.",
                   icon=":material/cloud_off:")
        return
    calls = data.get("calls", {})

    waiting = {k: v for k, v in calls.items()
               if v.get("state") == "consent_pending"}
    running = {k: v for k, v in calls.items()
               if v.get("state") not in ("consent_pending", "ended")}

    if not waiting and not running:
        st.info("No call yet. Press **Start monitoring** in the extension and "
                "it will appear here within a few seconds.",
                icon=":material/hourglass_empty:")
        return

    for cid, c in waiting.items():
        code = c.get("pairing_code") or "------"
        left = int(c.get("pairing_expires_in") or 0)
        a, b = st.columns([3, 1])
        with a:
            st.markdown(
                f'<div class="sx-card" style="padding:14px 18px">'
                f'<div class="sx-eyebrow">{cid} &middot; waiting for consent</div>'
                f'<div style="font-size:34px;font-weight:700;letter-spacing:.14em;'
                f'font-feature-settings:\'tnum\' 1;color:{ui.AMBER};'
                f'line-height:1.2">{code}</div>'
                f'<div class="sx-sub">'
                + (f"expires in {left}s" if left > 0 else
                   "<b>expired</b> — press New code in the extension panel")
                + "</div></div>", unsafe_allow_html=True)
        with b:
            st.write("")
            if st.button("Approve", key=f"appr_{cid}", type="primary",
                         use_container_width=True, disabled=left <= 0):
                ok, err = _approve(cid)
                if ok:
                    # Deliberately NOT st.rerun(). A full rerun re-executes
                    # this whole page -- model discovery, the live-mic
                    # fragment, every chart -- and Streamlit blanks the view
                    # while it does, which is the dark flash. The next poll is
                    # at most three seconds away and repaints this row anyway.
                    st.success(f"Approved {cid} — scoring starts now.",
                               icon=":material/check_circle:")
                else:
                    st.error(f"Approve failed — {err}", icon=":material/error:")

    for cid, c in running.items():
        rb = c.get("ringbuffer") or {}
        st.success(
            f"**{cid}** — {c.get('state', '?').upper()} · head "
            f"`{c.get('model', '?')}` · {rb.get('windows_emitted', 0)} windows "
            f"emitted · {c.get('duration', 0):.0f}s",
            icon=":material/check_circle:")


pending_calls()

with st.expander("Skip the consent gate entirely (demo only)"):
    st.markdown("Start the server with `--auto-approve` and calls begin "
                "scoring the moment they connect — nothing to approve.")
    st.code(f"cd {REPO}\n"
            "python -m realtime.server --ckpt outputs/models/head_v3.pt "
            "--ws-port 8000 --mode webrtc --auto-approve", language="bash")
    st.caption("This bypasses a real safety feature. Fine for a demo you "
               "control; not something to leave on.")

st.markdown("**3 — Load the extension into Chrome**")
st.markdown(
    f"1. Open `chrome://extensions` (Chrome or Edge 116+)\n"
    f"2. Turn on **Developer mode** — top-right toggle\n"
    f"3. Click **Load unpacked**\n"
    f"4. Select this exact folder: `{_ext_dir}`\n"
    f"5. Pin the SONIX icon so it stays visible during the call")

st.markdown("**4 — Use it on a call**")
st.markdown(
    "1. Join a Google Meet call, then **reload the tab** — the content script "
    "only injects on page load, so an already-open Meet tab will not show the "
    "panel until you refresh it.\n"
    "2. **Tell the other participants they are being monitored.** The panel is "
    "visible on your screen only; Meet gives them no indication.\n"
    "3. Click the SONIX icon → **Start monitoring**.\n"
    "4. Approve the six-digit pairing code in the live server UI. Codes expire "
    "after 120 seconds.\n"
    "5. The side panel appears on the right with the verdict, the live graph "
    "and a head picker. Changing the head does not restart the call.\n"
    "6. **Stop monitoring** ends the call and writes the audit record to "
    "`outputs/calls/`.")

with st.expander("It did not work — the four usual causes"):
    st.markdown(
        "- **Panel never appears.** The content script injects only on "
        "`https://meet.google.com/*`, and only at page load. Reload the Meet "
        "tab after installing.\n"
        "- **\"Cannot reach the SONIX server.\"** Nothing is on port 8000. "
        "Open `http://localhost:8000/api/status` to check.\n"
        "- **Stuck on CONSENT_PENDING.** The pairing code was never approved, "
        "or it expired after 120 s. Approve it and start again.\n"
        "- **The meeting went silent.** Tab capture mutes the tab; "
        "`source.connect(ctx.destination)` in `offscreen.js` is what re-plays "
        "it. Inspect the offscreen document from `chrome://extensions` → "
        "SONIX → **Inspect views: offscreen.html**.")

st.caption("Tab audio is the mixed output of the call: with three people "
           "speaking you get all three in one stream, and a score applies to "
           "the 4-second window, not to a named participant. Per-speaker "
           "attribution needs diarisation, which SONIX does not do.")
