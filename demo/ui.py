"""SONIX design system.

The idea this is built around: the product analyses a VOICE, so the voice is
the picture. The hero is the actual waveform of the uploaded clip, painted with
the risk band at each moment -- green where it reads as a real voice, red where
it reads as synthetic. You see the evidence and the verdict in one object
instead of an abstract line on an axis.

Separate from theme.py on purpose: realtime/live_ui.py imports that one.

Contrast, measured against the surface each colour actually sits on:
    GREEN 9.38  AMBER 10.80  RED 6.70  INK 16.38  INK_2 8.29  INK_3 4.71
"""

from __future__ import annotations

import numpy as np

# --- ground --------------------------------------------------------------
BG = "#0A0B10"
SURFACE = "#14161F"
RAISED = "#1B1E29"
EDGE = "#252938"
EDGE_C = "#3A4055"          # interactive control boundary

INK = "#F2F4F8"
INK_2 = "#A8B0C2"
INK_3 = "#7A8299"
ACCENT = "#7C8CFF"          # brand indigo
ACCENT_2 = "#22D3EE"        # cyan, second stop of the brand gradient

GREEN, AMBER, RED = "#34D399", "#FBBF24", "#FB7185"
BAND = {"GREEN": GREEN, "AMBER": AMBER, "RED": RED}

# One line colour per registered head in the compare chart. Must stay at least
# as long as realtime.models.REGISTRY -- a shorter list cycles, and two heads
# drawn in the same colour is a misread chart, not a cosmetic issue.
SERIES = [ACCENT, ACCENT_2, "#C084FC", "#34D399", "#FBBF24",
          "#FB7185", "#60A5FA", "#F472B6", "#A3E635"]
BAND_GLOW = {"GREEN": "52,211,153", "AMBER": "251,191,36", "RED": "251,113,133"}
BAND_ACTION = {
    "GREEN": "Consistent with a real voice. Proceed normally.",
    "AMBER": "Uncertain. Call back on a number you already have.",
    "RED": "Likely synthetic. Second-level approval required.",
}
BAND_MEANING = (("GREEN", "reads as a real voice"),
                ("AMBER", "uncertain"),
                ("RED", "reads as synthetic"))


def page_css() -> str:
    return f"""<style>
      @media (prefers-reduced-motion: reduce) {{
        *, *::before, *::after {{
          animation-duration:.001ms !important; transition-duration:.001ms !important;
        }}
      }}
      .stApp {{
        background:
          radial-gradient(900px 500px at 12% -8%, rgba(124,140,255,.10), transparent 60%),
          radial-gradient(760px 460px at 92% 0%, rgba(34,211,238,.07), transparent 62%),
          {BG};
      }}
      .block-container {{ max-width: 1180px; padding-top: 86px; }}

      /* Streamlit's header is 60px tall at z-index 999990 and would sit on top
         of the in-page nav. Collapse it to zero height and keep it transparent:
         its toolbar buttons still float top-right, the nav owns the bar. */
      header[data-testid="stHeader"] {{
        background: transparent; border: 0; height: 0; min-height: 0;
      }}


      /* in-page nav: one sticky bar, anchors to sections on this page */
      /* fixed, not sticky: Streamlit wraps every element, so the nav's parent
         is only as tall as the nav and sticky has nothing to stick inside.
         The scrolling ancestor is section.stMain, not the document. */
      /* The bar is translucent, so widgets scrolled under it stay faintly
         visible -- and a full-width fixed element eats every click in the top
         62px. That combination is a trap: you can see the file uploader's
         "Browse files" through the glass, aim at it, and the click lands on
         the nav instead, with no feedback. Only the brand and the links have
         any business receiving clicks; the bar itself passes them through. */
      .sx-nav {{
        position: fixed; top: 0; left: 0; right: 0; z-index: 999991;
        padding: 12px max(26px, calc((100vw - 1180px) / 2));
        display: flex; gap: 6px; align-items: center; flex-wrap: wrap;
        background: rgba(10,11,16,.82); backdrop-filter: blur(14px) saturate(140%);
        border-bottom: 1px solid {EDGE};
        pointer-events: none;
      }}
      .sx-nav a, .sx-nav .sx-brand {{ pointer-events: auto; }}
      .sx-nav .sx-brand {{
        font-family: 'Space Grotesk', system-ui, sans-serif; font-weight: 700;
        font-size: 17px; letter-spacing: -.02em; color: {INK}; margin-right: 14px;
      }}
      .sx-nav a {{
        font-size: 13.5px; font-weight: 500; color: {INK_2}; text-decoration: none;
        padding: 7px 13px; border-radius: 9px; border: 1px solid transparent;
        transition: color 140ms, background 140ms, border-color 140ms;
      }}
      .sx-nav a:hover {{
        color: {INK}; background: {RAISED}; border-color: {EDGE};
      }}
      .sx-nav a:focus-visible {{ outline: 2px solid {ACCENT}; outline-offset: 2px; }}
      /* the heading must clear the sticky bar when an anchor is jumped to */
      .sx-anchor {{ scroll-margin-top: 78px; }}
      .sx-rule {{ border: 0; border-top: 1px solid {EDGE}; margin: 40px 0 26px; }}

      /* type */
      h1, h2, h3 {{ letter-spacing: -.02em; }}
      .sx-display {{
        font-size: clamp(40px, 6vw, 68px); font-weight: 700; line-height: 1.02;
        letter-spacing: -.035em; color: {INK}; margin: 0;
      }}
      .sx-grad {{
        background: linear-gradient(96deg, {INK} 12%, {ACCENT} 58%, {ACCENT_2} 96%);
        -webkit-background-clip: text; background-clip: text; color: transparent;
      }}
      .sx-lede {{
        font-size: 18px; line-height: 1.6; color: {INK_2}; max-width: 52ch; margin-top: 14px;
      }}
      .sx-eyebrow {{
        font-size: 11px; font-weight: 600; letter-spacing: .16em;
        text-transform: uppercase; color: {INK_3};
      }}

      /* surfaces */
      .sx-card {{
        background: linear-gradient(180deg, {RAISED}, {SURFACE});
        border: 1px solid {EDGE}; border-radius: 16px; padding: 18px 20px;
        box-sizing: border-box;
      }}
      [data-testid="stHorizontalBlock"] {{ align-items: stretch; }}
      .sx-stat {{ font-size: 30px; font-weight: 700; color: {INK};
                  line-height: 1.12; margin-top: 6px;
                  font-feature-settings: "tnum" 1; }}
      .sx-sub {{ font-size: 12.5px; color: {INK_3}; margin-top: 5px; }}

      /* controls: config's borderColor is the quiet hairline; controls need 3:1 */
      [data-testid="stFileUploaderDropzone"],
      div[data-baseweb="select"] > div,
      .stTextInput input {{ border-color: {EDGE_C} !important; }}
      [data-testid="stFileUploaderDropzone"] {{
        background: {SURFACE}; border-style: dashed; border-radius: 14px;
      }}
      .stButton button {{
        border: 0; font-weight: 600; letter-spacing: .01em;
        background: linear-gradient(96deg, {ACCENT}, {ACCENT_2});
        color: #0A0B10;
        transition: transform 160ms cubic-bezier(.16,1,.3,1), box-shadow 160ms;
      }}
      .stButton button:hover {{
        transform: translateY(-1px); box-shadow: 0 8px 28px -10px rgba(124,140,255,.75);
      }}
      .stButton button:active {{ transform: translateY(0); }}
    </style>"""


def card(body: str, pad: str = "18px 20px") -> str:
    return f'<div class="sx-card" style="padding:{pad}">{body}</div>'


def stat(label: str, value: str, sub: str = "", tone: str = "") -> str:
    sub_html = f'<div class="sx-sub">{sub}</div>' if sub else ""
    return card(f'<div class="sx-eyebrow">{label}</div>'
                f'<div class="sx-stat" style="color:{tone or INK}">{value}</div>{sub_html}')


def verdict(band: str, score: float, model: str, share: float) -> str:
    """The call, at display size. Band is a word, a percentage and a sentence --
    never colour alone."""
    c, glow = BAND[band], BAND_GLOW[band]
    return f"""
    <div style="position:relative;overflow:hidden;border:1px solid {EDGE};
                border-radius:20px;padding:26px 28px;
                background:
                  radial-gradient(620px 200px at 0% 0%, rgba({glow},.16), transparent 70%),
                  linear-gradient(180deg,{RAISED},{SURFACE});">
      <div style="position:absolute;left:0;top:0;bottom:0;width:3px;background:{c}"></div>
      <div class="sx-eyebrow">{model} &middot; verdict</div>
      <div style="display:flex;align-items:flex-end;gap:26px;flex-wrap:wrap;margin-top:8px">
        <div style="font-size:60px;font-weight:700;letter-spacing:-.03em;
                    line-height:1;color:{c}">{band}</div>
        <div style="font-size:34px;font-weight:600;color:{INK};line-height:1.1;
                    font-feature-settings:'tnum' 1">{score:.1%}
          <div style="font-size:12px;color:{INK_3};font-weight:500;
                      letter-spacing:.1em;text-transform:uppercase">peak P(AI voice)</div>
        </div>
        <div style="font-size:34px;font-weight:600;color:{INK};line-height:1.1;
                    font-feature-settings:'tnum' 1">{share:.0%}
          <div style="font-size:12px;color:{INK_3};font-weight:500;
                      letter-spacing:.1em;text-transform:uppercase">of clip flagged</div>
        </div>
      </div>
      <div style="font-size:15px;color:{INK_2};margin-top:14px">{BAND_ACTION[band]}</div>
    </div>"""


def legend() -> str:
    cells = "".join(
        f'<span style="display:inline-flex;align-items:center;gap:7px;margin-right:22px">'
        f'<span style="width:22px;height:4px;border-radius:2px;background:{BAND[b]}"></span>'
        f'<b style="color:{INK};font-weight:600">{b}</b>'
        f'<span style="color:{INK_3}">{m}</span></span>'
        for b, m in BAND_MEANING)
    return f'<div style="font-size:12.5px;margin:12px 0 2px">{cells}</div>'


# --- the hero: waveform painted by risk ----------------------------------
def envelope(wav: np.ndarray, buckets: int = 900) -> np.ndarray:
    """Peak amplitude per bucket, normalised to 0..1. Peak, not RMS: RMS
    flattens speech into a smooth blob and loses the syllable structure that
    makes a waveform recognisable as a voice."""
    n = max(1, len(wav) // buckets)
    usable = len(wav) - (len(wav) % n)
    if usable <= 0:
        return np.zeros(1, dtype=float)
    env = np.abs(wav[:usable].reshape(-1, n)).max(axis=1).astype(float)
    peak = env.max()
    return env / peak if peak > 0 else env


def band_per_bucket(bands, n_buckets, dur, hop=0.5, win=4.0):
    """Map each waveform bucket to the band of the window covering it.

    Windows overlap (4 s window, 0.5 s hop), so a bucket sits under several.
    Take the most severe band covering it -- under-reporting risk on a detector
    is the wrong way to be wrong.
    """
    rank = {"GREEN": 0, "AMBER": 1, "RED": 2}
    order = ["GREEN", "AMBER", "RED"]
    out = np.zeros(n_buckets, dtype=int)
    if not bands:
        return out
    for i, b in enumerate(bands):
        t0, t1 = i * hop, i * hop + win
        lo = int(n_buckets * t0 / dur)
        hi = min(n_buckets, int(np.ceil(n_buckets * t1 / dur)))
        if hi > lo:
            out[lo:hi] = np.maximum(out[lo:hi], rank[b])
    return np.array([order[i] for i in out])


def waveform(wav, bands, dur, *, height=300):
    """Mirrored waveform, each contiguous band run drawn as its own filled
    shape. This is the picture: the voice, coloured where it stops sounding
    like one."""
    import plotly.graph_objects as go

    env = envelope(wav)
    n = len(env)
    per = band_per_bucket(bands, n, max(dur, 1e-6))
    t = np.linspace(0, dur, n)

    fig = go.Figure()
    start = 0
    for i in range(1, n + 1):
        if i == n or per[i] != per[start]:
            # overlap one bucket so neighbouring runs meet with no seam
            sl = slice(start, min(n, i + 1))
            b = per[start]
            fig.add_trace(go.Scatter(
                x=np.concatenate([t[sl], t[sl][::-1]]),
                y=np.concatenate([env[sl], -env[sl][::-1]]),
                fill="toself", mode="lines",
                line=dict(width=0), fillcolor=BAND[b],
                opacity=0.92, hoverinfo="skip", showlegend=False))
            start = i

    fig.update_layout(
        height=height, margin=dict(l=0, r=0, t=6, b=24),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=INK_3, size=11, family="Inter, system-ui, sans-serif"),
        showlegend=False)
    fig.update_xaxes(range=[0, dur], showgrid=False, zeroline=False,
                     linecolor=EDGE, ticks="outside", tickcolor=EDGE,
                     tickfont=dict(color=INK_3, size=11),
                     title_text="seconds", title_font=dict(color=INK_3, size=11))
    fig.update_yaxes(range=[-1.08, 1.08], showgrid=False, zeroline=False,
                     showticklabels=False, fixedrange=True)
    return fig


def timeline(times, raw, smoothed, amber, red, *, height=280):
    """The numbers behind the picture: per-window evidence and the smoothed
    series the band is actually derived from."""
    import plotly.graph_objects as go

    fig = go.Figure()
    for lo, hi, key in ((0, amber, "GREEN"), (amber, red, "AMBER"), (red, 1, "RED")):
        fig.add_hrect(y0=lo, y1=hi, fillcolor=BAND[key], opacity=0.05,
                      line_width=0, layer="below")
    for y, key in ((amber, "AMBER"), (red, "RED")):
        fig.add_hline(y=y, line=dict(color=BAND[key], width=1, dash="dot"),
                      annotation_text=f"{key} {y:.0%}", annotation_position="right",
                      annotation_font=dict(color=BAND[key], size=11))
    if len(raw):
        fig.add_trace(go.Scatter(
            x=list(times), y=list(raw), mode="markers", name="per-window",
            marker=dict(size=5, color=INK_3, opacity=0.5),
            hovertemplate="%{x:.1f}s &nbsp; raw %{y:.3f}<extra></extra>"))
    if len(smoothed):
        fig.add_trace(go.Scatter(
            x=list(times), y=list(smoothed), mode="lines", name="smoothed",
            line=dict(color=ACCENT, width=2.5, shape="spline"),
            hovertemplate="%{x:.1f}s &nbsp; <b>%{y:.0%}</b><extra></extra>"))

    span = max(5.0, (float(max(times)) + 1.0) if len(times) else 0.0)
    axis = dict(gridcolor=EDGE, zeroline=False, linecolor=EDGE, ticks="outside",
                tickcolor=EDGE, tickfont=dict(color=INK_3, size=11),
                title_font=dict(color=INK_3, size=11))
    fig.update_layout(
        height=height, margin=dict(l=4, r=76, t=8, b=8),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=INK_2, size=12, family="Inter, system-ui, sans-serif"),
        hovermode="x unified",
        hoverlabel=dict(bgcolor=RAISED, bordercolor=EDGE_C, font=dict(color=INK)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0,
                    bgcolor="rgba(0,0,0,0)", font=dict(size=11, color=INK_3)))
    fig.update_xaxes(title_text="seconds", range=[0, span], showgrid=False, **axis)
    fig.update_yaxes(title_text="P(AI voice)", range=[-0.02, 1.02],
                     tickformat=".0%", tickvals=[0.25, 0.5, 0.75], **axis)
    return fig


SECTIONS = (("overview", "Overview"), ("live", "Live mic"),
            ("analyze", "Analyze"), ("compare", "Compare"), ("method", "Method"))


def nav() -> str:
    """One sticky bar of in-page anchors. Not st.navigation -- that splits the
    app into separate pages; this is a single page you scroll."""
    links = "".join(f'<a href="#{slug}">{label}</a>' for slug, label in SECTIONS)
    return f'<div class="sx-nav"><span class="sx-brand">SONIX</span>{links}</div>'


def anchor(slug: str, title: str, kicker: str = "") -> str:
    k = f'<div class="sx-eyebrow">{kicker}</div>' if kicker else ""
    return (f'<div id="{slug}" class="sx-anchor"></div>{k}'
            f'<h2 style="margin:4px 0 2px">{title}</h2>')
