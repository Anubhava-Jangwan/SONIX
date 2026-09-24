"""SONIX design tokens - the single source of truth for every colour in the UI.

Both Streamlit apps (`demo/app.py`, `realtime/live_ui.py`) and the plain-HTML
capture page (`realtime/miccapture.py`, via `css_vars()`) render from this
palette - nothing copies a hex value out of here by hand.

The Chrome extension (`extension/popup.css`) is deliberately NOT on this
palette: it draws over whatever call page the user is on, where a mismatched
panel would clash with the page. It keeps its own dark tokens, which happen to
already match this one.

Rules this palette encodes:

* Colour carries meaning and nothing else. GOOD / WARN / CRIT are reserved for
  the risk band and its threshold rules. Structure - panels, borders, headers,
  dividers - is neutral ink.
* Every risk colour clears 4.5:1 against SURFACE, so it survives a projector.
* The band is never colour-only: the name (GREEN / AMBER / RED) is always
  printed next to it, which is what keeps it readable in grayscale and for
  red-green colour vision deficiency.

Nothing in here knows about scores, thresholds or models. It is styling only.
"""

from __future__ import annotations

# --- Neutrals ------------------------------------------------------------
# Dark "control room" scheme: near-black page, a slightly raised panel colour,
# one hairline. Nothing on screen is saturated except the risk band, so the
# band is the only thing that can pull the eye - and it still reads clearly
# under a projector.
BG = "#0b1112"              # page background
SURFACE = "#121a1b"         # cards, panels, chart plotting area
SURFACE_RAISED = "#182223"  # hovered / nested surfaces
LINE = "#263334"            # decorative hairlines: panels, dividers, axis spines
# WCAG 2.1 1.4.11 wants 3:1 on the boundary of an INTERACTIVE control when that
# boundary is what identifies it. LINE is 1.35:1 on SURFACE -- fine for a quiet
# panel edge, not for a control you have to find. Measured, not guessed:
#   LINE  #263334 on SURFACE = 1.35:1   LINE_STRONG #5a6767 on SURFACE = 3.00:1
LINE_STRONG = "#5a6767"     # expanders, inputs, any focusable bordered control
INK = "#eef4f4"             # primary text
INK_2 = "#b3c1c1"           # secondary text, axis labels
INK_3 = "#7e8d8d"           # muted text, tick labels, de-emphasised series
ACCENT = "#3fc2ce"          # brand teal: primary action, the smoothed series

# --- Risk band (reserved) ------------------------------------------------
GOOD = "#39c26a"
WARN = "#e0a92b"
CRIT = "#f0685f"

BAND_COLOUR = {"GREEN": GOOD, "AMBER": WARN, "RED": CRIT}

# --- Chart tokens --------------------------------------------------------
SERIES = ACCENT             # the smoothed score - the line the band comes from
SERIES_MUTED = INK_3        # the raw per-window score - evidence, not verdict
GRID = "rgba(126,141,141,0.20)"
GRID_MPL = INK_3            # matplotlib takes the colour and alpha separately
GRID_ALPHA = 0.16

# --- Shape and rhythm ----------------------------------------------------
RADIUS = "10px"
RADIUS_SM = "7px"
PAD = "16px"
PAD_SM = "10px"
BORDER = f"1px solid {LINE}"
BORDER_CONTROL = f"1px solid {LINE_STRONG}"

FONT_STACK = (
    '"IBM Plex Sans", system-ui, -apple-system, "Segoe UI", Roboto, sans-serif'
)
MONO_STACK = "ui-monospace, SFMono-Regular, Menlo, monospace"

# Type scale. Deliberately small - three sizes plus a muted caption.
FS_DISPLAY = "40px"   # the band verdict, and only that
FS_TITLE = "17px"
FS_BODY = "13.5px"
FS_CAPTION = "11px"

# --- Motion ----------------------------------------------------------------
# This is a call-risk dashboard, not a wedding invite: motion best practice for
# a "serious" surface is quick, no overshoot, and only on things that can
# actually be interrupted (hover, a live value changing). EASE is a plain
# ease-out curve (decelerate, no bounce) - see .claude/skills/motion.
EASE = "cubic-bezier(.16,1,.3,1)"
DUR_FAST = "120ms"
DUR_MED = "220ms"
DUR_SLOW = "380ms"

PALETTE = {
    "BG": BG, "SURFACE": SURFACE, "SURFACE_RAISED": SURFACE_RAISED,
    "LINE": LINE, "INK": INK, "INK_2": INK_2, "INK_3": INK_3,
    "ACCENT": ACCENT, "GOOD": GOOD, "WARN": WARN, "CRIT": CRIT,
}


def css_vars() -> str:
    """`:root{--bg:...}` for the plain-HTML surfaces (the /mic page).

    Those pages are served outside Streamlit, so they used to carry a hand-copied
    hex list that drifted the moment this file changed. They import this instead.
    """
    pairs = {
        "bg": BG, "surface": SURFACE, "raised": SURFACE_RAISED, "line": LINE,
        "ink": INK, "ink-2": INK_2, "muted": INK_3, "accent": ACCENT,
        "good": GOOD, "warning": WARN, "critical": CRIT,
    }
    body = ";".join(f"--{k}:{v}" for k, v in pairs.items())
    return (
        f":root{{{body};--font:{FONT_STACK};--mono:{MONO_STACK};"
        f"--ease:{EASE};--dur-fast:{DUR_FAST};--dur-med:{DUR_MED};--dur-slow:{DUR_SLOW}}}"
    )


# --- Streamlit -----------------------------------------------------------
def page_css() -> str:
    """Global CSS for a Streamlit page. Chrome + motion only - no layout is moved."""
    return f"""<style>
      @media (prefers-reduced-motion: reduce) {{
        *, *::before, *::after {{ animation-duration: .001ms !important;
          transition-duration: .001ms !important; }}
      }}
      @keyframes sonix-in {{
        from {{ opacity: 0; transform: translateY(4px); }}
        to   {{ opacity: 1; transform: translateY(0); }}
      }}
      @keyframes sonix-pulse {{
        0%, 100% {{ opacity: 1; }}
        50%      {{ opacity: .45; }}
      }}

      .stApp {{ background: {BG}; }}
      .block-container {{
          max-width: 1180px;
          padding-top: 1.4rem;
          padding-bottom: 3rem;
      }}
      html, body, [class*="css"] {{ font-family: {FONT_STACK}; }}
      h1 {{ font-size: 26px !important; font-weight: 650; letter-spacing: -.01em; }}
      h2 {{ font-size: {FS_TITLE} !important; font-weight: 650; }}
      h3 {{ font-size: 15px !important; font-weight: 650; }}
      hr, [data-testid="stDivider"] {{ border-color: {LINE}; }}
      iframe {{ border: 0; display: block; }}

      /* --- Sticky top navbar ------------------------------------------
         Streamlit's tab strip becomes the navbar: pinned to the viewport top,
         full-bleed, with a sliding underline instead of the default static one. */
      div[data-testid="stTabs"] {{
          position: sticky; top: 0; z-index: 999;
          background: {BG}; padding-top: 2px;
      }}
      .stTabs [data-baseweb="tab-list"] {{
          gap: 4px; border-bottom: {BORDER};
      }}
      .stTabs [data-baseweb="tab"] {{
          font-size: {FS_BODY}; font-weight: 550; color: {INK_3};
          transition: color {DUR_FAST} {EASE};
          height: 42px;
      }}
      .stTabs [data-baseweb="tab"]:hover {{ color: {INK_2}; }}
      .stTabs [aria-selected="true"] {{ color: {INK}; }}
      .stTabs [data-baseweb="tab-highlight"] {{
          background-color: {ACCENT} !important;
          transition: left {DUR_MED} {EASE}, width {DUR_MED} {EASE} !important;
      }}
      .stTabs [data-baseweb="tab-border"] {{ background-color: {LINE} !important; }}

      /* --- Cards: hover lift + entrance fade --------------------------
         transform/opacity only, so this stays cheap on a scrolling page. */
      [data-testid="stMetric"] {{
          background: {SURFACE};
          border: {BORDER};
          border-radius: {RADIUS};
          padding: {PAD_SM} {PAD};
          transition: transform {DUR_FAST} {EASE}, border-color {DUR_FAST} {EASE};
      }}
      [data-testid="stMetric"]:hover {{
          transform: translateY(-1px);
          border-color: {ACCENT}55;
      }}
      [data-testid="stMetricLabel"] {{
          font-size: {FS_CAPTION} !important;
          letter-spacing: .08em;
          text-transform: uppercase;
          color: {INK_3} !important;
      }}
      [data-testid="stMetricValue"] {{
          font-size: 22px !important; font-weight: 650; color: {INK} !important;
      }}
      [data-testid="stSidebar"] {{
          background: {SURFACE}; border-right: {BORDER};
      }}
      div[data-testid="stExpander"] details {{
          background: {SURFACE};
          border: {BORDER_CONTROL} !important;
          border-radius: {RADIUS};
          transition: border-color {DUR_FAST} {EASE};
      }}
      div[data-testid="stExpander"] details:hover {{ border-color: {ACCENT}; }}

      .stButton button {{
          border-radius: {RADIUS_SM}; font-weight: 600;
          transition: transform {DUR_FAST} {EASE}, filter {DUR_FAST} {EASE};
      }}
      .stButton button:hover {{ transform: translateY(-1px); filter: brightness(1.08); }}
      .stButton button:active {{ transform: translateY(0); }}

      .stAlert {{ border-radius: {RADIUS_SM}; border: {BORDER}; }}
      code {{ font-family: {MONO_STACK}; color: {INK_2}; }}
      .sonix-caption {{ color: {INK_3}; font-size: {FS_CAPTION}; }}

    </style>"""


def risk_card(band_name: str, action: str, colour: str, eyebrow: str = "",
              compact: bool = False) -> str:
    """The verdict block. One quiet hairline plus a 3px accent edge - the colour
    lives in the word, not in a frame around it.

    compact=True is for narrow columns (the live dashboard's metric row), where
    the display size would otherwise wrap the band name onto two lines and make
    the row ragged."""
    size = "22px" if compact else FS_DISPLAY
    pad = f"{PAD_SM} 14px" if compact else f"{PAD} 20px"
    eyebrow_html = (
        f'<div style="font-size:{FS_CAPTION};font-weight:600;letter-spacing:.1em;'
        f'text-transform:uppercase;color:{INK_3};margin-bottom:4px;">{eyebrow}</div>'
        if eyebrow else ""
    )
    return (
        f'<div style="border:{BORDER};border-left:3px solid {colour};'
        f'border-radius:{RADIUS_SM};padding:{pad};background:{SURFACE};'
        f'transition:border-color {DUR_MED} {EASE};">'
        f'{eyebrow_html}'
        f'<div style="font-size:{size};font-weight:700;line-height:1.15;'
        f'white-space:nowrap;color:{colour};transition:color {DUR_MED} {EASE};">{band_name}</div>'
        f'<div style="font-size:{FS_BODY};color:{INK_2};margin-top:5px;'
        f'line-height:1.4;">{action}</div>'
        f'</div>'
    )


# --- matplotlib ----------------------------------------------------------
# 150 dpi: crisp on a HiDPI laptop screen and a projector alike, without the
# multi-MB PNGs 300 dpi would push through Streamlit's websocket on every
# rerun.
MPL_DPI = 150


def style_axes(fig, ax, *, xlabel: str = "", ylabel: str = "") -> None:
    """Apply the palette to a matplotlib figure. Grid sits behind the data,
    thin and low contrast; spines are a single neutral hairline."""
    fig.set_dpi(MPL_DPI)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(SURFACE)
    ax.set_xlabel(xlabel, color=INK_2, fontsize=11)
    ax.set_ylabel(ylabel, color=INK_2, fontsize=11)
    ax.tick_params(axis="both", colors=INK_3, labelsize=9.5, length=3, width=0.8)
    for side, spine in ax.spines.items():
        spine.set_color(LINE)
        spine.set_linewidth(0.8)
        if side in ("top", "right"):
            spine.set_visible(False)
    ax.grid(True, alpha=GRID_ALPHA, color=GRID_MPL, linewidth=0.7)
    ax.set_axisbelow(True)


def style_legend(ax, **kwargs):
    legend = ax.legend(
        loc="upper left", ncols=2, fontsize=9,
        facecolor=SURFACE, edgecolor=LINE, framealpha=0.92, **kwargs
    )
    legend.get_frame().set_linewidth(0.8)
    for text in legend.get_texts():
        text.set_color(INK_2)
    return legend


# --- plotly --------------------------------------------------------------
def plotly_layout(fig, height: int = 300, y_title: str = ""):
    """Same visual language as style_axes, for the live dashboard's charts."""
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=8, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=INK_3, size=12, family="IBM Plex Sans, system-ui, sans-serif"),
        hovermode="x unified",
        hoverlabel=dict(bgcolor=SURFACE, bordercolor=LINE, font=dict(color=INK)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0,
                    bgcolor="rgba(0,0,0,0)", font=dict(size=11, color=INK_3)),
        transition=dict(duration=200, easing="cubic-in-out"),
    )
    axis = dict(gridcolor=GRID, zeroline=False, linecolor=LINE,
                ticks="outside", tickcolor=LINE, tickfont=dict(color=INK_3, size=11),
                title_font=dict(color=INK_3, size=11))
    fig.update_xaxes(title_text="Seconds into call", **axis)
    fig.update_yaxes(title_text=y_title, **axis)
    return fig


# --- risk timeline -------------------------------------------------------
# The live dashboard (realtime/live_ui.py) draws its timeline in plotly with
# the decision bands shaded behind the series; demo/app.py drew the same data
# in matplotlib and looked like a different product. One definition here so
# the two surfaces cannot drift again -- realtime/live_ui.py still has its own
# copy (_risk_bands/upload_chart) and should be pointed at these next.

BAND_LEGEND = (
    (GOOD, "Green", "consistent with real voice"),
    (WARN, "Amber", "uncertain"),
    (CRIT, "Red", "likely synthetic"),
)


def band_word(score, amber, red):
    """Band name for a score. Colour is never the only signal -- this is the
    word that goes in the hover text and the readout beside it."""
    if score is None:
        return "-"
    return "Red" if score >= red else ("Amber" if score >= amber else "Green")


def risk_bands(fig, amber, red, overlay=None):
    """Green/Amber/Red shading + dotted threshold lines, 0-100% y-axis with
    gridlines only at 25/50/75."""
    for lo, hi, colour in ((0, amber, GOOD), (amber, red, WARN), (red, 1, CRIT)):
        fig.add_hrect(y0=lo, y1=hi, fillcolor=colour, opacity=0.07,
                      line_width=0, layer="below")
    for y, label, colour in ((amber, "Amber", WARN), (red, "Red", CRIT)):
        fig.add_hline(y=y, line=dict(color=colour, width=1, dash="dot"),
                      annotation_text=f"{label} {y:.0%}", annotation_position="right",
                      annotation_font=dict(color=colour, size=11))
    fig.update_yaxes(range=[-0.02, 1.02], tickformat=".0%",
                     tickvals=[0.25, 0.50, 0.75])
    fig.update_layout(transition=dict(duration=350, easing="cubic-in-out"))
    if overlay:
        fig.add_annotation(text=overlay, showarrow=False, xref="paper", yref="paper",
                           x=0.5, y=0.5, font=dict(size=20, color=CRIT),
                           bgcolor="rgba(18,26,27,0.90)", bordercolor=CRIT,
                           borderwidth=1, borderpad=10)
    return fig


def risk_timeline(times, raw, smoothed, amber, red, *, height=380,
                  current_time=None, switch_time=None, overlay=None):
    """Per-window risk over a clip: raw scores as evidence, the smoothed line
    as the verdict, decision bands behind both."""
    import plotly.graph_objects as go

    fig = go.Figure()
    risk_bands(fig, amber, red, overlay)
    if len(raw):
        fig.add_trace(go.Scatter(
            x=list(times), y=list(raw), mode="markers", name="Per-window score",
            customdata=[band_word(v, amber, red) for v in raw],
            marker=dict(size=6, color=SERIES_MUTED, opacity=0.65),
            hovertemplate="%{x:.1f}s &nbsp; raw %{y:.3f} - %{customdata}<extra></extra>",
        ))
    if len(smoothed):
        fig.add_trace(go.Scatter(
            x=list(times), y=list(smoothed), mode="lines",
            name="Smoothed (5-window mean)",
            customdata=[[f"{v:.3f}", band_word(v, amber, red)] for v in smoothed],
            line=dict(color=SERIES, width=2.5),
            hovertemplate=("%{x:.1f}s &nbsp; <b>%{y:.0%}</b> - %{customdata[1]}"
                           " &nbsp;(raw %{customdata[0]})<extra></extra>"),
        ))
    if switch_time is not None:
        fig.add_vline(x=switch_time, line=dict(color=INK_2, width=1.5, dash="dot"),
                      annotation_text="15 s switch", annotation_position="top",
                      annotation_font=dict(color=INK_2, size=11))
    if current_time is not None:
        fig.add_vline(x=current_time, line=dict(color=ACCENT, width=2))
    fig = plotly_layout(fig, height=height, y_title="P(AI voice)")
    # The Amber/Red threshold annotations anchor right, outside the plotting
    # area; plotly_layout()'s 8px right margin clips them to "A" and "Re".
    # Set after plotly_layout(), which replaces the whole margin dict.
    fig.update_layout(margin_r=64)
    # A single-window clip has one x value and plotly pads it out to -1..1, which
    # reads as negative time. Always start at 0 and show at least 5 s.
    span = max(5.0, float(max(times)) + 1.0 if len(times) else 0.0,
               float(current_time or 0.0) + 1.0,
               float(switch_time or 0.0) + 1.0)
    fig.update_xaxes(title_text="Window start (seconds into clip)",
                     showgrid=False, range=[0, span])
    return fig


def band_legend() -> str:
    """One-line swatch + word + plain-English meaning, for under a timeline."""
    cells = " &nbsp;&nbsp; ".join(
        f'<span style="color:{c};font-weight:700">&#9632;</span> '
        f'<b style="color:{INK_2}">{n}</b> '
        f'<span style="color:{INK_3}">= {meaning}</span>'
        for c, n, meaning in BAND_LEGEND
    )
    return f'<div style="font-size:12.5px;margin:6px 0 2px">{cells}</div>'


def stat_tile(label: str, value: str, tone: str = "") -> str:
    """Compact labelled value for the analysis metric row -- replaces the raw
    dict st.write() was dumping while a clip streamed."""
    colour = tone or INK
    return (
        f'<div style="border:{BORDER};border-radius:{RADIUS_SM};'
        f'padding:{PAD_SM} 14px;background:{SURFACE};">'
        f'<div style="font-size:{FS_CAPTION};font-weight:600;letter-spacing:.1em;'
        f'text-transform:uppercase;color:{INK_3};margin-bottom:3px;">{label}</div>'
        f'<div style="font-size:19px;font-weight:700;color:{colour};'
        f'line-height:1.2;white-space:nowrap;">{value}</div></div>'
    )
