"""
utils.py — Shared helpers, styling constants, and UI primitives.
"""

from __future__ import annotations

import streamlit as st


# ── Design tokens ─────────────────────────────────────────────────────────────
GOLD = "#C9A84C"
GOLD_LIGHT = "#E8D5A3"
CHARCOAL = "#1C1C1E"
SURFACE = "#242428"
SURFACE_2 = "#2E2E34"
BORDER = "#3A3A42"
TEXT_PRIMARY = "#F0EDE8"
TEXT_MUTED = "#8A8A9A"
SUCCESS = "#4CAF7D"
WARNING = "#E8A838"
ERROR = "#E85C5C"
INFO = "#5C9BE8"


GLOBAL_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&family=DM+Sans:wght@300;400;500;600&display=swap');

*, *::before, *::after {{ box-sizing: border-box; }}

html, body, [class*="css"] {{
    font-family: 'DM Sans', sans-serif;
    color: {TEXT_PRIMARY};
}}

.stApp, .main {{
    background-color: {CHARCOAL} !important;
}}

/* Sidebar */
section[data-testid="stSidebar"] {{
    background: {SURFACE} !important;
    border-right: 1px solid {BORDER};
}}

section[data-testid="stSidebar"] * {{
    color: {TEXT_PRIMARY} !important;
}}

/* Headers */
h1,h2,h3,h4 {{
    font-family: 'Syne', sans-serif !important;
    letter-spacing: -0.02em;
}}

/* Inputs */
.stTextInput > div > div > input,
.stSelectbox > div > div,
.stMultiSelect > div > div {{
    background: {SURFACE_2} !important;
    border: 1px solid {BORDER} !important;
    color: {TEXT_PRIMARY} !important;
    border-radius: 6px !important;
}}

.stRadio > div {{ gap: 0.5rem; }}

/* Buttons */
.stButton > button {{
    font-family: 'DM Mono', monospace !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.06em !important;
    background: {SURFACE_2} !important;
    border: 1px solid {BORDER} !important;
    color: {TEXT_PRIMARY} !important;
    border-radius: 6px !important;
    padding: 0.55rem 1.2rem !important;
    transition: all 0.18s ease !important;
}}

.stButton > button:hover {{
    border-color: {GOLD} !important;
    color: {GOLD} !important;
    background: rgba(201,168,76,0.08) !important;
}}

/* Primary button override */
.stButton > button[kind="primary"] {{
    background: {GOLD} !important;
    border-color: {GOLD} !important;
    color: {CHARCOAL} !important;
    font-weight: 700 !important;
}}

.stButton > button[kind="primary"]:hover {{
    background: {GOLD_LIGHT} !important;
    border-color: {GOLD_LIGHT} !important;
    color: {CHARCOAL} !important;
}}

/* Download button */
.stDownloadButton > button {{
    font-family: 'DM Mono', monospace !important;
    font-size: 0.82rem !important;
    background: rgba(201,168,76,0.12) !important;
    border: 1px solid {GOLD} !important;
    color: {GOLD} !important;
    border-radius: 6px !important;
    transition: all 0.18s ease !important;
}}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {{
    background: {SURFACE} !important;
    border-radius: 8px 8px 0 0 !important;
    gap: 0 !important;
    border-bottom: 1px solid {BORDER};
}}

.stTabs [data-baseweb="tab"] {{
    font-family: 'DM Mono', monospace !important;
    font-size: 0.78rem !important;
    color: {TEXT_MUTED} !important;
    padding: 0.6rem 1.2rem !important;
    border-radius: 0 !important;
}}

.stTabs [aria-selected="true"] {{
    color: {GOLD} !important;
    border-bottom: 2px solid {GOLD} !important;
    background: transparent !important;
}}

/* Dataframes */
div[data-testid="stDataFrame"] {{
    border: 1px solid {BORDER};
    border-radius: 8px;
    overflow: hidden;
}}

/* Expanders */
.streamlit-expanderHeader {{
    font-family: 'DM Mono', monospace !important;
    font-size: 0.82rem !important;
    background: {SURFACE} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 6px !important;
    color: {TEXT_PRIMARY} !important;
}}

/* Metrics */
[data-testid="metric-container"] {{
    background: {SURFACE} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 8px !important;
    padding: 0.8rem 1rem !important;
}}

[data-testid="metric-container"] label {{
    font-family: 'DM Mono', monospace !important;
    font-size: 0.72rem !important;
    color: {TEXT_MUTED} !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}}

[data-testid="metric-container"] [data-testid="stMetricValue"] {{
    font-family: 'Syne', sans-serif !important;
    font-size: 1.6rem !important;
    color: {GOLD} !important;
}}

/* Alerts */
.stAlert {{
    border-radius: 6px !important;
    border: none !important;
}}

/* Multiselect tags */
span[data-baseweb="tag"] {{
    background: rgba(201,168,76,0.18) !important;
    border: 1px solid {GOLD} !important;
    color: {GOLD} !important;
    border-radius: 4px !important;
}}

/* Slider */
.stSlider [data-baseweb="slider"] div[role="slider"] {{
    background: {GOLD} !important;
    border-color: {GOLD} !important;
}}

/* Progress */
.stProgress > div > div > div {{
    background: {GOLD} !important;
}}

/* Separator */
hr {{ border-color: {BORDER} !important; }}

/* Tooltips & captions */
small, .caption, [data-testid="caption"] {{
    color: {TEXT_MUTED} !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.75rem !important;
}}
</style>
"""


def inject_css() -> None:
    """Inject global CSS into the Streamlit page."""
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)


def section_header(step: int, title: str, subtitle: str = "") -> None:
    """Render a styled section header with step number."""
    sub_html = f'<p style="color:{TEXT_MUTED};font-family:\'DM Mono\',monospace;font-size:0.8rem;margin:0.3rem 0 0 0;">{subtitle}</p>' if subtitle else ""
    st.markdown(f"""
    <div style="
        border-left: 3px solid {GOLD};
        padding: 0.6rem 0 0.6rem 1rem;
        margin: 1.5rem 0 1rem 0;
    ">
        <span style="
            font-family: 'DM Mono', monospace;
            font-size: 0.68rem;
            color: {GOLD};
            letter-spacing: 0.12em;
            text-transform: uppercase;
            font-weight: 500;
        ">STEP {step:02d}</span>
        <h3 style="
            font-family: 'Syne', sans-serif;
            font-size: 1.15rem;
            font-weight: 700;
            margin: 0.1rem 0 0 0;
            color: {TEXT_PRIMARY};
        ">{title}</h3>
        {sub_html}
    </div>
    """, unsafe_allow_html=True)


def badge(text: str, color: str = GOLD) -> str:
    """Return an inline HTML badge string."""
    return (
        f'<span style="'
        f'display:inline-block;background:rgba({_hex_to_rgb(color)},0.15);'
        f'color:{color};border:1px solid {color};'
        f'font-family:\'DM Mono\',monospace;font-size:0.7rem;'
        f'padding:2px 8px;border-radius:4px;letter-spacing:0.06em;'
        f'font-weight:500;">{text}</span>'
    )


def status_ok(msg: str) -> None:
    st.markdown(
        f'<p style="color:{SUCCESS};font-family:\'DM Mono\',monospace;font-size:0.82rem;margin:0.4rem 0;">✓ {msg}</p>',
        unsafe_allow_html=True,
    )


def status_warn(msg: str) -> None:
    st.markdown(
        f'<p style="color:{WARNING};font-family:\'DM Mono\',monospace;font-size:0.82rem;margin:0.4rem 0;">⚠ {msg}</p>',
        unsafe_allow_html=True,
    )


def status_info(msg: str) -> None:
    st.markdown(
        f'<p style="color:{INFO};font-family:\'DM Mono\',monospace;font-size:0.82rem;margin:0.4rem 0;">ℹ {msg}</p>',
        unsafe_allow_html=True,
    )


def divider() -> None:
    st.markdown(f'<hr style="border-color:{BORDER};margin:1.5rem 0;"/>', unsafe_allow_html=True)


def _hex_to_rgb(hex_color: str) -> str:
    """Convert '#RRGGBB' to 'R,G,B' string for rgba()."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"{r},{g},{b}"


def mono(text: str, color: str = TEXT_MUTED) -> str:
    """Return inline mono-spaced HTML span."""
    return f'<span style="font-family:\'DM Mono\',monospace;color:{color};font-size:0.85rem;">{text}</span>'