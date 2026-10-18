"""Shared UI helpers for the Streamlit pages."""
from __future__ import annotations

import os

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLES = os.path.join(ROOT, "samples")
MODELS = os.path.join(ROOT, "models")

SEV_COLOR = {"Critical": "#e66767", "High": "#d95926", "Medium": "#c98500", "Low": "#3987e5", "None": "#6b7785"}
TYPE_ORDER = ["DoS", "Reconnaissance", "Brute force", "Exploits", "Fuzzers", "Generic", "Backdoor",
              "Shellcode", "Worms", "Analysis", "Anomaly", "Unknown"]
# categorical palette (dark-mode steps), fixed order
PALETTE = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"]

pio.templates["relic"] = go.layout.Template(layout=dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#e6ebf0", family="Inter, system-ui, sans-serif", size=13),
    colorway=PALETTE,
    xaxis=dict(gridcolor="#26323f", zerolinecolor="#3a4756", linecolor="#3a4756"),
    yaxis=dict(gridcolor="#26323f", zerolinecolor="#3a4756", linecolor="#3a4756"),
    legend=dict(bgcolor="rgba(0,0,0,0)"), margin=dict(l=10, r=10, t=40, b=10)))
pio.templates.default = "plotly_dark+relic"


TYPE_COLOR = {"DoS": "#d95926", "Reconnaissance": "#c98500", "Brute force": "#d55181", "Exploits": "#9085e9",
              "Fuzzers": "#199e70", "Generic": "#3987e5", "Backdoor": "#e66767", "Shellcode": "#2fb5c9",
              "Worms": "#8bbf3a", "Analysis": "#b08a6a", "Anomaly": "#7aa7e0", "Unknown": "#6b7785"}


def type_color(t: str) -> str:
    return TYPE_COLOR.get(t, "#6b7785")


def style():
    st.markdown("""
    <style>
      .block-container{padding-top:2rem;max-width:1300px}
      [data-testid="stMetricValue"]{font-size:1.9rem}
      .sev{display:inline-block;padding:.1rem .55rem;border-radius:999px;font-size:.8rem;font-weight:600}
      .muted{color:#9aa7b4}
      .layer{border:1px solid #26323f;border-radius:10px;padding:1rem 1.1rem;height:100%}
      .layer h4{margin:0 0 .35rem 0}
    </style>""", unsafe_allow_html=True)


def sev_badge(sev: str) -> str:
    c = SEV_COLOR.get(sev, "#6b7785")
    return f'<span class="sev" style="background:{c}26;color:{c};border:1px solid {c}">{sev}</span>'


def ensure_state():
    for k, v in {"result": None, "result_meta": None, "flows": None}.items():
        st.session_state.setdefault(k, v)
