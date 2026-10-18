import json
import os

import streamlit as st

from relic.ui import style, MODELS

style()
card = json.load(open(os.path.join(MODELS, "model_card.json")))

st.title("RELIC-IDS")
st.markdown("#### A hybrid intrusion detection system for cloud and enterprise networks")
st.markdown(
    "Upload a packet capture, a flow CSV or capture live traffic. RELIC-IDS groups packets into flows, "
    "scores every flow with three independent detection layers, explains each alert and keeps a history "
    "you can export as a report.")

c1, c2, c3 = st.columns(3)
c1.page_link("pages/detect.py", label="Analyse traffic", icon=":material/radar:")
c2.page_link("pages/attack_lab.py", label="Try to evade the IDS", icon=":material/science:")
c3.page_link("pages/research.py", label="See the research results", icon=":material/insights:")

st.divider()
st.subheader("How detection works")
l1, l2, l3 = st.columns(3)
with l1:
    st.markdown('<div class="layer"><h4>1. Behaviour rules</h4>'
                '<p class="muted">Port scans, host sweeps, SYN floods, UDP and ICMP floods and brute-force '
                'logins, found from how a host behaves across many flows. They hold on any network.</p></div>',
                unsafe_allow_html=True)
with l2:
    st.markdown('<div class="layer"><h4>2. Network baseline</h4>'
                '<p class="muted">An Isolation Forest learns what normal traffic looks like on your network '
                'and flags flows that do not fit, which covers attacks no rule describes.</p></div>',
                unsafe_allow_html=True)
with l3:
    m = card["full"]
    st.markdown(f'<div class="layer"><h4>3. Dataset ML model</h4>'
                f'<p class="muted">XGBoost trained on UNSW-NB15 (F1 {m["f1"]:.3f} on the official test set). '
                f'It is the detector for UNSW-format records and gives an extra signal on your own traffic.</p></div>',
                unsafe_allow_html=True)

st.divider()
st.subheader("Why three layers")
st.markdown(
    "Models trained on lab datasets look excellent on their own test files but transfer poorly to new "
    "networks. On the bundled demo capture, the dataset model alone flagged most normal flows and missed "
    "most attacks, while the behaviour rules caught every scripted attack with no false alarms on normal "
    "hosts. RELIC-IDS therefore uses the dataset model as one signal, not the judge. The research behind "
    "this is in the paper *Beyond 99% Accuracy* (see Research results).")

with st.expander("What runs where"):
    st.markdown(
        "- **PCAP and CSV upload** work anywhere, including the hosted demo.\n"
        "- **Live capture** works only when you run the app on your own machine with capture rights: "
        "run as Administrator with [Npcap](https://npcap.com) installed on Windows, or with `sudo` on Linux.\n"
        "- Analysis history is stored in a local SQLite file (`relic_history.db`).")

st.caption("Jayveer Vasava, M.Tech Cyber Security, Parul University. Code: github.com/jayveerv/relic-ids")
