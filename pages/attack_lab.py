import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from relic import engine
from relic.ui import style, SAMPLES

style()
st.title("Attack lab")
st.markdown("Act as the attacker: take a real attack record from UNSW-NB15 and change what an attacker controls. "
            "Watch whether each model still detects it. The **standard model** uses all 42 features, including the IP "
            "time-to-live (TTL). The **hardened model** drops TTL and other testbed artefacts.")

data = pd.read_csv(os.path.join(SAMPLES, "unsw_test_sample.csv"))
full, live = engine.load("full"), engine.load("live")
attacks = data[data["label"] == 1]

c1, c2 = st.columns([1, 2])
cat = c1.selectbox("Attack type", sorted(attacks["attack_cat"].unique()), index=sorted(attacks["attack_cat"].unique()).index("Exploits"))
pool = attacks[attacks["attack_cat"] == cat].reset_index(drop=True)
# start from a record both models detect, so evasion is visible
p_full = engine.score(pool, "full")["attack_prob"]; p_live = engine.score(pool, "live")["attack_prob"]
both = pool[(p_full >= 0.5) & (p_live >= 0.5)]
base_pool = both if len(both) else pool
idx = c2.slider("Record", 0, len(base_pool) - 1, 0, help="Records both models detect are listed first.")
rec = base_pool.iloc[[idx]].copy()

med = full["benign_median"]
st.markdown(f"Benign median for reference: TTL {med['sttl']:.0f}, source bytes {med['sbytes']:.0f}, "
            f"mean packet size {med['smean']:.0f}")

key = f"{cat}-{idx}"
if st.session_state.get("lab_key") != key:
    st.session_state["lab_key"] = key
    for f in ("sttl", "sbytes", "smean", "dur", "ct_srv_src"):
        st.session_state[f"lab_{f}"] = float(rec.iloc[0][f])
b1, b2 = st.columns(2)
if b1.button("Set TTL to the benign median (one-line evasion)"):
    st.session_state["lab_sttl"] = float(med["sttl"])
if b2.button("Reset record"):
    for f in ("sttl", "sbytes", "smean", "dur", "ct_srv_src"):
        st.session_state[f"lab_{f}"] = float(rec.iloc[0][f])

s1, s2, s3, s4, s5 = st.columns(5)
mod = rec.copy()
mod["sttl"] = s1.slider("Source TTL", 0.0, 255.0, key="lab_sttl", step=1.0)
mod["sbytes"] = s2.number_input("Source bytes", 0.0, 1e8, key="lab_sbytes", step=50.0)
mod["smean"] = s3.number_input("Mean packet size", 0.0, 1500.0, key="lab_smean", step=10.0)
mod["dur"] = s4.number_input("Duration (s)", 0.0, 3600.0, key="lab_dur", step=0.1, format="%.3f")
mod["ct_srv_src"] = s5.number_input("Same-service connections", 0.0, 100.0, key="lab_ct_srv_src", step=1.0)
mod["ct_state_ttl"] = rec["ct_state_ttl"]  # keep as recorded

pf = float(engine.score(mod, "full")["attack_prob"].iloc[0])
pl = float(engine.score(mod, "live")["attack_prob"].iloc[0])

def gauge(p, title):
    color = "#d95926" if p >= 0.5 else "#199e70"
    fig = go.Figure(go.Indicator(mode="gauge+number", value=p * 100, number={"suffix": "%", "valueformat": ".1f"},
                                 title={"text": title},
                                 gauge={"axis": {"range": [0, 100]}, "bar": {"color": color},
                                        "threshold": {"line": {"color": "#e6ebf0", "width": 2}, "value": 50}}))
    fig.update_layout(height=260, margin=dict(l=20, r=20, t=60, b=10))
    return fig

g1, g2 = st.columns(2)
with g1:
    st.plotly_chart(gauge(pf, "Standard model: attack probability"), use_container_width=True)
    st.markdown(("**Detected**" if pf >= 0.5 else "**Missed: the attack evaded this model**"))
with g2:
    st.plotly_chart(gauge(pl, "Hardened model (no TTL): attack probability"), use_container_width=True)
    st.markdown(("**Detected**" if pl >= 0.5 else "**Missed: the attack evaded this model**"))

st.info("In the paper's test, setting the TTL of detected UNSW-NB15 attacks to the benign median let 99.8% of them "
        "evade the standard XGBoost model. Dropping TTL-type features cost only 0.002 F1 on the official test set "
        f"(standard {full['metrics']['f1']:.3f}, hardened {live['metrics']['f1']:.3f}), so the hardened model is the "
        "one RELIC-IDS uses for live traffic.")
