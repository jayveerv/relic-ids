import pandas as pd
import plotly.express as px
import streamlit as st

from relic import engine, store
from relic.ui import style, sev_badge, type_color

style()
st.title("Investigate alerts")

runs = store.runs()
if runs.empty:
    st.info("No analyses yet. Run one on the Detect page.")
    st.page_link("pages/detect.py", label="Go to Detect", icon=":material/radar:")
    st.stop()

labels = {r.id: f"Run {r.id}: {r.source} ({r.created}, {r.alerts:,} alerts of {r.flows:,})" for r in runs.itertuples()}
run_id = st.selectbox("Analysis run", list(labels), format_func=labels.get)
alerts = store.alerts(run_id)
if alerts.empty:
    st.success("This run raised no alerts.")
    st.stop()

f1, f2, f3 = st.columns(3)
types = f1.multiselect("Attack type", sorted(alerts["attack_type"].dropna().unique()))
sevs = f2.multiselect("Severity", ["Critical", "High", "Medium", "Low"])
srch = f3.text_input("Source or destination IP contains")
view = alerts.copy()
if types: view = view[view["attack_type"].isin(types)]
if sevs: view = view[view["severity"].isin(sevs)]
if srch: view = view[view["srcip"].fillna("").str.contains(srch) | view["dstip"].fillna("").str.contains(srch)]
st.caption(f"{len(view):,} alerts match")

has_ip = view["srcip"].notna().any()
if has_ip:
    top = view.groupby(["srcip", "attack_type"]).size().reset_index(name="flows").sort_values("flows", ascending=False).head(15)
    fig = px.bar(top, x="flows", y="srcip", color="attack_type", orientation="h", title="Top alerting sources",
                 color_discrete_map={t: type_color(t) for t in top["attack_type"].unique()},
                 labels={"srcip": "", "flows": "Alerting flows", "attack_type": "Type"})
    fig.update_layout(height=380, yaxis=dict(categoryorder="total ascending"))
    st.plotly_chart(fig, use_container_width=True)

st.markdown("**Select an alert to investigate**")
cols = [c for c in ["flow_id", "srcip", "sport", "dstip", "dsport", "proto", "service", "attack_type", "severity",
                    "layer", "attack_prob", "anomaly_score"] if c in view.columns]
ev = st.dataframe(view[cols], hide_index=True, use_container_width=True, height=300,
                  on_select="rerun", selection_mode="single-row",
                  column_config={"attack_prob": st.column_config.NumberColumn("ML prob", format="%.2f"),
                                 "anomaly_score": st.column_config.NumberColumn("Anomaly", format="%.2f")})
sel = ev.selection.rows
if not sel:
    st.caption("Click a row to see why it was flagged.")
    st.stop()

row = view.iloc[sel[0]]
st.divider()
st.markdown(f"### Flow {row.flow_id} {sev_badge(row.severity)}", unsafe_allow_html=True)
if has_ip:
    st.markdown(f"`{row.srcip}:{row.sport}` to `{row.dstip}:{row.dsport}` over {row.proto}/{row.service}, "
                f"{int(row.spkts or 0)} packets out and {int(row.dpkts or 0)} back, {row.dur:.3f} s")
c1, c2 = st.columns([2, 3])
with c1:
    st.markdown("**Why it was flagged**")
    for part in str(row.evidence or "").split("; "):
        if part:
            st.markdown(f"- {part}")
    st.markdown(f"- Detection layer: **{row.layer or 'ML model'}**")
    if pd.notna(row.attack_prob):
        st.markdown(f"- Dataset model attack probability: **{row.attack_prob:.2f}**")
    st.markdown("**What to do**")
    advice = {"DoS": "Rate-limit or block the source at the edge; for spoofed floods enable SYN cookies and upstream filtering.",
              "Reconnaissance": "Block the scanning source and check which open services it found.",
              "Brute force": "Lock or throttle the targeted account, block the source, and enforce MFA.",
              "Anomaly": "Check the host and the service: a new application or a misconfiguration can also look unusual."}
    st.markdown(advice.get(row.attack_type, "Review the source host and the targeted service; correlate with host logs."))

with c2:
    st.markdown("**Feature contributions (dataset model, SHAP)**")
    res = store.load_run_flows(run_id)
    run_mode = runs.set_index("id").loc[run_id, "mode"]
    if res is not None:
        full_row = res[res["flow_id"] == int(row.flow_id)].head(1)
        model = "full" if run_mode == "dataset" else "live"
        exp = engine.explain(full_row, model)
        exp["direction"] = ["towards attack" if c > 0 else "towards normal" for c in exp["contribution"]]
        exp["label"] = [f"{f} = {v}" if not isinstance(v, float) else f"{f} = {v:.3g}" for f, v in zip(exp.feature, exp.value)]
        fig = px.bar(exp[::-1], x="contribution", y="label", orientation="h", color="direction",
                     color_discrete_map={"towards attack": "#d95926", "towards normal": "#3987e5"},
                     labels={"contribution": "Contribution to attack score (log-odds)", "label": "", "direction": ""})
        fig.update_layout(height=360, legend=dict(orientation="h", y=-0.2),
                          yaxis=dict(categoryorder="array", categoryarray=list(exp["label"][::-1])))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Positive bars push the dataset model towards 'attack'. Rule and baseline alerts are explained by the evidence on the left.")
    else:
        st.caption("The saved flows for this run were not found, so feature contributions cannot be shown.")
