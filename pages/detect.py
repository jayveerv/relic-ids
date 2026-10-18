import os
import tempfile
import time

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from relic import baseline as bl_mod, engine, pipeline, report, store
from relic.features import extract_pcap, capture_live, FEATURE_COLS
from relic.rules import RuleConfig
from relic.ui import style, ensure_state, SAMPLES, MODELS, SEV_COLOR, type_color

style(); ensure_state()
st.title("Detect")
st.caption("Choose traffic to analyse, check the settings in the sidebar, then run the analysis.")

# ------------------------------------------------------------------ sidebar settings
with st.sidebar:
    st.header("Settings")
    bl_choice = st.radio("Network baseline", ["Demo office network", "Learn from my normal traffic (PCAP)", "None"],
                         help="The anomaly layer compares traffic with a capture of normal traffic from the same network.")
    bl_file = None
    if bl_choice.startswith("Learn"):
        bl_file = st.file_uploader("Normal-traffic capture", type=["pcap", "pcapng", "cap"], key="blfile")
    bl_fp = st.select_slider("Baseline false-alarm target", options=[0.005, 0.01, 0.02, 0.05], value=0.01,
                             format_func=lambda x: f"{x:.1%}")
    with st.expander("Behaviour rule thresholds"):
        cfg = RuleConfig(
            scan_ports=st.number_input("Port scan: distinct ports to one host", 5, 1000, 20),
            sweep_hosts=st.number_input("Host sweep: distinct hosts on one port", 5, 1000, 20),
            syn_flood_req=st.number_input("SYN flood: half-open connections in 10 s", 10, 100000, 100),
            udp_flood_flows=st.number_input("UDP flood: one-way flows in 10 s", 10, 100000, 200),
            brute_sessions=st.number_input("Brute force: short logins in 60 s", 5, 10000, 30))
    card = engine.load("full")
    preset = st.selectbox("Dataset model threshold", list(card["presets"].keys()))
    ml_thr = float(card["presets"][preset])
    use_ml = st.checkbox("Count dataset model in the verdict for PCAP/live traffic", value=False,
                         help="Off by default: a lab-trained model raises many false alarms on traffic it was not trained on.")

# ------------------------------------------------------------------ input
src = st.radio("Traffic source", ["Demo capture", "Upload PCAP", "Upload CSV", "Live capture"], horizontal=True)
data_label, flows, dataset_df, truth = None, None, None, None

if src == "Demo capture":
    st.info("A synthetic 2-minute capture of an office network: 20 clients using web, DNS and SSH, plus a port scan, "
            "a SYN flood, a UDP flood and an HTTP brute-force attack. Ground truth is known, so results are scored.")
    path, data_label = os.path.join(SAMPLES, "demo_traffic.pcap"), "demo_traffic.pcap"
    truth = pd.read_csv(os.path.join(SAMPLES, "demo_ground_truth.csv"))
elif src == "Upload PCAP":
    up = st.file_uploader("PCAP or PCAPNG file (Wireshark, tcpdump)", type=["pcap", "pcapng", "cap"])
    path, data_label = None, None
    if up:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(up.name)[1])
        tmp.write(up.getbuffer()); tmp.close(); path, data_label = tmp.name, up.name
elif src == "Upload CSV":
    st.caption("Either UNSW-NB15 records (all 42 features; optional label and attack_cat columns are used for scoring) "
               "or a flow CSV exported from this app.")
    c1, c2 = st.columns([3, 1])
    up = c1.file_uploader("CSV file", type=["csv"])
    use_sample = c2.button("Use UNSW-NB15 test sample")
    if use_sample:
        st.session_state["csv_sample"] = True
    if up is not None:
        st.session_state["csv_sample"] = False
        dataset_df, data_label = pd.read_csv(up), up.name
    elif st.session_state.get("csv_sample"):
        dataset_df, data_label = pd.read_csv(os.path.join(SAMPLES, "unsw_test_sample.csv")), "unsw_test_sample.csv"
    if dataset_df is not None:
        st.write(f"{len(dataset_df):,} rows, {dataset_df.shape[1]} columns")
    path = None
else:
    st.warning("Live capture needs the app to run on your own machine with capture rights "
               "(Administrator + Npcap on Windows, sudo on Linux). It does not work on the hosted demo.")
    secs = st.slider("Capture duration (seconds)", 10, 300, 30)
    iface = st.text_input("Interface (leave blank for default)", "")
    path = None

run = st.button("Run analysis", type="primary")

# ------------------------------------------------------------------ run
if run:
    t0 = time.time()
    try:
        with st.status("Analysing traffic", expanded=False) as status:
            baseline = None
            if bl_choice.startswith("Demo"):
                baseline = bl_mod.load(os.path.join(MODELS, "baseline_demo.joblib"))
                if baseline and baseline["fp_target"] != bl_fp:
                    status.write("Re-fitting demo baseline for the chosen false-alarm target")
                    baseline = bl_mod.fit(extract_pcap(os.path.join(SAMPLES, "demo_baseline_normal.pcap")), bl_fp)
            elif bl_choice.startswith("Learn"):
                if bl_file is None:
                    st.error("Upload a normal-traffic capture for the baseline, or choose another baseline option.")
                    st.stop()
                tmpb = tempfile.NamedTemporaryFile(delete=False, suffix=".pcap"); tmpb.write(bl_file.getbuffer()); tmpb.close()
                status.write("Learning the baseline from your normal traffic")
                baseline = bl_mod.fit(extract_pcap(tmpb.name), bl_fp)

            if src == "Upload CSV":
                if dataset_df is None:
                    st.error("Upload a CSV or use the UNSW-NB15 sample."); st.stop()
                if pipeline.is_unsw_format(dataset_df):
                    mode = "dataset"
                    status.write("UNSW-NB15 records detected: scoring with the dataset model")
                    res = pipeline.analyze_dataset(dataset_df, ml_thr)
                else:
                    missing = engine.missing_features(engine.load("live"), dataset_df)
                    if missing or "srcip" not in dataset_df:
                        st.error("This CSV is neither UNSW-NB15 format nor a flow export from this app. Missing: "
                                 + ", ".join((missing or ["srcip, dstip, start_time"])[:10]))
                        st.stop()
                    mode = "flows"
                    res = pipeline.analyze_flows(dataset_df, cfg, baseline, ml_thr, use_ml)
            else:
                mode = "flows"
                if src == "Live capture":
                    status.write(f"Capturing for {secs} s")
                    try:
                        flows = capture_live(secs, iface or None)
                    except (PermissionError, OSError, RuntimeError) as e:
                        st.error(f"Live capture failed: {e}. Run the app as Administrator with Npcap (Windows) "
                                 "or with sudo (Linux).")
                        st.stop()
                    data_label = f"live capture {time.strftime('%Y-%m-%d %H:%M')}"
                else:
                    if not path:
                        st.error("Upload a PCAP file first."); st.stop()
                    status.write("Reading packets and building flows")
                    flows = extract_pcap(path)
                if flows.empty:
                    st.error("No IPv4 flows found in this traffic."); st.stop()
                status.write(f"{len(flows):,} flows built; running the three detection layers")
                res = pipeline.analyze_flows(flows, cfg, baseline, ml_thr, use_ml)
            settings = dict(source=src, baseline=bl_choice, baseline_fp_target=bl_fp, rules=cfg.as_dict(),
                            ml_threshold=ml_thr, ml_in_verdict=use_ml)
            run_id = store.save_run(data_label, mode, res, settings)
            status.update(label=f"Done in {time.time() - t0:.1f} s (saved as run {run_id})", state="complete")
        st.session_state.update(result=res, result_meta=dict(label=data_label, mode=mode, settings=settings,
                                                             run_id=run_id, truth=truth))
    except Exception as e:  # keep the page usable and show what went wrong
        st.exception(e)

# ------------------------------------------------------------------ results
res, meta = st.session_state.get("result"), st.session_state.get("result_meta")
if res is not None:
    sm = pipeline.summary(res)
    st.divider()
    st.subheader(f"Results: {meta['label']}")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Flows analysed", f"{sm['flows']:,}")
    k2.metric("Alerts", f"{sm['alerts']:,}")
    k3.metric("Share of flows", f"{sm['alert_rate']:.1%}")
    k4.metric("Alerting sources" if meta["mode"] == "flows" else "Attack types",
              f"{sm['sources']:,}" if meta["mode"] == "flows" else f"{len(sm['by_type'])}")

    alerts = res[res["verdict"] == "Attack"]
    left, right = st.columns([3, 2])
    with left:
        if meta["mode"] == "flows" and not alerts.empty:
            t0 = res["start_time"].min()
            a = alerts.assign(sec=((alerts["start_time"] - t0) // 5 * 5).astype(int))
            g = a.groupby(["sec", "attack_type"]).size().reset_index(name="flows")
            fig = px.bar(g, x="sec", y="flows", color="attack_type", title="Alerts over time (5 s bins)",
                         color_discrete_map={t: type_color(t) for t in g["attack_type"].unique()},
                         labels={"sec": "Seconds from start of capture", "flows": "Alerting flows",
                                 "attack_type": "Type"})
            fig.update_layout(bargap=0.1, height=340)
            st.plotly_chart(fig, use_container_width=True)
        elif not alerts.empty:
            g = alerts["attack_type"].value_counts().reset_index()
            g.columns = ["attack_type", "flows"]
            fig = px.bar(g, x="flows", y="attack_type", orientation="h", title="Alerts by attack type",
                         color="attack_type", color_discrete_map={t: type_color(t) for t in g["attack_type"]},
                         labels={"attack_type": "", "flows": "Records"})
            fig.update_layout(showlegend=False, height=340, yaxis=dict(categoryorder="total ascending"))
            st.plotly_chart(fig, use_container_width=True)
    with right:
        sev = pd.Series(sm["by_severity"]).reindex(["Critical", "High", "Medium", "Low"]).dropna()
        if not sev.empty:
            fig = px.bar(x=sev.values, y=sev.index, orientation="h", title="Alerts by severity",
                         color=sev.index, color_discrete_map=SEV_COLOR, labels={"x": "Alerts", "y": ""})
            fig.update_layout(showlegend=False, height=340, yaxis=dict(categoryorder="array",
                                                                        categoryarray=list(sev.index)[::-1]))
            st.plotly_chart(fig, use_container_width=True)

    # ---- scoring against known truth
    if meta.get("truth") is not None and meta["mode"] == "flows":
        tr = meta["truth"]
        def label(ip):
            for _, r in tr.iterrows():
                if "/" in r.source:
                    import ipaddress
                    if ipaddress.ip_address(ip) in ipaddress.ip_network(r.source):
                        return r.label
                elif ip == r.source:
                    return r.label
            return "Other"
        t = res.assign(truth=res["srcip"].map(label))
        tab = (t.groupby("truth").agg(flows=("flow_id", "size"),
                                      flagged=("verdict", lambda v: (v == "Attack").mean()))
               .reset_index().rename(columns={"truth": "Scenario (ground truth)", "flows": "Flows",
                                              "flagged": "Flagged as attack"}))
        tab["Flagged as attack"] = tab["Flagged as attack"] * 100
        st.markdown("**Scored against the demo's ground truth**")
        st.dataframe(tab, hide_index=True, use_container_width=True,
                     column_config={"Flagged as attack": st.column_config.ProgressColumn(format="%.1f%%",
                                                                                           min_value=0, max_value=100)})
    if meta["mode"] == "dataset" and "label" in res.columns:
        y, p = res["label"].astype(int), (res["verdict"] == "Attack").astype(int)
        tp, fp = int(((p == 1) & (y == 1)).sum()), int(((p == 1) & (y == 0)).sum())
        fn, tn = int(((p == 0) & (y == 1)).sum()), int(((p == 0) & (y == 0)).sum())
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy vs labels", f"{(tp + tn) / len(y):.3f}")
        c2.metric("Recall", f"{tp / max(tp + fn, 1):.3f}")
        c3.metric("Precision", f"{tp / max(tp + fp, 1):.3f}")
        c4.metric("False-positive rate", f"{fp / max(fp + tn, 1):.3f}")

    # ---- incidents and alert table
    if meta["mode"] == "flows" and not alerts.empty:
        st.markdown("**Incidents** (related alerts grouped together)")
        def incident(r):
            rule = r.get("rule")
            if isinstance(rule, str) and rule == "SYN flood":
                return f"SYN flood on {r.dstip}:{r.dsport}"
            if isinstance(rule, str):
                return f"{rule}: {r.srcip} to {r.dstip}"
            return f"Anomaly: {r.srcip}"
        a2 = alerts.assign(incident=alerts.apply(incident, axis=1))
        inc = (a2.groupby(["incident", "attack_type", "severity"])
               .agg(flows=("flow_id", "size"), sources=("srcip", "nunique"), targets=("dstip", "nunique"),
                    first=("start_time", "min"), evidence=("evidence", "first"))
               .reset_index().sort_values("flows", ascending=False))
        inc["first"] = pd.to_datetime(inc["first"], unit="s").dt.strftime("%H:%M:%S")
        st.dataframe(inc, hide_index=True, use_container_width=True, height=260,
                     column_config={"incident": st.column_config.TextColumn("Incident", width="medium"),
                                    "attack_type": "Type", "severity": "Severity", "flows": "Flows",
                                    "sources": "Sources", "targets": "Targets", "first": "First seen",
                                    "evidence": st.column_config.TextColumn("Evidence", width="large")})

    with st.expander(f"All alerts ({len(alerts):,})", expanded=meta["mode"] == "dataset"):
        show = [c for c in ["flow_id", "srcip", "sport", "dstip", "dsport", "proto", "service", "state",
                            "attack_type", "severity", "layer", "evidence", "attack_prob", "anomaly_score",
                            "label", "attack_cat"] if c in alerts.columns]
        st.dataframe(alerts[show], hide_index=True, use_container_width=True, height=320,
                     column_config={"attack_prob": st.column_config.NumberColumn("ML prob", format="%.2f"),
                                    "anomaly_score": st.column_config.NumberColumn("Anomaly", format="%.2f")})

    d1, d2, d3 = st.columns(3)
    d1.download_button("Download alerts (CSV)", alerts.to_csv(index=False).encode(), "relic_alerts.csv", "text/csv")
    d2.download_button("Download all flows (CSV)", res.to_csv(index=False).encode(), "relic_flows.csv", "text/csv")
    d3.download_button("Download incident report (PDF)",
                       report.build_pdf(meta["label"], sm, res, meta["settings"]), "relic_report.pdf", "application/pdf")
    st.page_link("pages/investigate.py", label="Investigate these alerts", icon=":material/manage_search:")
