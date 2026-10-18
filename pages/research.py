import json
import os

import pandas as pd
import plotly.express as px
import streamlit as st

from relic.ui import style, ROOT, MODELS

style()
st.title("Research results")
st.markdown("Results from the paper *Beyond 99% Accuracy: A Critical Review and Re-evaluation of AI-Based Intrusion "
            "Detection for Cloud Environments*. Every number comes from the released experiment scripts.")

R = json.load(open(os.path.join(ROOT, "experiments", "results", "results.json")))
Q = json.load(open(os.path.join(ROOT, "experiments", "results", "relic.json")))
card = json.load(open(os.path.join(MODELS, "model_card.json")))
SHORT = {"Logistic Regression": "LR", "Decision Tree": "DT", "Random Forest": "RF", "XGBoost": "XGBoost", "MLP (DNN)": "MLP"}

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Random-split pitfall", "Official test results", "Zero-day", "False alarms",
                                        "Deployment cost"])
with tab1:
    g = pd.DataFrame(R["split_gap"])
    m = g.melt(id_vars=["dataset", "model"], value_vars=["random_split_acc", "official_acc"], var_name="protocol",
               value_name="accuracy")
    m["protocol"] = m["protocol"].map({"random_split_acc": "Random 80/20 split of training file",
                                       "official_acc": "Official held-out test set"})
    fig = px.bar(m, x="model", y="accuracy", color="protocol", barmode="group", facet_col="dataset",
                 color_discrete_sequence=["#3987e5", "#d95926"], range_y=[0.5, 1.02], text_auto=".3f",
                 title="The same models, two evaluation protocols")
    fig.update_layout(height=420, legend=dict(orientation="h", y=-0.15, title=""))
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    fig.update_xaxes(title=""); fig.update_yaxes(title="Accuracy", col=1)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("Testing on part of the training file inflates accuracy by about 20 points on NSL-KDD and 8-9 points "
                "on UNSW-NB15.")
with tab2:
    b = pd.DataFrame(R["binary"]).groupby(["dataset", "model"])[["accuracy", "precision", "recall", "f1", "fpr", "auc"]].mean().reset_index()
    b["model"] = b["model"].map(SHORT)
    st.dataframe(b.round(3), hide_index=True, use_container_width=True)
    st.caption("Binary detection, official test sets, mean of three seeds.")
with tab3:
    z = pd.DataFrame(Q["zero_day"]["models"]).groupby("model")[["known_dr", "novel_dr"]].mean().reset_index()
    z["model"] = z["model"].map(SHORT)
    zm = z.melt(id_vars="model", var_name="attacks", value_name="detection")
    zm["attacks"] = zm["attacks"].map({"known_dr": "Seen in training", "novel_dr": "Unseen (zero-day)"})
    fig = px.bar(zm, x="model", y="detection", color="attacks", barmode="group", text_auto=".2f", range_y=[0, 1],
                 color_discrete_sequence=["#3987e5", "#d95926"], title="NSL-KDD: detection of known vs unseen attack types")
    fig.update_layout(height=400, legend=dict(orientation="h", y=-0.15, title=""))
    st.plotly_chart(fig, use_container_width=True)
with tab4:
    t = pd.DataFrame(Q["threshold"])
    t["target"] = t["target_fpr"].map(lambda x: "default 0.5" if pd.isna(x) else f"{x:.0%}")
    st.dataframe(t[["dataset", "model", "target", "threshold", "fpr", "recall", "f1"]].round(3).rename(
        columns={"target": "FPR target (validation)", "fpr": "Test FPR", "recall": "Test recall", "f1": "Test F1"}),
        hide_index=True, use_container_width=True)
    st.markdown("Thresholds set on validation data missed their false-positive target by 2.2-10 times on the test set, "
                "so RELIC-IDS lets the operator choose and revisit the operating point.")
with tab5:
    c = pd.DataFrame(Q["cost"])
    c["model"] = c["model"].map(SHORT)
    st.dataframe(c.round(3).rename(columns={"size_kb": "Size (KB)", "latency_ms": "Latency per flow (ms)",
                                            "throughput_fps": "Throughput (flows/s)"}),
                 hide_index=True, use_container_width=True)

st.divider()
st.subheader("Models inside this app")
mc = pd.DataFrame({k: {m: v[m] for m in ("accuracy", "precision", "recall", "f1", "fpr", "auc", "type_accuracy", "n_features")}
                   for k, v in card.items()}).T.reset_index().rename(columns={"index": "model"})
mc["model"] = mc["model"].map({"full": "Standard (42 features)", "live": "Hardened, no TTL (35 features)"})
st.dataframe(mc.round(4), hide_index=True, use_container_width=True)
st.caption("XGBoost, UNSW-NB15 official partitions. type_accuracy = 10-class attack-type accuracy.")
