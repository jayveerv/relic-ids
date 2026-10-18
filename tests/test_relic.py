"""Run with: python -m pytest -q"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from relic.features import Flow, features_from_flows, extract_pcap
from relic import pipeline, rules, baseline

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_formulas_match_unsw_row():
    # First record of the official UNSW-NB15 training file:
    # dur 0.121478, spkts 6, dpkts 4, sbytes 258, dbytes 172 -> rate 74.08749, sload 14158.94, dload 8495.365, sinpkt 24.2956
    t0, dur = 1000.0, 0.121478
    f = Flow("tcp", "1.1.1.1", 1234, "2.2.2.2", 80, t0, t0 + dur)
    f.s_times = list(np.linspace(t0, t0 + dur, 6)); f.d_times = list(np.linspace(t0 + 0.01, t0 + 0.0351, 4))
    f.sbytes, f.dbytes = 258, 172
    r = features_from_flows([f]).iloc[0]
    assert abs(r.rate - 74.08749) < 1e-3
    assert abs(r.sload - 14158.94238) < 1e-2
    assert abs(r.dload - 8495.365234) < 1e-2
    assert abs(r.sinpkt - 24.2956) < 1e-3


def demo():
    return extract_pcap(os.path.join(ROOT, "samples", "demo_traffic.pcap"))


def truth(ip):
    if ip == "192.168.56.66": return "Reconnaissance"
    if ip == "192.168.56.77" or ip.startswith("172.16."): return "DoS"
    if ip == "192.168.56.88": return "Brute force"
    return "Normal"


def test_rules_catch_demo_attacks_without_false_alarms():
    df = demo(); hits = rules.run_rules(df)
    df = df.merge(hits, on="flow_id", how="left"); df["truth"] = df.srcip.map(truth)
    assert df[df.truth != "Normal"]["rule"].notna().mean() == 1.0
    assert df[df.truth == "Normal"]["rule"].notna().sum() == 0


def test_pipeline_on_demo_with_baseline():
    bl = baseline.load(os.path.join(ROOT, "models", "baseline_demo.joblib"))
    res = pipeline.analyze_flows(demo(), baseline=bl)
    res["truth"] = res.srcip.map(truth)
    assert (res[res.truth != "Normal"].verdict == "Attack").mean() == 1.0
    assert (res[res.truth == "Normal"].verdict == "Attack").mean() < 0.10


def test_dataset_model_on_unsw_sample():
    s = pd.read_csv(os.path.join(ROOT, "samples", "unsw_test_sample.csv"))
    res = pipeline.analyze_dataset(s)
    acc = ((res.verdict == "Attack").astype(int) == res.label).mean()
    assert acc > 0.80
