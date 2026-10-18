"""Combine the three detection layers into one verdict per flow."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import engine, rules, baseline as bl_mod
from .features import FEATURE_COLS

RULE_SEVERITY = {"DoS": "High", "Reconnaissance": "Medium", "Brute force": "High"}


def is_unsw_format(df: pd.DataFrame) -> bool:
    """A CSV with every UNSW-NB15 feature but no flow identifiers (the public dataset files)."""
    return all(c in df.columns for c in FEATURE_COLS) and "srcip" not in df.columns


def analyze_dataset(df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """UNSW-NB15-format records: the full-feature classifier is the detector."""
    out = engine.score(df, "full", threshold)
    out["layer"] = np.where(out["verdict"] == "Attack", "ML model", "")
    out["evidence"] = [f"Model attack probability {p:.2f}" if v == "Attack" else ""
                       for p, v in zip(out["attack_prob"], out["verdict"])]
    out["anomaly_score"] = np.nan
    if "flow_id" not in out.columns:
        out.insert(0, "flow_id", range(len(out)))
    return out


def analyze_flows(flows: pd.DataFrame, rule_cfg: rules.RuleConfig | None = None, baseline: dict | None = None,
                  ml_threshold: float = 0.5, use_ml_in_verdict: bool = False) -> pd.DataFrame:
    """Flows from a PCAP or live capture: behaviour rules + baseline anomaly (+ optional ML)."""
    if flows.empty:
        return flows.assign(verdict=[], layer=[], attack_type=[], severity=[], evidence=[],
                            attack_prob=[], anomaly_score=[])
    out = engine.score(flows, "live", ml_threshold).rename(
        columns={"verdict": "ml_verdict", "attack_type": "ml_type", "severity": "ml_severity"})
    hits = rules.run_rules(flows, rule_cfg)
    out = out.merge(hits, on="flow_id", how="left")
    if baseline is not None:
        out["anomaly_score"] = bl_mod.score(baseline, flows)
        anom = out["anomaly_score"] >= baseline["threshold"]
    else:
        out["anomaly_score"] = np.nan
        anom = pd.Series(False, index=out.index)
    rule_hit = out["rule"].notna()
    ml_hit = (out["ml_verdict"] == "Attack") & use_ml_in_verdict

    out["verdict"] = np.where(rule_hit | anom | ml_hit, "Attack", "Normal")
    layers = []
    for r, a, m in zip(rule_hit, anom, ml_hit):
        layers.append(" + ".join([n for n, on in (("Rule", r), ("Anomaly", a), ("ML model", m)) if on]))
    out["layer"] = layers
    out["attack_type"] = np.where(rule_hit, out["attack_type"],
                                  np.where(anom | ml_hit, "Anomaly", "Normal"))
    sev = []
    for t, r, v in zip(out["attack_type"], rule_hit, out["verdict"]):
        sev.append("None" if v == "Normal" else RULE_SEVERITY.get(t, "Medium") if r else "Low")
    out["severity"] = sev
    ev = []
    for e, a, s, v in zip(out["evidence"], anom, out["anomaly_score"], out["verdict"]):
        parts = [e] if isinstance(e, str) else []
        if a:
            parts.append(f"Unusual for this network (anomaly score {s:.2f})")
        ev.append("; ".join(parts) if v == "Attack" else "")
    out["evidence"] = ev
    return out.drop(columns=["ml_severity"])


def summary(res: pd.DataFrame) -> dict:
    alerts = res[res["verdict"] == "Attack"]
    return dict(flows=len(res), alerts=len(alerts),
                alert_rate=float(len(alerts) / len(res)) if len(res) else 0.0,
                by_type=alerts["attack_type"].value_counts().to_dict(),
                by_severity=alerts["severity"].value_counts().to_dict(),
                sources=int(alerts["srcip"].nunique()) if "srcip" in alerts else 0)
