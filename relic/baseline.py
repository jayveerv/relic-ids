"""
Baseline anomaly model: learn what normal looks like on *your* network, flag what differs.

An Isolation Forest is fitted on flows from a capture of normal traffic (the baseline).
Scores are calibrated so that a chosen share of the baseline flows (default 1%) would be
flagged, which sets the expected false-alarm rate on traffic like the baseline.
"""
from __future__ import annotations

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

NUM = ["dur", "spkts", "dpkts", "sbytes", "dbytes", "rate", "sload", "dload", "sinpkt", "dinpkt",
       "sjit", "djit", "tcprtt", "smean", "dmean", "trans_depth", "response_body_len",
       "ct_srv_src", "ct_dst_ltm", "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm",
       "ct_src_ltm", "ct_srv_dst"]
CAT = {"proto": ["tcp", "udp", "icmp"], "state": ["FIN", "CON", "INT", "REQ", "RST", "ECO"],
       "service": ["-", "http", "dns", "ssh", "ftp", "smtp", "ssl", "snmp", "dhcp", "pop3"]}


def _matrix(df: pd.DataFrame) -> np.ndarray:
    X = np.log1p(df[NUM].astype(float).clip(lower=0).values)
    oh = [((df[c].astype(str) == v).astype(float)).values[:, None] for c, vals in CAT.items() for v in vals]
    return np.hstack([X] + oh)


def fit(normal_flows: pd.DataFrame, fp_target: float = 0.01, seed: int = 0) -> dict:
    X = _matrix(normal_flows)
    iso = IsolationForest(n_estimators=200, contamination="auto", random_state=seed).fit(X)
    s = -iso.score_samples(X)              # higher = more anomalous
    thr = float(np.quantile(s, 1 - fp_target))
    return dict(model=iso, threshold=thr, fp_target=fp_target, n_flows=len(normal_flows))


def score(bl: dict, flows: pd.DataFrame) -> np.ndarray:
    return -bl["model"].score_samples(_matrix(flows))


def save(bl: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(bl, path)


def load(path: str) -> dict | None:
    return joblib.load(path) if os.path.exists(path) else None
