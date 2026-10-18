"""Detection engine: load a model bundle, score flows, explain alerts."""
from __future__ import annotations

import os
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")

SEVERITY = {"Normal": "None", "Reconnaissance": "Medium", "Analysis": "Medium", "Fuzzers": "Medium",
            "Generic": "High", "DoS": "High", "Exploits": "High", "Backdoor": "Critical",
            "Shellcode": "Critical", "Worms": "Critical"}


@lru_cache(maxsize=2)
def load(name: str = "live") -> dict:
    return joblib.load(os.path.join(MODEL_DIR, f"relic_{name}.joblib"))


def _prepare(bundle: dict, df: pd.DataFrame) -> np.ndarray:
    X = df.reindex(columns=bundle["features"]).copy()
    for c in bundle["cat"]:
        X[c] = X[c].fillna("-").astype(str).str.strip().str.lower().replace({"": "-"})
    for c in bundle["num"]:
        X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0.0)
    return bundle["pre"].transform(X).astype(np.float32)


def missing_features(bundle: dict, df: pd.DataFrame) -> list[str]:
    return [c for c in bundle["features"] if c not in df.columns]


def score(df: pd.DataFrame, model: str = "live", threshold: float = 0.5) -> pd.DataFrame:
    """Return df with attack probability, verdict, predicted attack type and severity."""
    b = load(model)
    X = _prepare(b, df)
    prob = b["binary"].predict_proba(X)[:, 1]
    types = np.asarray(b["classes"])[b["multiclass"].predict(X).astype(int)]
    out = df.copy()
    out["attack_prob"] = prob
    out["verdict"] = np.where(prob >= threshold, "Attack", "Normal")
    # attack type is only meaningful for flows flagged as attacks
    t = np.where(out["verdict"] == "Attack", types, "Normal")
    t = np.where((out["verdict"] == "Attack") & (t == "Normal"), "Unknown", t)
    out["attack_type"] = t
    out["severity"] = [SEVERITY.get(x, "Medium") if v == "Attack" else "None"
                       for x, v in zip(out["attack_type"], out["verdict"])]
    return out


def explain(df_row: pd.DataFrame, model: str = "live", top: int = 10) -> pd.DataFrame:
    """SHAP contributions (log-odds) of the top features for one flow."""
    import shap
    b = load(model)
    X = _prepare(b, df_row)
    sv = shap.TreeExplainer(b["binary"]).shap_values(X)[0]
    names = b["enc_names"]
    # fold one-hot columns back to their raw feature (e.g. service_http -> service)
    raw = []
    for n in names:
        base = next((c for c in b["cat"] if n.startswith(c + "_")), n)
        raw.append(base)
    s = pd.Series(sv, index=raw).groupby(level=0).sum()
    vals = df_row.iloc[0]
    res = pd.DataFrame({"feature": s.index, "contribution": s.values,
                        "value": [vals.get(f, None) for f in s.index]})
    res["abs"] = res["contribution"].abs()
    return res.sort_values("abs", ascending=False).head(top).drop(columns="abs").reset_index(drop=True)
