"""SQLite history of analysis runs and their alerts."""
from __future__ import annotations

import json
import os
import sqlite3
import time

import pandas as pd

DB = os.environ.get("RELIC_DB", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                             "relic_history.db"))
ALERT_COLS = ["flow_id", "start_time", "srcip", "sport", "dstip", "dsport", "proto", "service", "state",
              "attack_type", "severity", "layer", "evidence", "attack_prob", "anomaly_score",
              "spkts", "dpkts", "sbytes", "dbytes", "dur"]


def _conn():
    c = sqlite3.connect(DB)
    c.execute("""CREATE TABLE IF NOT EXISTS runs(id INTEGER PRIMARY KEY AUTOINCREMENT, created REAL,
                 source TEXT, mode TEXT, flows INTEGER, alerts INTEGER, settings TEXT)""")
    c.execute(f"""CREATE TABLE IF NOT EXISTS alerts(run_id INTEGER, {", ".join(c_ + " TEXT" for c_ in ALERT_COLS)})""")
    return c


RUN_DIR = os.path.join(os.path.dirname(DB), "runs")


def run_path(run_id: int) -> str:
    return os.path.join(RUN_DIR, f"run_{int(run_id)}.csv.gz")


def load_run_flows(run_id: int) -> pd.DataFrame | None:
    p = run_path(run_id)
    return pd.read_csv(p) if os.path.exists(p) else None


def save_run(source: str, mode: str, res: pd.DataFrame, settings: dict) -> int:
    alerts = res[res["verdict"] == "Attack"].reindex(columns=ALERT_COLS)
    with _conn() as c:
        cur = c.execute("INSERT INTO runs(created, source, mode, flows, alerts, settings) VALUES(?,?,?,?,?,?)",
                        (time.time(), source, mode, len(res), len(alerts), json.dumps(settings, default=str)))
        run_id = cur.lastrowid
        alerts = alerts.astype(object).where(alerts.notna(), None)
        alerts.insert(0, "run_id", run_id)
        alerts.to_sql("alerts", c, if_exists="append", index=False)
    os.makedirs(RUN_DIR, exist_ok=True)
    res.to_csv(run_path(run_id), index=False)  # full flows, so alerts can be explained later
    return run_id


def runs() -> pd.DataFrame:
    with _conn() as c:
        df = pd.read_sql("SELECT * FROM runs ORDER BY id DESC", c)
    if not df.empty:
        df["created"] = pd.to_datetime(df["created"], unit="s").dt.strftime("%Y-%m-%d %H:%M")
    return df


def alerts(run_id: int) -> pd.DataFrame:
    with _conn() as c:
        df = pd.read_sql("SELECT * FROM alerts WHERE run_id = ?", c, params=(int(run_id),))
    for col in ("attack_prob", "anomaly_score", "spkts", "dpkts", "sbytes", "dbytes", "dur", "start_time"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def delete_run(run_id: int):
    with _conn() as c:
        c.execute("DELETE FROM alerts WHERE run_id = ?", (int(run_id),))
        c.execute("DELETE FROM runs WHERE id = ?", (int(run_id),))
    if os.path.exists(run_path(run_id)):
        os.remove(run_path(run_id))
