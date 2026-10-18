"""
Behaviour rules: detect attacks from how hosts behave across many flows.

Lab-trained classifiers transfer poorly to new networks, but these patterns hold anywhere:
a scanner touches many ports, a flood sends many half-open or one-way flows, a brute-force
client opens many short sessions to one login service. Each rule works on flows produced by
relic.features and returns the flow ids it matches.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import pandas as pd


@dataclass
class RuleConfig:
    window_s: float = 60.0           # time window for counting
    scan_ports: int = 20             # distinct destination ports from one source to one host
    sweep_hosts: int = 20            # distinct destination hosts on one port from one source
    syn_flood_req: int = 100         # half-open (REQ) flows to one host:port within flood_window_s
    flood_window_s: float = 10.0
    udp_flood_flows: int = 200       # one-way UDP flows from one source to one host within flood_window_s
    icmp_flood_flows: int = 200
    brute_sessions: int = 30         # sessions from one source to one login service within window_s
    brute_max_dur_s: float = 2.0     # brute-force sessions are short

    def as_dict(self):
        return asdict(self)


LOGIN_SERVICES = {"http", "ssh", "ftp", "smtp", "pop3"}


def _windows(df: pd.DataFrame, size: float) -> pd.Series:
    t0 = df["start_time"].min()
    return ((df["start_time"] - t0) // size).astype(int)


def run_rules(flows: pd.DataFrame, cfg: RuleConfig | None = None) -> pd.DataFrame:
    """Return one row per matched flow: flow_id, rule, attack_type, evidence."""
    cfg = cfg or RuleConfig()
    if flows.empty:
        return pd.DataFrame(columns=["flow_id", "rule", "attack_type", "evidence"])
    f = flows.copy()
    f["w"] = _windows(f, cfg.window_s)
    f["fw"] = _windows(f, cfg.flood_window_s)
    hits = []

    def add(ids, rule, typ, evidence):
        for i in ids:
            hits.append((int(i), rule, typ, evidence))

    # 1. vertical port scan: one source -> one host, many ports
    g = f[f.proto == "tcp"].groupby(["srcip", "dstip", "w"])
    for (src, dst, _), grp in g:
        n = grp["dsport"].nunique()
        if n >= cfg.scan_ports and (grp["spkts"] <= 3).mean() > 0.8:
            add(grp.flow_id, "Port scan", "Reconnaissance", f"{src} probed {n} ports on {dst}")
    # 2. horizontal sweep: one source -> many hosts, same port
    for (src, port, _), grp in f.groupby(["srcip", "dsport", "w"]):
        n = grp["dstip"].nunique()
        if n >= cfg.sweep_hosts and (grp["spkts"] <= 3).mean() > 0.8:
            add(grp.flow_id, "Host sweep", "Reconnaissance", f"{src} probed port {port} on {n} hosts")
    # 3. SYN flood: many half-open connections to one service
    req = f[(f.proto == "tcp") & (f.state == "REQ")]
    for (dst, port, _), grp in req.groupby(["dstip", "dsport", "fw"]):
        if len(grp) >= cfg.syn_flood_req:
            add(grp.flow_id, "SYN flood", "DoS",
                f"{len(grp)} half-open connections to {dst}:{port} from {grp['srcip'].nunique()} sources "
                f"in {cfg.flood_window_s:.0f} s")
    # 4. UDP / ICMP flood: many one-way flows from one source to one host
    for proto, limit, name in (("udp", cfg.udp_flood_flows, "UDP flood"), ("icmp", cfg.icmp_flood_flows, "ICMP flood")):
        one_way = f[(f.proto == proto) & (f.dpkts == 0)]
        for (src, dst, _), grp in one_way.groupby(["srcip", "dstip", "fw"]):
            if len(grp) >= limit:
                add(grp.flow_id, name, "DoS",
                    f"{len(grp)} unanswered {proto.upper()} flows from {src} to {dst} in {cfg.flood_window_s:.0f} s")
    # 5. brute force: many short sessions to a login service
    login = f[f.service.isin(LOGIN_SERVICES) & (f.dur <= cfg.brute_max_dur_s) & (f.dpkts > 0)]
    for (src, dst, svc, _), grp in login.groupby(["srcip", "dstip", "service", "w"]):
        if len(grp) >= cfg.brute_sessions:
            add(grp.flow_id, "Brute force", "Brute force",
                f"{len(grp)} short {svc} sessions from {src} to {dst} in {cfg.window_s:.0f} s")

    out = pd.DataFrame(hits, columns=["flow_id", "rule", "attack_type", "evidence"])
    # one rule per flow (first match wins, in the order above)
    return out.drop_duplicates("flow_id").reset_index(drop=True)
