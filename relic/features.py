"""
Packet -> flow -> UNSW-NB15-style features.

The extractor groups packets into bidirectional flows (the side that sends the first
packet is the source), then computes the per-flow fields that UNSW-NB15 defines and that
can be recovered from packet headers and a light look at HTTP/FTP payloads, plus the
"ct_*" connection-count features over the last 100 flows.

Formulas for rate, sload, dload, sinpkt and dinpkt were checked against rows of the
official UNSW-NB15 training file (see tests/test_features.py).
"""
from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

IDLE_TIMEOUT = 120.0  # seconds without packets before a flow is closed

PORT_SERVICE = {80: "http", 8080: "http", 21: "ftp", 20: "ftp-data", 25: "smtp", 53: "dns",
                22: "ssh", 110: "pop3", 6667: "irc", 443: "ssl", 161: "snmp", 162: "snmp",
                67: "dhcp", 68: "dhcp", 1812: "radius", 1813: "radius"}
HTTP_METHODS = (b"GET ", b"POST ", b"PUT ", b"HEAD ", b"DELETE ", b"OPTIONS ", b"PATCH ", b"CONNECT ")
FTP_CMDS = (b"USER", b"PASS", b"RETR", b"STOR", b"LIST", b"CWD", b"PWD", b"QUIT", b"PORT", b"PASV", b"TYPE")

# Columns produced, in UNSW-NB15 order (identifiers first)
ID_COLS = ["flow_id", "start_time", "srcip", "sport", "dstip", "dsport"]
FEATURE_COLS = ["dur", "proto", "service", "state", "spkts", "dpkts", "sbytes", "dbytes", "rate",
                "sttl", "dttl", "sload", "dload", "sloss", "dloss", "sinpkt", "dinpkt", "sjit", "djit",
                "swin", "stcpb", "dtcpb", "dwin", "tcprtt", "synack", "ackdat", "smean", "dmean",
                "trans_depth", "response_body_len", "ct_srv_src", "ct_state_ttl", "ct_dst_ltm",
                "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm", "is_ftp_login", "ct_ftp_cmd",
                "ct_flw_http_mthd", "ct_src_ltm", "ct_srv_dst", "is_sm_ips_ports"]


@dataclass
class Flow:
    proto: str
    src: str
    sport: int
    dst: str
    dport: int
    start: float
    last: float
    s_times: list = field(default_factory=list)
    d_times: list = field(default_factory=list)
    sbytes: int = 0
    dbytes: int = 0
    sttl: int = 0
    dttl: int = 0
    swin: int = 0
    dwin: int = 0
    stcpb: int = 0
    dtcpb: int = 0
    s_seqs: set = field(default_factory=set)
    d_seqs: set = field(default_factory=set)
    sloss: int = 0
    dloss: int = 0
    syn_t: float | None = None
    synack_t: float | None = None
    ack_t: float | None = None
    fin: bool = False
    rst: bool = False
    http_reqs: int = 0
    http_methods: int = 0
    resp_body: int = 0
    ftp_login: int = 0
    ftp_cmds: int = 0

    @property
    def closed(self):
        return self.fin or self.rst


def _service(proto: str, sport: int, dport: int) -> str:
    if proto not in ("tcp", "udp"):
        return "-"
    return PORT_SERVICE.get(dport) or PORT_SERVICE.get(sport) or "-"


def _state(f: Flow) -> str:
    if f.proto == "tcp":
        if f.rst:
            return "RST"
        if f.fin:
            return "FIN"
        if f.syn_t is not None and f.synack_t is None:
            return "REQ"
        return "CON" if f.d_times else "INT"
    if f.proto == "icmp":
        return "ECO" if f.d_times else "INT"
    return "CON" if f.d_times else "INT"


def _iat_ms(times: list) -> tuple[float, float]:
    """Mean and standard deviation of inter-arrival times in milliseconds."""
    if len(times) < 2:
        return 0.0, 0.0
    d = np.diff(np.asarray(times)) * 1000.0
    return float(d.mean()), float(d.std())


def _payload_stats(f: Flow, payload: bytes, from_src: bool):
    if not payload:
        return
    if from_src:
        if payload.startswith(HTTP_METHODS):
            f.http_reqs += 1
            f.http_methods += 1
        if f.dport in (20, 21) or f.sport in (20, 21):
            head = payload[:4].upper()
            if head.startswith(FTP_CMDS):
                f.ftp_cmds += 1
            if head.startswith(b"PASS"):
                f.ftp_login = 1
    else:
        if payload.startswith(b"HTTP/"):
            low = payload.lower()
            i = low.find(b"content-length:")
            if i >= 0:
                try:
                    f.resp_body += int(low[i + 15:].split(b"\r\n", 1)[0].strip())
                except ValueError:
                    pass


def flows_from_packets(packets) -> list[Flow]:
    """Group scapy packets into bidirectional flows."""
    from scapy.layers.inet import IP, TCP, UDP, ICMP

    active: dict[tuple, Flow] = {}
    done: list[Flow] = []
    from scapy.packet import Raw
    for pkt in packets:
        if IP not in pkt:
            # raw-IP captures (link types 101/228) arrive undecoded
            raw = bytes(pkt) if isinstance(pkt, Raw) else b""
            if raw[:1] and raw[0] >> 4 == 4:
                try:
                    t_raw = pkt.time; pkt = IP(raw); pkt.time = t_raw
                except Exception:
                    continue
            if IP not in pkt:
                continue
        ip = pkt[IP]
        t = float(pkt.time)
        if TCP in pkt:
            proto, l4 = "tcp", pkt[TCP]
            sp, dp = int(l4.sport), int(l4.dport)
        elif UDP in pkt:
            proto, l4 = "udp", pkt[UDP]
            sp, dp = int(l4.sport), int(l4.dport)
        elif ICMP in pkt:
            proto, l4, sp, dp = "icmp", pkt[ICMP], 0, 0
        else:
            proto, l4, sp, dp = str(ip.proto), None, 0, 0
        a, b = (ip.src, sp), (ip.dst, dp)
        key = (proto,) + (a + b if a <= b else b + a)

        f = active.get(key)
        if f is not None and (t - f.last > IDLE_TIMEOUT or (f.closed and proto == "tcp"
                                                             and TCP in pkt and pkt[TCP].flags & 0x02
                                                             and not pkt[TCP].flags & 0x10)):
            done.append(active.pop(key)); f = None
        if f is None:
            f = Flow(proto, ip.src, sp, ip.dst, dp, t, t)
            active[key] = f
        f.last = t
        from_src = (ip.src == f.src and sp == f.sport)
        size = int(ip.len)
        if from_src:
            f.s_times.append(t); f.sbytes += size
            if len(f.s_times) == 1:
                f.sttl = int(ip.ttl)
        else:
            f.d_times.append(t); f.dbytes += size
            if len(f.d_times) == 1:
                f.dttl = int(ip.ttl)

        payload = b""
        if proto == "tcp":
            tcp = pkt[TCP]
            flags = int(tcp.flags)
            payload = bytes(tcp.payload)
            if from_src and len(f.s_times) == 1:
                f.swin, f.stcpb = int(tcp.window), int(tcp.seq)
            if not from_src and len(f.d_times) == 1:
                f.dwin, f.dtcpb = int(tcp.window), int(tcp.seq)
            if flags & 0x02 and not flags & 0x10 and f.syn_t is None:
                f.syn_t = t
            elif flags & 0x12 == 0x12 and f.synack_t is None:
                f.synack_t = t
            elif flags & 0x10 and from_src and f.synack_t is not None and f.ack_t is None:
                f.ack_t = t
            if flags & 0x01:
                f.fin = True
            if flags & 0x04:
                f.rst = True
            if payload:  # retransmission = same sequence number with data seen again
                seqs = f.s_seqs if from_src else f.d_seqs
                if int(tcp.seq) in seqs:
                    if from_src: f.sloss += 1
                    else: f.dloss += 1
                seqs.add(int(tcp.seq))
        elif proto == "udp":
            payload = bytes(pkt[UDP].payload)
        _payload_stats(f, payload, from_src)
    done.extend(active.values())
    done.sort(key=lambda x: x.start)
    return done


def features_from_flows(flows: list[Flow]) -> pd.DataFrame:
    rows = []
    for i, f in enumerate(flows):
        dur = max(f.last - f.start, 0.0)
        spk, dpk = len(f.s_times), len(f.d_times)
        sin_m, sjit = _iat_ms(f.s_times)
        din_m, djit = _iat_ms(f.d_times)
        synack = (f.synack_t - f.syn_t) if (f.synack_t is not None and f.syn_t is not None) else 0.0
        ackdat = (f.ack_t - f.synack_t) if (f.ack_t is not None and f.synack_t is not None) else 0.0
        rows.append(dict(
            flow_id=i, start_time=f.start, srcip=f.src, sport=f.sport, dstip=f.dst, dsport=f.dport,
            dur=dur, proto=f.proto, service=_service(f.proto, f.sport, f.dport), state=_state(f),
            spkts=spk, dpkts=dpk, sbytes=f.sbytes, dbytes=f.dbytes,
            rate=((spk + dpk - 1) / dur) if dur > 0 else 0.0,
            sttl=f.sttl, dttl=f.dttl,
            sload=(f.sbytes * 8 * (spk - 1) / spk / dur) if (dur > 0 and spk > 0) else 0.0,
            dload=(f.dbytes * 8 * (dpk - 1) / dpk / dur) if (dur > 0 and dpk > 0) else 0.0,
            sloss=f.sloss, dloss=f.dloss, sinpkt=sin_m, dinpkt=din_m, sjit=sjit, djit=djit,
            swin=f.swin, stcpb=f.stcpb, dtcpb=f.dtcpb, dwin=f.dwin,
            tcprtt=synack + ackdat, synack=synack, ackdat=ackdat,
            smean=int(f.sbytes / spk) if spk else 0, dmean=int(f.dbytes / dpk) if dpk else 0,
            trans_depth=f.http_reqs, response_body_len=f.resp_body,
            is_ftp_login=f.ftp_login, ct_ftp_cmd=f.ftp_cmds, ct_flw_http_mthd=f.http_methods,
            is_sm_ips_ports=int(f.src == f.dst and f.sport == f.dport)))
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=ID_COLS + FEATURE_COLS)
    _add_connection_counts(df)
    return df[ID_COLS + FEATURE_COLS]


def _ttl_band(ttl: int) -> int:
    return 0 if ttl <= 32 else 1 if ttl <= 64 else 2 if ttl <= 128 else 3


def _add_connection_counts(df: pd.DataFrame, window: int = 100):
    """ct_* features: counts among the last `window` flows (this flow included)."""
    keys = {
        "ct_srv_src": lambda r: (r.service, r.srcip),
        "ct_srv_dst": lambda r: (r.service, r.dstip),
        "ct_dst_ltm": lambda r: r.dstip,
        "ct_src_ltm": lambda r: r.srcip,
        "ct_src_dport_ltm": lambda r: (r.srcip, r.dsport),
        "ct_dst_sport_ltm": lambda r: (r.dstip, r.sport),
        "ct_dst_src_ltm": lambda r: (r.srcip, r.dstip),
        "ct_state_ttl": lambda r: (r.state, _ttl_band(r.sttl), _ttl_band(r.dttl)),
    }
    recent = deque()
    counts = {k: defaultdict(int) for k in keys}
    out = {k: [] for k in keys}
    for r in df.itertuples(index=False):
        kv = {k: fn(r) for k, fn in keys.items()}
        recent.append(kv)
        for k, v in kv.items():
            counts[k][v] += 1
        if len(recent) > window:
            old = recent.popleft()
            for k, v in old.items():
                counts[k][v] -= 1
        for k, v in kv.items():
            out[k].append(counts[k][v])
    for k, v in out.items():
        df[k] = v


def _load_layers():
    # register link-layer decoders (Ethernet, Linux cooked, raw IP) with scapy
    import scapy.layers.l2, scapy.layers.inet, scapy.layers.inet6  # noqa: F401
    try:
        import scapy.layers.sll  # noqa: F401  (older scapy keeps SLL elsewhere)
    except ImportError:
        pass


def extract_pcap(path: str, max_packets: int | None = None) -> pd.DataFrame:
    _load_layers()
    from scapy.utils import PcapReader
    pkts = []
    with PcapReader(path) as r:
        for i, p in enumerate(r):
            if max_packets and i >= max_packets:
                break
            pkts.append(p)
    return features_from_flows(flows_from_packets(pkts))


def capture_live(seconds: int = 30, iface: str | None = None) -> pd.DataFrame:
    """Sniff live traffic (needs admin/root; Npcap on Windows) and return flow features."""
    _load_layers()
    from scapy.sendrecv import sniff
    pkts = sniff(timeout=seconds, iface=iface, store=True)
    return features_from_flows(flows_from_packets(pkts))
