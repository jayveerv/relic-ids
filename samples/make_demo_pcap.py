"""
Generate a synthetic demo capture with known ground truth.

Scenario (120 s, one /24 office network with a web, DNS and SSH server):
  Normal       : clients 10.0.0.11-30 browse the web server, resolve names, use SSH
  Reconnaissance: 192.168.56.66 SYN-scans ports 1-1024 of the web server
  DoS          : spoofed 172.16.x.x hosts SYN-flood port 80; 192.168.56.77 UDP-floods DNS
  Brute force  : 192.168.56.88 sends rapid HTTP POST /login attempts

Writes samples/demo_traffic.pcap and samples/demo_ground_truth.csv (label per source IP).
All traffic is crafted offline with scapy; nothing is sent on a network.
"""
import os, random
import pandas as pd
from scapy.layers.inet import IP, TCP, UDP
from scapy.layers.l2 import Ether
from scapy.layers.dns import DNS, DNSQR, DNSRR
from scapy.packet import Raw
from scapy.utils import wrpcap

import sys
BASELINE = '--baseline' in sys.argv
random.seed(11 if BASELINE else 7)
HERE = os.path.dirname(os.path.abspath(__file__))
WEB, DNS_SRV, SSH = "10.0.0.100", "10.0.0.53", "10.0.0.22"
T0 = 1_760_000_000.0
pkts = []

def add(p, t):
    p = Ether(src='02:00:00:00:00:01', dst='02:00:00:00:00:02')/p
    p.time = t; pkts.append(p)

def tcp_session(cli, srv, dport, t, req=b"", resp=b"", cli_ttl=128, srv_ttl=64, rtt=0.02, extra=0):
    sp = random.randint(40000, 60000); cs, ss = random.randint(1, 2**31), random.randint(1, 2**31)
    add(IP(src=cli, dst=srv, ttl=cli_ttl)/TCP(sport=sp, dport=dport, flags="S", seq=cs, window=64240), t)
    add(IP(src=srv, dst=cli, ttl=srv_ttl)/TCP(sport=dport, dport=sp, flags="SA", seq=ss, ack=cs+1, window=65160), t+rtt)
    t += rtt * 1.5
    add(IP(src=cli, dst=srv, ttl=cli_ttl)/TCP(sport=sp, dport=dport, flags="A", seq=cs+1, ack=ss+1, window=64240), t)
    cs += 1; ss += 1
    for _ in range(1 + extra):
        if req:
            t += 0.002
            add(IP(src=cli, dst=srv, ttl=cli_ttl)/TCP(sport=sp, dport=dport, flags="PA", seq=cs, ack=ss)/Raw(req), t)
            cs += len(req)
        if resp:
            t += rtt
            add(IP(src=srv, dst=cli, ttl=srv_ttl)/TCP(sport=dport, dport=sp, flags="PA", seq=ss, ack=cs)/Raw(resp), t)
            ss += len(resp)
    t += 0.01
    add(IP(src=cli, dst=srv, ttl=cli_ttl)/TCP(sport=sp, dport=dport, flags="FA", seq=cs, ack=ss), t)
    add(IP(src=srv, dst=cli, ttl=srv_ttl)/TCP(sport=dport, dport=sp, flags="FA", seq=ss, ack=cs+1), t+rtt)
    add(IP(src=cli, dst=srv, ttl=cli_ttl)/TCP(sport=sp, dport=dport, flags="A", seq=cs+1, ack=ss+1), t+rtt*1.5)

labels = []
# ---------------- normal traffic
clients = [f"10.0.0.{i}" for i in range(11, 31)]
for c in clients:
    labels.append((c, "Normal"))
for _ in range(260):
    c = random.choice(clients); t = T0 + random.uniform(0, 118)
    kind = random.random()
    if kind < 0.55:
        body = b"x" * random.randint(300, 4000)
        resp = b"HTTP/1.1 200 OK\r\nContent-Type: text/html\r\nContent-Length: %d\r\n\r\n" % len(body) + body[:1200]
        tcp_session(c, WEB, 80, t, b"GET /index.html HTTP/1.1\r\nHost: intranet\r\n\r\n", resp,
                    rtt=random.uniform(0.005, 0.04), extra=random.randint(0, 3))
    elif kind < 0.85:
        sp = random.randint(40000, 60000); name = random.choice(["intranet.local", "mail.local", "update.local"])
        q = IP(src=c, dst=DNS_SRV, ttl=128)/UDP(sport=sp, dport=53)/DNS(id=random.randint(0, 65535), rd=1, qd=DNSQR(qname=name))
        add(q, t)
        add(IP(src=DNS_SRV, dst=c, ttl=64)/UDP(sport=53, dport=sp)/DNS(id=q[DNS].id if DNS in q else 0, qr=1, qd=DNSQR(qname=name),
            an=DNSRR(rrname=name, rdata="10.0.0.100")), t + random.uniform(0.001, 0.01))
    else:
        tcp_session(c, SSH, 22, t, b"\x00" * random.randint(40, 200), b"\x00" * random.randint(40, 400),
                    rtt=random.uniform(0.003, 0.02), extra=random.randint(5, 20))

if BASELINE:
    pkts.sort(key=lambda p: p.time)
    wrpcap(os.path.join(HERE, 'demo_baseline_normal.pcap'), pkts)
    print(f'wrote {len(pkts)} baseline packets'); sys.exit(0)

# ---------------- reconnaissance: SYN scan
scanner = "192.168.56.66"; labels.append((scanner, "Reconnaissance")); t = T0 + 20
for port in range(1, 1025):
    sp = random.randint(30000, 65000); t += 0.002
    add(IP(src=scanner, dst=WEB, ttl=64)/TCP(sport=sp, dport=port, flags="S", seq=random.randint(1, 2**31), window=1024), t)
    if port in (22, 80):
        add(IP(src=WEB, dst=scanner, ttl=64)/TCP(sport=port, dport=sp, flags="SA", seq=random.randint(1, 2**31), window=65160), t + 0.0004)
        add(IP(src=scanner, dst=WEB, ttl=64)/TCP(sport=sp, dport=port, flags="R"), t + 0.0006)
    else:
        add(IP(src=WEB, dst=scanner, ttl=64)/TCP(sport=port, dport=sp, flags="RA", seq=0), t + 0.0004)

# ---------------- DoS: SYN flood from spoofed sources
t = T0 + 50
for i in range(1500):
    src = f"172.16.{random.randint(0, 255)}.{random.randint(1, 254)}"; t += 0.0013
    add(IP(src=src, dst=WEB, ttl=random.choice([64, 128, 255]))/TCP(sport=random.randint(1024, 65535), dport=80,
        flags="S", seq=random.randint(1, 2**31), window=512), t)
labels.append(("172.16.0.0/16", "DoS"))

# ---------------- DoS: UDP flood on DNS
flooder = "192.168.56.77"; labels.append((flooder, "DoS")); t = T0 + 70
for i in range(1200):
    t += 0.0015
    add(IP(src=flooder, dst=DNS_SRV, ttl=64)/UDP(sport=random.randint(1024, 65535), dport=53)/Raw(os.urandom(512)), t)

# ---------------- brute force: rapid HTTP login attempts
brute = "192.168.56.88"; labels.append((brute, "Brute force")); t = T0 + 90
for i in range(150):
    t += 0.06
    form = b"user=admin&pass=%06d" % random.randint(0, 999999)
    req = b"POST /login HTTP/1.1\r\nHost: intranet\r\nContent-Length: %d\r\n\r\n" % len(form) + form
    tcp_session(brute, WEB, 80, t, req, b"HTTP/1.1 401 Unauthorized\r\nContent-Length: 0\r\n\r\n",
                cli_ttl=64, rtt=0.004)

pkts.sort(key=lambda p: p.time)
wrpcap(os.path.join(HERE, "demo_traffic.pcap"), pkts)
pd.DataFrame(labels, columns=["source", "label"]).to_csv(os.path.join(HERE, "demo_ground_truth.csv"), index=False)
print(f"wrote {len(pkts)} packets")
