#!/usr/bin/env python3
"""
Kernel check of the XDP counters that the models read (Linux, root).

Loads the real XDP program (BCC, generic mode) on a veth pair, sends packets
with known headers and timing from a separate network namespace, and compares
what the kernel maps report with what was sent:

  - flow map (level 3): packets, IP-length sum/min/max/sum of squares, first/last
    time, inter-arrival sum of squares and maximum, SYN/ACK/FIN/RST/PSH counts
  - per-source counters (level 2): packets, bytes (IP length), SYN-without-ACK
  - totals (level 1): packets and bytes (IP length)
  - a blacklisted source is dropped by XDP and never reaches the stack
  - SYN guard: a client sending 1,200 SYNs over more than a second loses nothing, a burst of
    SYNs above 1,000 per second is dropped, and the same source is served again a second later

    sudo python3 scripts/check_xdp_flows.py
Exits 1 if any check fails. Writes logs/check_xdp_flows.json.
"""

import json
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

NS, VA, VB = 'xdpcheck', 'xchk0', 'xchk1'
SRC_A, SRC_B, SRC_C, SRC_D, DST = '10.77.0.1', '10.77.0.2', '10.77.0.3', '10.77.0.4', '10.77.0.9'


def sh(cmd):
    return subprocess.run(cmd, shell=True, check=False, capture_output=True, text=True)


def setup():
    sh(f'ip link del {VB}; ip netns del {NS}')
    for c in (f'ip netns add {NS}', f'ip link add {VA} type veth peer name {VB}',
              f'ip link set {VA} netns {NS}', f'ip netns exec {NS} ip link set {VA} up',
              f'ip link set {VB} up'):
        r = sh(c)
        if r.returncode:
            sys.exit(f'{c}: {r.stderr}')


def send(script: str):
    """Run scapy in the sender namespace (packets leave VA and arrive on VB, where XDP runs)"""
    r = subprocess.run(['ip', 'netns', 'exec', NS, sys.executable, '-c', script], capture_output=True, text=True)
    if r.returncode:
        sys.exit('sender failed: ' + r.stderr[-800:])


def main():
    import numpy as np
    from src.config import EBPF_PROGRAM_PATH
    from src.core.traffic_monitor import TrafficMonitor

    setup()
    mon = TrafficMonitor(VB, 'generic')
    if not mon.load_xdp_program(EBPF_PROGRAM_PATH):
        sys.exit('XDP program did not load')
    checks = {}
    try:
        mon.get_flow_records(clear=True)
        s0 = mon.get_statistics()
        # Flow 1: TCP SYN, SYN, ACK, PSH+ACK, FIN+ACK with 0.2/0.3/0.1/0.4 s gaps, payloads 0/0/0/100/0
        # Flow 2: 3 UDP packets of 60/80/100 payload bytes; source C: 4 UDP packets, then blocked
        send(f'''
import time
from scapy.all import Ether, IP, TCP, UDP, Raw, sendp
e = Ether(dst="ff:ff:ff:ff:ff:ff")
tcp = [("S", 0, 0.0), ("S", 0, 0.2), ("A", 0, 0.3), ("PA", 100, 0.1), ("FA", 0, 0.4)]
for flags, n, gap in tcp:
    time.sleep(gap)
    sendp(e/IP(src="{SRC_A}", dst="{DST}")/TCP(sport=40000, dport=80, flags=flags)/Raw(b"x"*n), iface="{VA}", verbose=0)
for n in (60, 80, 100):
    sendp(e/IP(src="{SRC_B}", dst="{DST}")/UDP(sport=53, dport=3333)/Raw(b"y"*n), iface="{VA}", verbose=0)
for _ in range(4):
    sendp(e/IP(src="{SRC_C}", dst="{DST}")/UDP(sport=5000, dport=6000)/Raw(b"z"*10), iface="{VA}", verbose=0)
''')
        time.sleep(0.5)
        rec = mon.get_flow_records(clear=True)
        ips = mon.get_ip_interval_statistics(limit=100)
        s1 = mon.get_statistics()

        def flow(src):
            i = np.flatnonzero(rec['src'] == src)
            assert len(i) == 1, f'expected one flow from {src}, got {len(i)}'
            return {k: float(v[i[0]]) for k, v in rec.items() if k != 'src'}

        a, b = flow(SRC_A), flow(SRC_B)
        lens_a = [40, 40, 40, 140, 40]
        checks['tcp_flow'] = {
            'packets': (a['packets'], 5), 'bytes': (a['bytes'], sum(lens_a)),
            'len_min': (a['len_min'], 40), 'len_max': (a['len_max'], 140),
            'len_sumsq': (a['len_sumsq'], sum(x * x for x in lens_a)),
            'syn': (a['syn'], 2), 'ack': (a['ack'], 3), 'fin': (a['fin'], 1), 'psh': (a['psh'], 1),
            'sport': (a['sport'], 40000), 'dport': (a['dport'], 80), 'proto': (a['proto'], 6),
        }
        timing = {'duration_s': a['t_last'] - a['t_first'], 'iat_max_s': a['iat_max'],
                  'iat_sumsq_s2': a['iat_sumsq']}
        lens_b = [28 + n for n in (60, 80, 100)]
        checks['udp_flow'] = {'packets': (b['packets'], 3), 'bytes': (b['bytes'], sum(lens_b)),
                              'len_min': (b['len_min'], min(lens_b)), 'len_max': (b['len_max'], max(lens_b)),
                              'sport': (b['sport'], 53)}
        by_ip = {e['ip']: e for e in ips}
        checks['per_source'] = {'A packets': (by_ip[SRC_A]['packets'], 5),
                                'A syn (SYN without ACK)': (by_ip[SRC_A]['syn_count'], 2),
                                'B bytes (IP length)': (by_ip[SRC_B]['bytes'], sum(lens_b))}
        checks['totals'] = {'packets': (s1['total_packets'] - s0['total_packets'], 12),
                            'bytes (IP length)': (s1['total_bytes'] - s0['total_bytes'],
                                                  sum(lens_a) + sum(lens_b) + 4 * 38)}

        # Block C, send 5 more: all must be dropped by XDP
        mon.add_to_blacklist(SRC_C, ttl_s=30)
        d0 = mon.get_statistics()['dropped_packets']
        send(f'''
from scapy.all import Ether, IP, UDP, sendp
for _ in range(5):
    sendp(Ether(dst="ff:ff:ff:ff:ff:ff")/IP(src="{SRC_C}", dst="{DST}")/UDP(sport=5000, dport=6000), iface="{VA}", verbose=0)
''')
        time.sleep(0.3)
        checks['blacklist'] = {'dropped': (mon.get_statistics()['dropped_packets'] - d0, 5)}

        # SYN guard, source D: 600 + 600 SYNs with a 1.2 s pause (1,200 in total), a burst of
        # 3,000 as fast as scapy can send, then 5 SYNs a little over a second later
        def syns(n):
            send(f'''
from scapy.all import Ether, IP, TCP, sendp
sendp([Ether(dst="ff:ff:ff:ff:ff:ff")/IP(src="{SRC_D}", dst="{DST}")/TCP(sport=41000 + i, dport=80, flags="S")
       for i in range({n})], iface="{VA}", verbose=0)
''')
            time.sleep(0.2)

        d0 = mon.get_statistics()['dropped_packets']
        syns(600)
        time.sleep(1.2)
        syns(600)
        slow = mon.get_statistics()['dropped_packets'] - d0
        time.sleep(1.2)
        d1 = mon.get_statistics()['dropped_packets']
        syns(3000)
        burst = mon.get_statistics()['dropped_packets'] - d1
        time.sleep(1.2)
        d2 = mon.get_statistics()['dropped_packets']
        syns(5)
        after = mon.get_statistics()['dropped_packets'] - d2
        checks['syn_guard'] = {'1,200 slow SYNs dropped': (slow, 0), 'dropped a second after burst': (after, 0),
                               'burst dropped (any > 0)': (int(burst > 0), 1)}
        burst_dropped = burst
    finally:
        mon.unload_xdp_program()
        sh(f'ip link del {VB}; ip netns del {NS}')

    ok = True
    report = {'checks': {}, 'timing (sender sleeps 0.2+0.3+0.1+0.4 s)': timing,
              'syn_burst_packets_dropped_of_3000': int(burst_dropped)}
    for group, items in checks.items():
        report['checks'][group] = {}
        for name, (got, want) in items.items():
            good = abs(got - want) < 1e-6
            ok &= good
            report['checks'][group][name] = {'kernel': got, 'expected': want, 'ok': good}
            print(f"{'OK  ' if good else 'FAIL'} {group:10s} {name:26s} kernel {got:>10.0f}  expected {want}")
    # Sender sleeps are not exact; the gaps must add up and the largest must be the 0.4 s one
    t_ok = 0.95 <= timing['duration_s'] <= 1.3 and 0.38 <= timing['iat_max_s'] <= 0.6
    ok &= t_ok
    print(f"{'OK  ' if t_ok else 'FAIL'} timing     duration {timing['duration_s']:.3f} s (sent 1.0 s), "
          f"largest gap {timing['iat_max_s']:.3f} s (sent 0.4 s)")
    report['all_ok'] = bool(ok)
    out = PROJECT_ROOT / 'logs' / 'check_xdp_flows.json'
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    print('ALL OK' if ok else 'SOME CHECKS FAILED', '->', out)
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
