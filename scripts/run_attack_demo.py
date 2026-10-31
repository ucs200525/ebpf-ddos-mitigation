#!/usr/bin/env python3
"""
DDoS Attack Simulation Demo
Posts crafted attack traffic stats to the LIVE running system via
the /api/simulate_attack endpoint so alerts appear in the dashboard.

Usage:
  python3 run_attack_demo.py
  python3 run_attack_demo.py --url http://127.0.0.1:5000
"""

import sys
import time
import random
import argparse
import requests

BASE_URL = "http://127.0.0.1:5000"
STEP_SECONDS = 1.0  # one sample per second, like the live monitoring loop

# Box-drawing/emoji output must not crash on non-UTF-8 consoles or pipes
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding='utf-8', errors='replace')
    except (AttributeError, ValueError):
        pass

# ── ANSI colours ──────────────────────────────────────────────────────────────
C = {
    'red':    '\033[91m', 'green':  '\033[92m', 'yellow': '\033[93m',
    'blue':   '\033[94m', 'purple': '\033[95m', 'cyan':   '\033[96m',
    'bold':   '\033[1m',  'reset':  '\033[0m',
}
def c(color, text): return f"{C.get(color,'')}{text}{C['reset']}"

def banner(title, color='cyan'):
    w = 64
    print(f"\n{c(color,'═'*w)}")
    print(f"{c(color,'║')} {c('bold', title):<{w-3}}{c(color,'║')}")
    print(f"{c(color,'═'*w)}")


# ── Traffic generators ────────────────────────────────────────────────────────

def rand_ip():
    subnets = ['45.33', '185.220', '91.108', '198.51', '203.0', '89.248']
    s = random.choice(subnets)
    return f"{s}.{random.randint(0,255)}.{random.randint(1,254)}"

def make_ip_stats(num_ips, total_pkts, attack='udp'):
    ips, remaining = [], total_pkts
    for i in range(num_ips):
        share = remaining if i == num_ips - 1 else random.randint(
            1, max(1, int(remaining / max(1, num_ips - i) * 2)))
        share = min(share, remaining)
        remaining -= share
        syn = int(share * 0.9) if attack == 'syn' else random.randint(0, 3)
        udp = int(share * 0.9) if attack == 'udp' else 0
        ips.append({'ip': rand_ip(), 'packets': share, 'bytes': share * 900,
                    'flow_count': random.randint(1, 20),
                    'syn_count': syn, 'udp_count': udp,
                    'last_seen': int(time.time() * 1e9)})
        if remaining <= 0:
            break
    return ips

def normal_stats(cumulative, pps=50):
    pps = max(0, int(pps + random.gauss(0, 5)))
    tcp = int(pps * random.uniform(0.82, 0.88))
    udp = int(pps * random.uniform(0.06, 0.12))
    icmp = pps - tcp - udp
    bps  = pps * random.randint(400, 1400)
    return {
        'total_packets':   cumulative['total_packets']   + pps,
        'total_bytes':     cumulative['total_bytes']     + bps,
        'tcp_packets':     cumulative['tcp_packets']     + tcp,
        'udp_packets':     cumulative['udp_packets']     + udp,
        'icmp_packets':    cumulative['icmp_packets']    + icmp,
        'dropped_packets': cumulative['dropped_packets'],
        'dropped_bytes':   cumulative['dropped_bytes'],
        'passed_packets':  cumulative['passed_packets']  + pps,
        'passed_bytes':    cumulative['passed_bytes']    + bps,
        'other_packets':   cumulative['other_packets'],
    }, pps

def attack_stats(cumulative, pps, attack='udp'):
    pps = max(1, int(pps + random.gauss(0, pps * 0.03)))
    if attack == 'udp':
        tcp=int(pps*0.02); udp=int(pps*0.96); icmp=pps-tcp-udp
        bps=pps*random.randint(800,1400)
    elif attack == 'syn':
        tcp=pps; udp=0; icmp=0; bps=pps*64
    elif attack == 'icmp':
        tcp=0; udp=0; icmp=pps; bps=pps*64
    else:  # mixed
        tcp=int(pps*0.35); udp=int(pps*0.55); icmp=pps-tcp-udp
        bps=pps*random.randint(500,1200)
    return {
        'total_packets':   cumulative['total_packets']   + pps,
        'total_bytes':     cumulative['total_bytes']     + bps,
        'tcp_packets':     cumulative['tcp_packets']     + tcp,
        'udp_packets':     cumulative['udp_packets']     + udp,
        'icmp_packets':    cumulative['icmp_packets']    + icmp,
        'dropped_packets': cumulative['dropped_packets'],
        'dropped_bytes':   cumulative['dropped_bytes'],
        'passed_packets':  cumulative['passed_packets'],
        'passed_bytes':    cumulative['passed_bytes'],
        'other_packets':   cumulative['other_packets'],
    }, pps


# ── API helpers ───────────────────────────────────────────────────────────────

def post_stats(stats, ip_stats, label, url=BASE_URL):
    try:
        r = requests.post(
            f"{url}/api/simulate_attack",
            json={'stats': stats, 'ip_stats': ip_stats, 'label': label,
                  'interval_seconds': STEP_SECONDS},
            timeout=3,
        )
        if r.status_code == 200:
            return r.json()
        print(f"  {c('red','[!] API error:')} HTTP {r.status_code}: {r.text[:200]}")
    except Exception as e:
        print(f"  {c('red','[!] API error:')} {e}")
    return {}

def describe(r):
    """One-line verdict for a simulate_attack response"""
    if r.get('is_anomaly'):
        src = r.get('detection_source', '?')
        kind = r.get('attack_type', '')
        verdict = c('red', f"🚨 ATTACK ({src}{', ' + kind if kind and kind != 'BENIGN' else ''})")
    else:
        verdict = c('green', '✓ normal')
    alert = f"  {c('red','→ 🔔 ALERT FIRED!')}" if r.get('alert_fired') else ''
    return f"Score:{r.get('score', 0):>5.1f} | {verdict} | blacklist: {r.get('blacklist_size', 0)}{alert}"

def check_system(url=BASE_URL):
    try:
        r = requests.get(f"{url}/api/status", timeout=3)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


# ── Main simulation ───────────────────────────────────────────────────────────

def run(url=BASE_URL):
    banner("DDoS Mitigation System — Live Attack Simulation", 'cyan')
    print(f"  {c('bold','Target dashboard:')} {url}")
    print(f"  {c('bold','Watch the browser at:')} {c('yellow', url)}\n")

    status = check_system(url)
    if not status:
        print(c('red', f"  ✖ Cannot reach dashboard at {url}"))
        print(  "    Make sure main.py is running with --dashboard first!")
        sys.exit(1)

    print(f"  {c('green','✓')} Connected | "
          f"Interface: {c('bold', status.get('interface','?'))} | "
          f"Running: {status.get('running', False)}\n")

    cum = dict(total_packets=0, total_bytes=0, tcp_packets=0, udp_packets=0,
               icmp_packets=0, dropped_packets=0, dropped_bytes=0,
               passed_packets=0, passed_bytes=0, other_packets=0)

    # ── Phase 1: Build baseline ───────────────────────────────────────────────
    banner("Phase 1 — Normal Baseline (15s)", 'green')
    print(f"  {c('green','Sending normal traffic (~50 pps) — building baseline...')}\n")

    for i in range(15):
        stats, pps = normal_stats(cum)
        ip_stats = [{'ip': f"10.{random.randint(0,3)}.0.{random.randint(1,254)}",
                     'packets': random.randint(1,4), 'bytes': 500,
                     'flow_count': 1, 'syn_count': 0, 'udp_count': 0,
                     'last_seen': int(time.time()*1e9)}
                    for _ in range(random.randint(5,20))]
        r = post_stats(stats, ip_stats, 'normal', url)
        cum = stats
        print(f"  [{i+1:>2}/15] PPS:{pps:>4} | {describe(r)} | "
              f"Baseline: {r.get('baseline_pps',0):.1f} pps")
        time.sleep(STEP_SECONDS)

    # ── Phase 2: UDP Flood ────────────────────────────────────────────────────
    banner("Phase 2 — UDP Flood (escalating: 1000→5000→10000 pps)", 'red')
    print(f"  {c('red','Simulating UDP flood attack...')}\n")
    print(f"  {c('yellow','👉 Watch the browser dashboard for ALERTS!')} {url}\n")

    for target_pps in [1000, 5000, 10000]:
        print(f"  {c('yellow', f'  ↑ Ramping to {target_pps:,} pps...')}")
        for _ in range(5):
            stats, pps = attack_stats(cum, target_pps, 'udp')
            num_bots = max(20, target_pps // 100)

            ip_stats = make_ip_stats(num_bots, pps, 'udp')
            r = post_stats(stats, ip_stats, f'UDP Flood {target_pps} pps', url)
            cum = stats
            print(f"    UDP {pps:>6,} pps | Bots:{num_bots:>4} | {describe(r)}")
            time.sleep(STEP_SECONDS)
        print()

    # ── Phase 3: SYN Flood ────────────────────────────────────────────────────
    banner("Phase 3 — TCP SYN Flood (15,000 pps, 1000-bot botnet)", 'red')
    print(f"  {c('red','SYN flood — connection exhaustion attack...')}\n")

    for i in range(8):
        pps_t = 15000 + random.randint(-1000, 1000)
        stats, pps = attack_stats(cum, pps_t, 'syn')
        ip_stats = make_ip_stats(1000, pps, 'syn')

        r = post_stats(stats, ip_stats, f'SYN Flood {pps_t} pps', url)
        cum = stats
        print(f"  [{i+1}/8] SYN {pps:>6,} pps | Bots:1000 | {describe(r)}")
        time.sleep(STEP_SECONDS)

    # ── Phase 4: Multi-vector ─────────────────────────────────────────────────
    banner("Phase 4 — Multi-vector DDoS (20,000 pps, 3000-bot botnet)", 'red')
    print(f"  {c('red','UDP + SYN + ICMP simultaneously...')}\n")

    for i in range(8):
        pps_t = 20000 + random.randint(-1000, 1000)
        stats, pps = attack_stats(cum, pps_t, 'mixed')
        ip_stats = make_ip_stats(3000, pps, 'mixed')

        r = post_stats(stats, ip_stats, f'Mixed DDoS {pps_t} pps', url)
        cum = stats
        print(f"  [{i+1}/8] Mixed {pps:>6,} pps | Bots:3000 | {describe(r)}")
        time.sleep(STEP_SECONDS)

    # ── Final report ──────────────────────────────────────────────────────────
    banner("Simulation Complete — Final Dashboard Status", 'cyan')

    final = check_system(url)
    if final:
        st = final.get('statistics', {})
        bl = final.get('blacklist', [])
        ml = final.get('ml_stats', {})
        baseline = final.get('baseline', {})
        alerts = final.get('recent_alerts', [])
        
        print(f"\n  📊 {c('bold','SIMULATION RESULTS:')}")
        print(f"     Blacklisted IPs  : {c('red', len(bl))}")
        print(f"     ML Enabled       : {c('green', 'Yes' if final.get('ml_enabled') else 'No')}")
        if final.get('ml_enabled') and ml:
            print(f"     ML Predictions   : {ml.get('total_ml_predictions', 0)}")
            print(f"     Attacks Classified: {ml.get('ml_attacks_detected', 0)}")
            accuracy = ml.get('model_accuracy')
            if accuracy is not None:
                print(f"     Model test accuracy: {accuracy*100:.1f}%")
        print(f"     Alerts raised    : {len(alerts)}")
        
        print(f"\n  📉 {c('bold','TRAFFIC STATISTICS:')}")
        print(f"     Total Packets    : {st.get('total_packets',0):,}")
        print(f"     Dropped Packets  : {st.get('dropped_packets',0):,}")
        print(f"     Baseline Mean    : {baseline.get('mean_pps',0):.1f} pps")

        print(f"\n  🚫 {c('bold','BLACKLIST SAMPLES:')}")
        if bl:
            for ip in bl[:5]: print(f"     • {c('red', ip)}")
            if len(bl) > 5: print(f"     ... and {len(bl)-5} more")
        else:
            print(f"     None (top sources are blocked during an attack when they send > 15 packets/interval)")


    print(f"\n  {c('green','✓')} Open {c('yellow', url)} to see live dashboard")
    print(f"  {c('green','✓')} Re-run this script any time to simulate another attack\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DDoS Attack Simulation Demo')
    parser.add_argument('--url', default='http://127.0.0.1:5000',
                        help='Dashboard URL (default: http://127.0.0.1:5000)')
    args = parser.parse_args()
    BASE_URL = args.url
    run(args.url)
