#!/usr/bin/env python3
"""
Real packets for the XDP traffic lab (Linux, root). Two modes:

  send  (run inside the attacker network namespace)
        Reads a profile name per line on stdin (normal, flash_crowd, udp_flood, syn_flood,
        multi_vector, idle) and sends real Ethernet frames out of --iface, paced over each
        second. Prints "sent <legit> <attack>" once per second.

  recv  (run in the server namespace)
        Counts the frames that get past the XDP program on --iface (AF_PACKET sees a packet
        only after generic XDP has passed it). Prints "recv <legit> <attack>" once per second.

The populations and rates are the same as the dashboard's traffic profiles
(src/dashboard/lab.py): 80 regular clients at about 300 packets/s, a flash crowd of
1,500 new users, and 400 bots. Bots are in 45.32.0.0/14; scripts/lab_server.sh routes
these ranges back into the veth pair, so no reply ever leaves the machine.
"""

import argparse
import os
import random
import socket
import struct
import sys
import threading
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.dashboard.lab import BOTS, CLIENTS, CROWD, is_attack_ip  # noqa: E402

NORMAL_PPS = 300.0
ETH_P_ALL = 0x0003
PACKET_OUTGOING = 4


def _csum(data: bytes) -> int:
    if len(data) % 2:
        data += b'\0'
    s = sum(struct.unpack(f'!{len(data) // 2}H', data))
    s = (s >> 16) + (s & 0xFFFF)
    s += s >> 16
    return ~s & 0xFFFF


def frame(src: str, dst: str, kind: str, size: int) -> bytes:
    """Ethernet + IPv4 + TCP/UDP/ICMP frame, `size` bytes of IP payload in total"""
    rnd = random.randint
    if kind == 'udp':
        proto, l4 = 17, struct.pack('!HHHH', rnd(1024, 65535), rnd(1, 65535), 8 + max(0, size - 8), 0)
    elif kind == 'icmp':
        proto, l4 = 1, struct.pack('!BBHHH', 8, 0, 0, rnd(0, 65535), 1)
    else:   # syn or tcp (established, ACK|PSH)
        flags = 0x02 if kind == 'syn' else 0x18
        proto, l4 = 6, struct.pack('!HHIIBBHHH', rnd(1024, 65535), 443 if kind == 'tcp' else 80,
                                   rnd(0, 2**32 - 1), 0 if kind == 'syn' else rnd(0, 2**32 - 1),
                                   5 << 4, flags, 64240, 0, 0)
    payload = b'\0' * max(0, size - len(l4))
    if proto == 1:
        l4 = l4[:2] + struct.pack('!H', _csum(l4 + payload)) + l4[4:]
    total = 20 + len(l4) + len(payload)
    ip = struct.pack('!BBHHHBBH4s4s', 0x45, 0, total, rnd(0, 65535), 0, 64, proto, 0,
                     socket.inet_aton(src), socket.inet_aton(dst))
    ip = ip[:10] + struct.pack('!H', _csum(ip)) + ip[12:]
    eth = b'\xff' * 6 + b'\x02\x00\x00\x00\x00\x01' + b'\x08\x00'
    return eth + ip + l4 + payload


class Sender:
    def __init__(self, iface: str, target: str, seed: int = 7):
        self.sock = socket.socket(socket.AF_PACKET, socket.SOCK_RAW)
        self.sock.bind((iface, 0))
        self.rng = np.random.default_rng(seed)
        self.profile, self.t = 'idle', 0
        self.lock = threading.Lock()
        w = lambda n, a: (lambda x: x / x.sum())(self.rng.permutation(1.0 / np.arange(1, n + 1) ** a))  # noqa: E731
        self.client_w, self.crowd_w = w(len(CLIENTS), 0.9), w(len(CROWD), 0.4)
        # A few pre-built frames per source and kind (building per packet is too slow)
        mk = lambda ips, kinds, size: {(ip, k): [frame(ip, target, k, size) for _ in range(3)]  # noqa: E731
                                       for ip in ips for k in kinds}
        self.frames = mk(CLIENTS + CROWD, ('tcp', 'udp', 'icmp', 'syn'), 860)
        self.frames.update(mk(BOTS, ('udp', 'syn', 'icmp'), 40))

    def command(self, name: str):
        with self.lock:
            self.profile, self.t = name, 0

    def _split(self, ips, weights, total):
        counts = self.rng.multinomial(int(total), weights)
        return [(ip, int(c)) for ip, c in zip(ips, counts) if c > 0]

    def second(self, profile: str, t: int):
        """Frames for one second of `profile` and the legit / attack packet counts"""
        if profile == 'idle':
            return [], 0, 0
        rng, out, legit, attack = self.rng, [], 0, 0
        entries = list(self._split(CLIENTS, self.client_w, NORMAL_PPS * rng.uniform(0.9, 1.1)))
        if profile == 'flash_crowd':
            extra = NORMAL_PPS * min(4.0, 0.3 * (t + 1)) * rng.uniform(0.9, 1.1)
            entries += self._split(CROWD, self.crowd_w, extra)
        for ip, c in entries:
            tcp, udp, icmp = rng.multinomial(c, (0.85, 0.12, 0.03))
            syn = rng.binomial(tcp, 0.04)
            for kind, n in (('tcp', tcp - syn), ('syn', syn), ('udp', udp), ('icmp', icmp)):
                fr = self.frames[(ip, kind)]
                out += [fr[i % 3] for i in range(int(n))]
            legit += c
        if profile in ('udp_flood', 'syn_flood', 'multi_vector'):
            ramp = min(1.0, 0.5 + 0.25 * t)
            vectors = {'udp_flood': [(BOTS[:100], 8000, 'udp')],
                       'syn_flood': [(BOTS[100:400], 6000, 'syn')],
                       'multi_vector': [(BOTS[:200], 5000, 'udp'), (BOTS[200:400], 4000, 'syn'),
                                        (BOTS[:400:4], 2000, 'icmp')]}[profile]
            for bots, pps, kind in vectors:
                for ip, c in self._split(bots, np.full(len(bots), 1.0 / len(bots)), pps * ramp * rng.uniform(0.95, 1.05)):
                    fr = self.frames[(ip, kind)]
                    out += [fr[i % 3] for i in range(c)]
                    attack += c
        random.shuffle(out)
        return out, legit, attack

    def run(self):
        slices = 20
        nxt = time.monotonic()
        while True:
            with self.lock:
                profile, t = self.profile, self.t
                self.t += 1
            frames, legit, attack = self.second(profile, t)
            per = (len(frames) + slices - 1) // slices or 1
            for k in range(slices):
                for f in frames[k * per:(k + 1) * per]:
                    try:
                        self.sock.send(f)
                    except OSError:
                        pass
                time.sleep(max(0.0, nxt + (k + 1) / slices - time.monotonic()))
            nxt += 1.0
            print(f'sent {legit} {attack}', flush=True)
            if time.monotonic() > nxt + 1:      # fell behind: do not try to catch up
                nxt = time.monotonic()


def send(args):
    s = Sender(args.iface, args.target)

    def reader():
        for line in sys.stdin:
            name = line.strip()
            if name:
                s.command(name)
        os._exit(0)     # stdin closed: the server went away
    threading.Thread(target=reader, daemon=True).start()
    print('ready', flush=True)
    s.run()


def recv(args):
    sock = socket.socket(socket.AF_PACKET, socket.SOCK_RAW, socket.htons(ETH_P_ALL))
    sock.bind((args.iface, 0))
    sock.settimeout(0.2)
    legit = attack = 0
    nxt = time.monotonic() + 1
    print('ready', flush=True)
    while True:
        try:
            data, addr = sock.recvfrom(64)
            if addr[2] != PACKET_OUTGOING and len(data) >= 34 and data[12:14] == b'\x08\x00':
                if is_attack_ip(socket.inet_ntoa(data[26:30])):
                    attack += 1
                else:
                    legit += 1
        except socket.timeout:
            pass
        if time.monotonic() >= nxt:
            print(f'recv {legit} {attack}', flush=True)
            legit = attack = 0
            nxt += 1


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('mode', choices=['send', 'recv'])
    p.add_argument('--iface', required=True)
    p.add_argument('--target', default='192.168.99.2')
    args = p.parse_args()
    (send if args.mode == 'send' else recv)(args)


if __name__ == '__main__':
    main()
