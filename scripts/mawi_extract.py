#!/usr/bin/env python3
"""
Extract per-packet header fields from the start of a MAWI trace (WIDE backbone).

MAWI publishes 15-minute anonymized packet traces (pcap.gz, 96-byte snap length)
at https://mawi.wide.ad.jp/mawi/ for research use. This script streams one
trace, parses only the first --seconds seconds, and stops downloading there.
It keeps the IPv4 header fields the XDP program sees, one row per packet:

    sec, src, dst, proto, syn, ttl, value, length
    (value = TCP window for TCP, packet length for UDP/ICMP; syn = SYN without ACK)

Usage (from project/):
    python scripts/mawi_extract.py --trace 2008/200803181400 --seconds 180
Writes data/external/mawi/<trace id>.npz (not part of the repository).
"""

import argparse
import gzip
import struct
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / 'data' / 'external' / 'mawi'
BASE_URL = 'https://mawi.wide.ad.jp/mawi/samplepoint-F/{}.pcap.gz'
COLUMNS = ('sec', 'src', 'dst', 'proto', 'syn', 'ttl', 'value', 'length')


def parse(stream, seconds: float):
    import dpkt
    rows = {c: [] for c in COLUMNS}
    start = None
    n = 0
    for ts, buf in dpkt.pcap.Reader(stream):
        if start is None:
            start = ts
        sec = ts - start
        if sec >= seconds:
            break
        n += 1
        # Ethernet II (with an optional 802.1Q tag) carrying IPv4
        if len(buf) < 34:
            continue
        eth_type = struct.unpack_from('!H', buf, 12)[0]
        off = 14
        if eth_type == 0x8100:
            eth_type = struct.unpack_from('!H', buf, 16)[0]
            off = 18
        if eth_type != 0x0800 or len(buf) < off + 20:
            continue
        ihl = (buf[off] & 0x0F) * 4
        total_len = struct.unpack_from('!H', buf, off + 2)[0]
        ttl, proto = buf[off + 8], buf[off + 9]
        src, dst = struct.unpack_from('!II', buf, off + 12)
        l4 = off + ihl
        syn, value = 0, total_len
        if proto == 6 and len(buf) >= l4 + 16:
            flags = buf[l4 + 13]
            syn = int(flags & 0x02 and not flags & 0x10)
            value = struct.unpack_from('!H', buf, l4 + 14)[0]
        rows['sec'].append(int(sec))
        rows['src'].append(src)
        rows['dst'].append(dst)
        rows['proto'].append(proto)
        rows['syn'].append(syn)
        rows['ttl'].append(ttl)
        rows['value'].append(value)
        rows['length'].append(total_len)
        if n % 1_000_000 == 0:
            print(f'  {n:,} packets, {sec:.0f} s', flush=True)
    dtypes = dict(sec=np.int16, src=np.uint32, dst=np.uint32, proto=np.uint8, syn=np.uint8,
                  ttl=np.uint8, value=np.uint16, length=np.uint16)
    return {c: np.asarray(v, dtype=dtypes[c]) for c, v in rows.items()}, start


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--trace', default='2008/200803181400', help='YEAR/YYYYMMDDHHMM of a samplepoint-F trace')
    ap.add_argument('--seconds', type=float, default=180)
    args = ap.parse_args()
    url = BASE_URL.format(args.trace)
    # Older traces are published as .dump.gz; the server answers HTML for a missing file
    head = subprocess.run(['curl', '-sSI', url], capture_output=True, text=True).stdout.lower()
    if 'gzip' not in head:
        url = url.replace('.pcap.gz', '.dump.gz')
    out = OUT_DIR / (args.trace.split('/')[-1] + '.npz')
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f'streaming {url} (first {args.seconds:g} s only)')
    started = time.time()
    curl = subprocess.Popen(['curl', '-sSL', url], stdout=subprocess.PIPE)
    try:
        data, start_ts = parse(gzip.GzipFile(fileobj=curl.stdout), args.seconds)
    finally:
        curl.kill()
    np.savez_compressed(out, trace=args.trace, start_ts=start_ts, url=url, **data)
    print(f'wrote {out}: {len(data["sec"]):,} IPv4 packets in {time.time() - started:.0f} s')
    return 0


if __name__ == '__main__':
    sys.exit(main())
