#!/usr/bin/env python3
"""
Extract what the XDP program would see from the CIC-DDoS2019 packet captures.

The captures (PCAPs/01-12, PCAPs/03-11; 193 GB unzipped) are streamed member by
member straight out of the downloaded zips; nothing is unpacked to disk. Only
IPv4 packets that ENTER the protected network (destination inside --protected,
source outside it) are kept, because that is the traffic an XDP filter on the
network's ingress sees. Per capture file two tables are written:

  persec  one row per (second, source):  packets, bytes, SYN-without-ACK, TCP, UDP,
          ICMP packets, and the header signature (protocol class, TTL, TCP window
          or packet length) of the source's first packet in that second.
          -> level 1 (per-second window features) and level 2 (per-source scoring)

  flows   one row per (10-second window, 5-tuple): packets, bytes, length
          min/max/sum of squares, first/last time, sum of squared and largest
          inter-arrival time, SYN/ACK/FIN/RST/PSH packet counts. All columns can be
          merged when a flow crosses a capture-file boundary. The same counters are
          also kept per (second, 5-tuple) (fsec table), so that a flow can be rebuilt as
          it stood at any second of its window, as a read of the XDP flow map would see it.
          -> level 3 (per-flow classification). Flows of the attacker addresses are
          hash-sampled (--attack-flow-rate) because floods create tens of millions
          of them; every other flow is kept. The hash is of the flow key, so a flow
          split over two files is kept in both or in neither.

Labels are NOT decided here: the rows keep the source address and time, and
src/ml/cic_data.py labels them (attacker address during a labelled attack).

Usage (from project/):
    python scripts/cic_extract.py --cic-dir H:/CICDDoS2019 --workers 8
Writes data/external/cic/<day>/<member>.npz plus data/external/cic/<day>.npz (merged).
"""

import argparse
import ipaddress
import json
import struct
import sys
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.provenance import provenance  # noqa: E402

OUT_DIR = PROJECT_ROOT / 'data' / 'external' / 'cic'
FLOW_WINDOW = 10                      # seconds; the same window the detector uses
DAYS = {'01-12': 'PCAPs/01-12', '03-11': 'PCAPs/03-11'}
# The attack machine of the CIC testbed (Sharafaldin et al. 2019); only used to decide
# which flows are sampled, the labels come from the CSVs (src/ml/cic_data.py)
ATTACKER_IPS = ('172.16.0.5',)
_HASH_MULT = np.uint64(0x9E3779B97F4A7C15)
FORMAT = 3                            # bump when the per-file tables change (invalidates the cache)


def _ip(s: str) -> int:
    return int(ipaddress.IPv4Address(s))


def member_jobs(cic_dir: Path, days):
    jobs = []
    for day in days:
        for zp in sorted((cic_dir / DAYS[day]).glob('*.zip')):
            with zipfile.ZipFile(zp) as z:
                for info in z.infolist():
                    if not info.is_dir():
                        jobs.append((day, str(zp), info.filename, info.file_size))
    return jobs


def record_offsets(buf: bytes):
    """Offsets and capture lengths of every record of a little/big-endian pcap"""
    magic = struct.unpack_from('<I', buf, 0)[0]
    if magic in (0xA1B2C3D4, 0xA1B23C4D):
        endian = '<'
    elif magic in (0xD4C3B2A1, 0x4D3CB2A1):
        endian = '>'
    else:
        raise ValueError(f'not a pcap file (magic {magic:#x})')
    nano = magic in (0xA1B23C4D, 0x4D3CB2A1)
    link = struct.unpack_from(endian + 'I', buf, 20)[0]
    if link != 1:
        raise ValueError(f'link type {link} is not Ethernet')
    hdr = struct.Struct(endian + 'IIII')
    n = len(buf)
    off = 24
    sec, frac, offs, caps = [], [], [], []
    unpack = hdr.unpack_from
    while off + 16 <= n:
        s, f, incl, _ = unpack(buf, off)
        if off + 16 + incl > n:          # truncated last record
            break
        sec.append(s)
        frac.append(f)
        offs.append(off + 16)
        caps.append(incl)
        off += 16 + incl
    ts = np.asarray(sec, dtype=np.float64) + np.asarray(frac, dtype=np.float64) * (1e-9 if nano else 1e-6)
    return ts, np.asarray(offs, dtype=np.int64), np.asarray(caps, dtype=np.int64)


def parse_headers(buf: bytes, with_frames: bool = False):
    """Vectorised Ethernet(+802.1Q)/IPv4/TCP/UDP header fields of every IPv4 packet;
    with_frames=True also returns each kept packet's frame offset and length in buf"""
    ts, off, cap = record_offsets(buf)
    b = np.frombuffer(buf, dtype=np.uint8)
    last = len(b) - 1

    def u8(idx, ok):
        return np.where(ok, b[np.minimum(idx, last)], 0).astype(np.int64)

    def u16(idx, ok):
        return (u8(idx, ok) << 8) | u8(idx + 1, ok)

    def u32(idx, ok):
        return (u16(idx, ok) << 16) | u16(idx + 2, ok)

    ok = cap >= 34
    eth = u16(off + 12, ok)
    vlan = eth == 0x8100
    ip = off + np.where(vlan, 18, 14)
    eth = np.where(vlan, u16(off + 16, ok), eth)
    ok &= (eth == 0x0800) & (cap >= ip - off + 20)
    ok &= (u8(ip, ok) >> 4) == 4
    ihl = (u8(ip, ok) & 0x0F) * 4
    length = u16(ip + 2, ok)
    first_frag = (u16(ip + 6, ok) & 0x1FFF) == 0      # later fragments carry no L4 header
    ttl = u8(ip + 8, ok)
    proto = u8(ip + 9, ok)
    src = u32(ip + 12, ok)
    dst = u32(ip + 16, ok)
    l4 = ip + ihl
    l4_cap = cap - (l4 - off)
    has_ports = ok & first_frag & ((proto == 6) | (proto == 17)) & (l4_cap >= 4)
    sport = u16(l4, has_ports)
    dport = u16(l4 + 2, has_ports)
    is_tcp_hdr = ok & first_frag & (proto == 6) & (l4_cap >= 16)
    flags = u8(l4 + 13, is_tcp_hdr)
    window = u16(l4 + 14, is_tcp_hdr)
    keep = ok
    cols = dict(ts=ts, src=src, dst=dst, sport=sport, dport=dport, proto=proto, flags=flags,
                ttl=ttl, window=window, length=length)
    out = {k: v[keep] for k, v in cols.items()}
    if with_frames:
        out['frame_off'], out['frame_len'] = off[keep], cap[keep]
    return out


def flow_hash(*cols):
    """64-bit mixing hash of integer columns (deterministic, platform independent)"""
    h = np.zeros(len(cols[0]), dtype=np.uint64)
    with np.errstate(over='ignore'):
        for c in cols:
            h = (h ^ c.astype(np.uint64)) * _HASH_MULT
            h ^= h >> np.uint64(29)
    return h


def _group_starts(keys, tiebreak=None):
    """Sort order and group start mask for rows grouped by the key columns (last = primary);
    `tiebreak` (e.g. packet time) only orders rows inside a group"""
    order = np.lexsort(([tiebreak] if tiebreak is not None else []) + list(keys))
    sorted_keys = [k[order] for k in keys]
    start = np.ones(len(order), dtype=bool)
    if len(order):
        change = np.zeros(len(order) - 1, dtype=bool)
        for k in sorted_keys:
            change |= k[1:] != k[:-1]
        start[1:] = change
    return order, start


def per_second_table(p):
    sec = np.floor(p['ts']).astype(np.int64)
    order, start = _group_starts([p['src'], sec], tiebreak=p['ts'])   # (sec, src), packets in time order
    gid = np.cumsum(start) - 1
    first = order[start]
    flags, proto = p['flags'][order], p['proto'][order]
    syn = ((flags & 0x02) != 0) & ((flags & 0x10) == 0)
    n = int(gid[-1]) + 1 if len(gid) else 0

    def gsum(x):
        return np.bincount(gid, weights=x, minlength=n)

    fp = p['proto'][first]
    first_syn = ((p['flags'][first] & 0x02) != 0) & ((p['flags'][first] & 0x10) == 0)
    return dict(
        sec=sec[first].astype(np.int64),
        src=p['src'][first].astype(np.uint32),
        packets=np.bincount(gid, minlength=n).astype(np.int64),
        bytes=gsum(p['length'][order]).astype(np.int64),
        syn=gsum(syn).astype(np.int64),
        tcp=gsum(proto == 6).astype(np.int64),
        udp=gsum(proto == 17).astype(np.int64),
        icmp=gsum(proto == 1).astype(np.int64),
        sig_proto=fp.astype(np.uint8),
        sig_ttl=p['ttl'][first].astype(np.uint8),
        sig_value=np.where(fp == 6, p['window'][first], p['length'][first]).astype(np.uint16),
        sig_syn=first_syn.astype(np.uint8),
    )


def flow_table(p, attacker_ips, sample_rate, slot_s=FLOW_WINDOW):
    """Flow counters per (slot, 5-tuple); slot_s = FLOW_WINDOW gives one row per window.
    Sampling always uses the window, so all slots of a sampled flow are kept."""
    sec = np.floor(p['ts']).astype(np.int64)
    win = sec // FLOW_WINDOW
    key_cols = [win, p['src'], p['dst'], p['sport'], p['dport'], p['proto']]
    # Sample attacker flows by a hash of the flow key (same decision in every file)
    attacker = np.isin(p['src'], np.asarray(attacker_ips, dtype=np.int64))
    if sample_rate < 1.0:
        h = flow_hash(*key_cols)
        keep = ~attacker | (h < np.uint64(int(sample_rate * 2 ** 64)))
        p = {k: v[keep] for k, v in p.items()}
        win, sec = win[keep], sec[keep]
    slot = sec // slot_s
    key_cols = [slot, p['src'], p['dst'], p['sport'], p['dport'], p['proto']]
    order, start = _group_starts(key_cols[::-1], tiebreak=p['ts'])   # primary: slot, then src, ...
    gid = np.cumsum(start) - 1
    n = int(gid[-1]) + 1 if len(gid) else 0
    first = order[start]
    last = order[np.r_[start[1:], True]] if len(order) else order
    ts = p['ts'][order]
    length = p['length'][order].astype(np.float64)
    flags = p['flags'][order]
    iat = np.diff(ts, prepend=ts[:1])
    iat[start] = 0.0                       # first packet of each flow has no gap

    def gsum(x):
        return np.bincount(gid, weights=x, minlength=n)

    def gext(x, fn, init):
        out = np.full(n, init, dtype=np.float64)
        fn.at(out, gid, x)
        return out

    return dict(
        win=win[first].astype(np.int64), slot=slot[first].astype(np.int64),
        src=p['src'][first].astype(np.uint32), dst=p['dst'][first].astype(np.uint32),
        sport=p['sport'][first].astype(np.uint16), dport=p['dport'][first].astype(np.uint16),
        proto=p['proto'][first].astype(np.uint8),
        packets=np.bincount(gid, minlength=n).astype(np.int64),
        bytes=gsum(length).astype(np.int64),
        len_min=gext(length, np.minimum, np.inf).astype(np.float32),
        len_max=gext(length, np.maximum, 0.0).astype(np.float32),
        len_sumsq=gsum(length ** 2),
        t_first=p['ts'][first], t_last=p['ts'][last],
        iat_sumsq=gsum(iat ** 2),
        iat_max=gext(iat, np.maximum, 0.0),
        syn=gsum((flags & 0x02) != 0).astype(np.int32),
        ack=gsum((flags & 0x10) != 0).astype(np.int32),
        fin=gsum((flags & 0x01) != 0).astype(np.int32),
        rst=gsum((flags & 0x04) != 0).astype(np.int32),
        psh=gsum((flags & 0x08) != 0).astype(np.int32),
    )


def extract_member(job):
    day, zp, member, protected, attacker_ips, sample_rate, out_dir = job
    t0 = time.time()
    out = Path(out_dir) / day / (Path(member).name + '.npz')
    if out.exists():
        try:
            with np.load(out) as d:
                if int(d['format']) == FORMAT:
                    return day, member, 'cached', 0, 0.0
        except (KeyError, ValueError, OSError):
            pass
    net, mask = protected
    with zipfile.ZipFile(zp) as z:
        buf = z.read(member)
    p = parse_headers(buf)
    n_ipv4 = len(p['ts'])
    del buf
    inbound = ((p['dst'] & mask) == net) & ((p['src'] & mask) != net)
    p = {k: v[inbound] for k, v in p.items()}
    sec = per_second_table(p)
    flows = flow_table(p, attacker_ips, sample_rate, slot_s=1)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix('.tmp.npz')
    np.savez_compressed(tmp, **{f'persec_{k}': v for k, v in sec.items()},
                        **{f'flow_{k}': v for k, v in flows.items()},
                        n_ipv4=n_ipv4, n_inbound=len(p['ts']), format=FORMAT,
                        t_min=p['ts'].min() if len(p['ts']) else np.nan,
                        t_max=p['ts'].max() if len(p['ts']) else np.nan)
    tmp.replace(out)
    return day, member, 'ok', len(p['ts']), time.time() - t0


def merge_flows(f, by='win'):
    """Merge partial flow rows with the same (window or slot, 5-tuple): parts from different
    capture files, or the seconds of one window"""
    keys = [f['proto'], f['dport'], f['sport'], f['dst'], f['src'], f[by]]
    order, start = _group_starts(keys, tiebreak=f['t_first'])
    if start.all():
        return f
    f = {k: v[order] for k, v in f.items()}
    gid = np.cumsum(start) - 1
    n = int(gid[-1]) + 1
    # gap between the end of one part and the start of the next part of the same flow
    gap = np.where(start, 0.0, f['t_first'] - np.r_[f['t_first'][:1], f['t_last'][:-1]])
    gap = np.maximum(gap, 0.0)

    def gsum(x):
        return np.bincount(gid, weights=x, minlength=n)

    def gext(x, fn, init):
        out = np.full(n, init, dtype=np.float64)
        fn.at(out, gid, x)
        return out

    first = np.flatnonzero(start)
    out = {k: f[k][first] for k in ('win', 'slot', 'src', 'dst', 'sport', 'dport', 'proto', 't_first') if k in f}
    for k in ('packets', 'bytes', 'syn', 'ack', 'fin', 'rst', 'psh'):
        out[k] = gsum(f[k]).astype(f[k].dtype)
    out['len_sumsq'] = gsum(f['len_sumsq'])
    out['iat_sumsq'] = gsum(f['iat_sumsq']) + gsum(gap ** 2)
    out['iat_max'] = np.maximum(gext(f['iat_max'], np.maximum, 0.0), gext(gap, np.maximum, 0.0))
    out['len_min'] = gext(f['len_min'], np.minimum, np.inf).astype(np.float32)
    out['len_max'] = gext(f['len_max'], np.maximum, 0.0).astype(np.float32)
    out['t_last'] = gext(f['t_last'], np.maximum, 0.0)
    return out


def merge_day(day: str, out_dir: Path):
    parts = sorted(q for q in (out_dir / day).glob('*.npz') if not q.name.endswith('.tmp.npz'))
    persec, flows, meta = {}, {}, dict(n_ipv4=0, n_inbound=0, files=len(parts))
    t_min, t_max = np.inf, -np.inf
    for part in parts:
        with np.load(part) as d:
            for k in d.files:
                if k.startswith('persec_'):
                    persec.setdefault(k[7:], []).append(d[k])
                elif k.startswith('flow_'):
                    flows.setdefault(k[5:], []).append(d[k])
            meta['n_ipv4'] += int(d['n_ipv4'])
            meta['n_inbound'] += int(d['n_inbound'])
            if np.isfinite(d['t_min']):
                t_min, t_max = min(t_min, float(d['t_min'])), max(t_max, float(d['t_max']))
    persec = {k: np.concatenate(v) for k, v in persec.items()}
    seconds = merge_flows({k: np.concatenate(v) for k, v in flows.items()}, by='slot')   # across files
    flows = merge_flows(seconds, by='win')
    flows.pop('slot', None)
    # A (second, source) pair can also be split over two files: sum the counters
    keys = [persec['src'], persec['sec']]
    order, start = _group_starts(keys)
    if not start.all():
        gid = np.cumsum(start) - 1
        first = order[start]
        merged = {k: persec[k][first] for k in ('sec', 'src', 'sig_proto', 'sig_ttl', 'sig_value', 'sig_syn')}
        for k in ('packets', 'bytes', 'syn', 'tcp', 'udp', 'icmp'):
            merged[k] = np.bincount(gid, weights=persec[k][order]).astype(np.int64)
        persec = merged
    else:
        persec = {k: v[order] for k, v in persec.items()}
    out = out_dir / f'{day}.npz'
    tmp = out_dir / f'{day}.tmp.npz'
    np.savez_compressed(tmp, **{f'persec_{k}': v for k, v in persec.items()},
                        **{f'flow_{k}': v for k, v in flows.items()},
                        **{f'fsec_{k}': v for k, v in seconds.items()},
                        t_min=t_min, t_max=t_max, **{k: v for k, v in meta.items()})
    tmp.replace(out)
    return out, dict(meta, t_min=t_min, t_max=t_max, persec_rows=len(persec['sec']),
                     flow_rows=len(flows['win']), flow_second_rows=len(seconds['win']))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--cic-dir', type=Path, required=True, help='folder holding CSVs/ and PCAPs/')
    ap.add_argument('--days', nargs='+', default=list(DAYS), choices=list(DAYS))
    ap.add_argument('--protected', default='192.168.50.0/24',
                    help='the victim network of the CIC testbed: packets entering it are kept')
    ap.add_argument('--attack-flow-rate', type=float, default=1 / 32,
                    help='share of attacker-address flows kept (every other flow is kept)')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--limit', type=int, default=0, help='only the first N capture files per day (testing)')
    args = ap.parse_args()

    netw = ipaddress.IPv4Network(args.protected)
    protected = (int(netw.network_address), int(netw.netmask))
    attackers = tuple(_ip(a) for a in ATTACKER_IPS)
    jobs = member_jobs(args.cic_dir, args.days)
    if args.limit:
        by_day = {}
        for j in jobs:
            by_day.setdefault(j[0], []).append(j)
        jobs = [j for d in by_day.values() for j in sorted(d, key=lambda j: _member_index(j[2]))[:args.limit]]
    if not jobs:
        sys.exit(f'No PCAP zips found under {args.cic_dir / "PCAPs"}')
    print(f'{len(jobs)} capture files, {sum(j[3] for j in jobs) / 1e9:.1f} GB', flush=True)
    work = [(d, zp, m, protected, attackers, args.attack_flow_rate, str(OUT_DIR)) for d, zp, m, _ in
            sorted(jobs, key=lambda j: -j[3])]
    t0 = time.time()
    done = 0
    with ProcessPoolExecutor(args.workers) as pool:
        futures = [pool.submit(extract_member, w) for w in work]
        for fut in as_completed(futures):
            day, member, status, n, dt = fut.result()
            done += 1
            if done % 20 == 0 or done == len(futures):
                print(f'  {done}/{len(futures)} files  ({time.time() - t0:.0f} s)  last: {day}/{member} '
                      f'{status} {n:,} inbound packets', flush=True)
    summary = {}
    for day in args.days:
        out, meta = merge_day(day, OUT_DIR)
        summary[day] = meta
        print(f'{day}: {meta["n_ipv4"]:,} IPv4 packets, {meta["n_inbound"]:,} entering {args.protected}, '
              f'{meta["persec_rows"]:,} (second, source) rows, {meta["flow_rows"]:,} flow rows -> {out}')
    (OUT_DIR / 'extract_meta.json').write_text(json.dumps(dict(
        days=summary,
        provenance=provenance('scripts/cic_extract.py', dict(vars(args), cic_dir=str(args.cic_dir),
                                                             flow_window=FLOW_WINDOW,
                                                             attacker_ips=ATTACKER_IPS)),
    ), indent=2, default=str))


def _member_index(name: str) -> int:
    """SAT-01-12-2018_0123 -> 123 (members are numbered _0, _01, ..., _0818)"""
    tail = name.rsplit('_', 1)[-1]
    return int(tail) if tail.isdigit() else 0


if __name__ == '__main__':
    main()
