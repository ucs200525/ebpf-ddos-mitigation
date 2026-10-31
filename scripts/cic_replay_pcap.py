#!/usr/bin/env python3
"""
Replay files of real CIC-DDoS2019 traffic for the live XDP experiment.

For each run of the test day (03-11) it copies the original frames of every
IPv4 packet entering the victim network (192.168.50.0/24 from outside), in
capture order, out of the downloaded zips into one pcap:

  attack runs  every attack episode: PRE_S seconds before it, its first
               ATTACK_S seconds, POST_S seconds after (episodes closer than
               PRE_S + POST_S are one run, as in scripts/cic_experiment.py)
  quiet runs   the QUIET_RUNS 5-minute attack-free stretches with the most
               benign packets (the hardest for collateral damage)

Frames are cut to --snap bytes (headers kept; XDP never reads payload, and its
byte counters use the IP total length, which is unchanged). Writes
<out>/<run>.pcap and <out>/manifest.json.

Usage (from project/, any OS):
    python scripts/cic_replay_pcap.py --cic-dir H:/CICDDoS2019 --out /root/cic_replay
"""

import argparse
import ipaddress
import json
import struct
import sys
import zipfile
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'scripts'))

from cic_extract import DAYS, OUT_DIR as EXTRACT_DIR, parse_headers  # noqa: E402
from src.ml.cic_data import ATTACKER_IP, attack_episodes, attack_type_of, load_day  # noqa: E402

DAY = '03-11'
PRE_S, ATTACK_S, POST_S = 120, 180, 60
QUIET_S, QUIET_RUNS = 300, 2
PROTECTED = ipaddress.IPv4Network('192.168.50.0/24')


def plan_runs(persec):
    """[(name, start, onset or None, end, scenario)] in time order"""
    merged = []
    for a, b in attack_episodes(persec):
        if merged and a - merged[-1][1] < PRE_S + POST_S:
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))
    runs, prev_end = [], int(persec['sec'].min())
    quiet = []
    for a, b in merged:
        start = max(prev_end, a - PRE_S)
        while prev_end + QUIET_S <= start:
            quiet.append((prev_end, prev_end + QUIET_S))
            prev_end += QUIET_S
        kind = str(attack_type_of(DAY, np.array([a + 30]))[0]) or 'unlabelled'
        runs.append((f'{len(runs) + 1:02d}_{kind}', start, a, min(b, a + ATTACK_S) + POST_S, kind))
        prev_end = b + POST_S
    # quiet stretches with the most benign packets
    benign = persec['src'] != ATTACKER_IP
    vol = [(int(persec['packets'][benign & (persec['sec'] >= s) & (persec['sec'] < e)].sum()), s, e) for s, e in quiet]
    for i, (_, s, e) in enumerate(sorted(vol, reverse=True)[:QUIET_RUNS]):
        runs.append((f'q{i + 1}_no-attack', s, None, e, 'no attack'))
    return runs


def member_times(day):
    """capture member -> (first, last packet time), from the extraction's per-file tables"""
    out = {}
    for f in sorted((EXTRACT_DIR / day).glob('*.npz')):
        with np.load(f) as d:
            if np.isfinite(d['t_min']):
                out[f.name[:-4]] = (float(d['t_min']), float(d['t_max']))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--cic-dir', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--snap', type=int, default=128)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    persec, _ = load_day(DAY)
    runs = plan_runs(persec)
    times = member_times(DAY)
    zips = {}
    for zp in sorted((args.cic_dir / DAYS[DAY]).glob('*.zip')):
        with zipfile.ZipFile(zp) as z:
            for n in z.namelist():
                zips[n] = zp
    net, mask = int(PROTECTED.network_address), int(PROTECTED.netmask)
    manifest = {'day': DAY, 'snap': args.snap, 'pre_s': PRE_S, 'attack_s': ATTACK_S, 'post_s': POST_S,
                'quiet_s': QUIET_S, 'attacker': '172.16.0.5', 'protected': str(PROTECTED), 'runs': []}
    for name, start, onset, end, kind in runs:
        members = sorted((m for m, (lo, hi) in times.items() if hi >= start and lo < end), key=lambda m: times[m][0])
        path = args.out / f'{name}.pcap'
        n_att = n_leg = 0
        first_ts = None
        with open(path, 'wb') as out:
            out.write(struct.pack('<IHHiIII', 0xA1B2C3D4, 2, 4, 0, 0, args.snap, 1))
            for m in members:
                with zipfile.ZipFile(zips[m]) as z:
                    buf = z.read(m)
                p = parse_headers(buf, with_frames=True)
                sel = ((p['ts'] >= start) & (p['ts'] < end) & ((p['dst'] & mask) == net) & ((p['src'] & mask) != net))
                idx = np.flatnonzero(sel)
                atk = p['src'][idx] == ATTACKER_IP
                n_att += int(atk.sum())
                n_leg += int((~atk).sum())
                chunks = []
                for i in idx:
                    ts, o, ln = p['ts'][i], int(p['frame_off'][i]), int(p['frame_len'][i])
                    first_ts = ts if first_ts is None else first_ts
                    sec = int(ts)
                    usec = int(round((ts - sec) * 1e6))
                    if usec == 1_000_000:
                        sec, usec = sec + 1, 0
                    cut = min(ln, args.snap)
                    chunks.append(struct.pack('<IIII', sec, usec, cut, ln) + buf[o:o + cut])
                out.write(b''.join(chunks))
                del buf
        entry = {'name': name, 'scenario': kind, 'start': start, 'attack_onset': onset, 'end': end,
                 'seconds': end - start, 'packets': n_att + n_leg, 'attack_packets': n_att,
                 'legit_packets': n_leg, 'members': members, 'bytes': path.stat().st_size}
        manifest['runs'].append(entry)
        print(f"{name:20s} {end - start:4d} s  {n_att:>10,} attack + {n_leg:>8,} legit packets  "
              f"{path.stat().st_size / 1e6:7.1f} MB  ({len(members)} capture files)", flush=True)
    (args.out / 'manifest.json').write_text(json.dumps(manifest, indent=1))
    print('wrote', args.out / 'manifest.json')


if __name__ == '__main__':
    main()
