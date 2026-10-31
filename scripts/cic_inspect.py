#!/usr/bin/env python3
"""
Inventory of the CIC-DDoS2019 flow CSVs (Sharafaldin et al., 2019).

Streams every CSV straight out of the downloaded zips (nothing is extracted, the
two days are 31 GB unzipped) and records, per file and per label:
number of flows, time range, protocols, and the source/destination addresses
that carry the traffic. This shows what the labels mean before anything is
trained on them, e.g. whether "attack" is the same as "comes from the attacker host".

Usage (from project/):
    python scripts/cic_inspect.py --cic-dir H:/CICDDoS2019
Writes data/results/cic_inventory.json and prints a summary.
"""

import argparse
import json
import sys
import zipfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.provenance import provenance  # noqa: E402

OUT = PROJECT_ROOT / 'data' / 'results' / 'cic_inventory.json'
COLS = ['Source IP', 'Source Port', 'Destination IP', 'Destination Port', 'Protocol', 'Timestamp', 'Label']
CHUNK = 1_000_000


def csv_members(cic_dir: Path):
    """(zip path, member name) of every flow CSV in the two CSV zips"""
    out = []
    for zp in sorted((cic_dir / 'CSVs').glob('CSV-*.zip')):
        with zipfile.ZipFile(zp) as z:
            for m in z.namelist():
                name = m.rsplit('/', 1)[-1]
                if name.endswith('.csv') and not name.startswith('.~lock'):
                    out.append((str(zp), m))
    return out


def scan(job):
    zp, member = job
    per_label = {}
    with zipfile.ZipFile(zp) as z, z.open(member) as f:
        reader = pd.read_csv(f, usecols=lambda c: c.strip() in COLS, chunksize=CHUNK,
                             skipinitialspace=True, low_memory=False, dtype=str)
        for chunk in reader:
            chunk.columns = [c.strip() for c in chunk.columns]
            for label, g in chunk.groupby('Label', sort=False):
                s = per_label.setdefault(label, dict(n=0, src=Counter(), dst=Counter(), proto=Counter(),
                                                     t_min=None, t_max=None))
                s['n'] += len(g)
                s['src'].update(g['Source IP'].value_counts().to_dict())
                s['dst'].update(g['Destination IP'].value_counts().to_dict())
                s['proto'].update(g['Protocol'].value_counts().to_dict())
                t = g['Timestamp']
                lo, hi = t.min(), t.max()
                s['t_min'] = lo if s['t_min'] is None else min(s['t_min'], lo)
                s['t_max'] = hi if s['t_max'] is None else max(s['t_max'], hi)
    out = {}
    for label, s in per_label.items():
        out[label] = dict(
            flows=s['n'], t_min=s['t_min'], t_max=s['t_max'],
            protocols=dict(s['proto'].most_common()),
            n_sources=len(s['src']), top_sources=s['src'].most_common(5),
            n_destinations=len(s['dst']), top_destinations=s['dst'].most_common(5),
        )
    return f'{Path(zp).name}:{member}', out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--cic-dir', type=Path, required=True, help='folder holding CSVs/ and PCAPs/')
    ap.add_argument('--workers', type=int, default=8)
    args = ap.parse_args()

    jobs = csv_members(args.cic_dir)
    if not jobs:
        sys.exit(f'No CSV zips found under {args.cic_dir / "CSVs"}')
    # Largest files first so the pool finishes evenly
    with ProcessPoolExecutor(args.workers) as pool:
        results = dict(pool.map(scan, jobs))

    totals = Counter()
    for name, labels in sorted(results.items()):
        print(f'\n{name}')
        for label, s in sorted(labels.items(), key=lambda kv: -kv[1]['flows']):
            totals[(name.split(':')[1].split('/')[0], label)] += s['flows']
            top = ', '.join(f'{ip} ({n:,})' for ip, n in s['top_sources'][:3])
            print(f'  {label:<14} {s["flows"]:>11,} flows  {s["t_min"]} .. {s["t_max"]}  '
                  f'{s["n_sources"]:,} sources, top: {top}')
    print('\nFlows per day and label:')
    for (day, label), n in sorted(totals.items()):
        print(f'  {day}  {label:<14} {n:>12,}')

    zips = sorted({zp for zp, _ in jobs})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(dict(
        files=results,
        flows_per_day_label={f'{d}/{l}': n for (d, l), n in sorted(totals.items())},
        provenance=provenance('scripts/cic_inspect.py', vars(args) | {'cic_dir': str(args.cic_dir)},
                              {Path(z).name: z for z in zips}),
    ), indent=2, default=str))
    print(f'\nWrote {OUT}')


if __name__ == '__main__':
    main()
