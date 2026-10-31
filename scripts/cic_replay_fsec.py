#!/usr/bin/env python3
"""
Per-second flow table of the live replay windows, from the replay files themselves.

scripts/cic_extract.py writes the per-second flow table of whole days (it needs the
downloaded zips). The offline replay of the live windows (scripts/cic_xdp_analyze.py) only
needs those windows, and the replay files hold the same packets, so this builds the table
from them, with the same flow sampling. Writes data/external/cic/<day>_replay_fsec.npz.

    python scripts/cic_replay_fsec.py /root/cic_replay
"""

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'scripts'))

from cic_extract import OUT_DIR, flow_table, merge_flows, parse_headers  # noqa: E402
from src.ml.cic_data import ATTACKER_IP  # noqa: E402


def main(replay_dir: Path):
    manifest = json.loads((replay_dir / 'manifest.json').read_text())
    rate = float(json.loads((OUT_DIR / 'extract_meta.json').read_text())['provenance']['parameters']['attack_flow_rate'])
    parts = []
    for run in manifest['runs']:
        p = parse_headers((replay_dir / f"{run['name']}.pcap").read_bytes())
        parts.append(flow_table(p, (ATTACKER_IP,), rate, slot_s=1))
        print(run['name'], len(parts[-1]['win']), 'rows', flush=True)
    seconds = merge_flows({k: np.concatenate([q[k] for q in parts]) for k in parts[0]}, by='slot')
    out = OUT_DIR / f"{manifest['day']}_replay_fsec.npz"
    np.savez_compressed(out, **{f'fsec_{k}': v for k, v in seconds.items()})
    print('wrote', out, len(seconds['win']), 'rows')


if __name__ == '__main__':
    main(Path(sys.argv[1]))
