#!/usr/bin/env python3
"""
Read the XDP program's own counters (stats_map, summed over CPUs) for an interface,
through bpftool. Used by scripts/cic_xdp_replay.sh to cross-check the packet
counts measured with tcpdump. Linux, root.

    python3 scripts/xdp_counters.py rpb0
prints one JSON object: total/passed/dropped packets and bytes, TCP/UDP/ICMP packets.
"""

import json
import subprocess
import sys

FIELDS = ('total_packets', 'total_bytes', 'dropped_packets', 'dropped_bytes', 'passed_packets',
          'passed_bytes', 'tcp_packets', 'udp_packets', 'icmp_packets', 'other_packets')


def bpftool(*args):
    return json.loads(subprocess.run(['bpftool', '-j', *args], check=True, capture_output=True, text=True).stdout)


def main(dev):
    net = bpftool('net', 'show', 'dev', dev)
    progs = [x for entry in net for x in entry.get('xdp', [])]
    if not progs:
        sys.exit(f'no XDP program on {dev}')
    prog = bpftool('prog', 'show', 'id', str(progs[0]['id']))
    out = {'dev': dev, 'prog_id': prog['id'], 'prog_name': prog.get('name'),
           'xdp_mode': progs[0].get('mode')}
    for map_id in prog.get('map_ids', []):
        info = bpftool('map', 'show', 'id', str(map_id))
        if info.get('name') != 'stats_map':
            continue
        totals = dict.fromkeys(FIELDS, 0)
        for entry in bpftool('map', 'dump', 'id', str(map_id)):
            for cpu in entry.get('formatted', {}).get('values', []):
                val = cpu.get('value', {})
                for k in FIELDS:
                    totals[k] += int(val.get(k, 0))
        out.update(totals)
    print(json.dumps(out))


if __name__ == '__main__':
    main(sys.argv[1])
