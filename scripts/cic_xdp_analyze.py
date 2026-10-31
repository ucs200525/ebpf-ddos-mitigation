#!/usr/bin/env python3
"""
Results of the live XDP experiment (scripts/cic_xdp_replay.sh).

For every run and configuration it compares, per source, the packets sent
(the replay file) with the packets XDP let through (tcpdump on the victim
interface, which only sees passed packets in generic XDP mode), and reads:
  - attack / legitimate packets dropped, legitimate sources that lost packets
  - mitigation delay: attack onset -> first second with >= 50% of attack packets dropped
    (measured on the packets; alert lines are not used for timing because alerts have a
    60 s cooldown and some attacks ramp up before the onset mark)
  - checks: packets XDP saw vs sent, packets XDP passed vs captured, tcpdump and
    tcpreplay losses, achieved replay rate
Then it replays the same windows offline (the evaluation harness used by
scripts/cic_experiment.py) with the same four configurations, so the kernel
measurements can be compared with the offline model of the data path.

    python scripts/cic_xdp_analyze.py /root/cic_xdp_runs/full [--no-offline]
Writes data/results/cic_xdp_results.json.
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'scripts'))

from cic_extract import parse_headers  # noqa: E402
from src.ml.cic_data import ATTACKER_IP  # noqa: E402
from src.provenance import provenance  # noqa: E402

RESULTS = PROJECT_ROOT / 'data' / 'results' / 'cic_xdp_results.json'
CONFIGS = {'P0': 'top-5 blacklist, synthetic models',
           'P4': 'P4, synthetic models',
           'P4cic': 'P4, cic set without level 3 (detector: synthetic + CIC 01-12)',
           'P4L3': 'P4+L3, cic set (flow classifier verifies P4)'}
ANSI = re.compile(r'\x1b\[[0-9;]*m')


def per_source(pcap: Path):
    p = parse_headers(pcap.read_bytes())
    return p['ts'], p['src']


def counts(src):
    u, n = np.unique(src, return_counts=True)
    return dict(zip(u.tolist(), n.tolist()))


def log_times(console: Path, pattern: str):
    out = []
    for line in ANSI.sub('', console.read_text(errors='replace')).splitlines():
        if re.search(pattern, line):
            m = re.match(r'(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)', line)
            if m:
                out.append(time.mktime(time.strptime(m.group(1), '%Y-%m-%d %H:%M:%S')))
    return out


def analyze_run(run_dir: Path, run: dict, sent_ts, sent_src):
    sent = counts(sent_src)
    onset = run['attack_onset']
    out = {}
    for cfg in CONFIGS:
        d = run_dir / cfg
        if not (d / 'passed.pcap').exists():
            continue
        p_ts, p_src = per_source(d / 'passed.pcap')
        passed = counts(p_src)
        a_sent = sent.get(ATTACKER_IP, 0)
        a_pass = passed.get(ATTACKER_IP, 0)
        legit = [s for s in sent if s != ATTACKER_IP]
        l_sent = sum(sent[s] for s in legit)
        l_pass = sum(min(passed.get(s, 0), sent[s]) for s in legit)
        hit = [s for s in legit if passed.get(s, 0) < sent[s]]
        r = {'attack_sent': a_sent, 'attack_passed': a_pass, 'legit_sent': l_sent, 'legit_passed': l_pass,
             'attack_dropped_pct': 100.0 * (a_sent - a_pass) / a_sent if a_sent else None,
             'legit_dropped_pct': 100.0 * (l_sent - l_pass) / l_sent if l_sent else None,
             'legit_sources': len(legit), 'legit_sources_hit': len(hit)}
        # wall clock -> capture clock: the first packet of a run is benign and always passes
        offset = float(p_ts[0] - sent_ts[0]) if len(p_ts) else 0.0
        if onset is not None:
            atk_s = np.floor(sent_ts[sent_src == ATTACKER_IP] - onset).astype(int)
            atk_p = np.floor(p_ts[p_src == ATTACKER_IP] - offset - onset).astype(int)
            secs = np.arange(0, int(run['end'] - onset))
            s_n = np.bincount(np.clip(atk_s[atk_s >= 0], 0, len(secs) - 1), minlength=len(secs))
            p_n = np.bincount(np.clip(atk_p[(atk_p >= 0) & (atk_p < len(secs))], 0, len(secs) - 1),
                              minlength=len(secs))
            mitigated = np.flatnonzero((s_n > 0) & (p_n <= 0.5 * s_n))
            r['mitigation_delay_s'] = int(mitigated[0]) if len(mitigated) else None
        r['alerts'] = len(log_times(d / 'console.log', r'DDoS attack detected'))
        r['blocks_logged'] = sum(int(m) for m in re.findall(r'blocked (\d+) and slowed', (d / 'console.log').read_text(errors='replace')))
        r['releases_logged'] = sum(int(m) for m in re.findall(r'released (\d+) sources', (d / 'console.log').read_text(errors='replace')))
        # checks
        chk = {'sent': int(len(sent_src)), 'captured_after_xdp': int(len(p_src))}
        try:
            x = json.loads((d / 'xdp_counters.json').read_text())
            chk.update({'xdp_mode': x.get('xdp_mode'), 'xdp_total': x['total_packets'],
                        'xdp_passed': x['passed_packets'], 'xdp_dropped': x['dropped_packets']})
        except (OSError, ValueError, KeyError) as e:
            chk['xdp_counters_error'] = str(e)
        tl = (d / 'tcpdump.log').read_text(errors='replace')
        m = re.search(r'(\d+) packets dropped by kernel', tl)
        chk['tcpdump_dropped'] = int(m.group(1)) if m else None
        rl = (d / 'tcpreplay.log').read_text(errors='replace')
        m = re.search(r'Actual: (\d+) packets .*? in ([\d.]+) seconds', rl)
        if m:
            chk['replayed'], chk['replay_seconds'] = int(m.group(1)), float(m.group(2))
        m = re.search(r'Rated: [\d.]+ Bps, [\d.]+ Mbps, ([\d.]+) pps', rl)
        chk['replay_pps'] = float(m.group(1)) if m else None
        m = re.search(r'Failed packets:\s+(\d+)', rl)
        chk['replay_failed'] = int(m.group(1)) if m else None
        r['checks'] = chk
        out[cfg] = r
    return out


def offline(manifest):
    """Same windows and configurations through the offline harness (scripts/cic_experiment.py)"""
    import cic_experiment as E
    E.USE_REPLAY_WINDOWS = True
    from mitigation_experiment import RESULTS_PATH as SYN, load_scorers
    from src.mitigation import FlowVerifiedScoring, SourceScoringPolicy, TopKBlacklist
    from src.mitigation.evaluation import evaluate_policy
    params = json.loads(Path(SYN).read_text())['protocol']['chosen_params']
    th = {k: params[k] for k in ('t_block', 't_limit')}
    mode = json.loads(E.RESULTS_PATH.read_text())['level2']['systems']['cic_set']['p4_l3_on_legit_flows']
    scorer, _, _ = load_scorers()
    prepared = [E.prepare_cic_run((manifest['day'], r['start'], r['end'], r['scenario'])) for r in manifest['runs']]
    configs = {'P0': ('hybrid', lambda: TopKBlacklist(ttl=None)),
               'P4': ('hybrid', lambda: SourceScoringPolicy(scorer, **th)),
               'P4cic': ('hybrid-cic', lambda: SourceScoringPolicy(scorer, **th)),
               'P4L3': ('hybrid-cic', lambda: FlowVerifiedScoring(scorer, on_legit_flows=mode, **th))}
    out = {}
    for cfg, (det, make) in configs.items():
        runs = E._with_detector(prepared, det)
        out[cfg] = {'per_run': {}, 'mode': mode if cfg == 'P4L3' else None}
        for r, pr in zip(manifest['runs'], runs):
            res = evaluate_policy(make(), [pr], 'hybrid')
            out[cfg]['per_run'][r['name']] = {k: res[k] for k in ('attack_drop_pct', 'legit_loss_pct',
                                                                  'legit_sources_hit_pct', 'median_time_to_mitigate_s')}
        for group, sel in (('attack_runs', [pr for r, pr in zip(manifest['runs'], runs) if r['attack_onset']]),
                           ('quiet_runs', [pr for r, pr in zip(manifest['runs'], runs) if not r['attack_onset']]),
                           ('all_runs', runs)):
            res = evaluate_policy(make(), sel, 'hybrid')
            out[cfg][group] = {k: res[k] for k in ('attack_drop_pct', 'legit_loss_pct', 'legit_sources_hit_pct')}
    return out


def totals(per_run, names):
    t = {}
    for cfg in CONFIGS:
        rows = [per_run[n][cfg] for n in names if cfg in per_run.get(n, {})]
        if not rows:
            continue
        a_s, a_p = sum(r['attack_sent'] for r in rows), sum(r['attack_passed'] for r in rows)
        l_s, l_p = sum(r['legit_sent'] for r in rows), sum(r['legit_passed'] for r in rows)
        t[cfg] = {'attack_sent': a_s, 'legit_sent': l_s,
                  'attack_dropped_pct': 100.0 * (a_s - a_p) / a_s if a_s else None,
                  'legit_dropped_pct': 100.0 * (l_s - l_p) / l_s if l_s else None,
                  'legit_sources': sum(r['legit_sources'] for r in rows),
                  'legit_sources_hit': sum(r['legit_sources_hit'] for r in rows)}
        ml = [r['mitigation_delay_s'] for r in rows if r.get('mitigation_delay_s') is not None]
        if ml:
            t[cfg]['mitigation_delay_s_median'] = float(np.median(ml))
            t[cfg]['mitigation_delay_s_max'] = float(np.max(ml))
    return t


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('out_dir', type=Path)
    ap.add_argument('--replay-dir', type=Path, default=Path('/root/cic_replay'))
    ap.add_argument('--no-offline', action='store_true')
    args = ap.parse_args()
    manifest = json.loads((args.out_dir / 'manifest.json').read_text())
    per_run = {}
    for run in manifest['runs']:
        if not (args.out_dir / run['name']).exists():
            continue
        ts, src = per_source(args.replay_dir / f"{run['name']}.pcap")
        per_run[run['name']] = analyze_run(args.out_dir / run['name'], run, ts, src)
        for cfg, r in per_run[run['name']].items():
            print(f"{run['name']:16s} {cfg:6s} attack dropped {r['attack_dropped_pct'] if r['attack_dropped_pct'] is not None else float('nan'):6.2f}%  "
                  f"legit dropped {r['legit_dropped_pct']:6.2f}%  hit {r['legit_sources_hit']}/{r['legit_sources']}  "
                  f"alerts {r.get('alerts')}  mitigate {r.get('mitigation_delay_s')}  "
                  f"xdp {r['checks'].get('xdp_total')}/{r['checks']['sent']} passed {r['checks'].get('xdp_passed')}"
                  f"/{r['checks']['captured_after_xdp']} tcpdump-drop {r['checks']['tcpdump_dropped']} "
                  f"pps {r['checks'].get('replay_pps')}", flush=True)
    attack = [r['name'] for r in manifest['runs'] if r['attack_onset'] and r['name'] in per_run]
    quiet = [r['name'] for r in manifest['runs'] if not r['attack_onset'] and r['name'] in per_run]
    res = {'setup': {'what': 'real CIC-DDoS2019 test-day (03-11) packets replayed with tcpreplay at their '
                             'original timing into a veth pair; the system (main.py) runs on the victim end '
                             'with its XDP program in generic mode; tcpdump on the victim end records what '
                             'XDP passed', 'configs': CONFIGS,
                     'environment': (args.out_dir / 'environment.txt').read_text().strip().splitlines(),
                     'manifest': {k: v for k, v in manifest.items() if k != 'runs'},
                     'runs': [{k: r[k] for k in ('name', 'scenario', 'start', 'attack_onset', 'end', 'packets',
                                                 'attack_packets', 'legit_packets')} for r in manifest['runs']]},
           'per_run': per_run,
           'attack_runs': totals(per_run, attack), 'quiet_runs': totals(per_run, quiet),
           'all_runs': totals(per_run, attack + quiet)}
    if not args.no_offline:
        res['offline_same_windows'] = offline(manifest)
    res['provenance'] = provenance('scripts/cic_xdp_analyze.py', {'out_dir': str(args.out_dir)})
    RESULTS.write_text(json.dumps(res, indent=1, default=str))
    print('\nattack runs:', json.dumps(res['attack_runs'], indent=1))
    print('quiet runs:', json.dumps(res['quiet_runs'], indent=1))
    if 'offline_same_windows' in res:
        for cfg, o in res['offline_same_windows'].items():
            print(f"offline {cfg:6s} attack runs: drop {o['attack_runs']['attack_drop_pct']:.2f}% legit "
                  f"{o['attack_runs']['legit_loss_pct']:.2f}% | quiet runs legit {o['quiet_runs']['legit_loss_pct']:.2f}%")
    print('wrote', RESULTS)


if __name__ == '__main__':
    main()
