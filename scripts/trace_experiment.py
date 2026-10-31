#!/usr/bin/env python3
"""
Trace-driven mitigation experiment: real legitimate traffic, injected attacks.

Legitimate traffic comes from a real packet trace (MAWI, WIDE backbone; see
scripts/mawi_extract.py). Each of the busiest destinations in the trace plays
the protected server: every packet sent to it is a legitimate packet from a
real source, with real per-source rates, churn, TTLs and TCP windows. Attacks
are injected on top with the same models as the synthetic generator
(botnet / spoofed / low-rate UDP, SYN, ICMP and mixed floods; crafted or
OS-stack packet headers; bots that also send normal-looking traffic first).

Nothing is tuned on the trace: the detector, the P4 scorer and thresholds,
and the SIG classifier are the ones trained on synthetic data
(scripts/mitigation_experiment.py). Five injection seeds give mean +/- std.

Caveat: trace traffic that was itself malicious (scans, unknown attacks) is
counted as legitimate, since MAWI is not labeled at packet level.

Usage (from project/):
    python scripts/trace_experiment.py --trace 200803181400
Writes data/results/trace_results.json.
"""

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'scripts'))

from src.config import DATA_DIR, MODELS_DIR  # noqa: E402
from src.provenance import provenance  # noqa: E402

TRACE_DIR = DATA_DIR / 'external' / 'mawi'
RESULTS_PATH = DATA_DIR / 'results' / 'trace_results.json'
MAX_SOURCES = 1500                       # per-interval entries, as in the generator
ATTACKS = ['udp_flood', 'syn_flood', 'icmp_flood', 'mixed_flood',
           'udp_flood_spoofed', 'syn_flood_spoofed', 'low_rate_udp', 'low_rate_syn']
GROUPS = {'no attack': ['normal'],
          'botnet floods': ['udp_flood', 'syn_flood', 'icmp_flood', 'mixed_flood'],
          'low-rate floods': ['low_rate_udp', 'low_rate_syn'],
          'spoofed floods': ['udp_flood_spoofed', 'syn_flood_spoofed']}


def load_victims(trace: str, n_victims: int, window: int, min_pps: float):
    """Per-destination background traffic for the busiest destinations"""
    data = np.load(TRACE_DIR / f'{trace}.npz')
    sec, src, dst = data['sec'].astype(np.int64), data['src'], data['dst']
    proto, syn, ttl, value, length = data['proto'], data['syn'], data['ttl'], data['value'], data['length']
    duration = int(sec.max()) + 1
    n_windows = duration // window

    # Rank destinations by distinct sources, keep those busy enough to be a server
    order = np.lexsort((src, dst))
    d_s, s_s = dst[order], src[order]
    pair_start = np.r_[True, (d_s[1:] != d_s[:-1]) | (s_s[1:] != s_s[:-1])]
    dests, n_src = np.unique(d_s[pair_start], return_counts=True)
    pkts_per_dst = dict(zip(*np.unique(dst, return_counts=True)))
    ranked = [d for d in dests[np.argsort(-n_src)]
              if pkts_per_dst[d] / duration >= min_pps][:n_victims]

    victims = []
    for d in ranked:
        m = dst == d
        victims.append({
            'dst': int(d), 'sec': sec[m], 'src': src[m], 'proto': proto[m], 'syn': syn[m],
            'ttl': ttl[m], 'value': value[m], 'length': length[m],
        })
    return victims, n_windows


def _source_signatures(v):
    """One header signature per source: first SYN (TTL, window), else first packet"""
    from src.ml.traffic_dataset import signature_code
    cls = np.select([v['proto'] == 6, v['proto'] == 17, v['proto'] == 1], [1, 2, 3], 2)
    val = np.where(cls == 1, v['value'], v['length']).astype(np.int64)
    codes = (cls.astype(np.int64) << 40) | signature_code(0, v['ttl'], val)
    sig = {}
    # prefer a SYN packet's signature (as Dimolianis et al. do for TCP)
    for prefer_syn in (True, False):
        for s, c, y in zip(v['src'], codes, v['syn']):
            if (y == 1) == prefer_syn and int(s) not in sig:
                sig[int(s)] = int(c)
    return sig


def build_run(v, w, window, rng, scenario, bot_pre_activity=0.3):
    """One labeled TrafficRun: trace background of window w + injected attack"""
    from src.ml.traffic_dataset import (TrafficRun, TrafficStep, _OS_PROFILES, _OS_WEIGHTS,
                                        signature_code)
    lo, hi = w * window, (w + 1) * window
    m = (v['sec'] >= lo) & (v['sec'] < hi)
    t_rel = (v['sec'][m] - lo).astype(np.int64)
    src = v['src'][m].astype(np.int64)
    proto, syn, length = v['proto'][m], v['syn'][m], v['length'][m]
    sigs = _source_signatures({k: (x[m] if isinstance(x, np.ndarray) else x) for k, x in v.items()})
    base_pps = max(1.0, m.sum() / window)

    is_attack_run = scenario != 'normal'
    onset = int(rng.integers(window // 3, window // 2))
    end = window if rng.random() < 0.7 else min(window, onset + int(rng.integers(15, 35)))
    mult = rng.uniform(1.3, 2.5) if scenario.startswith('low_rate') else float(np.exp(rng.uniform(np.log(3), np.log(80))))
    ramp = int(rng.integers(1, 15))
    spoofed = scenario.endswith('_spoofed')
    n_bots = int(np.exp(rng.uniform(np.log(1), np.log(2000))))
    bot_w = 1.0 / np.arange(1, n_bots + 1) ** rng.uniform(0.0, 0.6)
    rng.shuffle(bot_w)
    bot_w /= bot_w.sum()
    taken = set(np.unique(src).tolist())
    bot_ips = []
    while len(bot_ips) < n_bots:          # random public-looking addresses not in the trace
        ip = int(rng.integers(1 << 24, 223 << 24))
        if ip not in taken:
            taken.add(ip)
            bot_ips.append(ip)
    bot_ips = np.array(bot_ips, dtype=np.int64)
    kind = scenario.replace('_spoofed', '').replace('low_rate_', '').replace('_flood', '')
    shares = {'udp': {'udp': 1.0}, 'syn': {'syn': 1.0}, 'icmp': {'icmp': 1.0}}.get(kind)
    if shares is None:
        u, s_ = rng.dirichlet([2, 2, 1])[:2]
        shares = {'udp': u, 'syn': s_, 'icmp': 1 - u - s_}
    names = list(shares)
    pcode = {'udp': 2, 'syn': 1, 'icmp': 3}
    bot_proto = np.array([pcode[names[i]] for i in rng.choice(len(names), n_bots, p=[shares[k] for k in names])])
    tool_os_like = bool(rng.random() < 0.5)
    tool_window, tool_size = int(rng.integers(1, 65536)), int(rng.integers(64, 1401))
    tool_hops, bot_hops = int(rng.integers(4, 21)), rng.integers(4, 21, n_bots)
    os_i = rng.choice(len(_OS_PROFILES), n_bots, p=_OS_WEIGHTS)
    bot_os_sig = signature_code(1, np.array([_OS_PROFILES[i][0] for i in os_i]) - bot_hops,
                                np.array([_OS_PROFILES[i][1] for i in os_i]))
    pre_rate = np.where(rng.random(n_bots) < bot_pre_activity, rng.uniform(0.2, 2.0, n_bots), 0.0) \
        if is_attack_run and not spoofed else np.zeros(n_bots)
    next_spoof = 1 << 32                 # spoofed ids above the IPv4 range (fresh every second)

    run = TrafficRun(scenario=scenario, tool_os_like=tool_os_like)
    cum = dict(total_packets=0, total_bytes=0, tcp_packets=0, udp_packets=0, icmp_packets=0)
    order = np.argsort(t_rel, kind='stable')
    t_sorted = t_rel[order]
    bounds = np.searchsorted(t_sorted, np.arange(window + 1))
    for t in range(window):
        sel = order[bounds[t]:bounds[t + 1]]
        ids, inv = np.unique(src[sel], return_inverse=True)
        pk = np.bincount(inv, minlength=len(ids)).astype(np.float64)
        sy = np.bincount(inv, weights=syn[sel], minlength=len(ids))
        cum['total_packets'] += len(sel)
        cum['total_bytes'] += int(length[sel].sum())
        cum['tcp_packets'] += int((proto[sel] == 6).sum())
        cum['udp_packets'] += int((proto[sel] == 17).sum())
        cum['icmp_packets'] += int((proto[sel] == 1).sum())
        ip_pk, ip_sy, ip_id = [pk], [sy], [ids]
        ip_atk = [np.zeros(len(ids), dtype=bool)]
        ip_sig = [np.array([sigs[int(i)] for i in ids], dtype=np.int64)]

        attack_now = False
        if is_attack_run and onset <= t < end:
            attack_now = True
            n_atk = int(rng.poisson(base_pps * mult * min(1.0, (t - onset + 1) / ramp)))
            if spoofed:
                n_src = min(n_atk, MAX_SOURCES)
                a_pk = np.floor(np.full(n_src, n_atk / max(n_src, 1)) + rng.random(n_src))
                a_id = next_spoof + np.arange(n_src)
                next_spoof += n_src
                a_proto = np.full(n_src, bot_proto[0])
                hops = np.full(n_src, tool_hops)
                tcp_sig = signature_code(1, 64 - hops, 64240 if tool_os_like else tool_window)
            else:
                a_pk = rng.multinomial(n_atk, bot_w).astype(np.float64)
                a_id, a_proto, hops = bot_ips, bot_proto, bot_hops
                tcp_sig = bot_os_sig if tool_os_like else signature_code(1, 64 - hops, tool_window)
            a_sy = np.where(a_proto == 1, a_pk, 0.0)
            ip_pk.append(a_pk)
            ip_sy.append(a_sy)
            ip_id.append(a_id)
            ip_atk.append(np.ones(len(a_pk), dtype=bool))
            ip_sig.append(np.where(a_proto == 1, tcp_sig,
                                   signature_code(2, 64 - hops, tool_size) * (a_proto == 2)
                                   + signature_code(3, 64 - hops, tool_size) * (a_proto == 3)))
            n_a = int(a_pk.sum())
            cum['total_packets'] += n_a
            cum['total_bytes'] += n_a * (60 if kind == 'syn' else tool_size)
            cum['tcp_packets'] += int(a_pk[a_proto == 1].sum())
            cum['udp_packets'] += int(a_pk[a_proto == 2].sum())
            cum['icmp_packets'] += int(a_pk[a_proto == 3].sum())
        if pre_rate.any():
            pre = rng.poisson(pre_rate).astype(np.float64)
            ip_pk.append(pre)
            ip_sy.append(np.zeros(n_bots))
            ip_id.append(bot_ips)
            ip_atk.append(np.ones(n_bots, dtype=bool))
            ip_sig.append(bot_os_sig)
            cum['total_packets'] += int(pre.sum())
            cum['tcp_packets'] += int(pre.sum())
            cum['total_bytes'] += int(pre.sum()) * 500

        pk_all = np.concatenate(ip_pk)
        keep = pk_all > 0
        run.steps.append(TrafficStep(
            timestamp=1_000_000.0 + t, stats=dict(cum), ip_packets=pk_all[keep],
            ip_syn=np.concatenate(ip_sy)[keep], is_attack=attack_now,
            ip_ids=np.concatenate(ip_id)[keep], ip_is_attack=np.concatenate(ip_atk)[keep],
            ip_sig=np.concatenate(ip_sig)[keep]))
    return run


def evaluate_seed(args):
    """Worker: build every run of one injection seed and evaluate all policies"""
    trace, seed, n_victims, window, min_pps = args
    import logging
    logging.disable(logging.WARNING)
    from mitigation_experiment import _detectors, build_policies, load_scorers, RESULTS_PATH as SYN
    from src.mitigation import AdaptiveHistoryFilter
    from src.mitigation.evaluation import prepare_run, evaluate_policy

    victims, n_windows = load_victims(trace, n_victims, window, min_pps)
    rng = np.random.default_rng(seed)
    detectors = _detectors()
    runs = []
    for v in victims:
        for w in range(n_windows):
            scenario = str(rng.choice(ATTACKS)) if rng.random() < 0.6 else 'normal'
            run = build_run(v, w, window, rng, scenario)
            prepared = prepare_run(run, detectors)
            prepared.src_ip = np.unique(np.concatenate([s.ip_ids for s in run.steps])) & 0xFFFFFFFF
            runs.append(prepared)

    params = json.loads(Path(SYN).read_text())['protocol']['chosen_params']
    scorer, sig_model, scorer_no_history = load_scorers()
    policies = build_policies(scorer, params, sig_model, scorer_no_history)
    policies['AHIF-/24'] = ('hybrid', AdaptiveHistoryFilter(margin=1.5, prefix_bits=24))
    out = {'n_runs': len(runs), 'n_attack_runs': sum(r.is_attack_run for r in runs),
           'policies': {}, 'groups': {}, 'profiles': {}}
    for key, (detector, policy) in policies.items():
        out['policies'][key] = {'name': policy.name, 'detector': detector,
                                **evaluate_policy(policy, runs, detector)}
        out['groups'][key] = {g: evaluate_policy(policy, [r for r in runs if r.scenario in sc], detector)
                              for g, sc in GROUPS.items() if any(r.scenario in sc for r in runs)}
        out['profiles'][key] = {
            name: evaluate_policy(policy, [r for r in runs if r.is_attack_run and r.tool_os_like == flag], detector)
            for name, flag in (('crafted headers', False), ('OS-stack headers', True))}
    benign = [r for r in runs if not r.is_attack_run]
    out['false_alarm_seconds_pct'] = {
        d: 100.0 * sum(sum(r.anomaly[d]) for r in benign) / max(1, sum(len(r.anomaly[d]) for r in benign))
        for d in detectors}
    attack = [r for r in runs if r.is_attack_run]
    out['detected_attack_runs_pct'] = {
        d: 100.0 * sum(any(a and s for a, s in zip(r.anomaly[d], r.attack_second)) for r in attack) / max(1, len(attack))
        for d in detectors}
    return seed, out


def summarize(values):
    vals = [v for v in values if v is not None]
    return {'mean': float(np.mean(vals)), 'std': float(np.std(vals))} if vals else None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--trace', default='200803181400')
    ap.add_argument('--victims', type=int, default=20)
    ap.add_argument('--window', type=int, default=90, help='seconds per run')
    ap.add_argument('--min-pps', type=float, default=5.0, help='minimum packets/s to a victim')
    ap.add_argument('--seeds', type=int, default=5)
    ap.add_argument('--workers', type=int, default=5)
    args = ap.parse_args()
    started = time.time()

    victims, n_windows = load_victims(args.trace, args.victims, args.window, args.min_pps)
    print(f'[trace] {args.trace}: {len(victims)} victims x {n_windows} windows of {args.window} s; '
          f'background pps per victim: ' + ', '.join(f'{len(v["sec"]) / (n_windows * args.window):.0f}' for v in victims))
    jobs = [(args.trace, 7000 + i, args.victims, args.window, args.min_pps) for i in range(args.seeds)]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        per_seed = dict(pool.map(evaluate_seed, jobs))

    seeds = sorted(per_seed)
    first = per_seed[seeds[0]]
    metric_keys = [k for k, v in first['policies']['P4'].items() if isinstance(v, (int, float)) or v is None]
    res = {
        'protocol': {
            'trace': f'MAWI samplepoint-F {args.trace} (first {n_windows * args.window} s)',
            'victims': len(victims), 'window_s': args.window, 'injection_seeds': seeds,
            'runs_per_seed': first['n_runs'], 'attack_runs_per_seed': first['n_attack_runs'],
            'note': 'legitimate traffic = real trace packets to each victim; attacks injected; '
                    'no tuning on the trace (detector, scorer, thresholds, SIG classifier from synthetic data)',
            'victim_background_pps': [round(len(v['sec']) / (n_windows * args.window), 1) for v in victims],
            'victim_sources': [int(len(np.unique(v['src']))) for v in victims],
            'provenance': provenance('scripts/trace_experiment.py', vars(args), {
                'trace': TRACE_DIR / f'{args.trace}.npz',
                'detection_model': MODELS_DIR / 'ddos_classifier.joblib',
                'source_scorer': MODELS_DIR / 'source_scorer.joblib',
                'source_scorer_no_history': MODELS_DIR / 'source_scorer_no_history.joblib',
                'signature_classifier': MODELS_DIR / 'signature_classifier.joblib',
                'mitigation_results (thresholds)': DATA_DIR / 'results' / 'mitigation_results.json'}),
        },
        'false_alarm_seconds_pct': {d: summarize([per_seed[s]['false_alarm_seconds_pct'][d] for s in seeds])
                                    for d in first['false_alarm_seconds_pct']},
        'detected_attack_runs_pct': {d: summarize([per_seed[s]['detected_attack_runs_pct'][d] for s in seeds])
                                     for d in first['detected_attack_runs_pct']},
        'policies': {k: {'name': v['name'], 'detector': v['detector'],
                         **{m: summarize([per_seed[s]['policies'][k][m] for s in seeds]) for m in metric_keys}}
                     for k, v in first['policies'].items()},
        'groups': {k: {g: {m: summarize([per_seed[s]['groups'][k][g][m] for s in seeds if g in per_seed[s]['groups'][k]])
                           for m in metric_keys} for g in gs} for k, gs in first['groups'].items()},
        'profiles': {k: {g: {m: summarize([per_seed[s]['profiles'][k][g][m] for s in seeds]) for m in metric_keys}
                         for g in gs} for k, gs in first['profiles'].items()},
    }
    RESULTS_PATH.write_text(json.dumps(res, indent=1))
    print(f'[done] {RESULTS_PATH} in {time.time() - started:.0f}s')
    print('detected attack runs %:', {d: round(v['mean'], 1) for d, v in res['detected_attack_runs_pct'].items()},
          '| false-alarm seconds %:', {d: round(v['mean'], 2) for d, v in res['false_alarm_seconds_pct'].items()})
    print(f"{'policy':18s} {'attack drop %':>16s} {'legit loss %':>14s} {'legit srcs hit %':>17s}")
    for k, r in res['policies'].items():
        f = lambda m: f"{r[m]['mean']:6.2f} ± {r[m]['std']:4.2f}"  # noqa: E731
        print(f"{k:18s} {f('attack_drop_pct'):>16s} {f('legit_loss_pct'):>14s} {f('legit_sources_hit_pct'):>17s}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
