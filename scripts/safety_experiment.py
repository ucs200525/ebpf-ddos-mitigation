#!/usr/bin/env python3
"""
Mitigation-safety experiment: traffic where legitimate users look like attackers.

Scenarios (synthetic, labelled; src/ml/traffic_dataset.py, "realistic" setting):
  flash_crowd                   many new legitimate users, no attack
  heavy_legit                   1-5 existing clients start bulk transfers (3-15x normal), no attack
  udp_flood_with_crowd          botnet UDP flood while a flash crowd of new users arrives
  udp_flood_spoofed_with_crowd  spoofed UDP flood while a flash crowd of new users arrives

None of these runs is used for training or tuning: the detector, the P4 scorer, its
thresholds and the SIG classifier are the ones produced by
scripts/mitigation_experiment.py. Low-rate and spoofed floods without a crowd are
already reported per scenario group by that script.

Seeds 5001-5005 (never used elsewhere), --runs runs per scenario and seed.

Usage (from project/):
    python scripts/safety_experiment.py                 # 5 seeds x 40 runs per scenario
    python scripts/safety_experiment.py --runs 5 --seeds 1 --workers 4   # quick check
Writes data/results/safety_results.json.
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

RESULTS_PATH = DATA_DIR / 'results' / 'safety_results.json'
SEED_BASE = 5001
SCENARIOS = ['flash_crowd', 'heavy_legit', 'udp_flood_with_crowd', 'udp_flood_spoofed_with_crowd']
POLICIES = ('P0', 'P1', 'P2', 'P3a', 'P3b', 'P4', 'AHIF', 'SIG', 'SIG-oracle',
            'P4-no-spoof-guard', 'P4-block-only', 'P4-no-expiry', 'P4-no-history')


def evaluate_seed(job):
    """Worker: all scenarios of one seed, every policy"""
    seed, n_runs = job
    import logging
    logging.disable(logging.WARNING)
    from mitigation_experiment import SETTINGS, _detectors, build_policies, load_scorers, \
        RESULTS_PATH as SYN
    from src.ml.traffic_dataset import generate_scenario_runs
    from src.mitigation.evaluation import evaluate_policy, prepare_run

    detectors = _detectors()
    params = json.loads(Path(SYN).read_text())['protocol']['chosen_params']
    scorer, sig_model, scorer_no_history = load_scorers()
    out = {}
    for i, scenario in enumerate(SCENARIOS):
        runs = []
        # one independent stream per (seed, scenario)
        for run in generate_scenario_runs(scenario, n_runs, seed=seed * 100 + i, **SETTINGS['realistic']):
            for step in run.steps:
                keep = step.ip_packets > 0
                step.ip_packets, step.ip_syn = step.ip_packets[keep], step.ip_syn[keep]
                step.ip_ids, step.ip_is_attack = step.ip_ids[keep], step.ip_is_attack[keep]
                step.ip_sig = step.ip_sig[keep]
            runs.append(prepare_run(run, detectors))
        entry = {'detector': {}}
        for d in detectors:
            flagged = sum(sum(r.anomaly[d]) for r in runs)
            seconds = sum(len(r.anomaly[d]) for r in runs)
            benign_secs = [(a, s) for r in runs for a, s in zip(r.anomaly[d], r.attack_second) if not s]
            entry['detector'][d] = {
                'flagged_seconds_pct': 100.0 * flagged / max(1, seconds),
                'benign_seconds_flagged_pct': 100.0 * sum(a for a, _ in benign_secs) / max(1, len(benign_secs)),
            }
        policies = build_policies(scorer, params, sig_model, scorer_no_history)
        entry['policies'] = {key: {'name': policies[key][1].name, 'detector': policies[key][0],
                                   **evaluate_policy(policies[key][1], runs, policies[key][0])}
                             for key in POLICIES if key in policies}
        out[scenario] = entry
    return seed, out


def summarize(values):
    vals = [v for v in values if v is not None]
    return {'mean': float(np.mean(vals)), 'std': float(np.std(vals))} if vals else None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--runs', type=int, default=40, help='runs per scenario and seed')
    ap.add_argument('--seeds', type=int, default=5)
    ap.add_argument('--workers', type=int, default=5)
    args = ap.parse_args()
    started = time.time()
    jobs = [(SEED_BASE + i, args.runs) for i in range(args.seeds)]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        per_seed = dict(pool.map(evaluate_seed, jobs))
    seeds = sorted(per_seed)
    first = per_seed[seeds[0]]

    res = {'protocol': {
        'scenarios': SCENARIOS, 'seeds': seeds, 'runs_per_scenario_per_seed': args.runs,
        'setting': 'realistic (client_churn=0.1, bot_pre_activity=0.3)',
        'data': 'synthetic (src/ml/traffic_dataset.py); no run used for training or tuning',
        'provenance': provenance('scripts/safety_experiment.py', vars(args), {
            'detection_model': MODELS_DIR / 'ddos_classifier.joblib',
            'source_scorer': MODELS_DIR / 'source_scorer.joblib',
            'source_scorer_no_history': MODELS_DIR / 'source_scorer_no_history.joblib',
            'signature_classifier': MODELS_DIR / 'signature_classifier.joblib',
            'mitigation_results (thresholds)': DATA_DIR / 'results' / 'mitigation_results.json'}),
    }, 'scenarios': {}}
    for scenario, entry in first.items():
        metric_keys = [k for k, v in entry['policies']['P4'].items()
                       if isinstance(v, (int, float)) or v is None]
        res['scenarios'][scenario] = {
            'detector': {d: {m: summarize([per_seed[s][scenario]['detector'][d][m] for s in seeds])
                             for m in entry['detector'][d]} for d in entry['detector']},
            'policies': {k: {'name': v['name'], 'detector': v['detector'],
                             **{m: summarize([per_seed[s][scenario]['policies'][k][m] for s in seeds])
                                for m in metric_keys}}
                         for k, v in entry['policies'].items()},
        }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(res, indent=1))
    print(f'[done] {RESULTS_PATH} in {time.time() - started:.0f}s')

    fmt = lambda v: '   -   ' if v is None else f"{v['mean']:6.2f}"  # noqa: E731
    for scenario, entry in res['scenarios'].items():
        det = entry['detector']['hybrid']
        print(f"\n== {scenario} (hybrid detector flags {fmt(det['flagged_seconds_pct'])}% of seconds) ==")
        print(f"{'policy':18s} {'attack drop %':>13s} {'legit loss %':>12s} {'legit srcs hit %':>16s} "
              f"{'post-attack loss %':>18s} {'recovery s':>10s}")
        for k, r in entry['policies'].items():
            print(f"{k:18s} {fmt(r['attack_drop_pct']):>13s} {fmt(r['legit_loss_pct']):>12s} "
                  f"{fmt(r['legit_sources_hit_pct']):>16s} {fmt(r['legit_loss_post_attack_pct']):>18s} "
                  f"{fmt(r['median_recovery_s']):>10s}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
