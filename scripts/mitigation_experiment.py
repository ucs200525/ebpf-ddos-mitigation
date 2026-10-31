#!/usr/bin/env python3
"""
Mitigation experiment: attack traffic stopped vs legitimate traffic lost.

Protocol (no test data is used for any fitting or tuning):
  1. Tuning, "realistic" setting only:
       seed 900  -> train the per-source scorer (SourceScorer)
       seed 901  -> choose P4 thresholds, maximizing attack_drop - 2 x legit_loss
       seed 900  -> also train the signature classifier of the SIG baseline
  2. Test: 5 fresh seeds (2001-2005) x 3 realism settings, every policy on the same
     cached detector verdicts (hybrid detector; P0 also with the
     statistical-only detector, i.e. the system before ML).

Published baselines: AHIF (Goldstein et al., ICN 2008) and SIG, signature-based
filtering (Dimolianis et al., 2021), with a trained classifier and with a
perfect one (upper bound). Results are also split by attack header profile
(crafted vs ordinary OS stack), which decides how well SIG can work.

Realism settings (see src/ml/traffic_dataset.generate_run):
  clean      client_churn=0.0, bot_pre_activity=0.0
  realistic  client_churn=0.1, bot_pre_activity=0.3
  hard       client_churn=0.2, bot_pre_activity=0.6

Usage (from project/):
    python scripts/mitigation_experiment.py                  # full run (~5-10 min on 16 cores)
    python scripts/mitigation_experiment.py --runs 20 --seeds 2 --workers 4   # quick check
Writes data/results/mitigation_results.json and data/models/source_scorer.joblib.
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

from src.config import DATA_DIR, MLConfig, MODELS_DIR  # noqa: E402
from src.provenance import provenance  # noqa: E402

SETTINGS = {
    'clean': dict(client_churn=0.0, bot_pre_activity=0.0),
    'realistic': dict(client_churn=0.1, bot_pre_activity=0.3),
    'hard': dict(client_churn=0.2, bot_pre_activity=0.6),
}
TUNE_TRAIN_SEED, TUNE_SELECT_SEED = 900, 901
TEST_SEED_BASE = 2001
DESIGN_NOTE = ('Version 2. Version 1 (test seeds 1001-1005, archived in '
               'data/results/mitigation_results_v1.json) used absolute-volume scorer features, '
               'unbalanced scorer training and a <=2-packet spoofing trigger; inspecting its results '
               'showed legitimate users of small networks being blocked and the spoofing fallback '
               'never firing. The scorer features, training sampling and spoofing trigger were '
               'revised, and seeds 1001-1005 were retired: these results use unseen seeds 2001-2005.')
SCORER_PATH = MODELS_DIR / 'source_scorer.joblib'
SCORER_NO_HISTORY_PATH = MODELS_DIR / 'source_scorer_no_history.joblib'   # ablation only
SIG_MODEL_PATH = MODELS_DIR / 'signature_classifier.joblib'
RESULTS_PATH = DATA_DIR / 'results' / 'mitigation_results.json'
T_BLOCK_GRID = [0.7, 0.8, 0.9, 0.95, 0.98]
T_LIMIT_GRID = [0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98]
LEGIT_WEIGHT = 2.0   # collateral damage counts double in the tuning objective
SCENARIO_GROUPS = {
    'normal / bursty': ['normal', 'normal_bursty'],
    'flash crowd': ['flash_crowd'],
    'botnet floods': ['udp_flood', 'syn_flood', 'icmp_flood', 'mixed_flood'],
    'low-rate floods': ['low_rate_udp', 'low_rate_syn'],
    'spoofed floods': ['udp_flood_spoofed', 'syn_flood_spoofed'],
}


def _detectors():
    from src.core.anomaly_detector import AnomalyDetector, MLEnhancedAnomalyDetector
    from src.ml.ml_classifier import DDoSClassifier
    classifier = DDoSClassifier()
    if not classifier.load(MLConfig.DEFAULT_MODEL_PATH):
        raise SystemExit('Trained detection model missing: python -m src.ml.model_trainer train')

    def hybrid():
        d = MLEnhancedAnomalyDetector()
        d.classifier, d.ml_enabled = classifier, True
        return d
    return {'hybrid': hybrid, 'statistical': AnomalyDetector}


def prepare(setting: str, seed: int, n_runs: int):
    """Worker: generate runs and cache detector verdicts (zero-packet entries removed)"""
    import logging
    logging.disable(logging.WARNING)
    from src.ml.traffic_dataset import generate_runs
    from src.mitigation.evaluation import prepare_run
    detectors = _detectors()
    prepared = []
    for run in generate_runs(n_runs, seed=seed, **SETTINGS[setting]):
        for step in run.steps:
            keep = step.ip_packets > 0
            step.ip_packets, step.ip_syn = step.ip_packets[keep], step.ip_syn[keep]
            step.ip_ids, step.ip_is_attack = step.ip_ids[keep], step.ip_is_attack[keep]
            step.ip_sig = step.ip_sig[keep]
        prepared.append(prepare_run(run, detectors))
    return prepared


MAIN_POLICIES = ('P0', 'P1', 'P2', 'P3a', 'P3b', 'P4', 'AHIF', 'SIG', 'SIG-oracle')


def build_policies(scorer, params, sig_model=None, scorer_no_history=None):
    """
    Every policy evaluated. Baselines P2, AHIF and SIG are our re-implementations
    from the published descriptions (no original code was available to us).
    Ablations reuse P4's tuned thresholds; only the named component changes.
    """
    from src.mitigation import (TopKBlacklist, HistoryFilter, AggregateRateLimit,
                                PerSourceRateLimit, SourceScoringPolicy,
                                AdaptiveHistoryFilter, SignatureFilter)
    p4 = dict(t_block=params['t_block'], t_limit=params['t_limit'])
    policies = {
        'P0': ('hybrid', TopKBlacklist(ttl=None)),
        'P0-statistical': ('statistical', TopKBlacklist(ttl=None)),
        'P1': ('hybrid', TopKBlacklist(ttl=30)),
        'P2': ('hybrid', HistoryFilter()),
        'P3a': ('hybrid', AggregateRateLimit(margin=1.5)),
        'P3b': ('hybrid', PerSourceRateLimit(quantile=0.99)),
        'P4': ('hybrid', SourceScoringPolicy(scorer, **p4)),
        'P4-no-spoof-guard': ('hybrid', SourceScoringPolicy(scorer, spoof_share=1.1, **p4)),
        'P4-block-only': ('hybrid', SourceScoringPolicy(scorer, t_block=params['t_block'],
                                                        t_limit=params['t_block'])),
        'P4-no-expiry': ('hybrid', SourceScoringPolicy(scorer, ttl=np.inf, **p4)),
        'AHIF': ('hybrid', AdaptiveHistoryFilter(margin=1.5)),
        'SIG': ('hybrid', SignatureFilter(classifier=sig_model)),
        'SIG-oracle': ('hybrid', SignatureFilter()),
    }
    if scorer_no_history is not None:
        no_hist = SourceScoringPolicy(scorer_no_history, **p4)
        no_hist.name = 'Source scoring without per-source history features (ablation)'
        policies['P4-no-history'] = ('hybrid', no_hist)
    return policies


def load_scorers():
    """Trained P4 scorer, SIG classifier and (if present) the no-history ablation scorer"""
    import joblib
    from src.mitigation import SourceScorer
    no_hist = SourceScorer.load(str(SCORER_NO_HISTORY_PATH)) if SCORER_NO_HISTORY_PATH.exists() else None
    return SourceScorer.load(str(SCORER_PATH)), joblib.load(SIG_MODEL_PATH), no_hist


def evaluate_seed(setting: str, seed: int, n_runs: int, params: dict):
    """Worker: prepare one seed and evaluate every policy, per scenario group too"""
    from src.mitigation import SourceScoringPolicy
    from src.mitigation.evaluation import evaluate_policy
    scorer, sig_model, scorer_no_history = load_scorers()
    runs = prepare(setting, seed, n_runs)
    out = {'policies': {}, 'groups': {}, 'profiles': {}, 'curve': []}
    for key, (detector, policy) in build_policies(scorer, params, sig_model, scorer_no_history).items():
        out['policies'][key] = {'name': policy.name, 'detector': detector,
                                **evaluate_policy(policy, runs, detector)}
        if key in MAIN_POLICIES:
            out['groups'][key] = {}
            for group, scenarios in SCENARIO_GROUPS.items():
                subset = [r for r in runs if r.scenario in scenarios]
                if subset:
                    out['groups'][key][group] = evaluate_policy(policy, subset, detector)
            # attack runs split by the attack tool's header profile
            out['profiles'][key] = {}
            for profile, flag in (('crafted headers', False), ('OS-stack headers', True)):
                subset = [r for r in runs if r.is_attack_run and r.tool_os_like == flag]
                if subset:
                    out['profiles'][key][profile] = evaluate_policy(policy, subset, detector)
    for t_block in T_BLOCK_GRID:
        for t_limit in (t for t in T_LIMIT_GRID if t <= t_block):
            res = evaluate_policy(SourceScoringPolicy(scorer, t_block=t_block, t_limit=t_limit), runs, 'hybrid')
            out['curve'].append({'t_block': t_block, 't_limit': t_limit,
                                 'attack_drop_pct': res['attack_drop_pct'],
                                 'legit_loss_pct': res['legit_loss_pct']})
    # One illustrative attack run with a time series per policy (first spoofed or botnet flood)
    if seed == TEST_SEED_BASE and setting == 'realistic':
        example = next((r for r in runs if r.scenario == 'udp_flood'), None) or \
            next((r for r in runs if r.is_attack_run), None)
        if example is not None:
            out['example'] = {'scenario': example.scenario, 'series': {}}
            for key in ('P0', 'P2', 'P4'):
                detector, policy = build_policies(scorer, params, sig_model)[key]
                res = evaluate_policy(policy, [example], detector, trace=True)
                out['example']['series'][key] = res['traces'][0]
    return setting, seed, out


def tune(n_runs: int, workers: int):
    from src.mitigation import SourceScorer, SourceScoringPolicy
    from src.mitigation.evaluation import collect_source_samples, evaluate_policy
    print(f'[tune] preparing seeds {TUNE_TRAIN_SEED}, {TUNE_SELECT_SEED} (realistic, {n_runs} runs each)')
    with ProcessPoolExecutor(max_workers=min(workers, 2)) as pool:
        train_runs, select_runs = pool.map(prepare, ['realistic'] * 2,
                                           [TUNE_TRAIN_SEED, TUNE_SELECT_SEED], [n_runs] * 2)

    import joblib
    from src.mitigation.policies import train_signature_classifier
    sig_model = train_signature_classifier(train_runs, 'hybrid')
    joblib.dump(sig_model, SIG_MODEL_PATH)
    print(f'[tune] signature classifier trained on seed {TUNE_TRAIN_SEED}')

    X, y = collect_source_samples(train_runs, 'hybrid')
    scorer = SourceScorer().fit(X, y, info={'train_seed': TUNE_TRAIN_SEED, 'setting': 'realistic',
                                             'runs': n_runs})
    scorer.save(str(SCORER_PATH))
    print(f'[tune] scorer trained on {len(y)} source-intervals ({100 * y.mean():.1f}% attack); '
          f'coefficients: ' + ', '.join(f'{k}={v:+.2f}' for k, v in scorer.coefficients().items()))
    # Ablation: the same training rows without the per-source history features
    from src.mitigation.source_scorer import HISTORY_FEATURES, SOURCE_FEATURES
    no_hist = SourceScorer([f for f in SOURCE_FEATURES if f not in HISTORY_FEATURES]).fit(
        X, y, info={'train_seed': TUNE_TRAIN_SEED, 'setting': 'realistic', 'runs': n_runs,
                    'ablation': 'no per-source history features'})
    no_hist.save(str(SCORER_NO_HISTORY_PATH))

    grid = []
    for t_block in T_BLOCK_GRID:
        for t_limit in (t for t in T_LIMIT_GRID if t <= t_block):
            res = evaluate_policy(SourceScoringPolicy(scorer, t_block=t_block, t_limit=t_limit),
                                  select_runs, 'hybrid')
            objective = res['attack_drop_pct'] - LEGIT_WEIGHT * res['legit_loss_pct']
            grid.append({'t_block': t_block, 't_limit': t_limit, 'objective': objective,
                         'attack_drop_pct': res['attack_drop_pct'], 'legit_loss_pct': res['legit_loss_pct']})
    best = max(grid, key=lambda g: g['objective'])
    print(f"[tune] chosen t_block={best['t_block']}, t_limit={best['t_limit']} "
          f"(selection seed: attack drop {best['attack_drop_pct']:.1f}%, legit loss {best['legit_loss_pct']:.2f}%)")
    return {'t_block': best['t_block'], 't_limit': best['t_limit']}, grid, scorer


def summarize(per_seed):
    """mean and std across seeds for every numeric metric"""
    keys = [k for k, v in per_seed[0].items() if isinstance(v, (int, float)) or v is None]
    out = {}
    for k in keys:
        vals = [s[k] for s in per_seed if s[k] is not None]
        out[k] = {'mean': float(np.mean(vals)), 'std': float(np.std(vals))} if vals else None
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    parser.add_argument('--runs', type=int, default=100, help='runs per seed')
    parser.add_argument('--seeds', type=int, default=5, help='test seeds per setting')
    parser.add_argument('--workers', type=int, default=15)
    args = parser.parse_args()
    started = time.time()

    params, grid, scorer = tune(args.runs, args.workers)

    jobs = [(s, TEST_SEED_BASE + i) for s in SETTINGS for i in range(args.seeds)]
    print(f'[test] {len(jobs)} jobs ({args.runs} runs each) on {args.workers} workers')
    raw = {s: {} for s in SETTINGS}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(evaluate_seed, s, seed, args.runs, params) for s, seed in jobs]
        for f in futures:
            setting, seed, out = f.result()
            raw[setting][seed] = out
            print(f'[test] done {setting} seed {seed}')

    results = {
        'protocol': {
            'runs_per_seed': args.runs, 'test_seeds': [TEST_SEED_BASE + i for i in range(args.seeds)],
            'tune_seeds': {'scorer_training': TUNE_TRAIN_SEED, 'threshold_selection': TUNE_SELECT_SEED},
            'settings': SETTINGS, 'objective': f'attack_drop - {LEGIT_WEIGHT:g} x legit_loss',
            'chosen_params': params, 'detection_model': MLConfig.DEFAULT_MODEL_PATH,
            'data': 'synthetic (src/ml/traffic_dataset.py)',
            'baselines': {'AHIF': 'Goldstein et al., ICN 2008 (per-address history, n=32)',
                          'SIG': 'Dimolianis et al., ICIN 2021 / IEEE Access 2021 (Random Forest on '
                                 'header signatures, trained on seed 900)',
                          'SIG-oracle': 'same filter with ground-truth signature labels (upper bound)'},
            'design_history': DESIGN_NOTE,
            'baseline_note': 'P2, AHIF and SIG are our re-implementations from the published '
                             'descriptions, not the original authors\' code. Every policy sees the '
                             'same runs, the same cached detector verdicts and the same per-source '
                             'counters; SIG additionally sees one header signature per source entry.',
            'provenance': provenance('scripts/mitigation_experiment.py', vars(args), {
                'detection_model': MLConfig.DEFAULT_MODEL_PATH, 'source_scorer': SCORER_PATH,
                'source_scorer_no_history': SCORER_NO_HISTORY_PATH, 'signature_classifier': SIG_MODEL_PATH}),
        },
        'scorer': {'coefficients': scorer.coefficients(), 'info': scorer.info},
        'tuning_grid': grid,
        'settings': {},
    }
    for setting, by_seed in raw.items():
        seeds = sorted(by_seed)
        first = by_seed[seeds[0]]
        entry = {'policies': {}, 'groups': {}, 'curve': []}
        for key, meta in first['policies'].items():
            entry['policies'][key] = {'name': meta['name'], 'detector': meta['detector'],
                                      **summarize([by_seed[s]['policies'][key] for s in seeds])}
        for key, groups in first['groups'].items():
            entry['groups'][key] = {g: summarize([by_seed[s]['groups'][key][g] for s in seeds])
                                    for g in groups}
        entry['profiles'] = {}
        for key, profiles in first['profiles'].items():
            entry['profiles'][key] = {g: summarize([by_seed[s]['profiles'][key][g] for s in seeds
                                                    if g in by_seed[s]['profiles'][key]])
                                      for g in profiles}
        for i, point in enumerate(first['curve']):
            entry['curve'].append({'t_block': point['t_block'], 't_limit': point['t_limit'],
                                   **summarize([{k: v for k, v in by_seed[s]['curve'][i].items()
                                                 if k not in ('t_block', 't_limit')} for s in seeds])})
        if 'example' in by_seed.get(TEST_SEED_BASE, {}):
            entry['example'] = by_seed[TEST_SEED_BASE]['example']
        results['settings'][setting] = entry

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results, indent=1))
    print(f'[done] {RESULTS_PATH} in {time.time() - started:.0f}s')

    for setting, entry in results['settings'].items():
        print(f'\n== {setting} (mean +/- std over {args.seeds} seeds) ==')
        print(f"{'policy':20s} {'attack drop %':>16s} {'legit loss %':>14s} {'legit srcs hit %':>17s} {'mitigated runs %':>17s}")
        for key, r in entry['policies'].items():
            f = lambda m: f"{r[m]['mean']:6.2f} ± {r[m]['std']:4.2f}"  # noqa: E731
            print(f"{key:20s} {f('attack_drop_pct'):>16s} {f('legit_loss_pct'):>14s} "
                  f"{f('legit_sources_hit_pct'):>17s} {f('mitigated_runs_pct'):>17s}")


if __name__ == '__main__':
    main()
