#!/usr/bin/env python3
"""
All three levels trained and tested on real attack traffic (CIC-DDoS2019).

Input: data/external/cic/<day>.npz from scripts/cic_extract.py (what the XDP
program would see at the victim network's ingress) and the CSV label
inventory from scripts/cic_inspect.py. Day 01-12 (the dataset's training day)
is used for every fit and every threshold; day 03-11 (its testing day, with
one attack type, Portmap, that 01-12 does not contain) only for testing.

  level1  per-second detection, three random forests on the 16 window features:
          shipped (synthetic traffic only), CIC-only (01-12) and combined
          (the shipped model's synthetic training runs + 01-12). Each is tested on
          03-11, on real benign MAWI traffic, on MAWI with injected attacks and on
          the synthetic test runs. The combined model is the window model of the
          `--models cic` set (a CIC-only model learns the testbed's packet rates and
          misses smaller attacks elsewhere). Then the detectors run second by second
          over the whole test day, as deployed.
  level3  per-flow classification of 10-second inbound flows (16 counter
          features, src/ml/flow_features.py), trained on 01-12, tested on 03-11,
          compared with two hand rules ("SYN-only TCP flow", "ACK-only TCP flow").
  level2  mitigation on real traffic: every policy replayed on attack episodes and
          quiet stretches of 03-11, for three systems:
            shipped     synthetic-trained detector, scorer and thresholds
            cic_scorer  combined detector + scorer and thresholds trained / chosen on
                        01-12 (analysis: this scorer does not transfer, see below)
            cic_set     what `main.py --models cic` runs: combined detector, shipped
                        scorer and thresholds, P4+L3 with the CIC flow model
          Each also with P4+L3 (P4 verified by the flow model); what P4+L3 does with
          sources whose flows look legitimate is chosen on 01-12. Then P4 with each
          scorer on synthetic traffic, to see whether the CIC scorer transfers.

Limit of this dataset, reported with the results: every attack packet comes
from ONE address (172.16.0.5), and benign traffic is light (tens of packets/s).
Blocking the single heaviest source therefore stops these attacks; the test
measures detection, flow classification and collateral damage on real
traffic, not how well policies separate many attack sources.

Usage (from project/):
    python scripts/cic_experiment.py                    # all stages (~20 min on 16 cores)
    python scripts/cic_experiment.py --stages level3    # one stage (others keep their results)
Writes data/results/cic_results.json and data/models/cic_{window_model,flow_model,source_scorer}.joblib.
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

from src.config import DATA_DIR, MLConfig, MODELS_DIR  # noqa: E402
from src.provenance import provenance  # noqa: E402

RESULTS_PATH = DATA_DIR / 'results' / 'cic_results.json'
WINDOW_MODEL_PATH = MODELS_DIR / 'cic_window_model.joblib'          # combined: synthetic + CIC 01-12
CIC_ONLY_MODEL_PATH = MODELS_DIR / 'cic_only_window_model.joblib'   # analysis only (does not transfer)
FLOW_MODEL_PATH = MODELS_DIR / 'cic_flow_model.joblib'
SCORER_PATH = MODELS_DIR / 'cic_source_scorer.joblib'
STAGES = ('data', 'level1', 'level3', 'level2')
SEED = 11
CV_BLOCK_S = 600            # grouped cross-validation on 10-minute blocks of the training day
FLOWS_PER_TYPE = 40_000     # training flows sampled per attack type (benign: all)
PRE_S, MAX_ATTACK_S, POST_S = 120, 600, 60      # mitigation runs around each attack episode
QUIET_RUN_S = 300
MAWI = dict(trace='200803181400', victims=20, window=90, min_pps=5.0)
WINDOW_GRID = [{'n_estimators': n, 'max_depth': d, 'min_samples_leaf': leaf}
               for n in (100, 200) for d in (8, 12, None) for leaf in (2, 5)]
FLOW_GRID = [{'max_depth': d, 'min_samples_leaf': leaf} for d in (12, 20, None) for leaf in (2, 5)]


def log(msg):
    print(msg, flush=True)


# ---------------------------------------------------------------- shared helpers
def _metrics(y, pred, score=None, weight=None):
    from sklearn.metrics import average_precision_score, roc_auc_score
    y, pred = np.asarray(y).astype(int), np.asarray(pred).astype(int)
    w = np.ones(len(y)) if weight is None else np.asarray(weight, dtype=np.float64)
    tp, fp = float(w[(y == 1) & (pred == 1)].sum()), float(w[(y == 0) & (pred == 1)].sum())
    fn, tn = float(w[(y == 1) & (pred == 0)].sum()), float(w[(y == 0) & (pred == 0)].sum())
    out = {'n': int(len(y)), 'n_attack': int(y.sum()), 'n_benign': int((y == 0).sum()),
           'recall': tp / (tp + fn) if tp + fn else None,
           'false_positive_rate': fp / (fp + tn) if fp + tn else None,
           'precision': tp / (tp + fp) if tp + fp else None,
           'accuracy': (tp + tn) / (tp + tn + fp + fn) if len(y) else None}
    if out['precision'] and out['recall']:
        out['f1'] = 2 * out['precision'] * out['recall'] / (out['precision'] + out['recall'])
    if score is not None and 0 < y.sum() < len(y):
        out['roc_auc'] = float(roc_auc_score(y, score, sample_weight=w))
        out['average_precision'] = float(average_precision_score(y, score, sample_weight=w))
    return out


def canonical(kind: str) -> str:
    """The two days name the same attack differently: DrDoS_LDAP / LDAP, UDP-lag / UDPLag"""
    return kind.replace('DrDoS_', '').replace('UDP-lag', 'UDPLag')


def _per_type(kinds, y, pred):
    out = {}
    for k in sorted(set(kinds[y == 1]) - {''}):
        m = (kinds == k) & (y == 1)
        out[k] = {'n': int(m.sum()), 'recall': float(pred[m].mean())}
    return out


def _choose(cv_results, simplicity):
    """Simplest setting within one standard deviation of the best cross-validated score"""
    best = max(cv_results, key=lambda r: r['cv_mean'])
    ok = [r for r in cv_results if r['cv_mean'] >= best['cv_mean'] - best['cv_std']]
    return min(ok, key=simplicity)


def load_window_classifier(path):
    from src.ml.ml_classifier import DDoSClassifier
    clf = DDoSClassifier()
    if not clf.load(str(path)):
        raise SystemExit(f'model missing: {path}')
    return clf


def _proba(clf, X):
    from src.ml.feature_extractor import FEATURE_NAMES
    X = X[:, [FEATURE_NAMES.index(n) for n in clf.feature_names]]   # extractor order -> model order
    if clf.scaler is not None:
        X = clf.scaler.transform(X)
    return clf.model.predict_proba(X)[:, list(clf.model.classes_).index(1)]


def day_run(day):
    from src.ml.cic_data import build_run, load_day, second_index
    persec, _ = load_day(day)
    t0, t1, bounds = second_index(persec)
    return build_run(persec, t0, t1, f'cic-{day}', bounds=bounds, t0=t0)


def day_windows(day):
    """Level-1 rows of a whole day: features, labels, time, attack type"""
    from src.ml.cic_data import attack_type_of
    from src.ml.traffic_dataset import run_to_windows
    run = day_run(day)
    X, y = run_to_windows(run)
    ts = np.array([s.timestamp for s in run.steps[3:]])      # run_to_windows skips 3 warm-up steps
    return day, X, y, ts, attack_type_of(day, ts)


# ---------------------------------------------------------------- level 1
def detector_factories(include_cic=True, include_cic_only=False):
    from src.core.anomaly_detector import AnomalyDetector, MLEnhancedAnomalyDetector
    shipped = load_window_classifier(MLConfig.DEFAULT_MODEL_PATH)
    cic = load_window_classifier(WINDOW_MODEL_PATH) if include_cic else None
    cic_only = load_window_classifier(CIC_ONLY_MODEL_PATH) if include_cic_only else None

    def hybrid(clf):
        def make():
            d = MLEnhancedAnomalyDetector()
            d.classifier, d.ml_enabled = clf, True
            return d
        return make
    out = {'statistical': AnomalyDetector, 'hybrid': hybrid(shipped)}
    if cic is not None:
        out['hybrid-cic'] = hybrid(cic)
    if cic_only is not None:
        out['hybrid-cic-only'] = hybrid(cic_only)
    return out


def detect_day(args):
    """Worker: one detector over a whole day, second by second, as deployed"""
    import logging
    logging.disable(logging.WARNING)
    day, name = args
    run = day_run(day)
    detector = detector_factories(include_cic=name == 'hybrid-cic',
                                  include_cic_only=name == 'hybrid-cic-only')[name]()
    verdicts = []
    for step in run.steps:
        r = detector.detect_anomaly(step.stats, step.ip_stats(), now=step.timestamp)
        detector.update_baseline(step.stats, now=step.timestamp, learn=not r.is_anomaly)
        verdicts.append(bool(r.is_anomaly))
    return name, np.array(verdicts), np.array([s.is_attack for s in run.steps]), \
        np.array([s.timestamp for s in run.steps])


def mawi_windows(inject: bool):
    """Level-1 rows of real MAWI traffic: benign only, or with the trace experiment's injected attacks"""
    from trace_experiment import ATTACKS, build_run as trace_run, load_victims
    from src.ml.traffic_dataset import run_to_windows
    victims, n_windows = load_victims(MAWI['trace'], MAWI['victims'], MAWI['window'], MAWI['min_pps'])
    rng = np.random.default_rng(SEED + inject)
    X, y = [], []
    for v in victims:
        for w in range(n_windows):
            scenario = str(rng.choice(ATTACKS)) if inject else 'normal'
            a, b = run_to_windows(trace_run(v, w, MAWI['window'], rng, scenario))
            X.append(a)
            y.append(b)
    return np.vstack(X), np.concatenate(y)


def cross_dataset(models):
    """Both window models on every test set we have (each model at its own 0.5 cut)"""
    from src.ml.model_trainer import TEST_SEED_OFFSET
    from src.ml.traffic_dataset import generate_dataset
    sets = {'mawi_benign': mawi_windows(False), 'mawi_injected_attacks': mawi_windows(True)}
    syn = generate_dataset(300, seed=7 + TEST_SEED_OFFSET)          # the shipped model's own test seed
    sets['synthetic_test'] = (syn['X'], syn['y'])
    out = {}
    for name, (X, y) in sets.items():
        out[name] = {'windows': int(len(y)), 'attack_windows': int(y.sum())}
        for m, clf in models.items():
            p = _proba(clf, X)
            out[name][m] = _metrics(y, p >= 0.5, p if 0 < y.sum() < len(y) else None)
    return out


def stage_data():
    """What the extracted captures contain (quoted in docs/REAL_DATA.md and the papers)"""
    from src.ml.cic_data import ATTACKER_IP, attack_episodes, attacker_pps, label_periods, load_day
    out = {}
    for day in ('01-12', '03-11'):
        persec, flows = load_day(day)
        secs, atk, oth = attacker_pps(persec)
        eps = attack_episodes(persec)
        medians = [float(np.median(atk[(secs >= a) & (secs < b)])) for a, b in eps]
        benign_src = persec['src'][persec['src'] != ATTACKER_IP]
        quiet = {}
        for label, lo, hi in label_periods(day):
            if label in ('Portmap', 'UDPLag'):
                # the label periods also cover the first seconds of the next flood: leave the episodes out
                m = (secs >= lo) & (secs <= hi)
                for a, b in eps:
                    m &= ~((secs >= a) & (secs < b))
                quiet[label] = {'attacker_pps_max': float(atk[m].max()), 'attacker_pps_mean': float(atk[m].mean())}
        long_eps = [md for (a, b), md in zip(eps, medians) if b - a >= 60]
        out[day] = {'seconds': int(len(secs)), 'benign_pps_mean': float(oth.mean()), 'benign_pps_max': float(oth.max()),
                    'episode_attacker_pps_median_min_60s': min(long_eps) if long_eps else None,
                    'episode_attacker_pps_median_max_60s': max(long_eps) if long_eps else None,
                    'benign_sources': int(len(np.unique(benign_src))),
                    'attacker_background_pps_median_outside_episodes': float(np.median(atk[atk < 100])),
                    'episodes': len(eps), 'episode_attacker_pps_median_min': min(medians) if medians else None,
                    'episode_attacker_pps_median_max': max(medians) if medians else None,
                    'attack_flow_rows': int((flows['src'] == ATTACKER_IP).sum()),
                    'benign_flow_rows': int((flows['src'] != ATTACKER_IP).sum()),
                    'low_rate_labels': quiet}
        log(f'[data] {day}: {out[day]}')
    return out


def stage_level1(workers):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GroupKFold, cross_val_score
    from src.ml.cic_data import TRAIN_DAY, TEST_DAY, attack_episodes, load_day
    from src.ml.feature_extractor import FEATURE_NAMES, FEATURE_SET_VERSION
    from src.ml.ml_classifier import DDoSClassifier

    with ProcessPoolExecutor(2) as pool:
        days = {d: (X, y, ts, k) for d, X, y, ts, k in pool.map(day_windows, [TRAIN_DAY, TEST_DAY])}
    Xtr, ytr, tstr, _ = days[TRAIN_DAY]
    Xte, yte, tste, kte = days[TEST_DAY]
    log(f'[level1] windows: {TRAIN_DAY} {len(ytr):,} ({ytr.mean():.1%} attack), '
        f'{TEST_DAY} {len(yte):,} ({yte.mean():.1%} attack)')

    groups = ((tstr - tstr.min()) // CV_BLOCK_S).astype(int)
    cv = []
    for params in WINDOW_GRID:
        model = RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced', **params)
        s = cross_val_score(model, Xtr, ytr, groups=groups, cv=GroupKFold(5), scoring='f1')
        cv.append({**params, 'cv_mean': float(s.mean()), 'cv_std': float(s.std())})
    chosen = _choose(cv, lambda r: (r['n_estimators'], r['max_depth'] or 99, -r['min_samples_leaf']))
    log(f'[level1] chosen {chosen}')
    clf = DDoSClassifier(model_type='random_forest', n_estimators=chosen['n_estimators'],
                         max_depth=chosen['max_depth'])
    clf.model.set_params(min_samples_leaf=chosen['min_samples_leaf'], min_samples_split=2)
    clf.scaler = None
    clf.feature_set_version = FEATURE_SET_VERSION
    clf.train(Xtr, ytr, None, None, list(FEATURE_NAMES))

    shipped = load_window_classifier(MLConfig.DEFAULT_MODEL_PATH)
    p_cic = _proba(clf, Xte)
    p_ship_te, p_ship_tr = _proba(shipped, Xte), _proba(shipped, Xtr)
    res = {
        'cic_model': {
            'train_day': TRAIN_DAY, 'test_day': TEST_DAY, 'cv': cv, 'chosen': chosen,
            'test': _metrics(yte, p_cic >= 0.5, p_cic),
            'test_per_attack_type': _per_type(kte, yte, (p_cic >= 0.5).astype(int)),
            'feature_importance': clf.get_feature_importance(len(FEATURE_NAMES)),
        },
        'shipped_model': {
            'note': 'trained on synthetic traffic only (data/models/ddos_classifier.joblib)',
            TRAIN_DAY: _metrics(ytr, p_ship_tr >= 0.5, p_ship_tr),
            TEST_DAY: _metrics(yte, p_ship_te >= 0.5, p_ship_te),
            'test_per_attack_type': _per_type(kte, yte, (p_ship_te >= 0.5).astype(int)),
        },
    }
    clf.evaluation = {'data_source': 'cic-ddos2019-pcap-window', 'train_day': TRAIN_DAY, 'test_day': TEST_DAY,
                      'test_metrics': res['cic_model']['test'], 'chosen_params': chosen,
                      'note': 'trained on CIC-DDoS2019 day 01-12 only; analysis model, does not transfer '
                              'to smaller networks (see level1.cross_dataset)'}
    clf.save(str(CIC_ONLY_MODEL_PATH))

    # Combined: the shipped model's synthetic training runs + CIC 01-12, the shipped model's
    # hyperparameters (no new search), so the only change is the added real traffic
    from src.ml.traffic_dataset import generate_dataset
    ship_rep = json.loads(Path(MLConfig.DEFAULT_REPORT_PATH).read_text())
    hp = ship_rep['chosen_params']
    syn = generate_dataset(ship_rep['train_runs'], seed=ship_rep['seed'])
    comb = DDoSClassifier(model_type='random_forest', n_estimators=hp['n_estimators'], max_depth=hp['max_depth'])
    comb.model.set_params(min_samples_leaf=hp['min_samples_leaf'], min_samples_split=2)
    comb.scaler = None
    comb.feature_set_version = FEATURE_SET_VERSION
    comb.train(np.vstack([syn['X'], Xtr]), np.concatenate([syn['y'], ytr]), None, None, list(FEATURE_NAMES))
    p_comb = _proba(comb, Xte)
    res['combined_model'] = {
        'train': f"{ship_rep['train_runs']} synthetic runs (seed {ship_rep['seed']}, {len(syn['y']):,} windows) "
                 f"+ CIC {TRAIN_DAY} ({len(ytr):,} windows)",
        'params': {k: hp[k] for k in ('n_estimators', 'max_depth', 'min_samples_leaf')},
        'test': _metrics(yte, p_comb >= 0.5, p_comb),
        'test_per_attack_type': _per_type(kte, yte, (p_comb >= 0.5).astype(int)),
        'feature_importance': comb.get_feature_importance(len(FEATURE_NAMES)),
    }
    comb.evaluation = {'data_source': 'synthetic-window + cic-ddos2019-pcap-window', 'train_day': TRAIN_DAY,
                       'test_day': TEST_DAY, 'test_metrics': res['combined_model']['test'],
                       'note': res['combined_model']['train'] + '; tested on CIC 03-11'}
    comb.save(str(WINDOW_MODEL_PATH))

    res['cross_dataset'] = cross_dataset({'cic_model': clf, 'shipped_model': shipped, 'combined_model': comb})
    res['cross_dataset']['note'] = (
        'window model alone at a 0.5 cut (not the hybrid detector). MAWI: real benign traffic to '
        'the busiest destinations of trace ' + MAWI['trace'] + ', with or without the trace '
        'experiment attacks injected; synthetic_test: 300 runs of the shipped model test seed (1007)')
    log(f"[level1] CIC model on {TEST_DAY}: recall {res['cic_model']['test']['recall']:.3f}, "
        f"FPR {res['cic_model']['test']['false_positive_rate']:.3f}; shipped model: recall "
        f"{res['shipped_model'][TEST_DAY]['recall']:.3f}, FPR {res['shipped_model'][TEST_DAY]['false_positive_rate']:.3f}")
    for name, r in res['cross_dataset'].items():
        if isinstance(r, dict):
            log(f"[level1]   {name:22s} " + '  '.join(
                f"{m}: recall {r[m]['recall'] if r[m]['recall'] is not None else float('nan'):.3f} "
                f"FPR {r[m]['false_positive_rate'] if r[m]['false_positive_rate'] is not None else float('nan'):.3f}"
                for m in ('cic_model', 'shipped_model', 'combined_model')))

    # Deployed detectors, second by second over the whole test day
    persec, _ = load_day(TEST_DAY)
    episodes = attack_episodes(persec)
    del persec
    with ProcessPoolExecutor(min(workers, 4)) as pool:
        series = list(pool.map(detect_day, [(TEST_DAY, n) for n in
                                            ('statistical', 'hybrid', 'hybrid-cic', 'hybrid-cic-only')]))
    res['end_to_end'] = {}
    for name, flag, attack, ts in series:
        delays = []
        for a, b in episodes:
            hit = np.flatnonzero(flag & attack & (ts >= a) & (ts < b))
            delays.append(float(ts[hit[0]] - a) if len(hit) else None)
        found = [d for d in delays if d is not None]
        res['end_to_end'][name] = {
            'attack_second_recall': float(flag[attack].mean()),
            'benign_second_fpr': float(flag[~attack].mean()),
            'episodes_detected': f'{len(found)}/{len(episodes)}',
            'median_detection_delay_s': float(np.median(found)) if found else None,
            'max_detection_delay_s': float(np.max(found)) if found else None,
        }
        log(f'[level1] {name:12s} over {TEST_DAY}: {res["end_to_end"][name]}')
    res['test_day_episodes'] = [[a, b] for a, b in episodes]
    return res


# ---------------------------------------------------------------- level 3
def flow_rows(day):
    from src.ml.cic_data import flow_labels, load_day
    from src.ml.flow_features import flow_feature_matrix
    _, flows = load_day(day)
    y, kind = flow_labels(day, flows)
    keep = ~(y & (kind == ''))                   # attacker flows outside every labelled period
    flows = {k: v[keep] for k, v in flows.items()}
    return flows, flow_feature_matrix(flows), y[keep].astype(int), kind[keep]


def stage_level3(attack_flow_rate):
    from sklearn.model_selection import GroupKFold
    from src.ml.cic_data import TRAIN_DAY, TEST_DAY
    from src.ml.flow_features import FLOW_FEATURES, FlowClassifier

    f_tr, Xtr, ytr, ktr = flow_rows(TRAIN_DAY)
    rng = np.random.default_rng(SEED)
    take = [np.flatnonzero(ytr == 0)]
    for k in sorted(set(ktr[ytr == 1])):
        idx = np.flatnonzero((ytr == 1) & (ktr == k))
        take.append(rng.choice(idx, min(len(idx), FLOWS_PER_TYPE), replace=False))
    take = np.sort(np.concatenate(take))
    Xs, ys = Xtr[take], ytr[take]
    groups = ((f_tr['t_first'][take] - f_tr['t_first'].min()) // CV_BLOCK_S).astype(int)
    log(f'[level3] training rows {len(ys):,} ({int((ys == 0).sum()):,} benign flows, '
        f'{int(ys.sum()):,} attack flows over {len(set(ktr[take][ys == 1]))} types)')

    cv = []
    for params in FLOW_GRID:
        scores = []
        for tr, va in GroupKFold(3).split(Xs, ys, groups):
            m = FlowClassifier(random_state=42, **params).fit(Xs[tr], ys[tr])
            p = m.predict_proba(Xs[va])
            scores.append(_metrics(ys[va], p >= 0.5, p).get('f1') or 0.0)
        cv.append({**params, 'cv_mean': float(np.mean(scores)), 'cv_std': float(np.std(scores))})
        log(f'[level3]   CV {params}: F1 {np.mean(scores):.4f} +/- {np.std(scores):.4f}')
    chosen = _choose(cv, lambda r: (r['max_depth'] or 99, -r['min_samples_leaf']))
    model = FlowClassifier(random_state=42, **{k: chosen[k] for k in ('max_depth', 'min_samples_leaf')})
    model.fit(Xs, ys, info={'train_day': TRAIN_DAY, 'flows_per_type': FLOWS_PER_TYPE, 'seed': SEED,
                            'source': 'CIC-DDoS2019 packet captures, inbound 10-s flows'})
    model.save(str(FLOW_MODEL_PATH))

    f_te, Xte, yte, kte = flow_rows(TEST_DAY)
    p = model.predict_proba(Xte)
    pred = (p >= 0.5).astype(int)
    # attack flows were sampled at attack_flow_rate: weight them back to the real mix
    w = np.where(yte == 1, 1.0 / attack_flow_rate, 1.0)
    pk = f_te['packets'].astype(np.float64)
    res = {
        'features': FLOW_FEATURES, 'train_day': TRAIN_DAY, 'test_day': TEST_DAY, 'cv': cv, 'chosen': chosen,
        'train_rows': int(len(ys)), 'attack_flow_sampling_rate': attack_flow_rate,
        'test_flows': _metrics(yte, pred, p),
        'test_flows_reweighted': _metrics(yte, pred, p, w),
        'test_packets': {'attack_packets_in_flagged_flows_pct': float(100 * pk[(yte == 1) & (pred == 1)].sum() / pk[yte == 1].sum()),
                         'benign_packets_in_flagged_flows_pct': float(100 * pk[(yte == 0) & (pred == 1)].sum() / pk[yte == 0].sum())},
        'test_per_attack_type': _per_type(kte, yte, pred),
        # CIC labels every flow of the attack machine as attack, also its background
        # traffic (a few packets/s) between attacks: recall split by that
        'test_recall_by_activity': _recall_by_activity(f_te, yte, pred),
        'unseen_in_training': sorted({canonical(k) for k in kte[yte == 1]} - {canonical(k) for k in ktr[ytr == 1]}),
        'feature_importance': model.feature_importance(),
        'benign_flagged_examples': _benign_examples(f_te, Xte, yte, pred),
        'partial_flows_train_day': partial_flow_check(TRAIN_DAY, {k: chosen[k] for k in ('max_depth', 'min_samples_leaf')}),
    }
    tcp = Xte[:, FLOW_FEATURES.index('protocol')] == 6
    syn_r, ack_r = Xte[:, FLOW_FEATURES.index('syn_ratio')], Xte[:, FLOW_FEATURES.index('ack_ratio')]
    rules = {'syn_only_tcp_flow': tcp & (syn_r > 0) & (ack_r == 0),
             'ack_only_tcp_flow': tcp & (syn_r == 0) & (ack_r == 1)}
    res['rules'] = {}
    for name, flag in rules.items():
        flag = flag.astype(int)
        res['rules'][name] = {'test_flows': _metrics(yte, flag), 'test_per_attack_type': _per_type(kte, yte, flag)}
    log(f"[level3] {TEST_DAY}: recall {res['test_flows']['recall']:.4f}, benign FPR "
        f"{res['test_flows']['false_positive_rate']:.4f}, per type "
        + ', '.join(f"{k} {v['recall']:.3f}" for k, v in res['test_per_attack_type'].items()))
    for name, r in res['rules'].items():
        log(f"[level3] rule {name}: recall {r['test_flows']['recall']:.4f}, "
            f"benign FPR {r['test_flows']['false_positive_rate']:.4f}")
    return res


def partial_flow_check(day, params):
    """How the flow model judges flows as they stood 1, 2, 3, 5 and 10 s into their window, which is
    what the block-time check sees. Training day only, cross-validated on 3 groups of 10-minute
    blocks: a flow is judged by a model trained on other blocks. Rule fixed in advance: the
    model stays as it is if, up to 3 s, recall >= 95% and benign flows flagged <= 1%."""
    from cic_extract import merge_flows
    from src.ml.cic_data import ATTACKER_IP, load_flow_seconds
    from src.ml.flow_features import FlowClassifier, flow_feature_matrix
    f_tr, X, y, kinds = flow_rows(day)
    fs = load_flow_seconds(day)
    t0 = float(f_tr['t_first'].min())
    fold_of = lambda t: ((np.asarray(t, dtype=np.float64) - t0) // CV_BLOCK_S).astype(int) % 3  # noqa: E731
    rng = np.random.default_rng(SEED)
    fold_tr, fold_fs = fold_of(f_tr['t_first']), fold_of(fs['win'] * 10.0)
    offset = fs['slot'] - fs['win'] * 10
    out = {}
    for sec in (1, 2, 3, 5, 10):
        tp = fn = fp = tn = 0
        for fold in range(3):
            pick = [np.flatnonzero((y == 0) & (fold_tr != fold))]
            for k in sorted(set(kinds[y == 1])):
                idx = np.flatnonzero((y == 1) & (kinds == k) & (fold_tr != fold))
                pick.append(rng.choice(idx, min(len(idx), FLOWS_PER_TYPE), replace=False))
            pick = np.concatenate(pick)
            model = FlowClassifier(random_state=42, **params).fit(X[pick], y[pick])
            m = (fold_fs == fold) & (offset < sec)
            sub = merge_flows({k: v[m] for k, v in fs.items()}, by='win')
            p = model.predict_proba(flow_feature_matrix(sub)) >= 0.5
            atk = sub['src'] == ATTACKER_IP
            tp, fn = tp + int(p[atk].sum()), fn + int((~p[atk]).sum())
            fp, tn = fp + int(p[~atk].sum()), tn + int((~p[~atk]).sum())
        out[f'{sec}s'] = {'recall': tp / max(1, tp + fn), 'false_positive_rate': fp / max(1, fp + tn),
                          'attack_flows': tp + fn, 'benign_flows': fp + tn}
        log(f"[level3] first {sec:2d} s of a window: attack flows found {100 * out[f'{sec}s']['recall']:.2f}%, "
            f"benign flows flagged {100 * out[f'{sec}s']['false_positive_rate']:.2f}%")
    return out


def _recall_by_activity(f, y, pred):
    from src.ml.cic_data import TEST_DAY, attack_episodes, load_day
    persec, _ = load_day(TEST_DAY)
    t = f['t_first']
    inside = np.zeros(len(t), dtype=bool)
    for a, b in attack_episodes(persec):
        inside |= (t >= a) & (t < b)
    out = {}
    for name, m in (('inside_attack_episodes', inside), ('attacker_background_outside_episodes', ~inside)):
        sel = (y == 1) & m
        out[name] = {'flows': int(sel.sum()), 'recall': float(pred[sel].mean()) if sel.any() else None}
    return out


def _benign_examples(f, X, y, pred, n=5):
    """A few benign flows the model flags (for the failure-case discussion)"""
    from src.ml.cic_data import ip_str
    from src.ml.flow_features import FLOW_FEATURES
    idx = np.flatnonzero((y == 0) & (pred == 1))
    if not len(idx):
        return []
    top = idx[np.argsort(-f['packets'][idx])[:n]]
    return [{'src': ip_str(f['src'][i]), 'sport': int(f['sport'][i]), 'dport': int(f['dport'][i]),
             **{k: round(float(X[i, j]), 4) for j, k in enumerate(FLOW_FEATURES)}} for i in top]


# ---------------------------------------------------------------- level 2
def mitigation_runs(day):
    """(start, end, scenario) of the runs of one day: one per attack episode, plus quiet stretches"""
    from src.ml.cic_data import attack_episodes, attack_type_of, load_day, second_index
    persec, _ = load_day(day)
    t0, t1, _ = second_index(persec)
    merged = []
    for a, b in attack_episodes(persec):
        if merged and a - merged[-1][1] < PRE_S + POST_S:      # no clean pre-attack period in between
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))
    runs, prev_end = [], t0
    for a, b in merged:
        start = max(prev_end, a - PRE_S)
        end = min(t1, min(b, a + MAX_ATTACK_S) + POST_S)
        kind = str(attack_type_of(day, np.array([a + min(30, (b - a) // 2)]))[0]) or 'unlabelled'
        while prev_end + QUIET_RUN_S <= start:              # quiet stretch before this episode
            runs.append((prev_end, prev_end + QUIET_RUN_S, 'no attack'))
            prev_end += QUIET_RUN_S
        runs.append((start, end, kind))
        prev_end = b + POST_S
    while prev_end + QUIET_RUN_S <= t1:
        runs.append((prev_end, prev_end + QUIET_RUN_S, 'no attack'))
        prev_end += QUIET_RUN_S
    return runs


def prepare_cic_run(args):
    """Worker: build one real-traffic run, cache detector verdicts, attach flow evidence"""
    import logging
    logging.disable(logging.WARNING)
    day, start, end, scenario = args
    from src.ml.cic_data import build_run, load_day, second_index, source_signatures
    from src.mitigation.evaluation import prepare_run
    persec, flows = _cached_day(day)
    t0, _, bounds = second_index(persec)
    run = build_run(persec, start, end, scenario, sigs=source_signatures(persec), bounds=bounds, t0=t0)
    prepared = prepare_run(run, detector_factories())
    prepared.src_ip = np.unique(np.concatenate([s.ip_ids for s in run.steps])) & 0xFFFFFFFF
    keys, share = _flow_evidence(day)
    prepared.flow_share = []
    for step in run.steps:
        k = ((int(step.timestamp) // 10 - 1) << 32) | step.ip_ids
        i = np.minimum(np.searchsorted(keys, k), len(keys) - 1)
        prepared.flow_share.append(np.where(keys[i] == k, share[i], np.nan))
    prepared.flow_now = [_flow_share_now(day, step) if any(v[t] for v in prepared.anomaly.values()) else None
                         for t, step in enumerate(run.steps)]
    return prepared


_DAY_CACHE = {}


def _cached_day(day):
    if day not in _DAY_CACHE:
        from src.ml.cic_data import load_day
        _DAY_CACHE.clear()
        _DAY_CACHE[day] = load_day(day)
    return _DAY_CACHE[day]


_EVIDENCE = {}


def _flow_evidence(day):
    """Per (10-s window, source): share of its packets in flows the flow model calls attack"""
    if day not in _EVIDENCE:
        from src.ml.flow_features import FlowClassifier, flow_feature_matrix
        _, flows = _cached_day(day)
        p = FlowClassifier.load(str(FLOW_MODEL_PATH)).predict_proba(flow_feature_matrix(flows))
        key = (flows['win'].astype(np.int64) << 32) | flows['src'].astype(np.int64)
        uk, inv = np.unique(key, return_inverse=True)
        pk = flows['packets'].astype(np.float64)
        _EVIDENCE.clear()
        _EVIDENCE[day] = (uk, np.bincount(inv, weights=pk * (p >= 0.5)) / np.bincount(inv, weights=pk))
    return _EVIDENCE[day]


_FLOW_SECONDS = {}
USE_REPLAY_WINDOWS = False      # scripts/cic_xdp_analyze.py: the per-second table of the replay windows


def _flow_share_now(day, step):
    """Per entry of a step: share of the source's packets in attack-like flows, from its flows so
    far in the open 10-s window (what reading the live flow map at that moment gives). Rebuilt from
    the per-second flow table up to and including this second; NaN for a source without flows."""
    from cic_extract import merge_flows
    from src.ml.cic_data import load_flow_seconds
    from src.ml.flow_features import FlowClassifier, flow_feature_matrix
    if day not in _FLOW_SECONDS:
        fs = load_flow_seconds(day, replay_windows=USE_REPLAY_WINDOWS)
        _FLOW_SECONDS.clear()
        _FLOW_SECONDS[day] = (fs, FlowClassifier.load(str(FLOW_MODEL_PATH)))
    fs, clf = _FLOW_SECONDS[day]
    sec = int(step.timestamp)
    lo_sec = sec - sec % 10
    out = np.full(len(step.ip_ids), np.nan)
    rows = []
    for k, ip in enumerate(step.ip_ids):
        a, b = np.searchsorted(fs['src'], [ip, ip + 1])
        if b > a:
            i, j = a + np.searchsorted(fs['slot'][a:b], [lo_sec, sec + 1])
            if j > i:
                rows.append(np.arange(i, j))
    if not rows:
        return out
    pick = np.concatenate(rows)
    sub = merge_flows({k: v[pick] for k, v in fs.items()}, by='win')
    p = clf.predict_proba(flow_feature_matrix(sub))
    pk = np.asarray(sub['packets'], dtype=np.float64)
    uniq, inv = np.unique(sub['src'].astype(np.int64), return_inverse=True)
    share = np.bincount(inv, weights=pk * (p >= 0.5)) / np.bincount(inv, weights=pk)
    pos = np.minimum(np.searchsorted(uniq, step.ip_ids), len(uniq) - 1)
    return np.where(uniq[pos] == step.ip_ids, share[pos], np.nan)


def _with_detector(runs, detector):
    """Copies of prepared runs whose 'hybrid' verdicts are those of another detector"""
    import copy
    out = []
    for r in runs:
        c = copy.copy(r)
        c.anomaly = dict(r.anomaly, hybrid=r.anomaly[detector])
        out.append(c)
    return out


def transfer_job(args):
    """Worker: synthetic runs (realistic test seed or a safety scenario) with cached verdicts"""
    import logging
    logging.disable(logging.WARNING)
    kind, seed, n = args
    from mitigation_experiment import SETTINGS, _detectors, prepare
    from src.ml.traffic_dataset import generate_scenario_runs
    from src.mitigation.evaluation import prepare_run
    if kind == 'synthetic realistic':
        return kind, prepare('realistic', seed, n)
    detectors = _detectors()
    runs = []
    for run in generate_scenario_runs(kind, n, seed=seed, **SETTINGS['realistic']):
        for step in run.steps:
            keep = step.ip_packets > 0
            step.ip_packets, step.ip_syn = step.ip_packets[keep], step.ip_syn[keep]
            step.ip_ids, step.ip_is_attack = step.ip_ids[keep], step.ip_is_attack[keep]
            step.ip_sig = step.ip_sig[keep]
        runs.append(prepare_run(run, detectors))
    return kind, runs


def transfer_check(workers, shipped, cic):
    """P4 with the shipped scorer vs with the CIC-trained scorer on synthetic traffic (both with the
    shipped detector; no flows there, so no P4+L3). Evaluation only: nothing is fitted or chosen here.
    Seeds: realistic test seed 2001 and the first seed of each safety scenario (as in safety_experiment)."""
    from src.mitigation import SourceScoringPolicy
    from src.mitigation.evaluation import evaluate_policy
    jobs = [('synthetic realistic', 2001, 100), ('heavy_legit', 5001 * 100 + 1, 40),
            ('udp_flood_with_crowd', 5001 * 100 + 2, 40), ('udp_flood_spoofed_with_crowd', 5001 * 100 + 3, 40)]
    with ProcessPoolExecutor(min(workers, len(jobs))) as pool:
        sets = dict(pool.map(transfer_job, jobs))
    out = {'note': transfer_check.__doc__.strip(), 'runs': {k: n for k, _, n in jobs}}
    for name, runs in sets.items():
        out[name] = {}
        for label, (scorer, params) in (('P4 shipped scorer', shipped), ('P4 CIC scorer', cic)):
            r = evaluate_policy(SourceScoringPolicy(scorer, **params), runs, 'hybrid')
            out[name][label] = {k: r[k] for k in ('attack_drop_pct', 'legit_loss_pct', 'legit_sources_hit_pct',
                                                  'n_runs', 'n_attack_runs')}
        log(f'[level2] transfer {name:30s} ' + '  '.join(
            f"{k}: atk {v['attack_drop_pct']:5.1f}% legit {v['legit_loss_pct']:5.1f}%" for k, v in out[name].items()))
    return out


def stage_level2(workers):
    from mitigation_experiment import (LEGIT_WEIGHT, MAIN_POLICIES, RESULTS_PATH as SYN_RESULTS,
                                       T_BLOCK_GRID, T_LIMIT_GRID, build_policies, load_scorers)
    from src.ml.cic_data import TRAIN_DAY, TEST_DAY
    from src.mitigation import FlowVerifiedScoring, SourceScorer, SourceScoringPolicy
    from src.mitigation.evaluation import collect_source_samples, evaluate_policy

    jobs = {d: [(d, a, b, s) for a, b, s in mitigation_runs(d)] for d in (TRAIN_DAY, TEST_DAY)}
    for d, j in jobs.items():
        log(f'[level2] {d}: {len(j)} runs: ' + ', '.join(f'{s} ({b - a}s)' for _, a, b, s in j))
    prepared = {}
    for d in (TRAIN_DAY, TEST_DAY):          # one day at a time: each worker keeps one day in memory
        with ProcessPoolExecutor(min(workers, 6)) as pool:
            prepared[d] = list(pool.map(prepare_cic_run, jobs[d]))

    # CIC system: scorer trained and thresholds chosen on the training day (CIC detector verdicts)
    train = _with_detector(prepared[TRAIN_DAY], 'hybrid-cic')
    X, y = collect_source_samples(train, 'hybrid')
    cic_scorer = SourceScorer().fit(X, y, info={'train_day': TRAIN_DAY, 'source': 'CIC-DDoS2019 captures'})
    cic_scorer.save(str(SCORER_PATH))
    grid = []
    for tb in T_BLOCK_GRID:
        for tl in (t for t in T_LIMIT_GRID if t <= tb):
            r = evaluate_policy(SourceScoringPolicy(cic_scorer, t_block=tb, t_limit=tl), train, 'hybrid')
            grid.append({'t_block': tb, 't_limit': tl, 'attack_drop_pct': r['attack_drop_pct'],
                         'legit_loss_pct': r['legit_loss_pct'],
                         'objective': r['attack_drop_pct'] - LEGIT_WEIGHT * r['legit_loss_pct']})
    best = max(grid, key=lambda g: g['objective'])
    cic_params = {'t_block': best['t_block'], 't_limit': best['t_limit']}
    log(f'[level2] CIC scorer: {len(y):,} rows ({y.mean():.1%} attack); thresholds {cic_params}; '
        + ', '.join(f'{k}={v:+.2f}' for k, v in cic_scorer.coefficients().items()))

    syn_params = json.loads(Path(SYN_RESULTS).read_text())['protocol']['chosen_params']
    scorer, sig_model, _ = load_scorers()
    test = prepared[TEST_DAY]
    systems = {}
    syn_thresholds = {k: syn_params[k] for k in ('t_block', 't_limit')}
    for system, (det, sc, params) in {'shipped': ('hybrid', scorer, syn_thresholds),
                                      'cic_scorer': ('hybrid-cic', cic_scorer, cic_params),
                                      'cic_set': ('hybrid-cic', scorer, syn_thresholds)}.items():
        runs = _with_detector(test, det)
        pol = {k: v for k, v in build_policies(sc, params, sig_model).items() if k in MAIN_POLICIES}
        train_runs = _with_detector(prepared[TRAIN_DAY], det)
        choice = {}
        for mode in ('limit', 'allow'):
            r = evaluate_policy(FlowVerifiedScoring(sc, on_legit_flows=mode, **params), train_runs, 'hybrid')
            choice[mode] = {'attack_drop_pct': r['attack_drop_pct'], 'legit_loss_pct': r['legit_loss_pct'],
                            'objective': r['attack_drop_pct'] - LEGIT_WEIGHT * r['legit_loss_pct']}
        mode = max(choice, key=lambda m: choice[m]['objective'])
        pol['P4+L3'] = ('hybrid', FlowVerifiedScoring(sc, on_legit_flows=mode, **params))
        pol['P4+L3-window'] = ('hybrid', FlowVerifiedScoring(sc, on_legit_flows=mode, block_time_check=False, **params))
        out = {}
        for key, (_, policy) in pol.items():
            out[key] = {'name': policy.name, 'all': evaluate_policy(policy, runs, 'hybrid'),
                        'per_scenario': {s: evaluate_policy(policy, [r for r in runs if r.scenario == s], 'hybrid')
                                         for s in sorted({r.scenario for r in runs})}}
        systems[system] = {'detector': det, 'thresholds': params, 'policies': out,
                           'p4_l3_on_legit_flows': mode, 'p4_l3_choice_on_train_day': choice}
        log(f'[level2] system "{system}" ({det}) on {TEST_DAY}:')
        for key, r in out.items():
            a = r['all']
            log(f"   {key:11s} attack drop {a['attack_drop_pct']:6.2f}%  legit loss {a['legit_loss_pct']:6.2f}%  "
                f"legit srcs hit {a['legit_sources_hit_pct']:5.1f}%")
    transfer = transfer_check(workers, (scorer, syn_thresholds), (cic_scorer, cic_params))
    return {
        'transfer_to_synthetic': transfer,
        'runs': {d: [{'start': a, 'end': b, 'scenario': s} for _, a, b, s in j] for d, j in jobs.items()},
        'test_attack_packets': float(sum(sum(p[a].sum() for p, a in zip(r.pkts, r.is_attack_src)) for r in test)),
        'test_legit_packets': float(sum(sum(p[~a].sum() for p, a in zip(r.pkts, r.is_attack_src)) for r in test)),
        'test_attack_sources': int(len({int(ip) for r in test for i, a in zip(r.idx, r.is_attack_src)
                                        for ip in r.src_ip[i[a]]})),
        'cic_scorer': {'rows': int(len(y)), 'attack_share': float(y.mean()),
                       'coefficients': cic_scorer.coefficients(), 'threshold_grid': grid},
        'systems': systems,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--stages', nargs='+', default=list(STAGES), choices=STAGES)
    ap.add_argument('--workers', type=int, default=8)
    args = ap.parse_args()
    from src.ml.cic_data import CIC_DIR, INVENTORY_PATH
    meta = json.loads((CIC_DIR / 'extract_meta.json').read_text())
    rate = float(meta['provenance']['parameters']['attack_flow_rate'])

    results = {}
    started = time.time()
    order = [s for s in STAGES if s in args.stages]   # level2 needs the level-1 and level-3 models
    for stage in order:
        t = time.time()
        results[stage] = {'data': stage_data,
                          'level1': lambda: stage_level1(args.workers),
                          'level3': lambda: stage_level3(rate),
                          'level2': lambda: stage_level2(args.workers)}[stage]()
        results[stage]['seconds'] = round(time.time() - t, 1)
        results[stage]['provenance'] = provenance('scripts/cic_experiment.py', {
            'stage': stage, 'seed': SEED, 'cv_block_s': CV_BLOCK_S, 'flows_per_type': FLOWS_PER_TYPE,
            'runs': dict(pre_s=PRE_S, max_attack_s=MAX_ATTACK_S, post_s=POST_S, quiet_run_s=QUIET_RUN_S),
            'mawi': MAWI, 'extract': meta['provenance']['parameters']}, {
            **{f'cic_{d}': CIC_DIR / f'{d}.npz' for d in ('01-12', '03-11')},
            'cic_inventory': INVENTORY_PATH,
            **({'shipped_model': MLConfig.DEFAULT_MODEL_PATH} if stage != 'level3' else {}),
            **({'cic_window_model': WINDOW_MODEL_PATH} if stage != 'level3' else {}),
            **({'cic_only_window_model': CIC_ONLY_MODEL_PATH} if stage == 'level1' else {}),
            **({'cic_flow_model': FLOW_MODEL_PATH} if stage != 'level1' else {})})
        results['protocol'] = {
            'dataset': 'CIC-DDoS2019 (Sharafaldin et al., ICCST 2019), packet captures',
            'train_day': '01-12', 'test_day': '03-11',
            'ingress': 'IPv4 packets entering 192.168.50.0/24 from outside it',
            'labels': 'attack = source 172.16.0.5, the only source of every attack-labelled CSV flow; '
                      'attack type = CSV label whose time range contains the packet; '
                      'attack second = attack machine sent >= 100 packets that second',
            'limitation': 'one attack source address and light benign traffic: tests detection, '
                          'flow classification and collateral damage on real traffic, not separation '
                          'of many attack sources',
            'extract': meta['days'],
        }
        # merge with the file as it is now (another stage may have finished meanwhile)
        current = json.loads(RESULTS_PATH.read_text()) if RESULTS_PATH.exists() else {}
        current.update({stage: results[stage], 'protocol': results['protocol']})
        RESULTS_PATH.write_text(json.dumps(current, indent=1, default=str))
        log(f'[{stage}] done in {time.time() - t:.0f}s -> {RESULTS_PATH}')
    log(f'[done] {time.time() - started:.0f}s')
    return 0


if __name__ == '__main__':
    sys.exit(main())
