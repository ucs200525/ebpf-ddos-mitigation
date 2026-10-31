"""
Model Trainer - Training script for DDoS classifier
Provides command-line interface for training and evaluation

Data sources:
  window          (default) Labeled synthetic traffic runs pushed through the
                  runtime FeatureExtractor. Produces the model main.py uses.
  cic             CIC-DDoS-2019 per-flow CSVs. Per-flow features cannot be
                  computed from eBPF counters at runtime, so these models are
                  for offline experiments only; main.py will refuse to load them.
  synthetic-flow  The original synthetic per-flow table (same limitation as cic).
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from src.config import MLConfig

DEFAULT_MODEL_PATH = MLConfig.DEFAULT_MODEL_PATH
TEST_SEED_OFFSET = 1000  # test runs come from an independent random stream


def _classification_metrics(y_true, y_pred, y_score=None) -> dict:
    from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                 f1_score, roc_auc_score, confusion_matrix)
    out = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'f1': float(f1_score(y_true, y_pred, zero_division=0)),
        'confusion_matrix': confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }
    if y_score is not None and len(set(y_true)) > 1:
        out['roc_auc'] = float(roc_auc_score(y_true, y_score))
    return out


def _per_scenario(scenarios, y_true, y_pred) -> dict:
    """Recall on attack windows / false-positive rate on benign windows, per scenario"""
    result = {}
    for sc in sorted(set(scenarios)):
        m = scenarios == sc
        attack = m & (y_true == 1)
        benign = m & (y_true == 0)
        result[sc] = {
            'windows': int(m.sum()),
            'attack_recall': float(y_pred[attack].mean()) if attack.any() else None,
            'benign_fpr': float(y_pred[benign].mean()) if benign.any() else None,
        }
    return result


def train_window_model(args):
    """Train on window-level features produced by the runtime FeatureExtractor"""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GroupKFold, cross_val_score
    from .traffic_dataset import generate_dataset
    from .feature_extractor import FEATURE_SET_VERSION
    from .ml_classifier import DDoSClassifier

    logger.info("=" * 70)
    logger.info("DDoS Classifier Training - window features (synthetic traffic runs)")
    logger.info("=" * 70)

    train = generate_dataset(args.train_runs, seed=args.seed)
    test = generate_dataset(args.test_runs, seed=args.seed + TEST_SEED_OFFSET)
    feature_names = train['feature_names']

    # Hyperparameter search with folds grouped by run: windows from one run
    # are strongly correlated, so a random split would leak and inflate scores
    grid = [
        {'n_estimators': n, 'max_depth': d, 'min_samples_leaf': leaf}
        for n in (100, 200) for d in (8, 12, None) for leaf in (2, 5)
    ]
    cv = GroupKFold(n_splits=5)
    cv_results = []
    for params in grid:
        model = RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced', **params)
        scores = cross_val_score(model, train['X'], train['y'], groups=train['groups'],
                                 cv=cv, scoring='f1')
        cv_results.append({**params, 'cv_f1_mean': float(scores.mean()), 'cv_f1_std': float(scores.std())})
        logger.info(f"  CV {params}: F1 {scores.mean():.4f} +/- {scores.std():.4f}")

    # Prefer the simplest model within one std of the best (smaller, faster)
    best = max(cv_results, key=lambda r: r['cv_f1_mean'])
    tolerance = best['cv_f1_mean'] - best['cv_f1_std']
    candidates = [r for r in cv_results if r['cv_f1_mean'] >= tolerance]
    chosen = min(candidates, key=lambda r: (r['n_estimators'], r['max_depth'] or 99, -r['min_samples_leaf']))
    logger.info(f"Best CV F1 {best['cv_f1_mean']:.4f}; chosen {chosen}")

    classifier = DDoSClassifier(
        model_type='random_forest',
        n_estimators=chosen['n_estimators'],
        max_depth=chosen['max_depth'],
    )
    classifier.model.set_params(min_samples_leaf=chosen['min_samples_leaf'], min_samples_split=2)
    classifier.scaler = None  # trees don't need scaling; keeps inference exact
    classifier.feature_set_version = FEATURE_SET_VERSION

    # Metrics stored with the model are computed on the independent test set
    classifier.train(train['X'], train['y'], test['X'], test['y'], feature_names)

    proba = classifier.model.predict_proba(test['X'])[:, 1]
    y_pred = (proba >= 0.5).astype(int)
    test_metrics = _classification_metrics(test['y'], y_pred, proba)
    per_scenario = _per_scenario(test['scenarios'], test['y'], y_pred)

    classifier.evaluation = {
        'data_source': 'synthetic-window',
        'train_runs': args.train_runs,
        'test_runs': args.test_runs,
        'seed': args.seed,
        'train_windows': int(len(train['y'])),
        'test_windows': int(len(test['y'])),
        'test_metrics': test_metrics,
        'per_scenario': per_scenario,
        'cv_results': cv_results,
        'chosen_params': chosen,
        'test_seed': args.seed + TEST_SEED_OFFSET,
        'note': 'Held-out windows come from independent synthetic runs; this is not an '
                'evaluation on real attack traffic.',
    }
    from src.provenance import provenance
    classifier.evaluation['provenance'] = provenance('python -m src.ml.model_trainer train', {
        'source': 'window', 'train_runs': args.train_runs, 'test_runs': args.test_runs, 'seed': args.seed})

    logger.info("=" * 50)
    logger.info("Held-out test set (independent runs)")
    logger.info("=" * 50)
    for key in ('accuracy', 'precision', 'recall', 'f1', 'roc_auc'):
        if key in test_metrics:
            logger.info(f"  {key:10s}: {test_metrics[key]:.4f}")
    logger.info(f"  confusion [[TN, FP], [FN, TP]]: {test_metrics['confusion_matrix']}")
    for sc, r in per_scenario.items():
        parts = []
        if r['attack_recall'] is not None:
            parts.append(f"attack recall {r['attack_recall']:.3f}")
        if r['benign_fpr'] is not None:
            parts.append(f"benign FPR {r['benign_fpr']:.3f}")
        logger.info(f"  {sc:18s} {', '.join(parts)}")

    logger.info("Feature importance:")
    for name, importance in classifier.get_feature_importance(len(feature_names)).items():
        logger.info(f"  {name:22s} {importance:.4f}")

    return classifier


def train_flow_model(args):
    """Train on CIC-style per-flow features (offline experiments only)"""
    from .data_loader import CICDataLoader, create_synthetic_dataset
    from .ml_classifier import DDoSClassifier
    from sklearn.model_selection import train_test_split

    loader = CICDataLoader(str(args.data_path))
    if args.source == 'cic':
        if not loader.find_csv_files():
            logger.error(f"No CSV files found in {args.data_path}; "
                         "download CIC-DDoS-2019 or use --source window")
            return None
        data = loader.prepare_data(
            max_files=args.max_files,
            samples_per_file=args.samples_per_file,
            binary=not args.multiclass,
            scale=True,
        )
    else:
        df = create_synthetic_dataset(args.samples_per_file * 2)
        X, y_str, feature_names = loader.preprocess(df)
        y = loader.encode_labels(y_str, binary=True)
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
        )
        X_train, _ = loader.scale_features(X_train, fit=True)
        X_val, _ = loader.scale_features(X_val, fit=False)
        X_test, _ = loader.scale_features(X_test, fit=False)
        data = {
            'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
            'y_train': y_train, 'y_val': y_val, 'y_test': y_test,
            'feature_names': feature_names, 'scaler': loader.scaler,
        }

    if not data:
        logger.error("Failed to prepare training data")
        return None

    classifier = DDoSClassifier(
        model_type=args.model_type,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
    )
    classifier.scaler = data['scaler']
    classifier.train(data['X_train'], data['y_train'], data['X_val'], data['y_val'],
                     data['feature_names'])

    from sklearn.metrics import classification_report
    y_pred = classifier.model.predict(data['X_test'])
    test_metrics = _classification_metrics(data['y_test'], y_pred)
    classifier.evaluation = {'data_source': args.source, 'test_metrics': test_metrics}
    logger.info(f"Test metrics: {test_metrics}")
    logger.info(f"\nClassification Report:\n{classification_report(data['y_test'], y_pred)}")
    logger.warning("Per-flow models are NOT usable by main.py (runtime features are window-level)")
    return classifier


def train_model(args):
    """Train the DDoS classifier and save it"""
    source = getattr(args, 'source', 'window')
    if source == 'window':
        classifier = train_window_model(args)
    else:
        classifier = train_flow_model(args)
    if not classifier:
        return None

    model_path = Path(args.model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    if classifier.save(str(model_path)):
        logger.info(f"Model saved to {model_path}")
    if getattr(args, 'report', None) and classifier.evaluation:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(classifier.evaluation, indent=2))
        logger.info(f"Evaluation report written to {args.report}")
    return classifier


def _detector_on_runs(make_detector, runs) -> dict:
    """
    Replay runs second-by-second through a detector, in the same order
    main.py uses (detect, then learn baseline only on normal samples).
    """
    second_tp = second_fn = second_fp = second_tn = 0
    attack_runs = detected_runs = benign_runs = false_alarm_runs = 0
    delays = []
    for run in runs:
        detector = make_detector()
        is_attack_run = any(s.is_attack for s in run.steps)
        first_detection = None
        onset = next((i for i, s in enumerate(run.steps) if s.is_attack), None)
        flagged_any_benign = False
        for i, step in enumerate(run.steps):
            result = detector.detect_anomaly(step.stats, step.ip_stats(), now=step.timestamp)
            detector.update_baseline(step.stats, now=step.timestamp, learn=not result.is_anomaly)
            flagged = result.is_anomaly
            if step.is_attack:
                second_tp += flagged
                second_fn += not flagged
                if flagged and first_detection is None:
                    first_detection = i
            else:
                second_fp += flagged
                second_tn += not flagged
                flagged_any_benign |= flagged
        if is_attack_run:
            attack_runs += 1
            if first_detection is not None:
                detected_runs += 1
                delays.append(first_detection - onset)
        else:
            benign_runs += 1
            false_alarm_runs += flagged_any_benign
    return {
        'attack_second_recall': second_tp / max(1, second_tp + second_fn),
        'benign_second_fpr': second_fp / max(1, second_fp + second_tn),
        'attack_runs_detected': f"{detected_runs}/{attack_runs}",
        'benign_runs_with_false_alarm': f"{false_alarm_runs}/{benign_runs}",
        'median_detection_delay_s': float(np.median(delays)) if delays else None,
    }


def evaluate_model(args):
    """Evaluate a trained model on fresh synthetic runs, vs the statistical detector"""
    from .ml_classifier import DDoSClassifier
    from .feature_extractor import FeatureExtractor
    from .traffic_dataset import generate_dataset, generate_runs
    from src.core.anomaly_detector import AnomalyDetector, MLEnhancedAnomalyDetector

    classifier = DDoSClassifier()
    if not classifier.load(args.model_path):
        logger.error("Failed to load model")
        return None

    from src.provenance import provenance
    report = {'model_path': args.model_path, 'training_date': classifier.training_date,
              'stored_evaluation_source': (classifier.evaluation or {}).get('data_source'),
              # This is a second, separate evaluation: its windows come from fresh runs
              # (seed `seed`), not from the held-out test set stored with the model
              'data': 'synthetic (src/ml/traffic_dataset.py)',
              'window_eval': {'runs': args.runs, 'seed': args.seed},
              'end_to_end': {'runs': args.e2e_runs, 'seed': args.seed + 1},
              'provenance': provenance('python -m src.ml.model_trainer evaluate', vars(args),
                                       {'model': args.model_path})}
    extractor = FeatureExtractor()

    if not extractor.supports(classifier.feature_names):
        logger.error(f"Model features are not produced by the runtime extractor "
                     f"({len(classifier.feature_names)} features, e.g. {classifier.feature_names[:3]}). "
                     "This model cannot run in main.py.")
        report['window_metrics'] = 'incompatible: model features unavailable at runtime'
    else:
        data = generate_dataset(args.runs, seed=args.seed)
        cols = [data['feature_names'].index(n) for n in classifier.feature_names]
        X = data['X'][:, cols]
        if classifier.scaler is not None:
            X = classifier.scaler.transform(X)
        proba = classifier.model.predict_proba(X)[:, list(classifier.model.classes_).index(1)]
        y_pred = (proba >= 0.5).astype(int)
        report['window_metrics'] = _classification_metrics(data['y'], y_pred, proba)
        report['window_metrics']['windows'] = int(len(data['y']))
        report['per_scenario'] = _per_scenario(data['scenarios'], data['y'], y_pred)
        logger.info(f"Window-level metrics: {report['window_metrics']}")

    # End-to-end, per-second detection on raw runs
    e2e_runs = generate_runs(args.e2e_runs, seed=args.seed + 1)
    logger.info(f"Replaying {len(e2e_runs)} runs through the detectors...")
    report['statistical_detector'] = _detector_on_runs(AnomalyDetector, e2e_runs)
    logger.info(f"Statistical only: {report['statistical_detector']}")
    if extractor.supports(classifier.feature_names):
        def make_hybrid():
            detector = MLEnhancedAnomalyDetector()
            detector.classifier = classifier
            detector.ml_enabled = True
            return detector
        report['hybrid_detector'] = _detector_on_runs(make_hybrid, e2e_runs)
        logger.info(f"Hybrid (stat + ML): {report['hybrid_detector']}")

    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(report, indent=2))
        logger.info(f"Report written to {args.report}")
    return report


def benchmark_inference(args):
    """Benchmark inference latency"""
    from .ml_classifier import DDoSClassifier

    classifier = DDoSClassifier()
    if not classifier.load(args.model_path):
        logger.error("Failed to load model")
        return

    n_features = len(classifier.feature_names) if classifier.feature_names else 64

    logger.info("=" * 50)
    logger.info("Inference Latency Benchmark")
    logger.info("=" * 50)

    for batch_size in [1, 10, 100, 1000]:
        X = np.random.randn(batch_size, n_features).astype(np.float32)
        for _ in range(10):  # warmup
            classifier.model.predict_proba(X)

        n_iterations = 100
        start = time.time()
        for _ in range(n_iterations):
            classifier.model.predict_proba(X)
        elapsed = time.time() - start

        avg_time = (elapsed / n_iterations) * 1000  # ms
        logger.info(f"Batch size {batch_size:4d}: "
                    f"Avg {avg_time:.3f}ms, "
                    f"Per sample {avg_time / batch_size:.4f}ms, "
                    f"Throughput {batch_size / (elapsed / n_iterations):.0f} samples/s")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='DDoS Classifier Training and Evaluation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest='command', help='Commands')

    train_parser = subparsers.add_parser('train', help='Train the classifier')
    train_parser.add_argument('--source', choices=['window', 'cic', 'synthetic-flow'],
                              default='window', help='Training data source (default: window)')
    train_parser.add_argument('--model-path', default=DEFAULT_MODEL_PATH,
                              help='Path to save trained model')
    train_parser.add_argument('--report', default=MLConfig.DEFAULT_REPORT_PATH,
                              help='Where to write the evaluation report (JSON)')
    train_parser.add_argument('--train-runs', type=int, default=600,
                              help='[window] Synthetic traffic runs for training')
    train_parser.add_argument('--test-runs', type=int, default=300,
                              help='[window] Independent runs for the held-out test set')
    train_parser.add_argument('--seed', type=int, default=7, help='[window] Random seed')
    train_parser.add_argument('--data-path', default=MLConfig.DATASET_DIR,
                              help='[cic] Path to CIC-DDoS-2019 dataset')
    train_parser.add_argument('--model-type', choices=['random_forest', 'gradient_boosting'],
                              default='random_forest', help='[cic/synthetic-flow] Model type')
    train_parser.add_argument('--n-estimators', type=int, default=100,
                              help='[cic/synthetic-flow] Number of trees')
    train_parser.add_argument('--max-depth', type=int, default=15,
                              help='[cic/synthetic-flow] Maximum tree depth')
    train_parser.add_argument('--max-files', type=int, default=5,
                              help='[cic] Maximum CSV files to load')
    train_parser.add_argument('--samples-per-file', type=int, default=50000,
                              help='[cic/synthetic-flow] Maximum samples per file')
    train_parser.add_argument('--multiclass', action='store_true',
                              help='[cic] Use multi-class classification')

    eval_parser = subparsers.add_parser('evaluate', help='Evaluate a trained model')
    eval_parser.add_argument('--model-path', default=DEFAULT_MODEL_PATH,
                             help='Path to trained model')
    eval_parser.add_argument('--runs', type=int, default=300,
                             help='Synthetic runs for window-level evaluation')
    eval_parser.add_argument('--e2e-runs', type=int, default=150,
                             help='Synthetic runs replayed through the full detectors')
    eval_parser.add_argument('--seed', type=int, default=2026,
                             help='Seed (use one not used for training)')
    eval_parser.add_argument('--report', default=None, help='Write JSON report here')

    bench_parser = subparsers.add_parser('benchmark', help='Benchmark inference latency')
    bench_parser.add_argument('--model-path', default=DEFAULT_MODEL_PATH,
                              help='Path to trained model')
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command == 'train':
        if not train_model(args):
            sys.exit(1)
    elif args.command == 'evaluate':
        if evaluate_model(args) is None:
            sys.exit(1)
    elif args.command == 'benchmark':
        benchmark_inference(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
