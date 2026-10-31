#!/usr/bin/env python3
"""
Control-plane benchmark: how long one monitoring interval takes in user space.

Each interval the system runs detection (rules, plus the Random Forest for the
hybrid detector), mitigation (top5 or P4) and bookkeeping in process_sample().
This script times process_sample() on labelled synthetic runs, in simulation
mode, with the per-source list capped at 1000 entries as the live monitor does.

What it does NOT measure: reading the eBPF maps through BCC and the in-kernel
per-packet cost of the XDP program. Those need Linux and root; see
scripts/live_xdp_demo.sh (it writes live_metrics.json).

Usage (from project/):
    python scripts/benchmark.py              # 20 runs per configuration, timed 5 times
Writes data/results/benchmark_results.json.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import psutil

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATA_DIR, MLConfig  # noqa: E402
from src.provenance import provenance  # noqa: E402

RESULTS_PATH = DATA_DIR / 'results' / 'benchmark_results.json'
MAX_SOURCES = 1000          # TrafficMonitor.get_ip_interval_statistics(limit=1000)
CONFIGS = [('statistical', 'top5'), ('hybrid', 'top5'), ('hybrid', 'p4')]


def make_system(detector: str, policy: str, tmp: Path):
    from src.core.anomaly_detector import AnomalyDetector, MLEnhancedAnomalyDetector
    from src.core.system import DDoSMitigationSystem
    from src.core.traffic_monitor import TrafficMonitor
    system = DDoSMitigationSystem('bench0', ml_model_path=None, auto_detect_model=False, policy=policy)
    system.anomaly_detector = (MLEnhancedAnomalyDetector(model_path=MLConfig.DEFAULT_MODEL_PATH)
                               if detector == 'hybrid' else AnomalyDetector())
    system.alert_system.alert_file = str(tmp / 'alerts.log')
    system.traffic_profiler.profile_file = str(tmp / 'profile.json')
    monitor = TrafficMonitor('bench0', blacklist_file=str(tmp / 'blacklist.json'))
    monitor.simulation_mode, monitor.loaded = True, True
    system.traffic_monitor = monitor
    return system


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--runs', type=int, default=20, help='synthetic runs per configuration')
    ap.add_argument('--seed', type=int, default=6001)
    ap.add_argument('--repeat', type=int, default=5, help='timed repetitions per configuration')
    args = ap.parse_args()
    logging.disable(logging.WARNING)
    from src.config import AlertConfig
    AlertConfig.ALERT_TO_CONSOLE = False   # printing alerts would be timed too

    import tempfile
    from src.ml.traffic_dataset import generate_runs
    runs = generate_runs(args.runs, seed=args.seed, client_churn=0.1, bot_pre_activity=0.3)
    # Pre-build per-interval inputs once (top 1000 sources, as the live monitor reports)
    samples = []
    for run in runs:
        steps = []
        for step in run.steps:
            ip_stats = sorted(step.ip_stats(), key=lambda e: e['packets'], reverse=True)[:MAX_SOURCES]
            steps.append((step.stats, ip_stats, step.timestamp))
        samples.append(steps)
    n_sources = [len(s[1]) for steps in samples for s in steps]

    proc = psutil.Process()
    results = {}
    for detector, policy in CONFIGS:
        key = f'{detector}+{policy}'
        repeats = []
        for rep in range(args.repeat):
            times = []
            with tempfile.TemporaryDirectory() as tmp:
                for steps in samples:
                    system = make_system(detector, policy, Path(tmp))
                    for stats, ip_stats, ts in steps:
                        t0 = time.perf_counter()
                        system.process_sample(stats, ip_stats, now=ts)
                        times.append((time.perf_counter() - t0) * 1000)
            times = np.array(times)
            repeats.append({'intervals': int(len(times)), 'mean_ms': float(times.mean()),
                            'p50_ms': float(np.percentile(times, 50)), 'p99_ms': float(np.percentile(times, 99)),
                            'max_ms': float(times.max())})
            print(f'{key:18s} repeat {rep + 1}: mean {times.mean():6.2f} ms  p50 {np.percentile(times, 50):6.2f}  '
                  f'p99 {np.percentile(times, 99):6.2f}  max {times.max():7.2f}  ({len(times)} intervals)')
        # Timing varies between repeats (CPU frequency, background load): report every repeat and the range
        results[key] = {
            'repeats': repeats,
            'intervals': repeats[0]['intervals'],
            **{f'{m}_{agg}': float(fn([r[m] for r in repeats]))
               for m in ('mean_ms', 'p50_ms', 'p99_ms', 'max_ms')
               for agg, fn in (('min', np.min), ('median', np.median), ('max', np.max))},
        }

    out = {
        'what': 'user-space time of one process_sample() call (detection + mitigation + bookkeeping), '
                'simulation mode; excludes BCC map reads and in-kernel XDP cost',
        'per_interval_sources': {'mean': float(np.mean(n_sources)), 'max': int(np.max(n_sources))},
        'results': results,
        'rss_mb_after': proc.memory_info().rss / 2**20,
        'provenance': provenance('scripts/benchmark.py', vars(args), {'detection_model': MLConfig.DEFAULT_MODEL_PATH}),
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(out, indent=1))
    print(f"RSS {out['rss_mb_after']:.0f} MB; wrote {RESULTS_PATH}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
