#!/usr/bin/env python3
"""
Reproduce every result file, figure and table used in the paper and the report.

Steps (each writes a result file that carries its own provenance block):
  1. train      detection model + held-out report (only with --retrain; otherwise the
                shipped model is kept, and --check-model retrains a copy and compares)
  2. evaluate   second synthetic evaluation + end-to-end detector replay
  3. mitigation synthetic mitigation experiment (tunes P4, trains the SIG classifier)
  4. trace      MAWI trace-driven experiment (skipped if the trace extract is missing;
                get it with scripts/mawi_extract.py)
  5. cic        all three levels trained and tested on real attacks, CIC-DDoS2019
                (skipped if the extracted captures are missing; get them with
                scripts/cic_inspect.py and scripts/cic_extract.py from the downloaded zips)
  5b. safety    mitigation-safety scenarios
  6. benchmark  control-plane time per interval
  7. figures    docs/figures/*.png and RESULTS_TABLES.md
  8. verify     docs/RESULT_VERIFICATION.md (every quoted number checked)
  9. tests      pytest

Writes data/results/reproduce_manifest.json: command, exit code, duration and
SHA-256 of every output of every step. Stops at the first failing step.

Not covered (needs Linux and root): the live XDP run (scripts/live_xdp_demo.sh)
and the kernel-verifier check (scripts/check_xdp_compile.py).

Usage (from project/):
    python scripts/reproduce_all.py                    # about 15-25 min on 16 cores
    python scripts/reproduce_all.py --check-model      # also retrain a copy of the model and compare
    python scripts/reproduce_all.py --only verify tests
"""

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.provenance import file_sha256, provenance  # noqa: E402

PY = sys.executable
D = PROJECT_ROOT / 'data'
TRACE = D / 'external' / 'mawi' / '200803181400.npz'
CIC = [D / 'external' / 'cic' / f'{day}.npz' for day in ('01-12', '03-11')]
STEPS = [
    ('train', [PY, '-m', 'src.ml.model_trainer', 'train'],
     [D / 'models' / 'ddos_classifier.joblib', D / 'models' / 'ddos_classifier_report.json']),
    ('evaluate', [PY, '-m', 'src.ml.model_trainer', 'evaluate', '--seed', '3030', '--runs', '300',
                  '--e2e-runs', '200', '--report', str(D / 'models' / 'ddos_classifier_eval.json')],
     [D / 'models' / 'ddos_classifier_eval.json']),
    ('mitigation', [PY, 'scripts/mitigation_experiment.py'],
     [D / 'results' / 'mitigation_results.json', D / 'models' / 'source_scorer.joblib',
      D / 'models' / 'source_scorer_no_history.joblib', D / 'models' / 'signature_classifier.joblib']),
    ('trace', [PY, 'scripts/trace_experiment.py', '--trace', '200803181400'], [D / 'results' / 'trace_results.json']),
    ('cic', [PY, 'scripts/cic_experiment.py'],
     [D / 'results' / 'cic_results.json', D / 'models' / 'cic_window_model.joblib',
      D / 'models' / 'cic_flow_model.joblib', D / 'models' / 'cic_source_scorer.joblib']),
    ('safety', [PY, 'scripts/safety_experiment.py'], [D / 'results' / 'safety_results.json']),
    ('benchmark', [PY, 'scripts/benchmark.py'], [D / 'results' / 'benchmark_results.json']),
    ('figures', [PY, 'scripts/make_figures.py'], [PROJECT_ROOT / 'docs' / 'figures' / 'RESULTS_TABLES.md']),
    ('verify', [PY, 'scripts/verify_results.py'], [PROJECT_ROOT / 'docs' / 'RESULT_VERIFICATION.md']),
    ('tests', [PY, '-m', 'pytest', '-q'], []),
]


def check_model() -> dict:
    """Retrain the detection model into a temporary file and compare with the shipped one"""
    import joblib
    import numpy as np
    from src.config import MLConfig
    from src.ml.traffic_dataset import generate_dataset
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / 'model.joblib'
        cmd = [PY, '-m', 'src.ml.model_trainer', 'train', '--model-path', str(path),
               '--report', str(Path(tmp) / 'report.json')]
        rc = subprocess.run(cmd, cwd=PROJECT_ROOT).returncode
        if rc != 0:
            return {'ok': False, 'error': f'training exited with {rc}'}
        new, old = joblib.load(path), joblib.load(MLConfig.DEFAULT_MODEL_PATH)
        data = generate_dataset(50, seed=4242)
        p_new = new['model'].predict_proba(data['X'])[:, 1]
        p_old = old['model'].predict_proba(data['X'])[:, 1]
        same_metrics = new['evaluation']['test_metrics'] == old['evaluation']['test_metrics']
        return {'ok': bool(np.array_equal(p_new, p_old) and same_metrics),
                'identical_probabilities_on_50_fresh_runs': bool(np.array_equal(p_new, p_old)),
                'max_probability_difference': float(np.abs(p_new - p_old).max()),
                'identical_stored_test_metrics': bool(same_metrics)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--retrain', action='store_true', help='retrain the shipped detection model in place')
    ap.add_argument('--check-model', action='store_true', help='retrain a copy and compare with the shipped model')
    ap.add_argument('--only', nargs='+', choices=[s[0] for s in STEPS], help='run only these steps')
    args = ap.parse_args()
    manifest = {'provenance': provenance('scripts/reproduce_all.py', vars(args)), 'steps': []}
    out_path = D / 'results' / 'reproduce_manifest.json'

    def save():
        out_path.write_text(json.dumps(manifest, indent=1))

    if args.check_model:
        print('== check-model: retraining a copy of the detection model')
        manifest['model_check'] = check_model()
        print(manifest['model_check'])
        save()

    for name, cmd, outputs in STEPS:
        if args.only and name not in args.only:
            continue
        if name == 'train' and not args.retrain:
            continue
        if name == 'trace' and not TRACE.exists():
            manifest['steps'].append({'step': name, 'skipped': f'{TRACE.name} not found (scripts/mawi_extract.py)'})
            print(f'== {name}: skipped, {TRACE} not found')
            continue
        missing = [p for p in CIC if not p.exists()] + ([] if TRACE.exists() else [TRACE])
        if name == 'cic' and missing:
            manifest['steps'].append({'step': name, 'skipped': f'{missing[0].name} not found '
                                                              '(scripts/cic_extract.py, scripts/mawi_extract.py)'})
            print(f'== {name}: skipped, {missing[0]} not found')
            continue
        print(f"== {name}: {' '.join(Path(c).name if c == PY else c for c in cmd)}")
        started = time.time()
        rc = subprocess.run(cmd, cwd=PROJECT_ROOT).returncode
        manifest['steps'].append({
            'step': name, 'command': ' '.join(['python' if c == PY else c for c in cmd]), 'exit_code': rc,
            'seconds': round(time.time() - started, 1),
            'outputs': {str(p.relative_to(PROJECT_ROOT)): file_sha256(p) for p in outputs}})
        save()
        if rc != 0:
            print(f'== {name} failed (exit {rc}); stopping. Manifest: {out_path}')
            return rc
    print(f'== done. Manifest: {out_path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
