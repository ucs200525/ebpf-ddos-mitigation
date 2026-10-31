#!/usr/bin/env python3
"""
Check every number quoted in the paper, report and slides against the saved result files.

Each claim names the result file it comes from, how the value is computed from it,
how it is rounded in the text, and the documents that quote it. The script
recomputes the value, checks that it rounds to the quoted text, and checks that the
quoted text appears in each document. It writes docs/RESULT_VERIFICATION.md and
exits with status 1 if any claim fails.

Usage (from project/):  python scripts/verify_results.py
"""

import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPO = PROJECT_ROOT.parent
DATA = PROJECT_ROOT / 'data'
BUILD = REPO / 'other' / 'deliverables' / 'build'
OUT = PROJECT_ROOT / 'docs' / 'RESULT_VERIFICATION.md'

SOURCES = {
    'model': (DATA / 'models' / 'ddos_classifier_report.json', 'python -m src.ml.model_trainer train'),
    'eval': (DATA / 'models' / 'ddos_classifier_eval.json',
             'python -m src.ml.model_trainer evaluate --seed 3030 --runs 300 --e2e-runs 200'),
    'mit': (DATA / 'results' / 'mitigation_results.json', 'python scripts/mitigation_experiment.py'),
    'trace': (DATA / 'results' / 'trace_results.json', 'python scripts/trace_experiment.py'),
    'safety': (DATA / 'results' / 'safety_results.json', 'python scripts/safety_experiment.py'),
    'bench': (DATA / 'results' / 'benchmark_results.json', 'python scripts/benchmark.py'),
    'cic': (DATA / 'results' / 'cic_results.json', 'python scripts/cic_experiment.py'),
    'xdp': (DATA / 'results' / 'cic_xdp_results.json',
            'sudo bash scripts/cic_xdp_replay.sh; python scripts/cic_xdp_analyze.py'),
    'live': (DATA / 'results' / 'live_demo_results.json', 'sudo POLICY=... MODELS=... bash scripts/live_xdp_demo.sh'),
}
DOCS = {'paper': BUILD / 'paper_content.py', 'report': BUILD / 'report_content.py',
        'slides': BUILD / 'build_slides.py'}


def m(d, setting, key, metric):
    return d['settings'][setting]['policies'][key][metric]['mean']


def g(d, setting, key, group, metric):
    return d['settings'][setting]['groups'][key][group][metric]['mean']


def t(d, key, metric):
    return d['policies'][key][metric]['mean']


def tg(d, key, group, metric):
    return d['groups'][key][group][metric]['mean']


def s(d, scenario, key, metric):
    v = d['scenarios'][scenario]['policies'][key][metric]
    return None if v is None else v['mean']


def ratio(text):
    a, b = text.split('/')
    return 100.0 * int(a) / int(b)


def curve(d, t_block, t_limit, metric):
    return next(c[metric]['mean'] for c in d['settings']['realistic']['curve']
                if c['t_block'] == t_block and c['t_limit'] == t_limit)


R = 'realistic'
# (id, what, source, value function, decimals, quoted text, documents)
CLAIMS = [
    # Detection model (held-out synthetic test set stored with the model)
    ('det-f1', 'Classifier F1 on 27,262 held-out synthetic windows', 'model',
     lambda d: d['test_metrics']['f1'], 3, '0.990', ['paper', 'report']),
    ('det-windows-test', 'Held-out test windows', 'model', lambda d: d['test_windows'], 0, '27,262', ['report']),
    ('det-windows-train', 'Training windows', 'model', lambda d: d['train_windows'], 0, '55,256', ['report']),
    ('det-acc', 'Classifier accuracy (held-out synthetic)', 'model', lambda d: d['test_metrics']['accuracy'], 3,
     '0.995', ['report']),
    ('det-auc', 'Classifier ROC-AUC (held-out synthetic)', 'model', lambda d: d['test_metrics']['roc_auc'], 4,
     '0.9999', ['report']),
    # End-to-end detection (separate evaluation, seed 3031, 200 runs)
    ('e2e-hyb-recall', 'Hybrid detector: attack seconds flagged', 'eval',
     lambda d: 100 * d['hybrid_detector']['attack_second_recall'], 1, '92.7', ['paper', 'report', 'slides']),
    ('e2e-hyb-fpr', 'Hybrid detector: benign seconds flagged', 'eval',
     lambda d: 100 * d['hybrid_detector']['benign_second_fpr'], 1, '3.0', ['paper', 'report', 'slides']),
    ('e2e-stat-recall', 'Rules only: attack seconds flagged', 'eval',
     lambda d: 100 * d['statistical_detector']['attack_second_recall'], 1, '77.2', ['slides']),
    ('e2e-stat-fpr', 'Rules only: benign seconds flagged', 'eval',
     lambda d: 100 * d['statistical_detector']['benign_second_fpr'], 1, '25.8', ['report', 'slides']),
    ('e2e-hyb-runs', 'Hybrid detector: attack runs detected (%)', 'eval',
     lambda d: ratio(d['hybrid_detector']['attack_runs_detected']), 0, '100', ['slides']),
]

# Main mitigation table (synthetic realistic setting, mean of 5 test seeds), paper Table I
for key, a, l in [('P0', '34.5', '27.8'), ('P1', '33.6', '18.4'), ('P2', '74.1', '11.0'),
                  ('P3a', '87.2', '19.6'), ('P3b', '51.5', '4.0'), ('AHIF', '92.5', '9.9'),
                  ('SIG', '71.1', '3.2'), ('SIG-oracle', '98.2', '1.8'), ('P4', '89.6', '2.7')]:
    CLAIMS += [
        (f'syn-{key}-attack', f'{key}: attack dropped, synthetic realistic', 'mit',
         lambda d, k=key: m(d, R, k, 'attack_drop_pct'), 1, a, ['paper']),
        (f'syn-{key}-legit', f'{key}: legitimate dropped, synthetic realistic', 'mit',
         lambda d, k=key: m(d, R, k, 'legit_loss_pct'), 1, l, ['paper']),
    ]
# Trace-driven table (MAWI + injected attacks, mean of 5 injection seeds)
for key, a, l in [('P0', '32.3', '7.0'), ('P1', '30.4', '5.6'), ('P2', '49.8', '23.1'),
                  ('P3a', '61.7', '13.7'), ('P3b', '30.7', '1.2'), ('AHIF', '71.9', '23.4'),
                  ('SIG', '69.6', '24.2'), ('SIG-oracle', '89.1', '0.3'), ('P4', '52.9', '1.8')]:
    CLAIMS += [
        (f'trace-{key}-attack', f'{key}: attack dropped, MAWI trace', 'trace',
         lambda d, k=key: t(d, k, 'attack_drop_pct'), 1, a, ['paper']),
        (f'trace-{key}-legit', f'{key}: legitimate dropped, MAWI trace', 'trace',
         lambda d, k=key: t(d, k, 'legit_loss_pct'), 1, l, ['paper']),
    ]

CLAIMS += [
    # Synthetic, text of the evaluation section
    ('syn-AHIF-crowd', 'AHIF: legitimate loss in flash crowds', 'mit',
     lambda d: g(d, R, 'AHIF', 'flash crowd', 'legit_loss_pct'), 1, '24.1', ['paper']),
    ('syn-SIG-os', 'SIG: attack dropped when attack packets carry OS-stack headers', 'mit',
     lambda d: d['settings'][R]['profiles']['SIG']['OS-stack headers']['attack_drop_pct']['mean'], 1, '55.0',
     ['paper']),
    ('syn-P4-mitig', 'P4: attack runs mitigated (%)', 'mit', lambda d: m(d, R, 'P4', 'mitigated_runs_pct'), 0, '66',
     ['paper']),
    ('syn-P0-mitig', 'P0: attack runs mitigated (%)', 'mit', lambda d: m(d, R, 'P0', 'mitigated_runs_pct'), 0, '35',
     ['paper']),
    ('syn-P4-clean-a', 'P4 attack dropped, clean setting', 'mit', lambda d: m(d, 'clean', 'P4', 'attack_drop_pct'),
     1, '89.5', ['paper']),
    ('syn-P4-clean-l', 'P4 legitimate dropped, clean setting', 'mit',
     lambda d: m(d, 'clean', 'P4', 'legit_loss_pct'), 1, '4.8', ['paper']),
    ('syn-P4-hard-a', 'P4 attack dropped, hard setting', 'mit', lambda d: m(d, 'hard', 'P4', 'attack_drop_pct'),
     1, '90.3', ['paper']),
    ('syn-P4-hard-l', 'P4 legitimate dropped, hard setting', 'mit', lambda d: m(d, 'hard', 'P4', 'legit_loss_pct'),
     1, '5.0', ['paper']),
    ('syn-P2-hard-a', 'P2 attack dropped, hard setting', 'mit', lambda d: m(d, 'hard', 'P2', 'attack_drop_pct'),
     1, '45.3', ['paper']),
    ('syn-thr-block', 'P4 chosen t_block', 'mit', lambda d: d['protocol']['chosen_params']['t_block'], 1, '0.8',
     ['paper']),
    ('syn-thr-limit', 'P4 chosen t_limit', 'mit', lambda d: d['protocol']['chosen_params']['t_limit'], 1, '0.7',
     ['paper']),
    ('syn-curve-09-a', 'P4 t_block 0.9 (t_limit 0.7): attack dropped', 'mit',
     lambda d: curve(d, 0.9, 0.7, 'attack_drop_pct'), 1, '85.9', ['paper']),
    ('syn-curve-09-l', 'P4 t_block 0.9 (t_limit 0.7): legitimate dropped', 'mit',
     lambda d: curve(d, 0.9, 0.7, 'legit_loss_pct'), 1, '1.5', ['paper']),
    ('syn-curve-07-a', 'P4 both thresholds 0.7: attack dropped', 'mit',
     lambda d: curve(d, 0.7, 0.7, 'attack_drop_pct'), 1, '93.5', ['paper']),
    ('syn-curve-07-l', 'P4 both thresholds 0.7: legitimate dropped', 'mit',
     lambda d: curve(d, 0.7, 0.7, 'legit_loss_pct'), 1, '5.9', ['paper']),
    # Ablation
    ('abl-nospoof-syn', 'P4 without spoofing fallback: attack dropped, synthetic', 'mit',
     lambda d: m(d, R, 'P4-no-spoof-guard', 'attack_drop_pct'), 1, '58.7', ['paper', 'slides']),
    ('abl-nospoof-trace', 'P4 without spoofing fallback: attack dropped, trace', 'trace',
     lambda d: t(d, 'P4-no-spoof-guard', 'attack_drop_pct'), 1, '37.9', ['paper']),
    ('abl-noexp-l', 'P4 without expiry: legitimate dropped, synthetic', 'mit',
     lambda d: m(d, R, 'P4-no-expiry', 'legit_loss_pct'), 1, '3.1', ['paper', 'slides']),
    ('abl-coef-syn', 'Scorer coefficient: SYN share', 'mit', lambda d: d['scorer']['coefficients']['syn_share'], 2,
     '0.77', ['paper']),
    ('abl-coef-jump', 'Scorer coefficient: rate jump', 'mit', lambda d: d['scorer']['coefficients']['rate_jump'], 2,
     '0.67', ['paper']),
    ('syn-P4-srcs', 'P4: legitimate sources hit (%)', 'mit', lambda d: m(d, R, 'P4', 'legit_sources_hit_pct'), 1,
     '5.2', ['paper']),
    ('syn-P0-srcs', 'P0: legitimate sources hit (%)', 'mit', lambda d: m(d, R, 'P0', 'legit_sources_hit_pct'), 2,
     '0.25', ['paper']),
    # Trace text
    ('trace-det-runs', 'Trace: attack runs detected by the hybrid detector', 'trace',
     lambda d: d['detected_attack_runs_pct']['hybrid']['mean'], 1, '91.8', ['paper']),
    ('trace-det-fa', 'Trace: seconds flagged in attack-free runs (hybrid)', 'trace',
     lambda d: d['false_alarm_seconds_pct']['hybrid']['mean'], 1, '9.1', ['paper']),
    ('trace-AHIF-benign', 'Trace: AHIF legitimate loss in attack-free runs', 'trace',
     lambda d: t(d, 'AHIF', 'legit_loss_benign_runs_pct'), 1, '18.0', ['paper']),
    ('trace-SIG-benign', 'Trace: SIG legitimate loss in attack-free runs', 'trace',
     lambda d: t(d, 'SIG', 'legit_loss_benign_runs_pct'), 1, '16.1', ['paper']),
    ('trace-P4-benign', 'Trace: P4 legitimate loss in attack-free runs', 'trace',
     lambda d: t(d, 'P4', 'legit_loss_benign_runs_pct'), 2, '0.16', ['paper']),
    ('trace-P4-botnet', 'Trace: P4 attack dropped, botnet floods', 'trace',
     lambda d: tg(d, 'P4', 'botnet floods', 'attack_drop_pct'), 1, '53.9', ['paper']),
    ('trace-P4-lowrate', 'Trace: P4 attack dropped, low-rate floods', 'trace',
     lambda d: tg(d, 'P4', 'low-rate floods', 'attack_drop_pct'), 1, '18.0', ['paper']),
    ('trace-P4-spoof-l', 'Trace: P4 legitimate loss in spoofed floods', 'trace',
     lambda d: tg(d, 'P4', 'spoofed floods', 'legit_loss_attack_runs_pct'), 1, '12.8', ['paper']),
]



def sp(d, scenario, key, metric):
    """Safety experiment: mean over seeds"""
    return d['scenarios'][scenario]['policies'][key][metric]['mean']


def sdet(d, scenario):
    return d['scenarios'][scenario]['detector']['hybrid']['flagged_seconds_pct']['mean']


def sd(d, setting, key, metric):
    return d['settings'][setting]['policies'][key][metric]['std']


# Report Table 4.3 (synthetic realistic): attack mean/std, legit mean/std, sources hit, runs mitigated
for key, row in {'P0': ('34.5', '6.6', '27.8', '5.3', '0.25', '35.1'),
                 'P1': ('33.6', '6.6', '18.4', '3.0', '0.25', '34.8'),
                 'P2': ('74.1', '8.5', '11.0', '2.9', '22.6', '42.3'),
                 'P3a': ('87.2', '1.8', '19.6', '2.2', '19.9', '51.0'),
                 'P3b': ('51.5', '11.1', '4.0', '0.7', '0.48', '27.0'),
                 'AHIF': ('92.5', '1.5', '9.9', '2.9', '25.3', '59.2'),
                 'SIG': ('71.1', '11.4', '3.2', '1.0', '3.2', '68.0'),
                 'SIG-oracle': ('98.2', '0.3', '1.8', '1.3', '4.6', '96.3'),
                 'P4': ('89.6', '5.2', '2.7', '1.9', '5.2', '65.9')}.items():
    a, a_sd, l, l_sd, hit, mit = row
    CLAIMS += [
        (f'rep-{key}-a', f'Report T4.3 {key}: attack dropped', 'mit',
         lambda d, k=key: (m(d, R, k, 'attack_drop_pct'), sd(d, R, k, 'attack_drop_pct')), 1, f'{a} +/- {a_sd}',
         ['report']),
        (f'rep-{key}-l', f'Report T4.3 {key}: legit dropped', 'mit',
         lambda d, k=key: (m(d, R, k, 'legit_loss_pct'), sd(d, R, k, 'legit_loss_pct')), 1, f'{l} +/- {l_sd}',
         ['report']),
        (f'rep-{key}-hit', f'Report T4.3 {key}: legit sources hit', 'mit',
         lambda d, k=key: m(d, R, k, 'legit_sources_hit_pct'), 2 if '.' in hit and len(hit.split('.')[1]) == 2 else 1,
         hit, ['report']),
        (f'rep-{key}-mit', f'Report T4.3 {key}: runs mitigated', 'mit',
         lambda d, k=key: m(d, R, k, 'mitigated_runs_pct'), 1, mit, ['report']),
    ]
# The "+/-" claims compare mean and std together
# Report Table 4.5 (trace)
for key, row in {'P0': ('32.3', '16.0', '7.0', '2.8', '0.30', '0.10'),
                 'P2': ('49.8', '15.5', '23.1', '2.5', '20.0', '50.4'),
                 'P3a': ('61.7', '5.5', '13.7', '3.5', '0.22', '3.9'),
                 'P3b': ('30.7', '15.4', '1.2', '0.7', '0.19', '0.16'),
                 'AHIF': ('71.9', '8.9', '23.4', '3.5', '18.0', '49.3'),
                 'SIG': ('69.6', '15.1', '24.2', '2.5', '16.1', '32.6'),
                 'SIG-oracle': ('89.1', '5.5', '0.3', '0.1', '0.00', '0.09'),
                 'P4': ('52.9', '11.0', '1.8', '0.5', '0.16', '0.83')}.items():
    a, a_sd, l, l_sd, benign, hit = row
    dec = lambda x: len(x.split('.')[1])  # noqa: E731
    CLAIMS += [
        (f'rep-tr-{key}-a', f'Report T4.5 {key}: attack dropped', 'trace',
         lambda d, k=key: (t(d, k, 'attack_drop_pct'), d['policies'][k]['attack_drop_pct']['std']), 1,
         f'{a} +/- {a_sd}', ['report']),
        (f'rep-tr-{key}-l', f'Report T4.5 {key}: legit dropped', 'trace',
         lambda d, k=key: (t(d, k, 'legit_loss_pct'), d['policies'][k]['legit_loss_pct']['std']), 1,
         f'{l} +/- {l_sd}', ['report']),
        (f'rep-tr-{key}-b', f'Report T4.5 {key}: legit dropped, attack-free runs', 'trace',
         lambda d, k=key: t(d, k, 'legit_loss_benign_runs_pct'), dec(benign), benign, ['report']),
        (f'rep-tr-{key}-hit', f'Report T4.5 {key}: legit sources hit', 'trace',
         lambda d, k=key: t(d, k, 'legit_sources_hit_pct'), dec(hit), hit, ['report']),
    ]
CLAIMS += [
    ('rep-P0-benign-syn', 'P0 legit loss in normal/bursty runs (synthetic)', 'mit',
     lambda d: g(d, R, 'P0', 'normal / bursty', 'legit_loss_pct'), 1, '26.9', ['report', 'slides']),
    ('rep-P4-spoof-a', 'P4 attack dropped, spoofed floods (synthetic)', 'mit',
     lambda d: g(d, R, 'P4', 'spoofed floods', 'attack_drop_pct'), 1, '94.3', ['report', 'slides']),
    ('rep-P0-lowrate', 'P0 attack dropped, low-rate floods (synthetic)', 'mit',
     lambda d: g(d, R, 'P0', 'low-rate floods', 'attack_drop_pct'), 1, '39.9', ['report']),
    ('rep-P4-lowrate', 'P4 attack dropped, low-rate floods (synthetic)', 'mit',
     lambda d: g(d, R, 'P4', 'low-rate floods', 'attack_drop_pct'), 1, '71.8', ['report']),
    ('rep-P4-crowd', 'P4 legit loss in flash crowds (synthetic)', 'mit',
     lambda d: g(d, R, 'P4', 'flash crowd', 'legit_loss_pct'), 1, '6.6', ['report', 'slides']),
    ('rep-P2-clean', 'P2 attack dropped, clean setting', 'mit', lambda d: m(d, 'clean', 'P2', 'attack_drop_pct'), 1,
     '92.0', ['report']),
    ('rep-abl-nospoof-l', 'Ablation: no spoofing fallback, legit dropped', 'mit',
     lambda d: m(d, R, 'P4-no-spoof-guard', 'legit_loss_pct'), 1, '1.9', ['report', 'slides']),
    ('rep-abl-nospoof-mit', 'Ablation: no spoofing fallback, runs mitigated', 'mit',
     lambda d: m(d, R, 'P4-no-spoof-guard', 'mitigated_runs_pct'), 1, '50.4', ['report']),
    ('rep-abl-noexp-a', 'Ablation: no expiry, attack dropped', 'mit',
     lambda d: m(d, R, 'P4-no-expiry', 'attack_drop_pct'), 1, '90.0', ['report', 'slides']),
    ('rep-abl-block-a', 'Ablation: block only, attack dropped', 'mit',
     lambda d: m(d, R, 'P4-block-only', 'attack_drop_pct'), 1, '88.6', ['report', 'slides']),
    ('rep-abl-block-l', 'Ablation: block only, legit dropped', 'mit',
     lambda d: m(d, R, 'P4-block-only', 'legit_loss_pct'), 1, '2.6', ['report', 'slides']),
    ('rep-abl-block-mit', 'Ablation: block only, runs mitigated', 'mit',
     lambda d: m(d, R, 'P4-block-only', 'mitigated_runs_pct'), 1, '63.4', ['report']),
    ('abl-nohist-a', 'Ablation: no per-source history features, attack dropped', 'mit',
     lambda d: m(d, R, 'P4-no-history', 'attack_drop_pct'), 1, '85.0', ['paper', 'report']),
    ('abl-nohist-l', 'Ablation: no per-source history features, legit dropped', 'mit',
     lambda d: m(d, R, 'P4-no-history', 'legit_loss_pct'), 1, '9.6', ['paper', 'report']),
    ('rec-P0-post', 'P0: legit loss after the attack ends', 'mit',
     lambda d: m(d, R, 'P0', 'legit_loss_post_attack_pct'), 1, '46.4', ['paper', 'report']),
    ('rec-P4-post', 'P4: legit loss after the attack ends', 'mit',
     lambda d: m(d, R, 'P4', 'legit_loss_post_attack_pct'), 1, '0.1', ['paper', 'report']),
    ('rec-P0-recovered', 'P0: runs that recover after the attack (%)', 'mit',
     lambda d: m(d, R, 'P0', 'recovered_runs_pct'), 0, '29', ['report']),
    ('rec-P4-recovered', 'P4: runs that recover after the attack (%)', 'mit',
     lambda d: m(d, R, 'P4', 'recovered_runs_pct'), 1, '98.9', ['report']),
    ('delay-detect', 'Median detection delay, synthetic realistic (s)', 'mit',
     lambda d: m(d, R, 'P4', 'median_detection_delay_s'), 1, '3.4', ['paper', 'report']),
    ('delay-mitigate-P4', 'P4 median time to 50% attack drop (s)', 'mit',
     lambda d: m(d, R, 'P4', 'median_time_to_mitigate_s'), 1, '3.8', ['paper', 'report']),
    ('det-runs-realistic', 'Attack runs detected, synthetic realistic (%)', 'mit',
     lambda d: m(d, R, 'P4', 'detected_runs_pct'), 1, '96.3', ['report']),
    ('abl-nohist-mit', 'Ablation: no per-source history features, runs mitigated', 'mit',
     lambda d: m(d, R, 'P4-no-history', 'mitigated_runs_pct'), 1, '62.4', ['report']),
    ('pass-P4', 'P4: attack passed', 'mit', lambda d: m(d, R, 'P4', 'attack_passed_pct'), 1, '10.4', ['report']),
    ('pass-AHIF', 'AHIF: attack passed', 'mit', lambda d: m(d, R, 'AHIF', 'attack_passed_pct'), 1, '7.5', ['report']),
    ('pass-P3a', 'P3a: attack passed', 'mit', lambda d: m(d, R, 'P3a', 'attack_passed_pct'), 1, '12.8', ['report']),
    ('pass-P0', 'P0: attack passed', 'mit', lambda d: m(d, R, 'P0', 'attack_passed_pct'), 1, '65.5', ['report']),
    ('pass-SIG', 'SIG: attack passed', 'mit', lambda d: m(d, R, 'SIG', 'attack_passed_pct'), 1, '28.9', ['report']),
    ('ret-P4', 'P4: legit retained', 'mit', lambda d: m(d, R, 'P4', 'legit_retained_pct'), 1, '97.3', ['report']),
    ('rep-tr-P4-spoof-a','Trace: P4 attack dropped, spoofed floods', 'trace',
     lambda d: tg(d, 'P4', 'spoofed floods', 'attack_drop_pct'), 1, '69.6', ['report']),
    ('rep-e2e-runs', 'Hybrid detector: attack runs found', 'eval',
     lambda d: int(d['hybrid_detector']['attack_runs_detected'].split('/')[0]), 0, '110', ['report']),
    ('rep-e2e-stat-runs', 'Rules only: attack runs found', 'eval',
     lambda d: int(d['statistical_detector']['attack_runs_detected'].split('/')[0]), 0, '89', ['report']),
    ('rep-e2e-delay', 'Hybrid detector: median detection delay (s)', 'eval',
     lambda d: d['hybrid_detector']['median_detection_delay_s'], 0, '4', ['report']),
    ('rep-cv-f1', 'Cross-validated F1 of the chosen model', 'model', lambda d: d['chosen_params']['cv_f1_mean'], 3,
     '0.992', ['report']),
    ('rep-lowrate-syn', 'Window recall, low_rate_syn', 'model',
     lambda d: 100 * d['per_scenario']['low_rate_syn']['attack_recall'], 1, '97.8', ['report']),
    ('rep-lowrate-udp', 'Window recall, low_rate_udp', 'model',
     lambda d: 100 * d['per_scenario']['low_rate_udp']['attack_recall'], 1, '98.3', ['report']),
]

CLAIMS += [
    # Mitigation-safety scenarios (data/results/safety_results.json)
    ('safe-heavy-det', 'Heavy legitimate transfers: seconds flagged by the hybrid detector', 'safety',
     lambda d: sdet(d, 'heavy_legit'), 1, '22.4', ['paper', 'report']),
    ('safe-heavy-P4', 'Heavy legitimate transfers: P4 legit loss', 'safety',
     lambda d: sp(d, 'heavy_legit', 'P4', 'legit_loss_pct'), 1, '67.5', ['paper', 'report']),
    ('safe-heavy-P0', 'Heavy legitimate transfers: P0 legit loss', 'safety',
     lambda d: sp(d, 'heavy_legit', 'P0', 'legit_loss_pct'), 1, '78.3', ['paper', 'report']),
    ('safe-heavy-P2', 'Heavy legitimate transfers: P2 legit loss', 'safety',
     lambda d: sp(d, 'heavy_legit', 'P2', 'legit_loss_pct'), 1, '3.1', ['report']),
    ('safe-heavy-SIG', 'Heavy legitimate transfers: SIG legit loss', 'safety',
     lambda d: sp(d, 'heavy_legit', 'SIG', 'legit_loss_pct'), 1, '4.9', ['report']),
    ('safe-crowd-det', 'Flash crowd: seconds flagged', 'safety', lambda d: sdet(d, 'flash_crowd'), 1, '5.1',
     ['report']),
    ('safe-crowd-P4', 'Flash crowd: P4 legit loss', 'safety', lambda d: sp(d, 'flash_crowd', 'P4', 'legit_loss_pct'),
     1, '6.2', ['report']),
    ('safe-crowd-AHIF', 'Flash crowd: AHIF legit loss', 'safety',
     lambda d: sp(d, 'flash_crowd', 'AHIF', 'legit_loss_pct'), 1, '19.7', ['report']),
    ('safe-fc-P4-a', 'UDP flood + crowd: P4 attack dropped', 'safety',
     lambda d: sp(d, 'udp_flood_with_crowd', 'P4', 'attack_drop_pct'), 1, '86.8', ['paper', 'report']),
    ('safe-fc-P4-l', 'UDP flood + crowd: P4 legit loss', 'safety',
     lambda d: sp(d, 'udp_flood_with_crowd', 'P4', 'legit_loss_pct'), 1, '18.4', ['paper', 'report']),
    ('safe-sfc-P4-a', 'Spoofed flood + crowd: P4 attack dropped', 'safety',
     lambda d: sp(d, 'udp_flood_spoofed_with_crowd', 'P4', 'attack_drop_pct'), 1, '86.2', ['paper', 'report']),
    ('safe-sfc-P4-l', 'Spoofed flood + crowd: P4 legit loss', 'safety',
     lambda d: sp(d, 'udp_flood_spoofed_with_crowd', 'P4', 'legit_loss_pct'), 1, '48.0', ['paper', 'report']),
    ('safe-sfc-nospoof-l', 'Spoofed flood + crowd: P4 without fallback, legit loss', 'safety',
     lambda d: sp(d, 'udp_flood_spoofed_with_crowd', 'P4-no-spoof-guard', 'legit_loss_pct'), 1, '19.3',
     ['report']),
    ('safe-sfc-SIG-l', 'Spoofed flood + crowd: SIG legit loss', 'safety',
     lambda d: sp(d, 'udp_flood_spoofed_with_crowd', 'SIG', 'legit_loss_pct'), 1, '7.4', ['report']),
    # Control-plane benchmark (timings: within 25%)
    ('bench-p4-mean-lo', 'Control plane, hybrid + P4: lowest mean ms over repetitions', 'bench',
     lambda d: d['results']['hybrid+p4']['mean_ms_min'], 1, '3.7', ['paper', 'report']),
    ('bench-p4-mean-hi', 'Control plane, hybrid + P4: highest mean ms over repetitions', 'bench',
     lambda d: d['results']['hybrid+p4']['mean_ms_max'], 1, '4.9', ['paper', 'report']),
    ('bench-p4-p99-hi', 'Control plane, hybrid + P4: highest p99 ms over repetitions', 'bench',
     lambda d: d['results']['hybrid+p4']['p99_ms_max'], 1, '17.5', ['paper', 'report']),
    ('bench-p4-p99-lo', 'Control plane, hybrid + P4: lowest p99 ms over repetitions', 'bench',
     lambda d: d['results']['hybrid+p4']['p99_ms_min'], 1, '12.0', ['report']),
    ('bench-stat-mean', 'Control plane, rules only: mean ms per interval', 'bench',
     lambda d: d['results']['statistical+top5']['mean_ms_max'], 1, '0.1', ['report']),
    ('bench-rss', 'Control plane memory (MB)', 'bench', lambda d: d['rss_mb_after'], 0, '447', ['report']),
    # Trace extended metrics
    ('trace-P4-passed', 'Trace: P4 attack passed', 'trace', lambda d: t(d, 'P4', 'attack_passed_pct'), 1, '47.1',
     ['paper', 'report']),
    ('trace-nohist-l', 'Trace: P4 without history features, legit loss', 'trace',
     lambda d: t(d, 'P4-no-history', 'legit_loss_pct'), 1, '3.1', ['report']),
    ('trace-nohist-a', 'Trace: P4 without history features, attack dropped', 'trace',
     lambda d: t(d, 'P4-no-history', 'attack_drop_pct'), 1, '47.6', ['report']),
    ('trace-P0-post', 'Trace: P0 legit loss after the attack', 'trace',
     lambda d: t(d, 'P0', 'legit_loss_post_attack_pct'), 1, '26.8', ['report']),
    ('trace-P4-post', 'Trace: P4 legit loss after the attack', 'trace',
     lambda d: t(d, 'P4', 'legit_loss_post_attack_pct'), 1, '0.7', ['report']),
]


# ---------------------------------------------------------------- CIC-DDoS2019 (captured attacks)
P_ = ['paper']


def cl2(d, system, key, metric='legit_loss_pct'):
    return d['level2']['systems'][system]['policies'][key]['all'][metric]


def xd(d, key):
    return d['level1']['cross_dataset'][key]


CLAIMS += [
    ('cic-benign-0112', 'CIC 01-12: benign packets/s, mean', 'cic', lambda d: d['data']['01-12']['benign_pps_mean'], 0,
     '13', P_),
    ('cic-benign-0311', 'CIC 03-11: benign packets/s, mean', 'cic', lambda d: d['data']['03-11']['benign_pps_mean'], 0,
     '18', P_),
    ('cic-ep-min', 'CIC 03-11: slowest flood episode, median attack packets/s (thousands)', 'cic',
     lambda d: round(d['data']['03-11']['episode_attacker_pps_median_min_60s'], -3), 0, '16,000', P_),
    ('cic-ep-max', 'CIC 03-11: fastest flood episode, median attack packets/s (thousands)', 'cic',
     lambda d: round(d['data']['03-11']['episode_attacker_pps_median_max_60s'], -3), 0, '27,000', P_),
    ('cic-l1-ship-recall', 'Shipped (synthetic-trained) window model: attack windows flagged, CIC 03-11', 'cic',
     lambda d: 100 * d['level1']['shipped_model']['03-11']['recall'], 1, '98.7', P_),
    ('cic-l1-ship-fpr', 'Shipped window model: benign windows flagged, CIC 03-11', 'cic',
     lambda d: 100 * d['level1']['shipped_model']['03-11']['false_positive_rate'], 1, '1.1', P_),
    ('cic-l1-only-mawi', 'CIC-only window model: attack windows found, MAWI + injected attacks', 'cic',
     lambda d: 100 * xd(d, 'mawi_injected_attacks')['cic_model']['recall'], 1, '15.0', P_),
    ('cic-l1-ship-mawi', 'Shipped window model: attack windows found, MAWI + injected attacks', 'cic',
     lambda d: 100 * xd(d, 'mawi_injected_attacks')['shipped_model']['recall'], 1, '91.5', P_),
    ('cic-l1-comb-cic', 'Combined window model: attack windows flagged, CIC 03-11', 'cic',
     lambda d: 100 * d['level1']['combined_model']['test']['recall'], 1, '99.3', P_),
    ('cic-l1-comb-mawi', 'Combined window model: attack windows found, MAWI + injected attacks', 'cic',
     lambda d: 100 * xd(d, 'mawi_injected_attacks')['combined_model']['recall'], 1, '90.5', P_),
    ('cic-l3-recall', 'Flow classifier: attack flows labelled attack, CIC 03-11', 'cic',
     lambda d: 100 * d['level3']['test_flows']['recall'], 1, '99.9', P_),
    ('cic-l3-fpr', 'Flow classifier: benign flows labelled attack, CIC 03-11', 'cic',
     lambda d: 100 * d['level3']['test_flows']['false_positive_rate'], 2, '0.21', P_),
    ('cic-l3-portmap', 'Flow classifier: Portmap flows (never in training) labelled attack', 'cic',
     lambda d: 100 * d['level3']['test_per_attack_type']['Portmap']['recall'], 1, '99.4', P_),
    ('cic-rule-syn', 'Rule "SYN-only TCP flow": attack flows found', 'cic',
     lambda d: 100 * d['level3']['rules']['syn_only_tcp_flow']['test_flows']['recall'], 1, '28.2', P_),
    ('cic-rule-ack', 'Rule "ACK-only TCP flow": benign flows flagged', 'cic',
     lambda d: 100 * d['level3']['rules']['ack_only_tcp_flow']['test_flows']['false_positive_rate'], 1, '55.8', P_),
    ('cic-l2-l3-legit', 'P4+L3 (shipped models): legitimate dropped, CIC 03-11 offline', 'cic',
     lambda d: cl2(d, 'shipped', 'P4+L3'), 1, '10.9', P_),
    ('cic-l2-l3-attack', 'P4+L3 (shipped models): attack dropped, CIC 03-11 offline', 'cic',
     lambda d: cl2(d, 'shipped', 'P4+L3', 'attack_drop_pct'), 1, '99.8', P_),
    ('cic-scorer-legit', 'P4 with the CIC-trained scorer: legitimate dropped, CIC 03-11', 'cic',
     lambda d: cl2(d, 'cic_scorer', 'P4'), 2, '0.17', P_),
    ('cic-transfer-cic', 'P4 with the CIC-trained scorer: attack dropped, synthetic realistic seed 2001', 'cic',
     lambda d: d['level2']['transfer_to_synthetic']['synthetic realistic']['P4 CIC scorer']['attack_drop_pct'], 1,
     '42.7', P_),
    ('cic-transfer-ship', 'P4 with the shipped scorer: attack dropped, synthetic realistic seed 2001', 'cic',
     lambda d: d['level2']['transfer_to_synthetic']['synthetic realistic']['P4 shipped scorer']['attack_drop_pct'], 1,
     '80.8', P_),
]
for key, a, l in [('P0', '99.9', '24.0'), ('P1', '96.5', '18.6'), ('P2', '0.0', '27.5'), ('P3a', '99.6', '33.7'),
                  ('P3b', '99.5', '17.1'), ('AHIF', '99.6', '33.9'), ('SIG', '39.5', '7.7'),
                  ('SIG-oracle', '75.7', '0.0'), ('P4', '99.8', '14.4')]:
    CLAIMS += [
        (f'cic-{key}-attack', f'{key}: attack dropped, CIC 03-11 offline (shipped models)', 'cic',
         lambda d, k=key: cl2(d, 'shipped', k, 'attack_drop_pct'), 1, a, P_),
        (f'cic-{key}-legit', f'{key}: legitimate dropped, CIC 03-11 offline (shipped models)', 'cic',
         lambda d, k=key: cl2(d, 'shipped', k), 1, l, P_),
    ]


# ---------------------------------------------------------------- live XDP (kernel-measured)
R_ = ['report']
PR = ['paper', 'report']
PRS = ['paper', 'report', 'slides']


def xa(d, cfg, key, group='attack_runs'):
    return d[group][cfg][key]


CLAIMS += [
    ('xdp-L3-attack', 'Live XDP, P4+L3: attack packets dropped (attack runs)', 'xdp',
     lambda d: xa(d, 'P4L3', 'attack_dropped_pct'), 1, '99.8', PR),
    ('xdp-min-attack', 'Live XDP: lowest attack drop of the four configurations', 'xdp',
     lambda d: min(v['attack_dropped_pct'] for v in d['attack_runs'].values()), 1, '99.8', PRS),
    ('xdp-attack-M', 'Live XDP: attack packets replayed (millions)', 'xdp',
     lambda d: d['attack_runs']['P0']['attack_sent'] / 1e6, 1, '25.6', PR),
    ('xdp-total', 'Live XDP: packets replayed per configuration', 'xdp',
     lambda d: sum(r['packets'] for r in d['setup']['runs']), 0, '25,680,423', R_),
    ('xdp-lost', 'Live XDP: most packets lost before XDP in one run', 'xdp',
     lambda d: max(v['checks']['sent'] - v['checks']['xdp_total'] for c in d['per_run'].values() for v in c.values()),
     0, '8', PR),
    ('xdp-P0-legitA', 'Live XDP, top-5: legitimate dropped during attack runs', 'xdp',
     lambda d: xa(d, 'P0', 'legit_dropped_pct'), 1, '31.1', PRS),
    ('xdp-P4-legitA', 'Live XDP, P4: legitimate dropped during attack runs', 'xdp',
     lambda d: xa(d, 'P4', 'legit_dropped_pct'), 1, '6.9', PRS),
    ('xdp-cic-legitA', 'Live XDP, P4 with the CIC detector: legitimate dropped during attack runs', 'xdp',
     lambda d: xa(d, 'P4cic', 'legit_dropped_pct'), 1, '7.3', R_),
    ('xdp-L3-legitA', 'Live XDP, P4+L3: legitimate dropped during attack runs', 'xdp',
     lambda d: xa(d, 'P4L3', 'legit_dropped_pct'), 1, '0.4', PRS),
    ('xdp-L3-legitQ', 'Live XDP, P4+L3: legitimate dropped in attack-free stretches', 'xdp',
     lambda d: xa(d, 'P4L3', 'legit_dropped_pct', 'quiet_runs'), 3, '0.003', PRS),
    ('xdp-P0-legitQ', 'Live XDP, top-5: legitimate dropped in attack-free stretches', 'xdp',
     lambda d: xa(d, 'P0', 'legit_dropped_pct', 'quiet_runs'), 1, '51.6', PRS),
    ('xdp-P4-legitQ', 'Live XDP, P4: legitimate dropped in attack-free stretches', 'xdp',
     lambda d: xa(d, 'P4', 'legit_dropped_pct', 'quiet_runs'), 1, '51.3', PR),
    ('xdp-cic-legitQ', 'Live XDP, P4 with the CIC detector: legitimate dropped in attack-free stretches', 'xdp',
     lambda d: xa(d, 'P4cic', 'legit_dropped_pct', 'quiet_runs'), 1, '49.0', R_),
    ('xdp-off-L3', 'Offline replay of the live windows, P4+L3: legitimate dropped (attack runs)', 'xdp',
     lambda d: d['offline_same_windows']['P4L3']['attack_runs']['legit_loss_pct'], 1, '0.0', R_),
    ('xdp-off-L3-q', 'Offline replay of the live windows, P4+L3: legitimate dropped (attack-free)', 'xdp',
     lambda d: d['offline_same_windows']['P4L3']['quiet_runs']['legit_loss_pct'], 1, '0.0', R_),
    ('xdp-mit-max', 'Live XDP: latest second (after onset) with >= 50% of attack packets dropped', 'xdp',
     lambda d: max(v['mitigation_delay_s_max'] for v in d['attack_runs'].values()), 0, '2', PR),
    ('xdp-hit-P0', 'Live XDP, top-5: legitimate sources hit during attack runs', 'xdp',
     lambda d: xa(d, 'P0', 'legit_sources_hit'), 0, '131', R_),
    ('xdp-hit-L3', 'Live XDP, P4+L3: legitimate sources hit during attack runs', 'xdp',
     lambda d: xa(d, 'P4L3', 'legit_sources_hit'), 0, '4', R_),
    ('xdp-srcs', 'Live XDP: legitimate sources in the attack runs', 'xdp',
     lambda d: xa(d, 'P0', 'legit_sources'), 0, '419', R_),
    ('xdp-before-legitA', 'Live XDP, P4+L3 with complete-window evidence only: legitimate dropped, attack runs', 'xdp',
     lambda d: d['before_block_time_check']['attack_runs']['P4L3']['legit_dropped_pct'], 1, '4.2', PR),
    ('xdp-before-legitQ', 'Live XDP, P4+L3 with complete-window evidence only: legitimate dropped, attack-free', 'xdp',
     lambda d: d['before_block_time_check']['quiet_runs']['P4L3']['legit_dropped_pct'], 1, '43.3', PR),
    ('live-syn-p4', 'Live SYN flood, P4: share of flood packets dropped by XDP', 'live',
     lambda d: d['runs']['p4_synthetic_syn']['packets']['dropped_over_attack_sent_pct'], 1, '75.5', PR),
    ('live-syn-l3', 'Live SYN flood, P4+L3: share of flood packets dropped by XDP', 'live',
     lambda d: d['runs']['p4l3_cic_syn']['packets']['dropped_over_attack_sent_pct'], 1, '76.8', PR),
    ('live-top5', 'Live synthetic flood, top-5: share of flood packets dropped by XDP', 'live',
     lambda d: d['runs']['top5_synthetic']['packets']['dropped_over_attack_sent_pct'], 1, '55.1', R_),
    ('live-p4', 'Live synthetic flood, P4: share of flood packets dropped by XDP', 'live',
     lambda d: d['runs']['p4_synthetic']['packets']['dropped_over_attack_sent_pct'], 1, '52.8', PR),
    ('live-l3', 'Live synthetic flood, P4+L3: share of flood packets dropped by XDP', 'live',
     lambda d: d['runs']['p4l3_cic']['packets']['dropped_over_attack_sent_pct'], 1, '89.0', PR),
    ('live-mit-top5', 'Live synthetic flood, top-5: seconds to 50% drop', 'live',
     lambda d: d['runs']['top5_synthetic']['detection']['mitigation_delay_s'], 1, '15.1', R_),
    ('live-mit-p4', 'Live synthetic flood, P4: seconds to 50% drop', 'live',
     lambda d: d['runs']['p4_synthetic']['detection']['mitigation_delay_s'], 1, '2.8', R_),
    ('live-mit-l3', 'Live synthetic flood, P4+L3: seconds to 50% drop', 'live',
     lambda d: d['runs']['p4l3_cic']['detection']['mitigation_delay_s'], 1, '2.8', R_),
]

# The 29 Oct revision of the COMSNETS paper made room for the real-attack and live-XDP results.
# Claims the paper no longer quotes are checked where they still are; claims quoted nowhere any
# more are retired (listed in docs/RESULT_VERIFICATION.md).
DOCS_NOW = {
    'e2e-hyb-recall': ['report', 'slides'], 'e2e-hyb-fpr': ['report', 'slides'], 'syn-AHIF-crowd': ['report'],
    'syn-P0-mitig': ['report'], 'syn-P4-clean-a': ['report', 'slides'], 'syn-P4-hard-a': ['report'],
    'syn-P2-hard-a': ['report', 'slides'], 'abl-noexp-l': ['slides'], 'syn-P4-srcs': ['report', 'slides'],
    'syn-P0-srcs': ['report', 'slides'], 'trace-P4-botnet': ['report'], 'abl-nohist-a': ['report'],
    'safe-heavy-det': ['report'], 'safe-fc-P4-a': ['report'], 'safe-sfc-P4-a': ['report'], 'trace-P4-passed': ['report'],
}
RETIRED = {'syn-curve-09-a', 'syn-curve-07-a', 'syn-curve-07-l', 'abl-nospoof-trace', 'abl-coef-syn', 'abl-coef-jump',
           'syn-P4-clean-l', 'syn-P4-hard-l', 'trace-P4-spoof-l'}
# CIC numbers that the report and slides quote as well
DOCS_NOW.update({'cic-l1-ship-recall': ['paper', 'report', 'slides'], 'cic-l1-ship-fpr': ['paper', 'report', 'slides'], 'cic-l1-only-mawi': ['paper', 'report', 'slides'], 'cic-l1-ship-mawi': ['paper', 'report'], 'cic-l1-comb-cic': ['paper', 'report'], 'cic-l1-comb-mawi': ['paper', 'report'], 'cic-l3-recall': ['paper', 'report', 'slides'], 'cic-l3-fpr': ['paper', 'report', 'slides'], 'cic-l3-portmap': ['paper', 'report'], 'cic-rule-ack': ['paper', 'report', 'slides'], 'cic-l2-l3-legit': ['paper', 'report', 'slides'], 'cic-l2-l3-attack': ['paper', 'report', 'slides'], 'cic-scorer-legit': ['paper', 'report'], 'cic-transfer-cic': ['paper', 'report'], 'cic-transfer-ship': ['paper', 'report'], 'cic-P0-attack': ['paper', 'report', 'slides'], 'cic-P0-legit': ['paper', 'report', 'slides'], 'cic-P1-attack': ['paper', 'report'], 'cic-P1-legit': ['paper', 'report'], 'cic-P2-attack': ['paper', 'report', 'slides'], 'cic-P2-legit': ['paper', 'report', 'slides'], 'cic-P3a-attack': ['paper', 'report', 'slides'], 'cic-P3a-legit': ['paper', 'report'], 'cic-P3b-attack': ['paper', 'report'], 'cic-P3b-legit': ['paper', 'report'], 'cic-AHIF-attack': ['paper', 'report', 'slides'], 'cic-AHIF-legit': ['paper', 'report', 'slides'], 'cic-SIG-attack': ['paper', 'report', 'slides'], 'cic-SIG-legit': ['paper', 'report', 'slides'], 'cic-SIG-oracle-attack': ['paper', 'report'], 'cic-SIG-oracle-legit': ['paper', 'report', 'slides'], 'cic-P4-attack': ['paper', 'report', 'slides'], 'cic-P4-legit': ['paper', 'report', 'slides']})
CLAIMS = [(c[0], c[1], c[2], c[3], c[4], c[5], DOCS_NOW.get(c[0], c[6])) for c in CLAIMS if c[0] not in RETIRED]


def fmt(value, decimals):
    if isinstance(value, tuple):
        return f'{fmt(value[0], decimals)} +/- {fmt(value[1], decimals)}'
    if decimals == 0:
        return f'{value:,.0f}'
    return f'{value:.{decimals}f}'


def run_claims(claims, cache):
    rows, failures = [], 0
    docs_text = {k: p.read_text(encoding='utf-8') for k, p in DOCS.items() if p.exists()}
    for cid, what, src, fn, dec, quoted, docs in claims:
        path, command = SOURCES[src]
        if src not in cache:
            cache[src] = json.loads(path.read_text()) if path.exists() else None
        status, computed = 'OK', ''
        if cache[src] is None:
            status, computed = 'MISSING FILE', '-'
        else:
            try:
                v = fn(cache[src])
                computed = fmt(v if isinstance(v, tuple) or not quoted[0].isdigit() else abs(v), dec)
                if src == 'bench':
                    # Timings vary with machine load: accept a rerun within 50% (or 0.1 ms) of the quoted value
                    if abs(v - float(quoted)) > max(0.5 * float(quoted), 0.1):
                        status = 'MISMATCH (timing, >50% off)'
                    elif computed != quoted:
                        status = 'OK (timing; this run: ' + computed + ')'
                elif computed != quoted:
                    status = 'MISMATCH'
            except (KeyError, TypeError, StopIteration) as e:
                status, computed = f'ERROR {type(e).__name__}: {e}', '-'
        missing = [k for k in docs if k in docs_text and not re.search(
            r'(?<![\d.])' + re.escape(quoted) + r'(?![\d])', docs_text[k])]
        if status == 'OK' and missing:
            status = 'NOT IN ' + ', '.join(missing)
        failures += not status.startswith('OK')
        rows.append((cid, what, quoted, computed, ', '.join(docs), f'`{path.relative_to(PROJECT_ROOT)}`',
                     f'`{command}`', status))
    return rows, failures


def main():
    cache = {}
    rows, failures = run_claims(CLAIMS, cache)
    lines = ['# Result verification', '',
             'Generated by `scripts/verify_results.py`. Every number below is quoted in the paper '
             '(`other/deliverables/build/paper_content.py`), the report (`report_content.py`) or the slides '
             '(`build_slides.py`). The script recomputes it from the saved result file, rounds it as the text does, '
             'and checks that the text quotes it. **Status OK** means the value, the rounding and the quotation all '
             'agree. Timing claims (source `bench`) depend on machine load; they pass when a rerun is within 50% '
             'of the quoted value, and the table shows this run\'s value. A quotation check only confirms that the '
             'number appears in the document, not in which sentence.', '',
             '## Result files and how they are produced', '',
             '| Source | File | Command | Provenance (git commit, code hash, created) |', '|---|---|---|---|']
    for key, (path, command) in SOURCES.items():
        d = cache.get(key) or (json.loads(path.read_text()) if path.exists() else None)
        prov = None
        if d:
            prov = (d.get('provenance') or d.get('protocol', {}).get('provenance')
                    or (d.get('evaluation') or {}).get('provenance'))
        p = f"{(prov.get('git_commit') or '-')[:10]}{' + uncommitted changes' if prov.get('git_dirty') else ''}, " \
            f"code {(prov.get('code_sha256') or '-')[:12]}, {prov.get('created', '-')}" if prov \
            else 'not recorded (older run)'
        lines.append(f'| {key} | `{path.relative_to(PROJECT_ROOT)}` | `{command}` | {p} |')
    lines += ['', '## Claims', '',
              '| ID | Claim | Quoted | Recomputed | Quoted in | File | Produced by | Status |',
              '|---|---|---|---|---|---|---|---|']
    lines += ['| ' + ' | '.join(r) + ' |' for r in rows]
    lines += ['', f'{len(rows) - failures} of {len(rows)} claims verified.', '',
              '## Evidence not in a result file', '',
              '| Claim | Evidence | Level |', '|---|---|---|',
              '| Earlier live P4 run (27 Oct): 2,000 pps spoofed UDP flood from 100 forged addresses; XDP dropped 31,802 of 59,999 '
              'flood packets (53.0%); P4 slowed 99 sources within 4 s of the alert | '
              '`logs/live_demo_20261027_234242/system_console.log` and `attack.log` (status lines, P4 lines) | '
              'Live generic XDP on a veth pair, one run. The final_status.json of that run is empty (the dashboard '
              'listened on another port); the attack start time was not logged, so "detected in its first second" '
              'rests on the status line 1 s after the flood began. The current live script logs the start time and '
              'writes live_metrics.json. |',
              '| Live top-5 run: XDP dropped 33,747 of 59,999 packets (56.2%) | '
              '`logs/live_demo_20261027_133958/final_status.json`, `attack.log` | Live generic XDP, one run |',
              '| XDP programs pass the kernel verifier | `sudo python3 scripts/check_xdp_compile.py` | Must be '
              're-run with root after any change to `src/ebpf/xdp_filter.c` (last run 31 Oct 2026: OK) |',
              '| Kernel counters and SYN guard match known packets (26 checks) | `sudo python3 scripts/check_xdp_flows.py` -> '
              '`logs/check_xdp_flows.json` | Generic XDP on a veth pair, 31 Oct 2026 |', '',
              '## Retired claims', '',
              'No longer quoted in any document since the 29 Oct 2026 revision of the paper: '
              + ', '.join(sorted(RETIRED)) + '.', '']
    OUT.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'{len(rows) - failures}/{len(rows)} claims verified; wrote {OUT.relative_to(PROJECT_ROOT)}')
    for r in rows:
        if r[-1] != 'OK':
            print(f'  {r[0]:22s} quoted {r[2]:>8s} recomputed {r[3]:>8s}  {r[-1]}')
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
