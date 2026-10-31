#!/usr/bin/env python3
"""
Report figures and result tables, generated only from saved result files:

  data/models/ddos_classifier_report.json    detection model training/test report
  data/models/ddos_classifier_eval.json      end-to-end statistical vs hybrid evaluation
  data/results/mitigation_results.json       mitigation experiment (scripts/mitigation_experiment.py)
  data/results/trace_results.json            trace-driven experiment (scripts/trace_experiment.py), optional
  + the held-out test windows regenerated with the training seed (for the ROC curve)

Output: docs/figures/*.png (static, for the printed report) and docs/figures/RESULTS_TABLES.md
(the same numbers as tables).

Usage (from project/):  python scripts/make_figures.py
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATA_DIR, MLConfig, MODELS_DIR  # noqa: E402

OUT = PROJECT_ROOT / 'docs' / 'figures'

# Reference palette (light mode; figures are for print). Categorical slots in fixed order.
SURFACE, INK, INK2, MUTED, GRID, AXIS = '#fcfcfb', '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7'
BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'
SEQ = ['#cde2fb', '#86b6ef', '#3987e5', '#1c5cab', '#0d366b']
DIV_POS, DIV_NEG = '#e34948', '#2a78d6'   # diverging poles (attacker-like vs legitimate-like)

POLICY_LABELS = {
    'P0': 'P0 Top-5 blacklist, permanent (current)',
    'P1': 'P1 Top-5 blacklist, 30 s TTL',
    'P2': 'P2 History-based filter',
    'P3a': 'P3a Aggregate rate limit',
    'P3b': 'P3b Per-source rate limit',
    'P4': 'P4 Source scoring (proposed)',
    'AHIF': 'AHIF Adaptive history filter (Goldstein 2008)',
    'SIG': 'SIG Signature filter, trained (Dimolianis 2021)',
    'SIG-oracle': 'SIG with a perfect classifier (upper bound)',
}
SHORT = {'P0': 'P0', 'P1': 'P1', 'P2': 'P2', 'P3a': 'P3a', 'P3b': 'P3b', 'P4': 'P4',
         'AHIF': 'AHIF', 'SIG': 'SIG', 'SIG-oracle': 'SIG*'}
PUBLISHED = ('AHIF', 'SIG', 'SIG-oracle')


def style():
    plt.rcParams.update({
        'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
        'axes.edgecolor': AXIS, 'axes.labelcolor': INK2, 'axes.titlecolor': INK,
        'axes.titlesize': 11, 'axes.titleweight': 'bold', 'axes.labelsize': 9.5,
        'xtick.color': MUTED, 'ytick.color': MUTED, 'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5,
        'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6,
        'axes.spines.top': False, 'axes.spines.right': False,
        'legend.frameon': False, 'legend.fontsize': 8.5, 'legend.labelcolor': INK2,
        'font.size': 9, 'text.color': INK, 'lines.linewidth': 2,
    })


def save(fig, name, caption, captions):
    path = OUT / name
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    captions.append((name, caption))
    print('wrote', path.relative_to(PROJECT_ROOT))


def bar_style():
    # 2px surface gap between adjacent fills
    return dict(edgecolor=SURFACE, linewidth=1.2)


# ---------------------------------------------------------------- detection figures
def fig_confusion(report, captions):
    cm = np.array(report['test_metrics']['confusion_matrix'])
    fig, ax = plt.subplots(figsize=(3.8, 3.3))
    ax.grid(False)
    norm = cm / cm.sum(axis=1, keepdims=True)
    ax.imshow(norm, cmap=matplotlib.colors.ListedColormap(SEQ), vmin=0, vmax=1)
    labels = ['Benign', 'Attack']
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f'{cm[i, j]:,}\n({100 * norm[i, j]:.1f}%)', ha='center', va='center',
                    color=SURFACE if norm[i, j] > 0.5 else INK, fontsize=9.5)
    ax.set_xticks([0, 1], labels)
    ax.set_yticks([0, 1], labels)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Detection model: held-out confusion matrix', loc='left')
    save(fig, 'fig01_confusion_matrix.png',
         f"Confusion matrix of the Random Forest on {report['test_windows']:,} held-out windows "
         f"from {report['test_runs']} independent synthetic runs (row-normalized percentages).", captions)


def fig_roc(report, captions):
    from sklearn.metrics import roc_curve, roc_auc_score
    from src.ml.ml_classifier import DDoSClassifier
    from src.ml.traffic_dataset import generate_dataset
    from src.ml.model_trainer import TEST_SEED_OFFSET
    clf = DDoSClassifier()
    clf.load(MLConfig.DEFAULT_MODEL_PATH)
    data = generate_dataset(report['test_runs'], seed=report['seed'] + TEST_SEED_OFFSET)
    cols = [data['feature_names'].index(n) for n in clf.feature_names]
    proba = clf.model.predict_proba(data['X'][:, cols])[:, list(clf.model.classes_).index(1)]
    fpr, tpr, _ = roc_curve(data['y'], proba)
    auc = roc_auc_score(data['y'], proba)
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.3))
    for ax, xmax in zip(axes, (1.0, 0.05)):
        ax.plot([0, xmax], [0, xmax] if xmax == 1 else [0, xmax], color=AXIS, lw=1, ls='--')
        ax.plot(fpr, tpr, color=BLUE)
        ax.set_xlim(0, xmax)
        ax.set_ylim(0 if xmax == 1 else 0.8, 1.0)
        ax.set_xlabel('False positive rate')
        ax.set_ylabel('True positive rate')
    axes[0].set_title(f'ROC, AUC {auc:.4f}', loc='left')
    axes[1].set_title('Zoom: FPR 0-5%', loc='left')
    fig.tight_layout(w_pad=2.5)
    save(fig, 'fig02_roc_curve.png',
         f'ROC curve of the detection model on the held-out windows (seed {report["seed"] + TEST_SEED_OFFSET}); '
         f'right panel zooms into low false-positive rates.', captions)


def fig_per_scenario(report, captions):
    per = report['per_scenario']
    attack = {k: v['attack_recall'] for k, v in per.items() if v['attack_recall'] is not None}
    benign_fpr = {k: v['benign_fpr'] for k, v in per.items() if k in ('normal', 'normal_bursty', 'flash_crowd')}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.2, 3.4), gridspec_kw={'width_ratios': [3, 2]})
    names = sorted(attack, key=attack.get)
    a1.barh(names, [100 * attack[n] for n in names], color=BLUE, height=0.6, **bar_style())
    a1.set_xlim(90, 100.5)
    a1.set_xlabel('Attack windows detected (%)')
    a1.set_title('Attack scenarios: recall', loc='left')
    for i, n in enumerate(names):
        a1.text(100 * attack[n] + 0.05, i, f'{100 * attack[n]:.1f}', va='center', color=INK2, fontsize=8)
    names_b = list(benign_fpr)
    a2.barh(names_b, [100 * benign_fpr[n] for n in names_b], color=BLUE, height=0.6, **bar_style())
    a2.set_xlim(0, 1)
    a2.set_xlabel('Benign windows flagged (%)')
    a2.set_title('Benign scenarios: false positives', loc='left')
    for i, n in enumerate(names_b):
        a2.text(100 * benign_fpr[n] + 0.02, i, f'{100 * benign_fpr[n]:.2f}', va='center', color=INK2, fontsize=8)
    a1.grid(axis='y', visible=False)
    a2.grid(axis='y', visible=False)
    fig.tight_layout()
    save(fig, 'fig03_per_scenario_detection.png',
         'Per-scenario window-level results on the held-out test runs: recall on attack scenarios '
         '(left, note the axis starts at 90%) and false-positive rate on benign scenarios (right).', captions)


def fig_detector_comparison(evaluation, captions):
    def pct(s):
        a, b = s.split('/')
        return 100 * int(a) / int(b)
    stat, hyb = evaluation['statistical_detector'], evaluation['hybrid_detector']
    metrics = [('Attack seconds\ndetected', 100 * stat['attack_second_recall'], 100 * hyb['attack_second_recall']),
               ('Attack runs\ndetected', pct(stat['attack_runs_detected']), pct(hyb['attack_runs_detected'])),
               ('Benign seconds\nfalsely flagged', 100 * stat['benign_second_fpr'], 100 * hyb['benign_second_fpr']),
               ('Benign runs with\n>=1 false alarm', pct(stat['benign_runs_with_false_alarm']),
                pct(hyb['benign_runs_with_false_alarm']))]
    x = np.arange(len(metrics))
    w = 0.36
    fig, ax = plt.subplots(figsize=(7.4, 3.5))
    for off, idx, color, label in ((-w / 2, 1, ORANGE, 'Statistical rules only'),
                                   (w / 2, 2, BLUE, 'Hybrid (rules + ML)')):
        vals = [m[idx] for m in metrics]
        ax.bar(x + off, vals, w, color=color, label=label, **bar_style())
        for xi, v in zip(x + off, vals):
            ax.text(xi, v + 1.5, f'{v:.1f}', ha='center', color=INK2, fontsize=8)
    ax.set_xticks(x, [m[0] for m in metrics])
    ax.set_ylabel('%')
    ax.set_ylim(0, 108)
    ax.grid(axis='x', visible=False)
    ax.legend(loc='upper center', ncol=2, bbox_to_anchor=(0.5, -0.16))
    ax.set_title('End-to-end detection: statistical vs hybrid', loc='left')
    save(fig, 'fig04_detector_comparison.png',
         'Second-by-second replay of fresh synthetic runs (seed 3031) through the statistical-only and hybrid '
         'detectors. Higher is better for the left two groups, lower for the right two.', captions)


def fig_feature_importance(report_model_path, captions):
    from src.ml.ml_classifier import DDoSClassifier
    clf = DDoSClassifier()
    clf.load(report_model_path)
    imp = clf.get_feature_importance(len(clf.feature_names))
    names = list(imp)[::-1]
    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    ax.barh(names, [100 * imp[n] for n in names], color=BLUE, height=0.65, **bar_style())
    for i, n in enumerate(names):
        ax.text(100 * imp[n] + 0.3, i, f'{100 * imp[n]:.1f}', va='center', color=INK2, fontsize=7.5)
    ax.set_xlabel('Importance (%)')
    ax.grid(axis='y', visible=False)
    ax.set_title('Detection model: feature importance', loc='left')
    save(fig, 'fig05_feature_importance.png',
         'Random Forest impurity-based feature importance for the 16 window-level features.', captions)


# ---------------------------------------------------------------- mitigation figures
def fig_tradeoff(res, captions, setting='realistic'):
    entry = res['settings'][setting]
    pol, curve = entry['policies'], entry['curve']
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    cx = np.array([c['legit_loss_pct']['mean'] for c in curve])
    cy = np.array([c['attack_drop_pct']['mean'] for c in curve])
    # Pareto front of the P4 threshold sweep
    order = np.argsort(cx)
    front_x, front_y, best = [], [], -1
    for i in order:
        if cy[i] > best:
            best = cy[i]
            front_x.append(cx[i])
            front_y.append(cy[i])
    ax.scatter(cx, cy, s=14, color=BLUE, alpha=0.35, linewidths=0, label='P4 threshold settings')
    ax.plot(front_x, front_y, color=BLUE, lw=1.5, alpha=0.8)
    for key in ('P0', 'P1', 'P2', 'P3a', 'P3b', 'P4') + tuple(k for k in PUBLISHED if k in pol):
        p = pol[key]
        x, y = p['legit_loss_pct'], p['attack_drop_pct']
        color = BLUE if key == 'P4' else (ORANGE if key == 'P0' else (AQUA if key in PUBLISHED else MUTED))
        marker = 'D' if key in PUBLISHED else 'o'
        face = SURFACE if key == 'SIG-oracle' else color      # hollow: upper bound, not a real system
        ax.errorbar(x['mean'], y['mean'], xerr=x['std'], yerr=y['std'], fmt=marker, ms=7 if marker == 'D' else 8,
                    color=color, mfc=face, mec=color if key == 'SIG-oracle' else SURFACE, mew=2,
                    ecolor=color, elinewidth=1, capsize=0, zorder=5)
        ax.annotate(SHORT[key], (x['mean'], y['mean']), xytext=(7, 4), textcoords='offset points',
                    color=INK, fontsize=9, fontweight='bold' if key in ('P0', 'P4') else 'normal')
    ax.set_xlabel('Legitimate packets dropped (%)   (lower is better)')
    ax.set_ylabel('Attack packets dropped (%)   (higher is better)')
    ax.set_xlim(left=0)
    ax.set_ylim(0, 100)
    ax.set_title('Mitigation trade-off: attack stopped vs collateral damage', loc='left')
    handles = [plt.Line2D([], [], marker='o', ls='', color=ORANGE, ms=7, label='P0: current system'),
               plt.Line2D([], [], marker='o', ls='', color=MUTED, ms=7, label='Baselines P1-P3'),
               plt.Line2D([], [], marker='D', ls='', color=AQUA, ms=6, label='Published: AHIF, SIG (SIG* = perfect classifier)'),
               plt.Line2D([], [], marker='o', ls='', color=BLUE, ms=7, label='P4 proposed (chosen point)'),
               plt.Line2D([], [], marker='o', ls='-', color=BLUE, alpha=0.5, ms=4, label='P4 threshold sweep')]
    ax.legend(handles=handles, loc='lower right')
    n = len(res['protocol']['test_seeds'])
    save(fig, 'fig06_mitigation_tradeoff.png',
         f'Attack traffic dropped versus legitimate traffic dropped for each mitigation policy '
         f'("{setting}" setting, hybrid detector, mean ± std over {n} test seeds x '
         f'{res["protocol"]["runs_per_seed"]} runs). Up-and-left is better. The light blue points are '
         f'other P4 thresholds; the line is their Pareto front. Diamonds are published baselines: AHIF '
         f'(Goldstein et al. 2008) and signature filtering (Dimolianis et al. 2021); the hollow SIG* gives the '
         f'signature filter a perfect classifier and is an upper bound, not a deployable system.', captions)


def fig_trace(trace, captions):
    """Trace-driven results: real legitimate traffic (MAWI) with injected attacks"""
    keys = [k for k in ('P0', 'P2', 'P3a', 'P3b', 'AHIF', 'SIG', 'SIG-oracle', 'P4') if k in trace['policies']]
    labels = [SHORT[k] for k in keys]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4))
    for ax, metric, title in ((axes[0], 'attack_drop_pct', 'Attack packets dropped (%)'),
                              (axes[1], 'legit_loss_pct', 'Legitimate packets dropped (%)')):
        vals = [trace['policies'][k][metric] for k in keys]
        colors = [BLUE if k == 'P4' else ORANGE if k == 'P0' else AQUA if k in PUBLISHED else MUTED for k in keys]
        bars = ax.bar(np.arange(len(keys)), [v['mean'] for v in vals], 0.62, yerr=[v['std'] for v in vals],
                      color=colors, error_kw=dict(ecolor=INK2, elinewidth=0.8), **bar_style())
        if 'SIG-oracle' in keys:
            bars[keys.index('SIG-oracle')].set_hatch('//')
        for i, v in enumerate(vals):
            ax.text(i, v['mean'] + v['std'] + 0.6, f"{v['mean']:.1f}", ha='center', color=INK2, fontsize=7.5)
        ax.set_xticks(np.arange(len(keys)), labels)
        ax.set_title(title, loc='left')
        ax.grid(axis='x', visible=False)
    axes[0].set_ylim(0, 110)
    fig.tight_layout()
    p = trace['protocol']
    save(fig, 'fig11_trace_driven.png',
         f"Trace-driven evaluation: real legitimate traffic from {p['trace']} toward {p['victims']} busy "
         f"destinations, with injected attacks; mean ± std over {len(p['injection_seeds'])} injection seeds x "
         f"{p['runs_per_seed']} runs. Nothing was tuned on the trace. SIG* (hatched) uses a perfect "
         f"signature classifier and is an upper bound.", captions)


def fig_settings(res, captions):
    keys = ['P0', 'P1', 'P2', 'P3a', 'P3b', 'P4']
    settings = list(res['settings'])
    colors = [BLUE, ORANGE, AQUA]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
    x = np.arange(len(keys))
    w = 0.26
    for ax, metric, title in ((axes[0], 'attack_drop_pct', 'Attack packets dropped (%)'),
                              (axes[1], 'legit_loss_pct', 'Legitimate packets dropped (%)')):
        for j, (setting, color) in enumerate(zip(settings, colors)):
            vals = [res['settings'][setting]['policies'][k][metric] for k in keys]
            ax.bar(x + (j - 1) * w, [v['mean'] for v in vals], w, yerr=[v['std'] for v in vals],
                   color=color, label=setting, error_kw=dict(ecolor=INK2, elinewidth=0.8), **bar_style())
        ax.set_xticks(x, keys)
        ax.set_title(title, loc='left')
        ax.grid(axis='x', visible=False)
    axes[0].set_ylim(0, 105)
    axes[1].legend(title='Traffic realism', title_fontsize=8.5, loc='upper right')
    fig.tight_layout()
    save(fig, 'fig07_mitigation_by_setting.png',
         'Mitigation policies under three traffic-realism settings (clean: no client churn, no pre-attack '
         'bot traffic; realistic: 10% churn, 30% of bots active beforehand; hard: 20% / 60%). '
         'Bars are means, whiskers ± std over the test seeds. Exact values are in RESULTS_TABLES.md.', captions)


def fig_groups(res, captions, setting='realistic'):
    groups = res['settings'][setting]['groups']
    names = list(groups['P0'])
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
    y = np.arange(len(names))
    h = 0.36
    for ax, metric, title in ((axes[0], 'attack_drop_pct', 'Attack packets dropped (%)'),
                              (axes[1], 'legit_loss_pct', 'Legitimate packets dropped (%)')):
        for off, key, color, label in ((-h / 2, 'P0', ORANGE, 'P0 current'), (h / 2, 'P4', BLUE, 'P4 proposed')):
            vals = [(groups[key][n][metric] or {'mean': 0.0})['mean'] for n in names]
            ax.barh(y + off, vals, h, color=color, label=label, **bar_style())
            for yi, v in zip(y + off, vals):
                ax.text(v + 0.8, yi, f'{v:.1f}', va='center', color=INK2, fontsize=7.5)
        ax.set_yticks(y, names)
        ax.invert_yaxis()
        ax.set_title(title, loc='left')
        ax.grid(axis='y', visible=False)
    axes[1].set_yticklabels([])
    axes[0].set_xlim(0, 110)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.legend(handles, labels, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.0))
    save(fig, 'fig08_mitigation_by_scenario.png',
         f'Current rule (P0) versus proposed source scoring (P4) per scenario group ("{setting}" setting, mean '
         f'over test seeds). Benign groups contain no attack packets, so only legitimate loss applies there.',
         captions)


def fig_example(res, captions, setting='realistic'):
    ex = res['settings'][setting].get('example')
    if not ex:
        return
    keys = [k for k in ('P0', 'P2', 'P4') if k in ex['series']]
    fig, axes = plt.subplots(2, len(keys), figsize=(10, 4.8), sharex=True)
    for col, key in enumerate(keys):
        s = np.array(ex['series'][key], dtype=float)
        t = np.arange(len(s))
        detect = s[:, 4] > 0
        for row, (offered, dropped, label) in enumerate(((0, 1, 'Attack'), (2, 3, 'Legitimate'))):
            ax = axes[row, col]
            for ti in t[detect]:
                ax.axvspan(ti - 0.5, ti + 0.5, color=GRID, lw=0, zorder=0)
            ax.plot(t, s[:, offered], color=MUTED, lw=1.2, ls='--', label='offered')
            ax.plot(t, s[:, offered] - s[:, dropped], color=ORANGE if row == 0 else BLUE, label='admitted')
            ax.set_ylim(bottom=0)
            if col == 0:
                ax.set_ylabel(f'{label} pkts/s')
            if row == 0:
                ax.set_title(POLICY_LABELS[key].split(' (')[0], loc='left', fontsize=9.5)
            if row == 1:
                ax.set_xlabel('Time (s)')
    handles = [plt.Line2D([], [], color=MUTED, lw=1.2, ls='--', label='offered'),
               plt.Line2D([], [], color=ORANGE, label='attack admitted'),
               plt.Line2D([], [], color=BLUE, label='legitimate admitted'),
               matplotlib.patches.Patch(color=GRID, label='detector flagged')]
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.legend(handles=handles, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1.0))
    save(fig, 'fig09_example_timeline.png',
         f'One "{ex["scenario"]}" test run ({setting} setting): offered vs admitted attack traffic (top) and '
         f'legitimate traffic (bottom) under three policies. Gray bands mark seconds the detector flagged.',
         captions)


def fig_scorer(res, captions):
    coef = res['scorer']['coefficients']
    names = sorted(coef, key=coef.get)
    vals = [coef[n] for n in names]
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    ax.barh(names, vals, color=[DIV_POS if v > 0 else DIV_NEG for v in vals], height=0.6, **bar_style())
    ax.axvline(0, color=AXIS, lw=1)
    for i, v in enumerate(vals):
        ax.text(v + (0.08 if v >= 0 else -0.08), i, f'{v:+.2f}', va='center',
                ha='left' if v >= 0 else 'right', color=INK2, fontsize=8)
    lim = max(abs(min(vals)), abs(max(vals))) * 1.35
    ax.set_xlim(-lim, lim)
    ax.set_xlabel('Standardized coefficient  (left: legitimate-like, right: attacker-like)')
    ax.grid(axis='y', visible=False)
    ax.set_title('Source scorer: what marks a source as an attacker', loc='left')
    save(fig, 'fig10_source_scorer_coefficients.png',
         'Logistic-regression coefficients of the per-source scorer (features standardized). '
         'Trained on tuning seed 900 only.', captions)


# ---------------------------------------------------------------- tables
def tables(report, evaluation, res, trace=None, safety=None, bench=None):
    lines = ['# Result tables', '',
             'Generated by `scripts/make_figures.py` from the saved result files. Data is synthetic '
             '(`src/ml/traffic_dataset.py`) except the trace-driven section at the end, whose legitimate '
             'traffic is a real MAWI trace.', '']
    tm = report['test_metrics']
    lines += ['## Detection model (held-out test windows)', '',
              '| Metric | Value |', '|---|---|'] + \
             [f'| {k} | {tm[k]:.4f} |' for k in ('accuracy', 'precision', 'recall', 'f1', 'roc_auc')] + \
             [f"| windows (train / test) | {report['train_windows']:,} / {report['test_windows']:,} |",
              f"| chosen hyperparameters | {report['chosen_params']['n_estimators']} trees, max_depth "
              f"{report['chosen_params']['max_depth']}, min_samples_leaf {report['chosen_params']['min_samples_leaf']} |", '']
    lines += ['## End-to-end detection (seed 3031)', '', '| Metric | Statistical only | Hybrid |', '|---|---|---|']
    s, h = evaluation['statistical_detector'], evaluation['hybrid_detector']
    for k, label in (('attack_second_recall', 'Attack seconds detected'),
                     ('benign_second_fpr', 'Benign seconds falsely flagged')):
        lines.append(f'| {label} | {100 * s[k]:.1f}% | {100 * h[k]:.1f}% |')
    for k, label in (('attack_runs_detected', 'Attack runs detected'),
                     ('benign_runs_with_false_alarm', 'Benign runs with >=1 false alarm'),
                     ('median_detection_delay_s', 'Median detection delay (s)')):
        lines.append(f'| {label} | {s[k]} | {h[k]} |')
    lines.append('')
    p = res['protocol']
    lines += ['## Mitigation policies', '',
              f"Mean ± std over test seeds {p['test_seeds']} x {p['runs_per_seed']} runs. "
              f"P4 thresholds chosen on seed {p['tune_seeds']['threshold_selection']} "
              f"(objective: {p['objective']}): t_block={p['chosen_params']['t_block']}, "
              f"t_limit={p['chosen_params']['t_limit']}. Scorer trained on seed {p['tune_seeds']['scorer_training']}.", '']
    if p.get('design_history'):
        lines += [f"> **Design history.** {p['design_history']}", '']
    fmt = lambda v: '—' if v is None else f"{v['mean']:.2f} ± {v['std']:.2f}"  # noqa: E731
    for setting, entry in res['settings'].items():
        lines += [f'### Setting: {setting} ({p["settings"][setting]})', '',
                  '| Policy | Detector | Attack dropped % | Legit dropped % | Legit dropped % (attack runs) | '
                  'Legit dropped % (benign runs) | Legit sources hit % | Attack runs mitigated % | Median time to mitigate (s) |',
                  '|---|---|---|---|---|---|---|---|---|']
        for key, r in entry['policies'].items():
            lines.append(f"| {key}: {r['name']} | {r['detector']} | {fmt(r['attack_drop_pct'])} | "
                         f"{fmt(r['legit_loss_pct'])} | {fmt(r['legit_loss_attack_runs_pct'])} | "
                         f"{fmt(r['legit_loss_benign_runs_pct'])} | {fmt(r['legit_sources_hit_pct'])} | "
                         f"{fmt(r['mitigated_runs_pct'])} | {fmt(r['median_time_to_mitigate_s'])} |")
        lines.append('')
    if 'profiles' in res['settings'].get('realistic', {}):
        lines += ['### Attack runs by attack header profile (realistic setting)', '',
                  'Crafted: the flood tool sets header values no client stack uses. OS-stack: attack packets carry '
                  'ordinary operating-system values, so they can share signatures with legitimate clients.', '',
                  '| Policy | Crafted: attack dropped % | Crafted: legit dropped % | OS-stack: attack dropped % | '
                  'OS-stack: legit dropped % |', '|---|---|---|---|---|']
        for key, prof in res['settings']['realistic']['profiles'].items():
            c, o = prof.get('crafted headers'), prof.get('OS-stack headers')
            lines.append(f"| {key} | {fmt(c['attack_drop_pct'])} | {fmt(c['legit_loss_attack_runs_pct'])} | "
                         f"{fmt(o['attack_drop_pct'])} | {fmt(o['legit_loss_attack_runs_pct'])} |")
        lines.append('')
    entry = res['settings']['realistic']['policies']
    if 'attack_passed_pct' in entry['P4']:
        lines += ['### Extended metrics (realistic setting)', '',
                  'Attack passed = attack packets that reached the server. Detection delay: attack onset to the first '
                  'flagged attack second (same for every policy using the hybrid detector). Mitigation delay: onset to '
                  'the first second with >= 50% of attack packets dropped. Post-attack loss and recovery: runs whose '
                  'attack ends before the run does; recovery = seconds until legitimate loss in a second falls below 1%.',
                  '', '| Policy | Attack passed % | Legit retained % | Attack runs detected % | Detection delay (s) | '
                  'Mitigation delay (s) | Legit loss after attack % | Recovery (s) | Runs recovered % |',
                  '|---|---|---|---|---|---|---|---|---|']
        for key in ('P0', 'P1', 'P2', 'P3a', 'P3b', 'AHIF', 'SIG', 'SIG-oracle', 'P4'):
            r = entry[key]
            lines.append(f"| {key} | {fmt(r['attack_passed_pct'])} | {fmt(r['legit_retained_pct'])} | "
                         f"{fmt(r['detected_runs_pct'])} | {fmt(r['median_detection_delay_s'])} | "
                         f"{fmt(r['median_time_to_mitigate_s'])} | {fmt(r['legit_loss_post_attack_pct'])} | "
                         f"{fmt(r['median_recovery_s'])} | {fmt(r['recovered_runs_pct'])} |")
        lines.append('')
        lines += ['### Ablation (realistic setting; every variant uses the tuned thresholds)', '',
                  '| Variant | Attack dropped % | Legit dropped % | Legit sources hit % |', '|---|---|---|---|']
        for key in ('P4', 'P4-no-spoof-guard', 'P4-block-only', 'P4-no-expiry', 'P4-no-history', 'P0'):
            if key in entry:
                r = entry[key]
                lines.append(f"| {key}: {r['name']} | {fmt(r['attack_drop_pct'])} | {fmt(r['legit_loss_pct'])} | "
                             f"{fmt(r['legit_sources_hit_pct'])} |")
        lines.append('')
    lines += ['### Source scorer coefficients (standardized)', '', '| Feature | Coefficient |', '|---|---|'] + \
             [f'| {k} | {v:+.3f} |' for k, v in res['scorer']['coefficients'].items()] + ['']
    if trace:
        tp = trace['protocol']
        lines += ['## Trace-driven evaluation (real legitimate traffic)', '',
                  f"{tp['trace']}: {tp['victims']} destinations x {tp['runs_per_seed'] // tp['victims']} windows of "
                  f"{tp['window_s']} s per injection seed ({tp['attack_runs_per_seed']} attack runs), seeds "
                  f"{tp['injection_seeds']}. {tp['note']}.", '',
                  f"Detection on the trace: attack runs detected (hybrid) "
                  f"{fmt(trace['detected_attack_runs_pct']['hybrid'])}%; seconds falsely flagged in attack-free runs "
                  f"(hybrid) {fmt(trace['false_alarm_seconds_pct']['hybrid'])}%.", '',
                  '| Policy | Attack dropped % | Legit dropped % | Legit dropped % (attack runs) | '
                  'Legit dropped % (benign runs) | Legit sources hit % |', '|---|---|---|---|---|---|']
        for key, r in trace['policies'].items():
            lines.append(f"| {key}: {r['name']} | {fmt(r['attack_drop_pct'])} | {fmt(r['legit_loss_pct'])} | "
                         f"{fmt(r['legit_loss_attack_runs_pct'])} | {fmt(r['legit_loss_benign_runs_pct'])} | "
                         f"{fmt(r['legit_sources_hit_pct'])} |")
        lines.append('')
    if safety:
        sp = safety['protocol']
        lines += ['## Mitigation safety (legitimate traffic that looks like an attack)', '',
                  f"Synthetic, {sp['setting']}; seeds {sp['seeds']} x {sp['runs_per_scenario_per_seed']} runs per "
                  f"scenario. No run was used for training or tuning.", '']
        for scenario, e in safety['scenarios'].items():
            det = e['detector']['hybrid']
            lines += [f"### {scenario}", '',
                      f"Hybrid detector flags {fmt(det['flagged_seconds_pct'])}% of all seconds and "
                      f"{fmt(det['benign_seconds_flagged_pct'])}% of attack-free seconds.", '',
                      '| Policy | Attack dropped % | Legit dropped % | Legit sources hit % | '
                      'Legit loss after attack % | Recovery (s) |', '|---|---|---|---|---|---|']
            for key, r in e['policies'].items():
                lines.append(f"| {key} | {fmt(r['attack_drop_pct']) if r['n_attack_runs']['mean'] else '—'} | "
                             f"{fmt(r['legit_loss_pct'])} | {fmt(r['legit_sources_hit_pct'])} | "
                             f"{fmt(r['legit_loss_post_attack_pct'])} | {fmt(r['median_recovery_s'])} |")
            lines.append('')
    if bench:
        lines += ['## Control-plane cost per monitoring interval', '', bench['what'] + '.', '',
                  f"Sources per interval: mean {bench['per_interval_sources']['mean']:.0f}, max "
                  f"{bench['per_interval_sources']['max']} (capped at 1000). Machine: "
                  f"{bench['provenance']['platform']}, {bench['provenance']['cpu_count']} logical CPUs.", '',
                  f"Each configuration was timed {len(next(iter(bench['results'].values()))['repeats'])} times; "
                  'values are the range (min-max) over the repetitions.', '',
                  '| Detector + policy | Intervals | Mean ms | p50 ms | p99 ms |', '|---|---|---|---|---|']
        rng = lambda r, m: f"{r[m + '_min']:.1f}-{r[m + '_max']:.1f}"  # noqa: E731
        for key, r in bench['results'].items():
            lines.append(f"| {key} | {r['intervals']:,} | {rng(r, 'mean_ms')} | {rng(r, 'p50_ms')} | "
                         f"{rng(r, 'p99_ms')} |")
        lines.append('')
    return '\n'.join(lines)


def cic_tables(cic):
    """Tables for the real-attack experiment (data/results/cic_results.json)"""
    pct = lambda v: '—' if v is None else f'{100 * v:.1f}%'  # noqa: E731
    p2 = lambda v: '—' if v is None else f'{100 * v:.2f}%'  # noqa: E731
    lines = ['## Real attacks: CIC-DDoS2019 (train day 01-12, test day 03-11)', '',
             cic['protocol']['labels'] + '.', '', 'Limitation: ' + cic['protocol']['limitation'] + '.', '']
    l1 = cic['level1']
    lines += ['### Level 1: per-second window model', '',
              'Window model alone at a 0.5 cut:', '',
              '| Model | Test set | Windows (attack) | Recall | False-positive rate |', '|---|---|---|---|---|']
    c, sh, cb = l1['cic_model']['test'], l1['shipped_model']['03-11'], l1['combined_model']['test']
    lines.append(f"| CIC-only | CIC 03-11 | {c['n']:,} ({c['n_attack']:,}) | {p2(c['recall'])} | "
                 f"{p2(c['false_positive_rate'])} |")
    lines.append(f"| combined (synthetic + CIC 01-12) | CIC 03-11 | {cb['n']:,} ({cb['n_attack']:,}) | "
                 f"{p2(cb['recall'])} | {p2(cb['false_positive_rate'])} |")
    lines.append(f"| shipped (synthetic-trained) | CIC 03-11 | {sh['n']:,} ({sh['n_attack']:,}) | "
                 f"{p2(sh['recall'])} | {p2(sh['false_positive_rate'])} |")
    for name, label in (('mawi_benign', 'MAWI, benign only'), ('mawi_injected_attacks', 'MAWI + injected attacks'),
                        ('synthetic_test', 'synthetic test runs')):
        r = l1['cross_dataset'][name]
        for m, ml in (('cic_model', 'CIC-only'), ('shipped_model', 'shipped (synthetic-trained)'),
                      ('combined_model', 'combined (synthetic + CIC 01-12)')):
            lines.append(f"| {ml} | {label} | {r['windows']:,} ({r['attack_windows']:,}) | "
                         f"{p2(r[m]['recall'])} | {p2(r[m]['false_positive_rate'])} |")
    lines += ['', 'Deployed detectors, second by second over the whole test day:', '',
              '| Detector | Attack seconds flagged | Benign seconds flagged | Episodes detected | '
              'Median delay (s) |', '|---|---|---|---|---|']
    for name, r in l1['end_to_end'].items():
        lines.append(f"| {name} | {p2(r['attack_second_recall'])} | {p2(r['benign_second_fpr'])} | "
                     f"{r['episodes_detected']} | {r['median_detection_delay_s']} |")
    lines.append('')

    l3 = cic['level3']
    t, tp = l3['test_flows'], l3['test_packets']
    lines += ['### Level 3: per-flow classifier (10-s inbound flows, 16 counter features)', '',
              f"Test day flows: {t['n']:,} ({t['n_benign']:,} benign, {t['n_attack']:,} attack; attack flows "
              f"sampled at 1/{round(1 / l3['attack_flow_sampling_rate'])}). Attack types not in training: "
              f"{', '.join(l3['unseen_in_training']) or 'none'}.", '',
              '| Classifier | Attack-flow recall | Benign-flow FPR | Attack packets caught | '
              'Benign packets flagged |', '|---|---|---|---|---|',
              f"| random forest | {p2(t['recall'])} | {p2(t['false_positive_rate'])} | "
              f"{tp['attack_packets_in_flagged_flows_pct']:.2f}% | {tp['benign_packets_in_flagged_flows_pct']:.2f}% |"]
    for name, r in l3['rules'].items():
        lines.append(f"| rule: {name.replace('_', ' ')} | {p2(r['test_flows']['recall'])} | "
                     f"{p2(r['test_flows']['false_positive_rate'])} | — | — |")
    lines += ['', '| Attack type | Flows | Recall (forest) |', '|---|---|---|']
    for k, r in l3['test_per_attack_type'].items():
        lines.append(f"| {k} | {r['n']:,} | {pct(r['recall'])} |")
    act = l3['test_recall_by_activity']
    ins, out = act['inside_attack_episodes'], act['attacker_background_outside_episodes']
    lines += ['', f"Recall on attack-machine flows inside attack episodes: {pct(ins['recall'])} "
                  f"({ins['flows']:,} flows); on its background flows outside them: {pct(out['recall'])} "
                  f"({out['flows']:,} flows).", '']

    l2 = cic['level2']
    lines += ['### Level 2 + mitigation on real traffic (test day 03-11)', '',
              f"{len(l2['runs']['03-11'])} runs (attack episodes and 5-minute quiet stretches); "
              f"{l2['test_attack_packets']:,.0f} attack and {l2['test_legit_packets']:,.0f} legitimate packets; "
              f"{l2['test_attack_sources']} attack source address(es).", '',
              '| System | Policy | Attack dropped | Legit dropped | Legit sources hit |', '|---|---|---|---|---|']
    for system, sysres in l2['systems'].items():
        for key, r in sysres['policies'].items():
            a = r['all']
            lines.append(f"| {system} | {key} | {a['attack_drop_pct']:.2f}% | {a['legit_loss_pct']:.2f}% | "
                         f"{a['legit_sources_hit_pct']:.1f}% |")
    tr = l2.get('transfer_to_synthetic')
    if tr:
        lines += ['', 'Does the CIC-trained scorer transfer? P4 on synthetic traffic, shipped detector:', '',
                  '| Traffic | Scorer | Attack dropped | Legit dropped |', '|---|---|---|---|']
        for name, res in tr.items():
            if isinstance(res, dict) and name != 'runs':
                for label, r in res.items():
                    lines.append(f"| {name} | {label} | {r['attack_drop_pct']:.1f}% | {r['legit_loss_pct']:.1f}% |")
    lines.append('')
    return '\n'.join(lines)


def fig_safety(safety, captions):
    """Legitimate loss per policy in the safety scenarios"""
    keys = [k for k in ('P0', 'P2', 'P3a', 'P3b', 'AHIF', 'SIG', 'P4') if k in safety['scenarios']['heavy_legit']['policies']]
    scen = list(safety['scenarios'])
    fig, ax = plt.subplots(figsize=(9.2, 3.6))
    w = 0.8 / len(scen)
    colors = [BLUE, ORANGE, AQUA, MUTED]
    x = np.arange(len(keys))
    for j, sc in enumerate(scen):
        vals = [safety['scenarios'][sc]['policies'][k]['legit_loss_pct'] for k in keys]
        ax.bar(x + (j - (len(scen) - 1) / 2) * w, [v['mean'] for v in vals], w, yerr=[v['std'] for v in vals],
               color=colors[j % len(colors)], label=sc.replace('_', ' '),
               error_kw=dict(ecolor=INK2, elinewidth=0.8), **bar_style())
    ax.set_xticks(x, [SHORT[k] for k in keys])
    ax.set_ylabel('Legitimate packets dropped (%)')
    ax.grid(axis='x', visible=False)
    ax.legend(ncol=4, loc='upper center', bbox_to_anchor=(0.5, -0.1))
    ax.set_title('Mitigation safety: legitimate loss when real users look like attackers', loc='left')
    sp = safety['protocol']
    save(fig, 'fig12_safety_scenarios.png',
         f"Legitimate traffic dropped in the safety scenarios (synthetic, realistic setting, mean ± std over "
         f"{len(sp['seeds'])} seeds x {sp['runs_per_scenario_per_seed']} runs per scenario; no run used for tuning).",
         captions)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    style()
    report = json.loads((MODELS_DIR / 'ddos_classifier_report.json').read_text())
    evaluation = json.loads((MODELS_DIR / 'ddos_classifier_eval.json').read_text())
    res_path = DATA_DIR / 'results' / 'mitigation_results.json'
    res = json.loads(res_path.read_text())
    captions = []
    fig_confusion(report, captions)
    fig_roc(report, captions)
    fig_per_scenario(report, captions)
    fig_detector_comparison(evaluation, captions)
    fig_feature_importance(MLConfig.DEFAULT_MODEL_PATH, captions)
    fig_tradeoff(res, captions)
    fig_settings(res, captions)
    fig_groups(res, captions)
    fig_example(res, captions)
    fig_scorer(res, captions)
    trace_path = DATA_DIR / 'results' / 'trace_results.json'
    trace = json.loads(trace_path.read_text()) if trace_path.exists() else None
    if trace:
        fig_trace(trace, captions)
    load = lambda p: json.loads(p.read_text()) if p.exists() else None  # noqa: E731
    safety = load(DATA_DIR / 'results' / 'safety_results.json')
    bench = load(DATA_DIR / 'results' / 'benchmark_results.json')
    if safety:
        fig_safety(safety, captions)
    text = tables(report, evaluation, res, trace, safety, bench)
    cic = load(DATA_DIR / 'results' / 'cic_results.json')
    if cic and all(k in cic for k in ('level1', 'level2', 'level3')):
        text += '\n' + cic_tables(cic)
    (OUT / 'RESULTS_TABLES.md').write_text(text, encoding='utf-8')
    index = ['# Figures', '', 'Generated by `scripts/make_figures.py`. Captions are suggestions for the report.', '']
    index += [f'- **{name}** — {caption}' for name, caption in captions]
    (OUT / 'README.md').write_text('\n'.join(index) + '\n', encoding='utf-8')
    print('wrote docs/figures/RESULTS_TABLES.md and docs/figures/README.md')


if __name__ == '__main__':
    main()
