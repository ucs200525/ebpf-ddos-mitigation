"""
Evaluation harness for mitigation policies.

Detection does not depend on mitigation (the XDP program counts packets
before dropping them), so each run is replayed through the detector once and
its per-second verdicts are cached; every policy is then evaluated on the
same cached verdicts.

Metrics (aggregated over runs):
  attack_drop_pct          attack packets dropped / attack packets
  legit_loss_pct           legitimate packets dropped / legitimate packets
  legit_loss_attack_runs   same, only in runs that contain an attack (collateral damage)
  legit_loss_benign_runs   same, only in attack-free runs (damage from false alarms)
  legit_sources_hit_pct    legitimate sources that lost >= 1 packet / legitimate sources seen
  mitigated_runs_pct       attack runs where >= 50% of attack packets were dropped
                           in at least one attack second
  median_time_to_mitigate  seconds from attack onset to that first second
  attack_passed_pct        100 - attack_drop_pct (attack packets that reached the server)
  legit_retained_pct       100 - legit_loss_pct
  detected_runs_pct        attack runs the detector flagged in at least one attack second
  median_detection_delay   seconds from attack onset to the first flagged attack second
  legit_loss_post_attack   legitimate packets dropped / legitimate packets in the seconds
                           after an attack ended (runs whose attack ends before the run does)
  median_recovery_s        seconds from the end of the attack until legitimate loss in a
                           second first falls below 1% (same runs; runs that never recover
                           are excluded here and counted by recovered_runs_pct)
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

from src.ml.traffic_dataset import TrafficRun
from .policies import MitigationPolicy
from .source_scorer import source_features


@dataclass
class PreparedRun:
    scenario: str
    n_sources: int
    idx: List[np.ndarray]          # per step: dense source index of each entry
    pkts: List[np.ndarray]
    syn: List[np.ndarray]
    is_attack_src: List[np.ndarray]
    attack_second: List[bool]
    anomaly: Dict[str, List[bool]] = field(default_factory=dict)  # detector name -> verdicts
    sig: Optional[List[np.ndarray]] = None     # per step: header signature of each entry
    src_ip: Optional[np.ndarray] = None        # IPv4 address (uint32) of each dense source index
    tool_os_like: bool = False                 # attack tool reuses OS stack signatures
    # per step: share of each entry's packets in flows the level-3 classifier judged
    # attack (NaN = no flow evidence); real captures only (CIC-DDoS2019)
    flow_share: Optional[List[np.ndarray]] = None
    # per step: the same share from the flows so far in the open window (what a read of the
    # flow map at that moment gives); None for seconds in which no detector raised an alarm
    flow_now: Optional[List[Optional[np.ndarray]]] = None

    @property
    def is_attack_run(self) -> bool:
        return any(self.attack_second)


def prepare_run(run: TrafficRun, detectors: Dict[str, Callable]) -> PreparedRun:
    """Index sources densely and cache each detector's per-second verdicts"""
    all_ids = np.unique(np.concatenate([s.ip_ids for s in run.steps]))
    prepared = PreparedRun(
        scenario=run.scenario,
        n_sources=len(all_ids),
        idx=[np.searchsorted(all_ids, s.ip_ids) for s in run.steps],
        pkts=[s.ip_packets.astype(np.float64) for s in run.steps],
        syn=[s.ip_syn.astype(np.float64) for s in run.steps],
        is_attack_src=[s.ip_is_attack for s in run.steps],
        attack_second=[s.is_attack for s in run.steps],
        sig=[s.ip_sig for s in run.steps] if run.steps and run.steps[0].ip_sig is not None else None,
        src_ip=(10 << 24) | (all_ids.astype(np.int64) & 0xFFFFFF),   # matches traffic_dataset.source_ip
        tool_os_like=getattr(run, 'tool_os_like', False),
    )
    for name, make_detector in detectors.items():
        detector = make_detector()
        verdicts = []
        for step in run.steps:
            result = detector.detect_anomaly(step.stats, step.ip_stats(), now=step.timestamp)
            detector.update_baseline(step.stats, now=step.timestamp, learn=not result.is_anomaly)
            verdicts.append(bool(result.is_anomaly))
        prepared.anomaly[name] = verdicts
    return prepared


def evaluate_policy(policy: MitigationPolicy, runs: List[PreparedRun], detector: str,
                    trace: bool = False) -> Dict:
    """Replay prepared runs through a policy; returns aggregate metrics (and per-second trace)"""
    tot = dict(attack_pkts=0.0, attack_dropped=0.0, legit_pkts=0.0, legit_dropped=0.0,
               legit_pkts_attack_runs=0.0, legit_dropped_attack_runs=0.0,
               legit_pkts_benign_runs=0.0, legit_dropped_benign_runs=0.0,
               legit_sources=0, legit_sources_hit=0, attack_runs=0, mitigated_runs=0,
               detected_runs=0, legit_pkts_post=0.0, legit_dropped_post=0.0, post_attack_runs=0)
    delays, traces, detect_delays, recoveries = [], [], [], []

    for run in runs:
        policy.reset(run.n_sources)
        policy.set_addresses(run.src_ip)
        legit_seen = np.zeros(run.n_sources, dtype=bool)
        legit_hit = np.zeros(run.n_sources, dtype=bool)
        onset = next((t for t, a in enumerate(run.attack_second) if a), None)
        attack_end = (max(t for t, a in enumerate(run.attack_second) if a) + 1) if onset is not None else None
        verdicts = run.anomaly[detector]
        if onset is not None:
            first_flag = next((t for t in range(onset, len(verdicts))
                               if verdicts[t] and run.attack_second[t]), None)
            if first_flag is not None:
                tot['detected_runs'] += 1
                detect_delays.append(first_flag - onset)
        recovered_at = None
        first_mitigated = None
        run_trace = []
        kind = 'attack_runs' if run.is_attack_run else 'benign_runs'

        for t, (idx, pkts, syn, atk) in enumerate(zip(run.idx, run.pkts, run.syn, run.is_attack_src)):
            policy.observe(run.sig[t] if run.sig is not None else None, atk)
            if run.flow_share is not None:
                now = run.flow_now[t] if run.flow_now is not None else None
                policy.observe_flows(run.flow_share[t], None if now is None else (lambda pos, a=now: a[pos]))
            dropped = policy.enforce(t, idx, pkts)
            legit = ~atk
            a_p, a_d = pkts[atk].sum(), dropped[atk].sum()
            l_p, l_d = pkts[legit].sum(), dropped[legit].sum()
            tot['attack_pkts'] += a_p
            tot['attack_dropped'] += a_d
            tot['legit_pkts'] += l_p
            tot['legit_dropped'] += l_d
            tot['legit_pkts_' + kind] += l_p
            tot['legit_dropped_' + kind] += l_d
            active = legit & (pkts > 0)
            legit_seen[idx[active]] = True
            legit_hit[idx[active & (dropped >= 1)]] = True
            if (first_mitigated is None and run.attack_second[t] and a_p > 0 and a_d / a_p >= 0.5):
                first_mitigated = t
            if attack_end is not None and t >= attack_end:
                tot['legit_pkts_post'] += l_p
                tot['legit_dropped_post'] += l_d
                if recovered_at is None and l_p > 0 and l_d / l_p < 0.01:
                    recovered_at = t
            if trace:
                run_trace.append((a_p, a_d, l_p, l_d, run.anomaly[detector][t]))
            policy.update(t, idx, pkts, syn, run.anomaly[detector][t])

        tot['legit_sources'] += int(legit_seen.sum())
        tot['legit_sources_hit'] += int(legit_hit.sum())
        if run.is_attack_run:
            tot['attack_runs'] += 1
            if first_mitigated is not None:
                tot['mitigated_runs'] += 1
                delays.append(first_mitigated - onset)
            if attack_end < len(run.attack_second):
                tot['post_attack_runs'] += 1
                if recovered_at is not None:
                    recoveries.append(recovered_at - attack_end)
        if trace:
            traces.append(run_trace)

    pct = lambda a, b: 100.0 * a / b if b else 0.0  # noqa: E731
    attack_drop = pct(tot['attack_dropped'], tot['attack_pkts'])
    legit_loss = pct(tot['legit_dropped'], tot['legit_pkts'])
    result = {
        'attack_drop_pct': attack_drop,
        'legit_loss_pct': legit_loss,
        'attack_passed_pct': 100.0 - attack_drop if tot['attack_pkts'] else 0.0,
        'legit_retained_pct': 100.0 - legit_loss if tot['legit_pkts'] else 0.0,
        'legit_loss_attack_runs_pct': pct(tot['legit_dropped_attack_runs'], tot['legit_pkts_attack_runs']),
        'legit_loss_benign_runs_pct': pct(tot['legit_dropped_benign_runs'], tot['legit_pkts_benign_runs']),
        'legit_sources_hit_pct': pct(tot['legit_sources_hit'], tot['legit_sources']),
        'mitigated_runs_pct': pct(tot['mitigated_runs'], tot['attack_runs']),
        'median_time_to_mitigate_s': float(np.median(delays)) if delays else None,
        'detected_runs_pct': pct(tot['detected_runs'], tot['attack_runs']),
        'median_detection_delay_s': float(np.median(detect_delays)) if detect_delays else None,
        'legit_loss_post_attack_pct': (pct(tot['legit_dropped_post'], tot['legit_pkts_post'])
                                       if tot['legit_pkts_post'] else None),
        'median_recovery_s': float(np.median(recoveries)) if recoveries else None,
        'recovered_runs_pct': (pct(len(recoveries), tot['post_attack_runs'])
                               if tot['post_attack_runs'] else None),
        'n_runs': len(runs),
        'n_attack_runs': tot['attack_runs'],
    }
    if trace:
        result['traces'] = traces
    return result


def collect_source_samples(runs: List[PreparedRun], detector: str, per_class_per_second: int = 50,
                           seed: int = 0):
    """
    Training rows for the SourceScorer: features of active sources in
    anomalous intervals, labeled with ground truth (attack source or not).

    At most `per_class_per_second` attack and legitimate sources are sampled
    per interval, so a spoofed flood with thousands of fresh sources per
    second cannot dominate the training set. History is learned exactly as
    the policies learn it.
    """
    from .policies import MitigationPolicy
    from .source_scorer import SOURCE_FEATURES
    rng = np.random.default_rng(seed)
    X, y = [], []
    for run in runs:
        tracker = MitigationPolicy()
        tracker.reset(run.n_sources)
        for t, (idx, pkts, syn, atk) in enumerate(zip(run.idx, run.pkts, run.syn, run.is_attack_src)):
            anomaly = run.anomaly[detector][t]
            if anomaly:
                chosen = []
                for cls in (True, False):
                    pos = np.nonzero((pkts > 0) & (atk == cls))[0]
                    if len(pos) > per_class_per_second:
                        pos = rng.choice(pos, per_class_per_second, replace=False)
                    chosen.append(pos)
                pos = np.concatenate(chosen)
                if len(pos):
                    i = idx[pos]
                    X.append(source_features(tracker.hist.known(i), pkts[pos], tracker.hist.mean[i],
                                             syn[pos], tracker.hist.typical_rate(),
                                             t - np.minimum(tracker.hist.first_seen[i], t)))
                    y.append(atk[pos].astype(int))
            tracker.update(t, idx, pkts, syn, anomaly)
    if not X:
        return np.zeros((0, len(SOURCE_FEATURES))), np.zeros(0, dtype=int)
    return np.vstack(X), np.concatenate(y)
