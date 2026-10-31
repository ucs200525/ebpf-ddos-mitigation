"""
Live source-scoring mitigation (P4) for the running system.

Runs the same SourceScoringPolicy that the mitigation experiment evaluates,
on the per-source counters of each monitoring interval, and turns its
decisions into kernel map entries through the TrafficMonitor:

  block for TTL        -> blacklist_map  (expiry time; XDP drops, then deletes when expired)
  slow down for TTL    -> ratelimit_map  (per-source token bucket in XDP)
  known sources        -> known_map      (sources seen in >= 2 normal intervals)
  spoofing guard       -> guard_map      (unknown sources share what is left of the budget)

With a flow classifier (LiveFlowEvidence, level 3) the policy is P4+L3
(FlowVerifiedScoring): every 10 s the flow map is read and classified, and
each source's share of packets in attack-like flows confirms or vetoes its
per-source score. Before acting on a source that is not blocked yet, the map is also
read without clearing it, so that a burst shorter than the window can be judged by its
flows so far.

The policy keeps dense per-source arrays; this wrapper maps IP addresses to
indices. When the index space is full it starts over (history is relearned).
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .policies import FlowVerifiedScoring, SourceScoringPolicy
from .source_scorer import SourceScorer

logger = logging.getLogger(__name__)

DEFAULT_THRESHOLDS = {'t_block': 0.8, 't_limit': 0.7}


def tuned_thresholds(results_path: Path) -> Dict[str, float]:
    """Thresholds chosen on the tuning seed by scripts/mitigation_experiment.py"""
    try:
        params = json.loads(Path(results_path).read_text())['protocol']['chosen_params']
        return {'t_block': float(params['t_block']), 't_limit': float(params['t_limit'])}
    except (OSError, KeyError, ValueError, TypeError):
        return dict(DEFAULT_THRESHOLDS)


class LiveSourceScoring:
    """Applies P4 decisions to the kernel (or the simulation) once per interval"""

    name = 'P4 source scoring'

    def __init__(self, scorer: SourceScorer, t_block: float, t_limit: float, ttl: float = 30.0,
                 margin: float = 1.5, capacity: int = 1 << 18, flow_verified: bool = False,
                 on_legit_flows: str = 'limit'):
        if flow_verified:
            self.policy = FlowVerifiedScoring(scorer, t_block=t_block, t_limit=t_limit, ttl=ttl, margin=margin,
                                              on_legit_flows=on_legit_flows)
        else:
            self.policy = SourceScoringPolicy(scorer, t_block=t_block, t_limit=t_limit, ttl=ttl, margin=margin)
        if flow_verified:
            self.name = 'P4+L3 source scoring verified by flows'
        self.capacity = capacity
        self.ttl = ttl
        self._reset_index()

    @classmethod
    def load(cls, scorer_path: Path, results_path: Path, thresholds: Optional[Dict[str, float]] = None,
             **kwargs) -> Optional['LiveSourceScoring']:
        if not Path(scorer_path).exists():
            logger.warning(f"Source scorer not found ({scorer_path}); P4 unavailable")
            return None
        return cls(SourceScorer.load(str(scorer_path)), **(thresholds or tuned_thresholds(results_path)), **kwargs)

    def _reset_index(self):
        self.policy.reset(self.capacity)
        self.index: Dict[str, int] = {}
        self.ips: List[str] = []
        self._known_sent = np.zeros(self.capacity, dtype=bool)

    def _ids(self, ips: List[str]) -> np.ndarray:
        if len(self.index) + len(ips) > self.capacity:
            logger.info("Source index full; starting a new history")
            self._reset_index()
        out = np.empty(len(ips), dtype=np.int64)
        for k, ip in enumerate(ips):
            i = self.index.get(ip)
            if i is None:
                i = self.index[ip] = len(self.ips)
                self.ips.append(ip)
            out[k] = i
        return out

    def step(self, now: float, ip_stats: List[Dict], anomaly: bool, monitor,
             flows: Optional['LiveFlowEvidence'] = None) -> Dict:
        """One interval: update the policy, then install its new decisions in the monitor"""
        entries = [e for e in ip_stats if e.get('packets', 0) > 0 and e.get('ip')]
        ips = [e['ip'] for e in entries]
        idx = self._ids(ips)
        pkts = np.array([float(e['packets']) for e in entries])
        syn = np.array([float(e.get('syn_count', 0)) for e in entries])
        pol = self.policy
        if flows is None:
            pol.observe_flows(None)
        else:
            pol.observe_flows(flows.shares_for(ips),
                              lambda pos: flows.probe(now, monitor, [ips[k] for k in pos]))

        blocked_before = pol.blocked_until[idx].copy()
        limited_before = pol.limit_until[idx].copy()
        pol.update(now, idx, pkts, syn, anomaly)

        # Installed or extended this interval (extensions refresh the kernel expiry);
        # "new" ones were not active before, for the event log
        blocks = np.nonzero(pol.blocked_until[idx] > blocked_before)[0]
        limits = [k for k in np.nonzero(pol.limit_until[idx] > limited_before)[0]
                  if pol.blocked_until[idx[k]] <= now]
        # Only entries the monitor installed count (protected addresses are refused)
        blocked_ok = [k for k in blocks if monitor.add_to_blacklist(ips[k], ttl_s=self.ttl)]
        # P4+L3 releases blocks of sources whose flows turned out legitimate
        released = [ips[k] for k in np.nonzero((blocked_before > now) & (pol.blocked_until[idx] <= now))[0]]
        for ip in released:
            monitor.remove_from_blacklist(ip)
        limited_ok = [k for k in limits if monitor.rate_limit_ip(ips[k], pol.limit_rate[idx[k]], self.ttl)]
        new_blocks = [ips[k] for k in blocked_ok if blocked_before[k] <= now]
        new_limits = [ips[k] for k in limited_ok if limited_before[k] <= now]

        # Sources that just became "known" are exempt from the spoofing guard
        known = pol.hist.known(idx) & ~self._known_sent[idx]
        if known.any():
            monitor.add_known_sources([ips[k] for k in np.nonzero(known)[0]])
            self._known_sent[idx[known]] = True

        guard = now < pol.spoof_until
        if guard:
            # Budget left for unknown sources after the known ones (as in the evaluated policy)
            known_now = pol.hist.known(idx)
            budget = pol._budget(pol.margin)
            remaining = max(0.0, (budget if np.isfinite(budget) else pkts.sum()) - float(pkts[known_now].sum()))
            monitor.set_spoof_guard(pol.spoof_until - now, remaining)

        if new_blocks or new_limits:
            logger.warning(f"P4: blocked {len(new_blocks)} and slowed {len(new_limits)} sources "
                           f"for {self.ttl:g}s" + (" (spoofing guard on)" if guard else ""))
        if released:
            logger.info(f"P4+L3: released {len(released)} sources whose flows look legitimate")
        return {'blocked': new_blocks, 'limited': new_limits, 'released': released, 'spoof_guard': guard}


class LiveFlowEvidence:
    """
    Level 3 live: classify the flows of the last window, per source the share of
    packets in flows judged attack. Read from the kernel flow map every
    `window_s` seconds (the window the flow classifier was trained on).
    """

    # Flows classified per window at most; in a flood the map holds up to 262k flows, and a
    # uniform sample keeps every source's attack share unbiased while bounding the time
    MAX_FLOWS = 50_000

    def __init__(self, classifier, window_s: float = None, cut: float = 0.5):
        from src.ml.flow_features import FLOW_WINDOW_S
        self.classifier = classifier
        self._rng = np.random.default_rng(0)
        self.window_s = window_s or FLOW_WINDOW_S
        self.cut = cut
        self.last_read: Optional[float] = None
        self.share: Dict[str, float] = {}
        self.flows_classified = 0
        self.attack_flows = 0
        self.probes = 0
        self._peek_time: Optional[float] = None
        self._peek: Optional[Dict] = None

    @classmethod
    def load(cls, path: Path) -> Optional['LiveFlowEvidence']:
        from src.ml.flow_features import FlowClassifier
        if not Path(path).exists():
            logger.warning(f"Flow classifier not found ({path}); level 3 off "
                           "(python scripts/cic_experiment.py --stages level3)")
            return None
        return cls(FlowClassifier.load(str(path)))

    def maybe_update(self, now: float, monitor) -> bool:
        """Read and classify the flow map once per window; True when it was refreshed"""
        if self.last_read is None:
            self.last_read = now
            monitor.get_flow_records(clear=True)        # start the first window clean
            return False
        if now - self.last_read < self.window_s:
            return False
        self.last_read = now
        records = monitor.get_flow_records(clear=True)
        if not records or not len(records['packets']):
            self.share = {}
            return True
        self.update_from_records(records)
        return True

    def _classify(self, records: Dict) -> Dict[str, float]:
        """Per source: share of its packets in flows the classifier calls attack"""
        from src.ml.flow_features import flow_feature_matrix
        n = len(records['packets'])
        if n > self.MAX_FLOWS:
            pick = np.sort(self._rng.choice(n, self.MAX_FLOWS, replace=False))
            records = {k: np.asarray(v)[pick] for k, v in records.items()}
        p = self.classifier.predict_proba(flow_feature_matrix(records))
        names, inv = np.unique(np.asarray(records['src']).astype(str), return_inverse=True)
        pk = np.asarray(records['packets'], dtype=np.float64)
        total = np.bincount(inv, weights=pk, minlength=len(names))
        attack = np.bincount(inv, weights=pk * (p >= self.cut), minlength=len(names))
        self.flows_classified += len(p)
        self.attack_flows += int((p >= self.cut).sum())
        return {ip: a / t for ip, a, t in zip(names, attack, total) if t > 0}

    def update_from_records(self, records: Dict) -> None:
        self.share = self._classify(records)

    def probe(self, now: float, monitor, ips: List[str]) -> np.ndarray:
        """Attack share of these sources from their flows so far in the open window (the map
        is read without clearing it, at most once per interval); NaN where a source has none"""
        if self._peek_time != now:
            self._peek_time, self._peek = now, monitor.get_flow_records(clear=False)
        rec = self._peek
        out = np.full(len(ips), np.nan)
        if not rec or not len(rec['packets']):
            return out
        keep = np.isin(np.asarray(rec['src']).astype(str), np.asarray(ips, dtype=str))
        if not keep.any():
            return out
        self.probes += 1
        share = self._classify({k: np.asarray(v)[keep] for k, v in rec.items()})
        return np.array([share.get(ip, np.nan) for ip in ips], dtype=np.float64)

    def shares_for(self, ips: List[str]) -> np.ndarray:
        return np.array([self.share.get(ip, np.nan) for ip in ips], dtype=np.float64)
