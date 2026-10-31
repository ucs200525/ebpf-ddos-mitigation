"""
Mitigation policies, evaluated one monitoring interval (1 s) at a time.

Model of the data path (matches the XDP program):
  1. packets arrive; enforce() decides how many of each source's packets
     are dropped, using the state built from PREVIOUS intervals
     (blocks and limits installed in kernel maps);
  2. the kernel counts every packet (dropped or not) and the detector runs;
  3. update() lets the policy install new blocks/limits and learn history.

State is held in dense arrays indexed by source (0..n_sources-1), so a policy
handles thousands of sources per interval without Python loops.

Policies:
  TopKBlacklist        P0 (current system rule, permanent) / P1 (with TTL)
  HistoryFilter        P2 history-based IP filtering (Peng et al., ICC 2003):
                       during an attack admit only sources seen in normal times
  AggregateRateLimit   P3a: cap total admitted traffic at a multiple of the baseline
  PerSourceRateLimit   P3b: cap every source at a normal per-source rate
  SourceScoringPolicy  P4 (proposed): score each source; block confident attackers
                       for a limited time, throttle uncertain ones, and fall back
                       to history-based admission when most traffic comes from
                       brand-new sources (spoofing)
  FlowVerifiedScoring  P4+L3: P4, with each source's own flows (level-3 flow
                       classifier) confirming or vetoing its per-source score

Published baselines:
  AdaptiveHistoryFilter  AHIF (Goldstein et al., ICN 2008): admit sources in order
                         of how often their prefix sent traffic in normal times,
                         until a capacity budget is used; drop the rest
  SignatureFilter        SIG (Dimolianis et al., ICIN 2021 / IEEE Access 2021):
                         source-agnostic; drop every packet whose header signature
                         is classified malicious. Signatures are labeled with ground
                         truth (a perfect classifier), so it is an optimistic upper
                         bound of the approach.
"""

from typing import Optional

import numpy as np

from .source_scorer import SourceScorer, source_features

NEG_INF = -np.inf


class SourceHistory:
    """Per-source behaviour learned from intervals judged normal"""

    def __init__(self, n_sources: int, alpha: float = 0.2, reservoir: int = 20000):
        self.alpha = alpha
        self.seen = np.zeros(n_sources, dtype=np.int32)       # normal intervals with traffic
        self.mean = np.zeros(n_sources)                      # EWMA packets per interval
        self.first_seen = np.full(n_sources, np.inf)
        self.baseline_total: Optional[float] = None          # EWMA total packets per interval
        self._rates = np.zeros(reservoir)
        self._n_rates = 0

    def note_present(self, t: float, idx: np.ndarray, pkts: np.ndarray) -> None:
        present = idx[pkts > 0]
        self.first_seen[present] = np.minimum(self.first_seen[present], t)

    def learn(self, idx: np.ndarray, pkts: np.ndarray) -> None:
        mask = pkts > 0
        i, p = idx[mask], pkts[mask]
        fresh = self.seen[i] == 0
        self.mean[i] = np.where(fresh, p, (1 - self.alpha) * self.mean[i] + self.alpha * p)
        self.seen[i] += 1
        total = float(pkts.sum())
        self.baseline_total = total if self.baseline_total is None else \
            (1 - self.alpha) * self.baseline_total + self.alpha * total
        # Keep a sample of normal per-source rates (for per-source limits)
        room = len(self._rates) - self._n_rates
        if room > 0:
            take = p[:room]
            self._rates[self._n_rates:self._n_rates + len(take)] = take
            self._n_rates += len(take)

    def known(self, idx: np.ndarray) -> np.ndarray:
        return self.seen[idx] >= 2

    def typical_rate(self) -> float:
        """Median packets per interval of an active source in normal intervals"""
        if self._n_rates == 0:
            return 1.0
        return float(np.median(self._rates[:self._n_rates]))

    def per_source_limit(self, quantile: float = 0.99, floor: float = 10.0) -> float:
        if self._n_rates == 0:
            return floor
        return max(floor, float(np.quantile(self._rates[:self._n_rates], quantile)))


class MitigationPolicy:
    """Base class: blocks, per-source limits and attack-mode bookkeeping"""

    name = 'base'

    def __init__(self, hold: int = 5):
        self.hold = hold   # seconds a detection keeps "attack mode" active

    def reset(self, n_sources: int) -> None:
        self.hist = SourceHistory(n_sources)
        self.attack_until = NEG_INF
        self.blocked_until = np.full(n_sources, NEG_INF)
        self.limit_until = np.full(n_sources, NEG_INF)
        self.limit_rate = np.zeros(n_sources)

    def in_attack_mode(self, t: float) -> bool:
        return t < self.attack_until

    def observe(self, sig: Optional[np.ndarray], is_attack: Optional[np.ndarray]) -> None:
        """Per-interval packet-header signatures and ground truth (only SignatureFilter uses them)"""

    def set_addresses(self, src_ip: Optional[np.ndarray]) -> None:
        """IPv4 address (uint32) of each dense source index (only AdaptiveHistoryFilter uses it)"""

    def observe_flows(self, attack_share: Optional[np.ndarray], probe=None) -> None:
        """Per entry of this interval: share of the source's packets in flows the level-3
        classifier judged attack in the last complete window (NaN = no flow evidence).
        probe(positions), if given, returns the same share for those entries from the flows
        so far in the open window. Only FlowVerifiedScoring uses them."""

    def enforce(self, t: float, idx: np.ndarray, pkts: np.ndarray) -> np.ndarray:
        """Packets dropped per source entry this interval"""
        dropped = np.zeros_like(pkts, dtype=np.float64)
        blocked = self.blocked_until[idx] > t
        dropped[blocked] = pkts[blocked]
        limited = ~blocked & (self.limit_until[idx] > t)
        dropped[limited] = np.maximum(0.0, pkts[limited] - self.limit_rate[idx[limited]])
        return self._extra_enforce(t, idx, pkts, dropped)

    def update(self, t: float, idx: np.ndarray, pkts: np.ndarray, syn: np.ndarray,
               anomaly: bool) -> None:
        self.hist.note_present(t, idx, pkts)
        if anomaly:
            self.attack_until = t + self.hold
            self._on_anomaly(t, idx, pkts, syn)
        elif not self.in_attack_mode(t):
            self.hist.learn(idx, pkts)

    # Hooks for subclasses
    def _extra_enforce(self, t, idx, pkts, dropped):
        return dropped

    def _on_anomaly(self, t, idx, pkts, syn):
        pass

    def _aggregate_cap(self, scope: np.ndarray, pkts: np.ndarray, dropped: np.ndarray,
                       budget: float) -> np.ndarray:
        """Drop the same fraction from every source in scope so admitted <= budget"""
        admitted = (pkts - dropped)[scope]
        total = float(admitted.sum())
        if total > budget:
            dropped = dropped.copy()
            dropped[scope] += admitted * (1.0 - budget / total)
        return dropped

    def _budget(self, margin: float) -> float:
        base = self.hist.baseline_total
        return np.inf if base is None else margin * base


class TopKBlacklist(MitigationPolicy):
    """Block the K heaviest sources of each anomalous interval (P0 / P1)"""

    def __init__(self, k: int = 5, min_pkts: float = 15, ttl: Optional[float] = None, hold: int = 5):
        super().__init__(hold)
        self.k, self.min_pkts, self.ttl = k, min_pkts, ttl
        self.name = f'Top-{k} blacklist ' + ('(permanent)' if ttl is None else f'(TTL {ttl:g}s)')

    def _on_anomaly(self, t, idx, pkts, syn):
        top = np.argsort(-pkts, kind='stable')[:self.k]
        top = top[pkts[top] > self.min_pkts]
        targets = idx[top]
        fresh = self.blocked_until[targets] <= t
        self.blocked_until[targets[fresh]] = np.inf if self.ttl is None else t + self.ttl


class HistoryFilter(MitigationPolicy):
    """P2: during an attack, admit only sources seen in normal intervals"""

    name = 'History-based filter'

    def _extra_enforce(self, t, idx, pkts, dropped):
        if self.in_attack_mode(t):
            unknown = ~self.hist.known(idx)
            dropped = dropped.copy()
            dropped[unknown] = pkts[unknown]
        return dropped


class AggregateRateLimit(MitigationPolicy):
    """P3a: during an attack, cap total admitted traffic at margin x baseline"""

    def __init__(self, margin: float = 1.5, hold: int = 5):
        super().__init__(hold)
        self.margin = margin
        self.name = f'Aggregate rate limit ({margin:g}x baseline)'

    def _extra_enforce(self, t, idx, pkts, dropped):
        if self.in_attack_mode(t):
            dropped = self._aggregate_cap(np.ones(len(pkts), dtype=bool), pkts, dropped,
                                          self._budget(self.margin))
        return dropped


class PerSourceRateLimit(MitigationPolicy):
    """P3b: during an attack, cap every source at a high normal per-source rate"""

    def __init__(self, quantile: float = 0.99, hold: int = 5):
        super().__init__(hold)
        self.quantile = quantile
        self.name = f'Per-source rate limit (p{quantile * 100:g} of normal)'

    def _extra_enforce(self, t, idx, pkts, dropped):
        if self.in_attack_mode(t):
            limit = self.hist.per_source_limit(self.quantile)
            dropped = np.maximum(dropped, pkts - limit)
        return dropped


class SourceScoringPolicy(MitigationPolicy):
    """
    P4 (proposed): collateral-damage-aware mitigation.

    On each anomalous interval every active source gets P(attacker) from a
    SourceScorer using counter-only features. Sources with p >= t_block are
    blocked for `ttl` seconds; sources with t_limit <= p < t_block are
    throttled to a normal per-source rate for `ttl` seconds; everyone else
    is untouched. If most of the interval's traffic comes from sources that
    appeared for the first time in this very interval (the signature of
    source spoofing, where blocking individual addresses is useless; a
    legitimate crowd keeps reusing its addresses), unknown sources are
    additionally held to whatever remains of a margin x baseline budget after
    known sources. Blocks and limits expire, so mistakes heal on their own.
    """

    def __init__(self, scorer: SourceScorer, t_block: float = 0.9, t_limit: float = 0.6,
                 ttl: float = 30, margin: float = 1.5, spoof_share: float = 0.5,
                 spoof_min_sources: int = 100, hold: int = 5):
        super().__init__(hold)
        self.scorer = scorer
        self.t_block, self.t_limit, self.ttl = t_block, t_limit, ttl
        self.margin, self.spoof_share, self.spoof_min_sources = margin, spoof_share, spoof_min_sources
        self.name = f'Source scoring (block>={t_block:g}, limit>={t_limit:g}, TTL {ttl:g}s)'

    def reset(self, n_sources: int) -> None:
        super().reset(n_sources)
        self.spoof_until = NEG_INF

    def score(self, t, idx, pkts, syn):
        """P(attacker) for the active sources; returns (positions, probabilities)"""
        pos = np.nonzero(pkts > 0)[0]
        i = idx[pos]
        feats = source_features(self.hist.known(i), pkts[pos], self.hist.mean[i], syn[pos],
                                self.hist.typical_rate(), t - np.minimum(self.hist.first_seen[i], t))
        return pos, self.scorer.predict_proba(feats)

    def _on_anomaly(self, t, idx, pkts, syn):
        pos, prob = self.score(t, idx, pkts, syn)
        block = idx[pos[prob >= self.t_block]]
        self.blocked_until[block] = t + self.ttl
        limit = idx[pos[(prob >= self.t_limit) & (prob < self.t_block)]]
        self.limit_until[limit] = t + self.ttl
        self.limit_rate[limit] = self.hist.per_source_limit()

        # update() has already recorded first_seen for this interval
        brand_new = (self.hist.first_seen[idx] == t) & (pkts > 0)
        total = float(pkts.sum())
        if (brand_new.sum() >= self.spoof_min_sources and total > 0
                and pkts[brand_new].sum() / total >= self.spoof_share):
            self.spoof_until = t + self.hold

    def _extra_enforce(self, t, idx, pkts, dropped):
        if t < self.spoof_until:
            known = self.hist.known(idx)
            budget = self._budget(self.margin)
            remaining = max(0.0, budget - float((pkts - dropped)[known].sum()))
            dropped = self._aggregate_cap(~known, pkts, dropped, remaining)
        return dropped


class FlowVerifiedScoring(SourceScoringPolicy):
    """
    P4+L3: source scoring verified by the source's own flows.

    The level-3 flow classifier labels each flow of the last complete 10-second
    window; a source's flow evidence is the share of its packets in flows judged
    attack. On an anomalous interval, per source:
      no flow evidence      P4's decision unchanged
      evidence < 0.5        never blocked (its flows look legitimate)
      evidence >= 0.5       blocked already when its score reaches t_limit
    The last complete window says nothing about a burst that has just started, so
    before acting on a source that is not blocked yet the policy probes its flows so
    far in the open window (fresh evidence replaces the window's). A block whose
    source then turns out legitimate in a complete window is released, unless
    fresh evidence still says attack. on_legit_flows says what a source with
    legitimate-looking flows gets instead: 'limit' (throttled to a normal
    per-source rate while its score reaches t_limit / for the rest of a released
    block) or 'allow' (nothing). The 0.5 cut is the classifier's own decision
    threshold; on_legit_flows is chosen on the training data.
    """

    def __init__(self, scorer: SourceScorer, flow_cut: float = 0.5, on_legit_flows: str = 'limit',
                 block_time_check: bool = True, **kwargs):
        super().__init__(scorer, **kwargs)
        self.block_time_check = block_time_check     # False: complete windows only (ablation)
        if on_legit_flows not in ('limit', 'allow'):
            raise ValueError("on_legit_flows must be 'limit' or 'allow'")
        self.flow_cut = flow_cut
        self.on_legit_flows = on_legit_flows
        self._share: Optional[np.ndarray] = None
        self._probe = None
        self.name = f'Source scoring verified by per-flow classification (P4+L3, legit flows: {on_legit_flows})'

    def observe_flows(self, attack_share, probe=None):
        self._share = attack_share
        self._probe = probe if self.block_time_check else None

    def _evidence(self, idx):
        if self._share is not None and len(self._share) == len(idx):
            return self._share
        return np.full(len(idx), np.nan)

    def update(self, t, idx, pkts, syn, anomaly):
        super().update(t, idx, pkts, syn, anomaly)
        share = self._evidence(idx)
        legit = ~np.isnan(share) & (share < self.flow_cut)
        if self._probe is not None and (legit & (self.blocked_until[idx] > t)).any():
            # the last window looked legitimate, but the source may have changed since
            where = np.nonzero(legit & (self.blocked_until[idx] > t))[0]
            legit = legit.copy()
            legit[where[self._probe(where) >= self.flow_cut]] = False
        held = idx[legit & (self.blocked_until[idx] > t)]
        if len(held):
            if self.on_legit_flows == 'limit':
                self.limit_until[held] = np.maximum(self.limit_until[held], self.blocked_until[held])
                self.limit_rate[held] = self.hist.per_source_limit()
            self.blocked_until[held] = NEG_INF
        if self.on_legit_flows == 'allow':
            self.limit_until[idx[legit]] = NEG_INF

    def _on_anomaly(self, t, idx, pkts, syn):
        share = self._evidence(idx)
        pos, prob = self.score(t, idx, pkts, syn)
        ev = share[pos].copy()
        if self._probe is not None:
            new = np.nonzero((prob >= self.t_limit) & (self.blocked_until[idx[pos]] <= t))[0]
            if len(new):
                fresh = self._probe(pos[new])
                seen = ~np.isnan(fresh)
                ev[new[seen]] = fresh[seen]
        has = ~np.isnan(ev)
        looks_attack = has & (ev >= self.flow_cut)
        looks_legit = has & (ev < self.flow_cut)
        block_mask = ((prob >= self.t_block) & ~looks_legit) | ((prob >= self.t_limit) & looks_attack)
        limit_mask = ~block_mask & (prob >= self.t_limit)
        if self.on_legit_flows == 'allow':
            limit_mask &= ~looks_legit
        block = idx[pos[block_mask]]
        self.blocked_until[block] = t + self.ttl
        limit = idx[pos[limit_mask]]
        self.limit_until[limit] = t + self.ttl
        self.limit_rate[limit] = self.hist.per_source_limit()

        brand_new = (self.hist.first_seen[idx] == t) & (pkts > 0)
        total = float(pkts.sum())
        if (brand_new.sum() >= self.spoof_min_sources and total > 0
                and pkts[brand_new].sum() / total >= self.spoof_share):
            self.spoof_until = t + self.hold


class AdaptiveHistoryFilter(MitigationPolicy):
    """
    AHIF (Goldstein et al., ICN 2008): Bayes-optimal admission under a capacity limit.

    In normal intervals the filter learns P(x|L), the share of legitimate
    traffic sent by each source prefix x (netmask `prefix_bits`). During an
    attack it admits sources in decreasing order of P(x|L) until the admitted
    traffic reaches the capacity budget (margin x baseline), and drops the rest;
    prefixes never seen in normal times have P = 0 and are dropped. This is the
    decision boundary theta of the paper, set by the capacity constraint.
    """

    def __init__(self, margin: float = 1.5, prefix_bits: int = 32, hold: int = 5):
        super().__init__(hold)
        self.margin, self.prefix_bits = margin, prefix_bits
        self.name = f'Adaptive history filter (AHIF, /{prefix_bits}, {margin:g}x baseline)'

    def reset(self, n_sources: int) -> None:
        super().reset(n_sources)
        self.prefix = np.arange(n_sources, dtype=np.int64)   # without addresses: one prefix per source
        self.prefix_pkts = np.zeros(n_sources)

    def set_addresses(self, src_ip):
        if src_ip is None:
            return
        shift = 32 - self.prefix_bits
        raw = np.asarray(src_ip, dtype=np.int64) >> shift
        _, self.prefix = np.unique(raw, return_inverse=True)
        self.prefix_pkts = np.zeros(int(self.prefix.max()) + 1 if len(self.prefix) else 0)

    def update(self, t, idx, pkts, syn, anomaly):
        learning = not anomaly and not self.in_attack_mode(t)
        super().update(t, idx, pkts, syn, anomaly)
        if learning:
            np.add.at(self.prefix_pkts, self.prefix[idx], pkts)

    def _extra_enforce(self, t, idx, pkts, dropped):
        if not self.in_attack_mode(t):
            return dropped
        budget = self._budget(self.margin)
        prob = self.prefix_pkts[self.prefix[idx]]
        order = np.argsort(-prob, kind='stable')
        remaining = pkts - dropped
        cum = np.cumsum(remaining[order])
        # admit whole sources while the budget lasts (the marginal source partly); P = 0 never
        admit = np.clip(budget - (cum - remaining[order]), 0.0, remaining[order])
        admit[prob[order] <= 0] = 0.0
        dropped = dropped.copy()
        dropped[order] += remaining[order] - admit
        return dropped


class SignatureFilter(MitigationPolicy):
    """
    SIG (Dimolianis et al., 2021): source-agnostic signature-based filtering.

    A signature is a combination of packet-header values (here: protocol,
    observed TTL and TCP window or packet size). On each anomalous interval,
    signatures classified as malicious become XDP drop rules for every source.
    As in the paper, a supervised classifier trained on labeled signatures
    decides (`classifier`, see train_signature_classifier). Without one, each
    signature is labeled with ground truth (malicious if most of its packets in
    the interval are attack packets): a perfect classifier and an optimistic
    upper bound, whose only collateral damage comes from legitimate clients
    that share a header signature with the attack. Rules are removed when
    attack mode ends.
    """

    def __init__(self, classifier=None, hold: int = 5):
        super().__init__(hold)
        self.classifier = classifier
        self.name = ('Signature filter (SIG, trained classifier)' if classifier is not None
                     else 'Signature filter (SIG, perfect signature classifier)')

    def reset(self, n_sources: int) -> None:
        super().reset(n_sources)
        self.bad = np.zeros(0, dtype=np.int64)
        self._sig = self._atk = None

    def observe(self, sig, is_attack):
        self._sig, self._atk = sig, is_attack

    def _on_anomaly(self, t, idx, pkts, syn):
        if self._sig is None or not len(pkts):
            return
        sigs, inv = np.unique(self._sig[pkts > 0], return_inverse=True)
        if self.classifier is not None:
            malicious = self.classifier.predict(signature_features(sigs)).astype(bool)
        else:
            active = pkts[pkts > 0]
            attack_pkts = np.bincount(inv, weights=active * self._atk[pkts > 0], minlength=len(sigs))
            malicious = attack_pkts > 0.5 * np.bincount(inv, weights=active, minlength=len(sigs))
        self.bad = np.union1d(self.bad, sigs[malicious])

    def _extra_enforce(self, t, idx, pkts, dropped):
        if not self.in_attack_mode(t):
            self.bad = np.zeros(0, dtype=np.int64)
            return dropped
        if self._sig is None or not len(self.bad):
            return dropped
        hit = np.isin(self._sig, self.bad)
        dropped = dropped.copy()
        dropped[hit] = pkts[hit]
        return dropped


def signature_features(sigs: np.ndarray) -> np.ndarray:
    """Header values of packed signatures: protocol class, observed TTL, window or size"""
    sigs = np.asarray(sigs, dtype=np.int64)
    return np.column_stack([sigs >> 40, (sigs >> 24) & 0xFFFF, sigs & 0xFFFFFF]).astype(np.float64)


def train_signature_classifier(runs, detector: str, seed: int = 0):
    """
    Supervised signature classifier (Dimolianis et al.): a Random Forest on the
    header values of signatures seen in anomalous intervals, each labeled by
    whether most of its packets were attack packets.
    """
    from sklearn.ensemble import RandomForestClassifier
    X, y = [], []
    for run in runs:
        for t, (pkts, atk) in enumerate(zip(run.pkts, run.is_attack_src)):
            if not run.anomaly[detector][t] or run.sig is None:
                continue
            on = pkts > 0
            sigs, inv = np.unique(run.sig[t][on], return_inverse=True)
            a = np.bincount(inv, weights=pkts[on] * atk[on], minlength=len(sigs))
            n = np.bincount(inv, weights=pkts[on], minlength=len(sigs))
            X.append(signature_features(sigs))
            y.append((a > 0.5 * n).astype(int))
    X, y = np.vstack(X), np.concatenate(y)
    model = RandomForestClassifier(n_estimators=100, min_samples_leaf=2, class_weight='balanced',
                                   random_state=seed, n_jobs=1)
    return model.fit(X, y)
