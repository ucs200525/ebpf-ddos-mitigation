"""
Tests for mitigation policies, the source scorer features, the evaluation
harness, and the generator's source identities / realism options.
"""

import numpy as np
import pytest

from src.mitigation import (AdaptiveHistoryFilter, AggregateRateLimit, HistoryFilter,
                            PerSourceRateLimit, SignatureFilter, SourceScoringPolicy, TopKBlacklist,
                            SOURCE_FEATURES)
from src.mitigation.policies import signature_features, train_signature_classifier
from src.mitigation.evaluation import collect_source_samples, evaluate_policy, prepare_run
from src.mitigation.source_scorer import SourceScorer, source_features
from src.ml.traffic_dataset import generate_dataset, generate_run, generate_runs


def _learn(policy, n=10, seconds=5, pkts=10.0):
    """Feed `seconds` normal intervals where sources 0..n-1 each send `pkts`"""
    policy.reset(2000)                  # room for every index used in these tests
    idx = np.arange(n)
    for t in range(seconds):
        p = np.full(n, pkts)
        policy.enforce(t, idx, p)
        policy.update(t, idx, p, np.zeros(n), anomaly=False)
    return seconds


class TestBaselinePolicies:
    def test_topk_blocks_heaviest_sources_permanently(self):
        pol = TopKBlacklist(k=2, min_pkts=15, ttl=None)
        t = _learn(pol)
        idx = np.arange(5)
        pkts = np.array([100.0, 50, 5, 20, 1])
        pol.update(t, idx, pkts, np.zeros(5), anomaly=True)
        dropped = pol.enforce(t + 1000, idx, pkts)
        np.testing.assert_array_equal(dropped, [100, 50, 0, 0, 0])

    def test_topk_ttl_expires(self):
        pol = TopKBlacklist(k=1, min_pkts=0, ttl=30)
        t = _learn(pol)
        idx, pkts = np.array([3]), np.array([40.0])
        pol.update(t, idx, pkts, np.zeros(1), anomaly=True)
        assert pol.enforce(t + 29, idx, pkts)[0] == 40
        assert pol.enforce(t + 30, idx, pkts)[0] == 0

    def test_topk_ignores_sources_below_min_packets(self):
        pol = TopKBlacklist(k=5, min_pkts=15)
        t = _learn(pol)
        idx, pkts = np.array([0, 1]), np.array([15.0, 16.0])
        pol.update(t, idx, pkts, np.zeros(2), anomaly=True)
        np.testing.assert_array_equal(pol.enforce(t + 1, idx, pkts), [0, 16])

    def test_history_filter_drops_only_unknown_during_attack(self):
        pol = HistoryFilter(hold=5)
        t = _learn(pol, n=4)
        idx = np.array([0, 1, 12, 13])            # 12, 13 never seen before
        pkts = np.array([10.0, 10, 99, 99])
        assert pol.enforce(t, idx, pkts).sum() == 0  # no attack mode yet
        pol.update(t, idx, pkts, np.zeros(4), anomaly=True)
        np.testing.assert_array_equal(pol.enforce(t + 1, idx, pkts), [0, 0, 99, 99])
        assert pol.enforce(t + 5, idx, pkts).sum() == 0  # attack mode ended

    def test_aggregate_rate_limit_caps_total(self):
        pol = AggregateRateLimit(margin=1.5)
        t = _learn(pol, n=10, pkts=10.0)           # baseline total = 100 per interval
        idx, pkts = np.arange(10), np.full(10, 100.0)
        pol.update(t, idx, pkts, np.zeros(10), anomaly=True)
        dropped = pol.enforce(t + 1, idx, pkts)
        assert (pkts - dropped).sum() == pytest.approx(150.0)
        assert np.allclose(dropped, dropped[0])     # same fraction from everyone

    def test_per_source_rate_limit(self):
        pol = PerSourceRateLimit(quantile=0.99)
        t = _learn(pol, n=10, pkts=12.0)            # normal per-source rate 12 -> limit 12
        idx, pkts = np.array([0, 1]), np.array([5.0, 100.0])
        pol.update(t, idx, pkts, np.zeros(2), anomaly=True)
        np.testing.assert_array_equal(pol.enforce(t + 1, idx, pkts), [0, 88])

    def test_attack_intervals_are_not_learned_as_history(self):
        pol = HistoryFilter(hold=5)
        t = _learn(pol, n=2)
        attacker = np.array([7])
        pol.update(t, attacker, np.array([500.0]), np.zeros(1), anomaly=True)
        for dt in range(1, 4):                      # still attack mode
            pol.update(t + dt, attacker, np.array([500.0]), np.zeros(1), anomaly=False)
        assert not pol.hist.known(attacker)[0]


class TestPublishedBaselines:
    def test_ahif_admits_by_history_until_budget(self):
        pol = AdaptiveHistoryFilter(margin=1.5, hold=5)
        pol.reset(20)
        idx = np.arange(4)
        for t in range(5):                          # source 0 sends most in normal times
            p = np.array([6.0, 3, 1, 0])
            pol.enforce(t, idx, p)
            pol.update(t, idx, p, np.zeros(4), anomaly=False)
        # baseline total = 10 packets -> budget 15
        pkts = np.array([6.0, 6, 6, 50])            # source 3 never seen before
        pol.update(5, idx, pkts, np.zeros(4), anomaly=True)
        dropped = pol.enforce(6, idx, pkts)
        np.testing.assert_allclose(dropped, [0, 0, 3, 50])   # 0 and 1 admitted, 2 partly, 3 not at all
        assert pol.enforce(11, idx, pkts).sum() == 0          # attack mode over

    def test_ahif_generalizes_within_a_prefix(self):
        pol = AdaptiveHistoryFilter(prefix_bits=24, hold=5)
        pol.reset(3)
        pol.set_addresses(np.array([0x0A000001, 0x0A000002, 0x0B000001]))   # 10.0.0.1, 10.0.0.2, 11.0.0.1
        idx = np.array([0])
        for t in range(3):
            pol.update(t, idx, np.array([10.0]), np.zeros(1), anomaly=False)
        idx = np.array([1, 2])
        pol.update(3, idx, np.array([5.0, 5.0]), np.zeros(2), anomaly=True)
        np.testing.assert_allclose(pol.enforce(4, idx, np.array([5.0, 5.0])), [0, 5])

    def test_signature_filter_drops_every_source_with_a_bad_signature(self):
        pol = SignatureFilter(hold=5)
        pol.reset(10)
        idx = np.arange(4)
        sig = np.array([7, 7, 9, 9])
        atk = np.array([True, True, False, False])
        pkts = np.array([50.0, 50, 5, 5])
        pol.observe(sig, atk)
        pol.update(0, idx, pkts, np.zeros(4), anomaly=True)
        # new sources (spoofed) with the same signature are dropped too; signature 9 is not
        idx2 = np.array([5, 6, 2])
        pol.observe(np.array([7, 7, 9]), np.array([True, True, False]))
        np.testing.assert_allclose(pol.enforce(1, idx2, np.array([40.0, 40, 5])), [40, 40, 0])

    def test_signature_classifier_learns_crafted_headers(self, prepared):
        sigs = np.concatenate([r.sig[0] for r in prepared])
        feats = signature_features(sigs)
        assert feats.shape == (len(sigs), 3) and set(np.unique(feats[:, 0])) <= {1, 2, 3}
        model = train_signature_classifier(prepared, 'statistical')
        assert set(model.classes_) <= {0, 1}

    def test_signatures_do_not_change_packet_counts(self):
        a = generate_run(np.random.default_rng(3), 'syn_flood')
        b = generate_run(np.random.default_rng(3), 'syn_flood')
        for sa, sb in zip(a.steps, b.steps):
            np.testing.assert_array_equal(sa.ip_packets, sb.ip_packets)
            np.testing.assert_array_equal(sa.ip_sig, sb.ip_sig)
            assert len(sa.ip_sig) == len(sa.ip_packets)


class _StubScorer:
    """Scores sources by a fixed table keyed on packets, for deterministic tests.
    Packets are recovered from rel_rate, given the typical rate learned in _learn()."""

    def __init__(self, by_pkts, typical_rate):
        self.by_pkts, self.typical = by_pkts, typical_rate

    def predict_proba(self, X):
        col = SOURCE_FEATURES.index('rel_rate')
        pkts = np.expm1(X[:, col] + np.log1p(self.typical))
        return np.array([self.by_pkts.get(int(round(p)), 0.0) for p in pkts])


class TestSourceScoringPolicy:
    def test_blocks_high_limits_medium_spares_low(self):
        scorer = _StubScorer({100: 0.99, 50: 0.7, 30: 0.1}, typical_rate=12.0)
        pol = SourceScoringPolicy(scorer, t_block=0.9, t_limit=0.6, ttl=30)
        t = _learn(pol, n=10, pkts=12.0)            # per-source limit becomes 12
        idx, pkts = np.array([20, 21, 22]), np.array([100.0, 50, 30])
        pol.update(t, idx, pkts, np.zeros(3), anomaly=True)
        np.testing.assert_array_equal(pol.enforce(t + 1, idx, pkts), [100, 38, 0])
        assert pol.enforce(t + 30, idx, pkts).sum() == 0   # everything expires

    def test_spoof_guard_limits_unknown_sources_only(self):
        scorer = _StubScorer({}, typical_rate=10.0)  # scorer flags nobody
        pol = SourceScoringPolicy(scorer, spoof_min_sources=100, spoof_share=0.5, margin=1.5)
        t = _learn(pol, n=10, pkts=10.0)            # baseline total 100 -> budget 150
        known = np.arange(10)
        spoofed = np.arange(1000, 1400)             # 400 brand-new one-packet sources
        idx = np.concatenate([known, spoofed])
        pkts = np.concatenate([np.full(10, 10.0), np.ones(400)])
        pol.update(t, idx, pkts, np.zeros(len(idx)), anomaly=True)
        dropped = pol.enforce(t + 1, idx, pkts)
        assert dropped[:10].sum() == 0              # known sources untouched
        assert (pkts - dropped)[10:].sum() == pytest.approx(50.0)   # unknown get the remaining budget

    def test_spoof_guard_ignores_returning_sources(self):
        scorer = _StubScorer({}, typical_rate=10.0)
        pol = SourceScoringPolicy(scorer, spoof_min_sources=100, spoof_share=0.5)
        t = _learn(pol, n=10, pkts=10.0)
        crowd = np.arange(1000, 1400)
        pol.update(t, crowd, np.ones(400), np.zeros(400), anomaly=False)  # crowd arrives (normal)
        pol.update(t + 1, crowd, np.ones(400), np.zeros(400), anomaly=True)  # same addresses again
        assert pol.spoof_until == -np.inf


@pytest.fixture(scope='module')
def prepared():
    from src.core.anomaly_detector import AnomalyDetector
    runs = generate_runs(6, seed=77, client_churn=0.1, bot_pre_activity=0.3)
    return [prepare_run(r, {'statistical': AnomalyDetector}) for r in runs]


class TestScorerAndHarness:
    def test_source_features_shape_and_values(self):
        X = source_features(np.array([True, False]), np.array([10.0, 1.0]), np.array([10.0, 0.0]),
                            np.array([0.0, 1.0]), 10.0, np.array([120.0, 0.0]))
        assert X.shape == (2, len(SOURCE_FEATURES))
        assert X[0, SOURCE_FEATURES.index('known')] == 1
        assert X[0, SOURCE_FEATURES.index('rate_jump')] == pytest.approx(0.0)
        assert X[0, SOURCE_FEATURES.index('rel_rate')] == pytest.approx(0.0)
        assert X[1, SOURCE_FEATURES.index('syn_share')] == 1.0
        assert X[0, SOURCE_FEATURES.index('age')] == 1.0

    def test_evaluate_policy_metrics_in_range(self, prepared):
        res = evaluate_policy(TopKBlacklist(), prepared, 'statistical')
        for key in ('attack_drop_pct', 'legit_loss_pct', 'legit_sources_hit_pct', 'mitigated_runs_pct'):
            assert 0.0 <= res[key] <= 100.0

    def test_extended_metrics_are_consistent(self, prepared):
        res = evaluate_policy(TopKBlacklist(), prepared, 'statistical')
        assert res['attack_passed_pct'] + res['attack_drop_pct'] == pytest.approx(100.0)
        assert res['legit_retained_pct'] + res['legit_loss_pct'] == pytest.approx(100.0)
        assert res['n_runs'] == len(prepared)
        assert res['n_attack_runs'] == sum(r.is_attack_run for r in prepared)
        if res['median_detection_delay_s'] is not None:
            assert res['median_detection_delay_s'] >= 0
        if res['median_recovery_s'] is not None:
            assert res['median_recovery_s'] >= 0

    def test_scorer_feature_subset(self, prepared, tmp_path):
        from src.mitigation.source_scorer import HISTORY_FEATURES
        X, y = collect_source_samples(prepared, 'statistical', per_class_per_second=20)
        if len(set(y)) < 2:
            pytest.skip('sample runs produced a single class')
        subset = [f for f in SOURCE_FEATURES if f not in HISTORY_FEATURES]
        scorer = SourceScorer(subset).fit(X, y)
        assert list(scorer.coefficients()) == subset
        path = tmp_path / 'subset.joblib'
        scorer.save(str(path))
        loaded = SourceScorer.load(str(path))
        assert loaded.features == subset
        np.testing.assert_allclose(loaded.predict_proba(X[:20]), scorer.predict_proba(X[:20]))
        # the history columns must not influence the ablation scorer
        X2 = X[:20].copy()
        X2[:, [SOURCE_FEATURES.index(f) for f in HISTORY_FEATURES]] = 123.0
        np.testing.assert_allclose(loaded.predict_proba(X2), loaded.predict_proba(X[:20]))

    def test_scorer_trains_and_round_trips(self, prepared, tmp_path):
        X, y = collect_source_samples(prepared, 'statistical', per_class_per_second=20)
        assert X.shape[1] == len(SOURCE_FEATURES) and len(X) == len(y)
        assert max(np.sum(y == 1), np.sum(y == 0)) <= 20 * sum(sum(r.anomaly['statistical']) for r in prepared)
        if len(set(y)) < 2:
            pytest.skip('sample runs produced a single class')
        scorer = SourceScorer().fit(X, y)
        path = tmp_path / 'scorer.joblib'
        scorer.save(str(path))
        loaded = SourceScorer.load(str(path))
        np.testing.assert_allclose(loaded.predict_proba(X[:20]), scorer.predict_proba(X[:20]))


class TestGeneratorSourceIdentities:
    def test_defaults_unchanged_and_ids_consistent(self):
        rng = np.random.default_rng(5)
        run = generate_run(rng, 'udp_flood')
        for step in run.steps:
            assert len(step.ip_ids) == len(step.ip_packets) == len(step.ip_is_attack)
            if not step.is_attack:
                assert step.ip_packets[step.ip_is_attack].sum() == 0   # no bot traffic by default
        a = generate_dataset(3, seed=11)
        b = generate_dataset(3, seed=11)
        np.testing.assert_array_equal(a['X'], b['X'])

    def test_realism_adds_churn_and_pre_attack_bots(self):
        run = generate_run(np.random.default_rng(9), 'udp_flood', client_churn=0.2, bot_pre_activity=0.8)
        pre = [s for s in run.steps if not s.is_attack][:20]
        assert sum(s.ip_packets[s.ip_is_attack].sum() for s in pre) > 0
        churn_ids = np.concatenate([s.ip_ids[(s.ip_ids >= 50_000_000) & (s.ip_packets > 0)] for s in run.steps])
        assert len(np.unique(churn_ids)) > 0

    def test_safety_scenarios(self):
        from src.ml.traffic_dataset import generate_scenario_runs
        (heavy,) = generate_scenario_runs('heavy_legit', 1, seed=3)
        assert not any(s.is_attack for s in heavy.steps)
        assert sum(s.ip_packets[s.ip_is_attack].sum() for s in heavy.steps) == 0
        # the heavy clients are existing clients: no new ids appear
        assert all(s.ip_ids.max() < 1_000_000 for s in heavy.steps)
        for scenario in ('udp_flood_with_crowd', 'udp_flood_spoofed_with_crowd'):
            (run,) = generate_scenario_runs(scenario, 1, seed=3)
            attack = [s for s in run.steps if s.is_attack]
            assert attack
            crowd = [s.ip_packets[(s.ip_ids >= 1_000_000) & (s.ip_ids < 2_000_000)].sum() for s in attack]
            assert sum(crowd) > 0                                   # new legitimate users arrive
            assert all(not s.ip_is_attack[(s.ip_ids >= 1_000_000) & (s.ip_ids < 2_000_000)].any()
                       for s in attack)

    def test_spoofed_sources_are_fresh_every_second(self):
        run = generate_run(np.random.default_rng(4), 'udp_flood_spoofed')
        atk = [set(s.ip_ids[s.ip_is_attack]) for s in run.steps if s.is_attack]
        assert all(not (a & b) for a, b in zip(atk, atk[1:]))
