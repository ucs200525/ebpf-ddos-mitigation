"""
Tests for the real-capture pipeline (CIC-DDoS2019): packet header parsing,
per-second and per-flow tables, merging flows split over capture files,
the level-3 flow features, the P4+L3 policy and live flow evidence.
These use hand-built packets; they do not need the dataset.
"""

import struct
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts'))

from cic_extract import flow_table, merge_flows, parse_headers, per_second_table  # noqa: E402
from src.ml.flow_features import FLOW_FEATURES, FlowClassifier, flow_feature_matrix  # noqa: E402
from src.mitigation import FlowVerifiedScoring, SourceScoringPolicy  # noqa: E402

ATTACKER = (172 << 24) | (16 << 16) | 5
CLIENT = (8 << 24) | 8


def _ip(src, dst, proto, payload, frag=0, ttl=64):
    total = 20 + len(payload)
    return struct.pack('!BBHHHBBHII', 0x45, 0, total, 1, frag, ttl, proto, 0, src, dst) + payload


def _tcp(sport, dport, flags, window=1024):
    return struct.pack('!HHIIBBHHH', sport, dport, 0, 0, 5 << 4, flags, window, 0, 0)


def _udp(sport, dport, n=8):
    return struct.pack('!HHHH', sport, dport, 8 + n, 0) + bytes(n)


def _eth(ip, vlan=False):
    head = bytes(12) + (struct.pack('!HHH', 0x8100, 7, 0x0800) if vlan else struct.pack('!H', 0x0800))
    return head + ip


def _pcap(packets):
    out = struct.pack('<IHHiIII', 0xA1B2C3D4, 2, 4, 0, 0, 262144, 1)
    for ts, frame in packets:
        out += struct.pack('<IIII', int(ts), int(round((ts % 1) * 1e6)), len(frame), len(frame)) + frame
    return out


DST = (192 << 24) | (168 << 16) | (50 << 8) | 1


def _sample():
    return [
        (100.0, _eth(_ip(CLIENT, DST, 6, _tcp(40000, 80, 0x02)))),             # SYN
        (100.5, _eth(_ip(CLIENT, DST, 6, _tcp(40000, 80, 0x10)), vlan=True)),  # ACK, VLAN tagged
        (101.5, _eth(_ip(CLIENT, DST, 6, _tcp(40000, 80, 0x18)))),             # PSH+ACK
        (100.2, _eth(_ip(ATTACKER, DST, 17, _udp(53, 3333, 100)))),            # reflected DNS
        (100.3, _eth(_ip(ATTACKER, DST, 17, b'\x00' * 50, frag=10))),          # later fragment
    ]


class TestHeaderParsing:
    def test_fields(self):
        p = parse_headers(_pcap(_sample()))
        assert len(p['ts']) == 5
        assert list(p['src']) == [CLIENT, CLIENT, CLIENT, ATTACKER, ATTACKER]
        assert list(p['proto']) == [6, 6, 6, 17, 17]
        assert list(p['flags'][:3]) == [0x02, 0x10, 0x18]
        assert list(p['sport']) == [40000, 40000, 40000, 53, 0]   # no ports in a later fragment
        assert list(p['dport']) == [80, 80, 80, 3333, 0]
        assert p['length'][0] == 40 and p['length'][3] == 20 + 8 + 100
        assert p['window'][0] == 1024

    def test_rejects_non_ethernet(self):
        bad = bytearray(_pcap(_sample()))
        bad[20:24] = struct.pack('<I', 101)
        with pytest.raises(ValueError):
            parse_headers(bytes(bad))


class TestTables:
    def test_per_second_counts_syn_without_ack(self):
        sec = per_second_table(parse_headers(_pcap(_sample())))
        rows = {(int(s), int(i)): k for k, (s, i) in enumerate(zip(sec['sec'], sec['src']))}
        c100 = rows[(100, CLIENT)]
        assert sec['packets'][c100] == 2 and sec['syn'][c100] == 1 and sec['tcp'][c100] == 2
        a100 = rows[(100, ATTACKER)]
        assert sec['packets'][a100] == 2 and sec['udp'][a100] == 2 and sec['syn'][a100] == 0

    def test_flow_counters(self):
        f = flow_table(parse_headers(_pcap(_sample())), (ATTACKER,), sample_rate=1.0)
        i = int(np.flatnonzero((f['src'] == CLIENT))[0])
        assert f['packets'][i] == 3
        assert f['syn'][i] == 1 and f['ack'][i] == 2 and f['psh'][i] == 1
        assert f['t_last'][i] - f['t_first'][i] == pytest.approx(1.5)
        assert f['iat_max'][i] == pytest.approx(1.0)
        assert f['iat_sumsq'][i] == pytest.approx(0.5 ** 2 + 1.0 ** 2)

    def test_flow_split_over_files_merges_to_the_same_counters(self):
        pk = parse_headers(_pcap(_sample()))
        whole = flow_table(pk, (ATTACKER,), sample_rate=1.0)
        order = np.argsort(pk['ts'])
        halves = [{k: v[order[:2]] for k, v in pk.items()}, {k: v[order[2:]] for k, v in pk.items()}]
        parts = [flow_table(h, (ATTACKER,), sample_rate=1.0) for h in halves]
        merged = merge_flows({k: np.concatenate([p[k] for p in parts]) for k in parts[0]})
        key = lambda f: np.lexsort((f['dport'], f['sport'], f['src']))  # noqa: E731
        for k in ('packets', 'bytes', 'syn', 'ack', 'len_min', 'len_max', 't_first', 't_last'):
            np.testing.assert_allclose(merged[k][key(merged)], whole[k][key(whole)])
        np.testing.assert_allclose(merged['iat_sumsq'][key(merged)], whole['iat_sumsq'][key(whole)])
        np.testing.assert_allclose(merged['iat_max'][key(merged)], whole['iat_max'][key(whole)])

    def test_attacker_flow_sampling_keeps_other_flows(self):
        f = flow_table(parse_headers(_pcap(_sample())), (ATTACKER,), sample_rate=0.0)
        assert set(f['src'].tolist()) == {CLIENT}


class TestFlowFeatures:
    def test_values(self):
        rec = dict(proto=[6], sport=[40000], dport=[80], packets=[3], bytes=[120], len_min=[40],
                   len_max=[40], len_sumsq=[3 * 40 ** 2], t_first=[10.0], t_last=[13.0],
                   iat_sumsq=[1.0 + 4.0], iat_max=[2.0], syn=[1], ack=[2], fin=[0], rst=[1], psh=[0])
        x = dict(zip(FLOW_FEATURES, flow_feature_matrix({k: np.array(v) for k, v in rec.items()})[0]))
        assert x['src_port_low'] == 0 and x['dst_port_low'] == 1
        assert x['duration'] == 3.0 and x['len_mean'] == 40 and x['len_std'] == 0
        assert x['iat_mean'] == 1.5 and x['iat_std'] == pytest.approx(0.5)
        assert x['syn_ratio'] == pytest.approx(1 / 3) and x['fin_rst_ratio'] == pytest.approx(1 / 3)

    def test_single_packet_flow_has_no_gaps(self):
        rec = dict(proto=[17], sport=[53], dport=[3333], packets=[1], bytes=[128], len_min=[128],
                   len_max=[128], len_sumsq=[128 ** 2], t_first=[5.0], t_last=[5.0],
                   iat_sumsq=[0.0], iat_max=[0.0], syn=[0], ack=[0], fin=[0], rst=[0], psh=[0])
        x = dict(zip(FLOW_FEATURES, flow_feature_matrix({k: np.array(v) for k, v in rec.items()})[0]))
        assert x['iat_mean'] == 0 and x['iat_std'] == 0 and x['src_port_low'] == 1

    def test_classifier_round_trip(self, tmp_path):
        rng = np.random.default_rng(0)
        X = rng.random((200, len(FLOW_FEATURES)))
        y = (X[:, 3] > 0.5).astype(int)
        clf = FlowClassifier(n_estimators=10).fit(X, y)
        clf.save(str(tmp_path / 'f.joblib'))
        again = FlowClassifier.load(str(tmp_path / 'f.joblib'))
        np.testing.assert_allclose(again.predict_proba(X), clf.predict_proba(X))


class _Fixed:
    """Scorer stub: P(attacker) given per source position"""

    def __init__(self, p):
        self.p = np.asarray(p, dtype=np.float64)

    def predict_proba(self, X):
        return self.p[:len(X)]


class TestFlowVerifiedScoring:
    def _decide(self, prob, share):
        pol = FlowVerifiedScoring(_Fixed(prob), t_block=0.8, t_limit=0.5, ttl=30)
        pol.reset(10)
        idx = np.arange(len(prob))
        pkts = np.full(len(prob), 100.0)
        pol.observe_flows(np.asarray(share, dtype=np.float64))
        pol.update(0, idx, pkts, np.zeros(len(prob)), anomaly=True)
        blocked = pol.blocked_until[idx] > 1
        limited = ~blocked & (pol.limit_until[idx] > 1)
        return blocked.tolist(), limited.tolist()

    def test_veto_confirm_and_no_evidence(self):
        #           high score,   high score,     mid score,      mid score,    low score
        #           no flows      benign flows    attack flows    no flows      attack flows
        blocked, limited = self._decide([0.9, 0.9, 0.6, 0.6, 0.2], [np.nan, 0.1, 0.9, np.nan, 1.0])
        assert blocked == [True, False, True, False, False]
        assert limited == [False, True, False, True, False]

    def test_without_evidence_equals_p4(self):
        prob = [0.95, 0.7, 0.3]
        p4 = SourceScoringPolicy(_Fixed(prob), t_block=0.8, t_limit=0.5, ttl=30)
        p4.reset(10)
        p4.update(0, np.arange(3), np.full(3, 100.0), np.zeros(3), anomaly=True)
        blocked, limited = self._decide(prob, [np.nan] * 3)
        assert blocked == (p4.blocked_until[:3] > 1).tolist()
        assert limited == ((p4.limit_until[:3] > 1) & ~(p4.blocked_until[:3] > 1)).tolist()


class TestLiveFlowEvidence:
    def test_share_per_source(self):
        from src.mitigation.live import LiveFlowEvidence

        class Clf:
            def predict_proba(self, X):
                return (X[:, FLOW_FEATURES.index('syn_ratio')] == 1).astype(float)

        ev = LiveFlowEvidence(Clf())
        n = 3
        rec = {k: np.zeros(n) for k in ('proto', 'sport', 'dport', 'bytes', 'len_min', 'len_max', 'len_sumsq',
                                        't_first', 't_last', 'iat_sumsq', 'iat_max', 'ack', 'fin', 'rst', 'psh')}
        rec['src'] = np.array(['1.1.1.1', '1.1.1.1', '2.2.2.2'], dtype=object)
        rec['packets'] = np.array([30.0, 10.0, 5.0])
        rec['syn'] = np.array([30.0, 0.0, 0.0])          # first flow is SYN-only
        ev.update_from_records(rec)
        s = ev.shares_for(['1.1.1.1', '2.2.2.2', '3.3.3.3'])
        assert s[0] == pytest.approx(0.75) and s[1] == 0 and np.isnan(s[2])


def test_local_time_conversion():
    from src.ml.cic_data import _local_to_epoch
    # 03-11 is Atlantic daylight time (UTC-3): 09:18:16 local = 12:18:16 UTC
    assert _local_to_epoch('03-11', '2018-11-03 09:18:16') == 1541247496.0
    # 01-12 is Atlantic standard time (UTC-4)
    assert _local_to_epoch('01-12', '2018-12-01 09:17:11') == 1543670231.0


def test_flow_evidence_spares_heavy_legitimate_source_in_evaluation():
    """End to end through evaluate_policy: an attacker and a heavy legitimate client both
    score as attackers; their flows tell them apart, so only P4+L3 spares the client"""
    from src.mitigation.evaluation import PreparedRun, evaluate_policy
    n = 40
    idx = [np.array([0, 1])] * n
    pkts = [np.array([1000.0, 800.0]) if t >= 20 else np.array([1.0, 5.0]) for t in range(n)]
    run = PreparedRun(scenario='udp_flood', n_sources=2, idx=idx, pkts=pkts,
                      syn=[np.zeros(2)] * n, is_attack_src=[np.array([True, False])] * n,
                      attack_second=[t >= 20 for t in range(n)],
                      anomaly={'hybrid': [t >= 20 for t in range(n)]},
                      flow_share=[np.array([1.0, 0.0])] * n)
    scorer = _Fixed([0.95, 0.95])
    p4 = evaluate_policy(SourceScoringPolicy(scorer, t_block=0.8, t_limit=0.5), [run], 'hybrid')
    l3 = evaluate_policy(FlowVerifiedScoring(scorer, t_block=0.8, t_limit=0.5), [run], 'hybrid')
    assert p4['attack_drop_pct'] > 90 and p4['legit_loss_pct'] > 90
    assert l3['attack_drop_pct'] > 90
    assert l3['legit_loss_pct'] < p4['legit_loss_pct']


def test_block_released_when_flows_look_legitimate():
    """A block installed before flow evidence existed is lifted once the flows look legitimate"""
    for mode, limited_after in (('limit', True), ('allow', False)):
        pol = FlowVerifiedScoring(_Fixed([0.95]), t_block=0.8, t_limit=0.5, ttl=30, on_legit_flows=mode)
        pol.reset(4)
        idx, pk = np.array([0]), np.array([500.0])
        pol.observe_flows(np.array([np.nan]))
        pol.update(0, idx, pk, np.zeros(1), anomaly=True)
        assert pol.blocked_until[0] > 1                      # no evidence yet: P4 blocks
        pol.observe_flows(np.array([0.0]))                   # next window: its flows are legitimate
        pol.update(1, idx, pk, np.zeros(1), anomaly=True)
        assert pol.blocked_until[0] <= 2
        assert (pol.limit_until[0] > 2) == limited_after


class TestBlockTimeProbe:
    """Fresh evidence from the open window, read just before acting on a source"""

    def _decide(self, prob, share, fresh, mode='allow'):
        pol = FlowVerifiedScoring(_Fixed(prob), t_block=0.8, t_limit=0.5, ttl=30, on_legit_flows=mode)
        pol.reset(10)
        idx, pkts = np.arange(len(prob)), np.full(len(prob), 100.0)
        asked = []

        def probe(pos):
            asked.extend(pos.tolist())
            return np.asarray(fresh, dtype=np.float64)[pos]

        pol.observe_flows(np.asarray(share, dtype=np.float64), probe)
        pol.update(0, idx, pkts, np.zeros(len(prob)), anomaly=True)
        return (pol.blocked_until[idx] > 1).tolist(), asked

    def test_burst_with_legitimate_flows_is_not_blocked(self):
        blocked, _ = self._decide([0.95], [np.nan], [0.1])
        assert blocked == [False]

    def test_attack_flows_are_blocked_at_the_lower_threshold(self):
        blocked, _ = self._decide([0.6], [np.nan], [0.9])
        assert blocked == [True]

    def test_no_flows_yet_leaves_p4_unchanged(self):
        blocked, _ = self._decide([0.95, 0.6], [np.nan, np.nan], [np.nan, np.nan])
        assert blocked == [True, False]

    def test_fresh_evidence_replaces_the_last_window(self):
        # legitimate last window, attack now: a source that turned into a flood source
        blocked, _ = self._decide([0.95], [0.1], [0.9])
        assert blocked == [True]

    def test_sources_below_the_thresholds_are_not_probed(self):
        _, asked = self._decide([0.9, 0.2], [np.nan, np.nan], [0.9, 0.9])
        assert 1 not in asked

    def test_blocked_source_is_not_probed_again(self):
        pol = FlowVerifiedScoring(_Fixed([0.95]), t_block=0.8, t_limit=0.5, ttl=30)
        pol.reset(4)
        idx, pk = np.array([0]), np.array([500.0])
        pol.observe_flows(np.array([np.nan]), lambda pos: np.array([0.9]))
        pol.update(0, idx, pk, np.zeros(1), anomaly=True)
        asked = []
        pol.observe_flows(np.array([np.nan]), lambda pos: asked.append(1) or np.array([0.9]))
        pol.update(1, idx, pk, np.zeros(1), anomaly=True)
        assert asked == [] and pol.blocked_until[0] > 2


class TestLiveProbe:
    def test_reads_the_map_once_per_interval_without_clearing(self):
        from src.mitigation.live import LiveFlowEvidence

        class Clf:
            def predict_proba(self, X):
                return (X[:, FLOW_FEATURES.index('syn_ratio')] == 1).astype(float)

        class Monitor:
            calls = []

            def get_flow_records(self, clear=True):
                self.calls.append(clear)
                n = 2
                rec = {k: np.zeros(n) for k in ('proto', 'sport', 'dport', 'bytes', 'len_min', 'len_max', 'len_sumsq',
                                                't_first', 't_last', 'iat_sumsq', 'iat_max', 'ack', 'fin', 'rst', 'psh')}
                rec['src'] = np.array(['1.1.1.1', '2.2.2.2'], dtype=object)
                rec['packets'] = np.array([10.0, 10.0])
                rec['syn'] = np.array([10.0, 0.0])
                return rec

        ev, mon = LiveFlowEvidence(Clf()), Monitor()
        s = ev.probe(5.0, mon, ['1.1.1.1', '2.2.2.2', '3.3.3.3'])
        assert s[0] == 1 and s[1] == 0 and np.isnan(s[2])
        ev.probe(5.0, mon, ['2.2.2.2'])
        assert mon.calls == [False]


def test_flow_seconds_rebuild_the_window():
    """Per-second flow rows merged over a window equal the row of the whole window"""
    pk = parse_headers(_pcap(_sample()))
    whole = flow_table(pk, (ATTACKER,), sample_rate=1.0)
    seconds = flow_table(pk, (ATTACKER,), sample_rate=1.0, slot_s=1)
    assert len(seconds['win']) > len(whole['win'])
    merged = merge_flows(seconds, by='win')
    key = lambda f: np.lexsort((f['dport'], f['sport'], f['src']))  # noqa: E731
    for k in ('packets', 'bytes', 'syn', 'ack', 'iat_sumsq', 'iat_max', 't_first', 't_last'):
        np.testing.assert_allclose(merged[k][key(merged)], whole[k][key(whole)])
