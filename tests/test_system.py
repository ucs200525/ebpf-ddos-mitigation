"""
System-level tests: monitoring pipeline, mitigation, and the dashboard API.
Runs in simulation mode (no eBPF), which is how the project runs on Windows.
"""

import json

import pytest

from src.core.system import DDoSMitigationSystem
from src.dashboard import app as dashboard
from src.core.traffic_monitor import TrafficMonitor, validate_blockable_ip


def _stats(total, tcp=0.85, udp=0.10):
    return {
        'total_packets': total,
        'total_bytes': total * 500,
        'tcp_packets': int(total * tcp),
        'udp_packets': int(total * udp),
        'icmp_packets': total - int(total * tcp) - int(total * udp),
    }


@pytest.fixture
def system(tmp_path):
    # Statistical detector only, with all state files in a temp folder
    sys_ = DDoSMitigationSystem('test0', ml_model_path=None, auto_detect_model=False, policy='top5')
    sys_.alert_system.alert_file = str(tmp_path / 'alerts.log')
    sys_.traffic_profiler.profile_file = str(tmp_path / 'traffic_profile.json')
    # Simulation-mode monitor regardless of host OS
    monitor = TrafficMonitor('test0', blacklist_file=str(tmp_path / 'blacklist.json'))
    monitor.simulation_mode = True
    monitor.loaded = True
    sys_.traffic_monitor = monitor
    sys_.allow_simulation_api = True
    return sys_


def _normal_then_flood(system, flood_pps=20000, normal_seconds=15):
    t, total = 1000.0, 0
    for _ in range(normal_seconds):
        total += 50
        ips = [{'ip': f'10.0.0.{i}', 'packets': 5} for i in range(10)]
        result = system.process_sample(_stats(total), ips, now=t)
        assert not result.is_anomaly
        t += 1
    total += flood_pps
    bots = [{'ip': f'203.0.113.{i}', 'packets': flood_pps // 20, 'udp_count': flood_pps // 20}
            for i in range(20)]
    return system.process_sample(_stats(total, tcp=0.02, udp=0.97), bots, now=t)


class TestProcessSample:
    def test_flood_detected_and_mitigated(self, system):
        result = _normal_then_flood(system)
        assert result.is_anomaly
        assert len(system.traffic_monitor.get_blacklist()) == 5  # top 5 sources
        assert len(system.alert_system.get_recent_alerts()) == 1
        assert system._current_pps == pytest.approx(20000)

    def test_baseline_not_poisoned_by_attack(self, system):
        _normal_then_flood(system)
        baseline_after = system.anomaly_detector.baseline.mean_pps
        assert baseline_after == pytest.approx(50, rel=0.05)

    def test_pps_measured_before_baseline_update(self, system):
        """Regression: detection used to run after update_baseline and always saw 0 pps"""
        result = _normal_then_flood(system)
        assert result.metrics['pps'] == pytest.approx(20000)

    def test_already_blocked_sources_are_skipped(self, system):
        """Regression: XDP still counts blocked sources, so they kept the top-5 slots"""
        _normal_then_flood(system)
        assert len(system.traffic_monitor.get_blacklist()) == 5
        bots = [{'ip': f'203.0.113.{i}', 'packets': 1000, 'udp_count': 1000} for i in range(20)]
        system.process_sample(_stats(system.anomaly_detector.last_stats['total_packets'] + 20000,
                                     tcp=0.02, udp=0.97), bots, now=1016.0)
        assert len(system.traffic_monitor.get_blacklist()) == 10  # 5 new each second

    def test_idle_host_never_flagged(self, system):
        """Regression: lifetime counters kept a quiet single-IP host 'under attack' forever"""
        t = 1000.0
        for i in range(30):
            result = system.process_sample(_stats(1921), [], now=t + i)
            assert not result.is_anomaly
        assert system.traffic_monitor.get_blacklist() == []

    def test_status_reports_simulation_sample(self, system):
        _normal_then_flood(system)
        status = system.get_status()
        assert status['simulation_mode'] is True
        assert status['statistics']['total_packets'] > 0
        assert status['last_detection']['is_anomaly'] is True
        assert len(status['ip_stats']) == 10


class TestBlacklist:
    @pytest.mark.parametrize('ip', ['127.0.0.1', '0.0.0.0', '224.0.0.1', '255.255.255.255',
                                    '1', '300.1.1.1', '<script>', ''])
    def test_protected_or_invalid_ips_rejected(self, system, ip):
        assert validate_blockable_ip(ip) is None
        assert system.traffic_monitor.add_to_blacklist(ip) is False

    def test_add_is_idempotent_and_removable(self, system):
        monitor = system.traffic_monitor
        assert monitor.add_to_blacklist('203.0.113.9')
        assert monitor.add_to_blacklist('203.0.113.9')
        assert monitor.get_blacklist() == ['203.0.113.9']
        assert monitor.remove_from_blacklist('203.0.113.9')
        assert monitor.get_blacklist() == []


class _FakeMap(dict):
    Key = int
    Leaf = int


class _FakeBPF:
    """Mimics BCC's BPF: __len__ counts only tables cached by bpf[name], so the
    object is falsy when maps are read through get_table()"""

    def __init__(self):
        self.maps = {'blacklist_map': _FakeMap()}
        self.removed = []

    def __len__(self):
        return 0

    def get_table(self, name):
        return self.maps[name]

    def remove_xdp(self, interface):
        self.removed.append(interface)


class TestLiveMonitorWithFalsyBPF:
    @pytest.fixture
    def monitor(self, tmp_path):
        monitor = TrafficMonitor('veth1', blacklist_file=str(tmp_path / 'blacklist.json'))
        monitor.simulation_mode = False
        monitor.bpf = _FakeBPF()
        monitor.loaded = True
        return monitor

    def test_is_blacklisted_sees_kernel_map(self, monitor):
        """Regression: 'if not self.bpf' made is_blacklisted always False on Linux"""
        assert not monitor.is_blacklisted('203.0.113.9')
        assert monitor.add_to_blacklist('203.0.113.9')
        assert monitor.is_blacklisted('203.0.113.9')

    def test_unload_detaches_xdp(self, monitor):
        monitor.unload_xdp_program()
        assert monitor.bpf.removed == ['veth1']
        assert monitor.loaded is False


class TestIpIntervalDeltas:
    def test_deltas_from_cumulative_counters(self, monkeypatch):
        monitor = TrafficMonitor('test0')
        snapshots = iter([
            [{'ip': '1.1.1.1', 'packets': 100, 'bytes': 1000, 'syn_count': 10, 'udp_count': 0}],
            [{'ip': '1.1.1.1', 'packets': 130, 'bytes': 1300, 'syn_count': 12, 'udp_count': 0},
             {'ip': '2.2.2.2', 'packets': 5, 'bytes': 50, 'syn_count': 0, 'udp_count': 5}],
            [{'ip': '1.1.1.1', 'packets': 130, 'bytes': 1300, 'syn_count': 12, 'udp_count': 0},
             {'ip': '2.2.2.2', 'packets': 3, 'bytes': 30, 'syn_count': 0, 'udp_count': 3}],
        ])
        monkeypatch.setattr(monitor, 'get_ip_statistics', lambda limit=10000: next(snapshots))

        first = monitor.get_ip_interval_statistics()
        assert first[0]['packets'] == 100

        second = {d['ip']: d for d in monitor.get_ip_interval_statistics()}
        assert second['1.1.1.1']['packets'] == 30
        assert second['1.1.1.1']['syn_count'] == 2
        assert second['2.2.2.2']['packets'] == 5

        # 1.1.1.1 idle -> omitted; 2.2.2.2 counter went backwards (LRU re-insert)
        third = monitor.get_ip_interval_statistics()
        assert [d['ip'] for d in third] == ['2.2.2.2']
        assert third[0]['packets'] == 3


@pytest.fixture
def client(system):
    dashboard.ddos_system = system
    dashboard.app.config['TESTING'] = True
    with dashboard.app.test_client() as c:
        yield c
    dashboard.ddos_system = None


def _payload(total=100, ips=None, **extra):
    return {'stats': _stats(total), 'ip_stats': ips or [], 'label': 'test', **extra}


class TestDashboardApi:
    def test_status(self, client):
        r = client.get('/api/status')
        assert r.status_code == 200
        assert r.get_json()['interface'] == 'test0'

    def test_index_escapes_untrusted_values(self, client):
        html = client.get('/').get_data(as_text=True)
        # Every value is inserted with textContent; markup is never built from data
        assert 'innerHTML' not in html and 'insertAdjacentHTML' not in html
        assert 'textContent' in html
        assert '95.3%' not in html  # accuracy comes from the model, not hardcoded

    def test_simulate_runs_full_pipeline(self, client):
        for i in range(15):
            r = client.post('/api/simulate_attack', json=_payload(
                50 * (i + 1), [{'ip': '10.0.0.1', 'packets': 5}], interval_seconds=1))
            assert r.status_code == 200
        bots = [{'ip': f'203.0.113.{i}', 'packets': 1000} for i in range(20)]
        stats = _stats(750 + 20000, tcp=0.02, udp=0.97)
        r = client.post('/api/simulate_attack',
                        json={'stats': stats, 'ip_stats': bots, 'interval_seconds': 1})
        body = r.get_json()
        assert body['is_anomaly'] is True
        assert body['alert_fired'] is True
        assert body['blacklist_size'] == 5

    def test_simulate_disabled_on_live_interface(self, client, system):
        system.allow_simulation_api = False
        r = client.post('/api/simulate_attack', json=_payload())
        assert r.status_code == 403

    def test_simulate_requires_json_content_type(self, client):
        r = client.post('/api/simulate_attack', data=json.dumps(_payload()),
                        content_type='text/plain')
        assert r.status_code == 400

    @pytest.mark.parametrize('body', [
        {'stats': {}},
        {'stats': {'total_packets': -5}},
        {'stats': {'total_packets': 'lots'}},
        {'stats': {'total_packets': 1}, 'ip_stats': [{'ip': '<img src=x onerror=alert(1)>', 'packets': 99}]},
        {'stats': {'total_packets': 1}, 'ip_stats': [{'ip': '127.0.0.1', 'packets': 99}]},
        {'stats': {'total_packets': 1}, 'interval_seconds': 0},
    ])
    def test_simulate_rejects_bad_input(self, client, body):
        r = client.post('/api/simulate_attack', json=body)
        assert r.status_code == 400

    def test_simulate_not_cors_enabled(self, client):
        r = client.post('/api/simulate_attack', json=_payload(),
                        headers={'Origin': 'http://evil.example'})
        assert 'Access-Control-Allow-Origin' not in r.headers
        r = client.get('/api/status', headers={'Origin': 'http://evil.example'})
        assert r.headers.get('Access-Control-Allow-Origin') in ('*', 'http://evil.example')


class TestDemo:
    """Built-in dashboard demo: scenarios run through the real detector"""

    @pytest.fixture
    def demo(self, system):
        from src.config import MLConfig
        from src.core.anomaly_detector import MLEnhancedAnomalyDetector
        from src.dashboard.demo import DemoRunner
        # The demo runs with the shipped model, as main.py does
        system.anomaly_detector = MLEnhancedAnomalyDetector(model_path=MLConfig.DEFAULT_MODEL_PATH)
        assert system.anomaly_detector.ml_enabled
        clock = iter(range(10_000, 20_000))
        system.demo = DemoRunner(system, step_seconds=0, clock=lambda: float(next(clock)))
        return system.demo

    def test_rules_alone_would_flag_the_flash_crowd(self, system):
        """Why the ML model is needed: the statistical rules alone call it an attack"""
        from src.dashboard.demo import DemoRunner
        clock = iter(range(10_000, 20_000))
        demo = DemoRunner(system, step_seconds=0, clock=lambda: float(next(clock)))
        demo.play('flash_crowd')
        self._run(demo, system, 15)
        assert any(self._run(demo, system, 30))

    @staticmethod
    def _run(demo, system, seconds):
        verdicts = []
        for _ in range(seconds):
            demo.tick()
            verdicts.append(system.last_result.is_anomaly)
        return verdicts

    def test_flash_crowd_is_not_blocked(self, demo, system):
        demo.play('flash_crowd')                 # warm-up normal traffic is inserted first
        assert demo.status()['scenario'] == 'normal'
        assert not any(self._run(demo, system, 15))
        assert demo.status()['scenario'] == 'flash_crowd'
        assert not any(self._run(demo, system, 30))
        assert system.traffic_monitor.get_blacklist() == []

    def test_udp_flood_detected_and_bots_dropped(self, demo, system):
        demo.play('udp_flood')
        self._run(demo, system, 15)
        verdicts = self._run(demo, system, 30)
        assert all(verdicts)
        status = demo.status()
        assert status['blocked_bots'] == 100     # every bot blocked within 30 s
        assert system._current_drop_pps > 0.9 * 8000
        kinds = [e['kind'] for e in system.get_status()['events']]
        assert 'attack' in kinds and 'block' in kinds

    def test_reset_clears_blocks_and_chart(self, demo, system):
        demo.play('udp_flood')
        self._run(demo, system, 25)
        demo.reset()
        assert system.traffic_monitor.get_blacklist() == []
        assert system.get_status()['timeline'] == []
        assert not demo.status()['running']

    def test_demo_api_requires_json_and_simulation(self, client, system):
        assert client.post('/api/demo', data='action=reset').status_code == 400
        r = client.post('/api/demo', json={'action': 'play', 'scenario': 'nope'})
        assert r.status_code == 400
        r = client.post('/api/demo', json={'action': 'stop'})
        assert r.status_code == 200 and r.get_json()['running'] is False
        system.allow_simulation_api = False
        system.demo = None
        assert client.post('/api/demo', json={'action': 'stop'}).status_code == 403

    def test_policy_switch_via_api(self, client, system):
        r = client.post('/api/demo', json={'action': 'policy', 'policy': 'p4'})
        assert r.status_code == 200 and r.get_json()['policy'] == 'p4'
        assert system.get_status()['mitigation']['title'] == 'P4 source scoring'
        assert client.post('/api/demo', json={'action': 'policy', 'policy': 'x'}).status_code == 400


class TestLiveP4:
    """P4 running in the system loop: decisions go to the (simulated) kernel maps"""

    @pytest.fixture
    def p4(self, system):
        system.set_policy('p4')
        assert system.p4 is not None, 'needs data/models/source_scorer.joblib'
        return system

    @staticmethod
    def _normal(system, seconds=20, t0=1000.0):
        total = 0
        for i in range(seconds):
            total += 100
            ips = [{'ip': f'10.0.0.{k}', 'packets': 10, 'syn_count': 0} for k in range(10)]
            system.process_sample(_stats(total), ips, now=t0 + i)
        return t0 + seconds, total

    def test_flood_blocks_bots_for_ttl_and_spares_clients(self, p4):
        t, total = self._normal(p4)
        bots = [{'ip': f'203.0.113.{k}', 'packets': 1000, 'udp_count': 1000} for k in range(20)]
        clients = [{'ip': f'10.0.0.{k}', 'packets': 10} for k in range(10)]
        for i in range(3):
            total += 20100
            p4.process_sample(_stats(total, tcp=0.02, udp=0.97), bots + clients, now=t + i)
        monitor = p4.traffic_monitor
        blocked = set(monitor.get_blacklist())
        assert {b['ip'] for b in bots} <= blocked
        assert not any(c['ip'] in blocked for c in clients)
        assert p4.get_status()['mitigation']['blocked'] >= 20
        # Blocks expire on their own: 30 s TTL
        monitor.sim_now = t + 2 + 31
        assert monitor.get_blacklist() == []
        assert monitor.get_blacklist() == [] and not monitor.is_blacklisted('203.0.113.1')

    def test_spoofed_flood_turns_on_guard_and_keeps_known_clients(self, p4):
        t, total = self._normal(p4)
        clients = [{'ip': f'10.0.0.{k}', 'packets': 10} for k in range(10)]
        for i in range(3):
            fresh = [{'ip': f'198.18.{i}.{k}', 'packets': 30, 'udp_count': 30} for k in range(1, 201)]
            total += 6100
            p4.process_sample(_stats(total, tcp=0.02, udp=0.97), fresh + clients, now=t + i)
        monitor = p4.traffic_monitor
        assert monitor.spoof_guard_active()
        flood = {f'198.19.0.{k}': 30 for k in range(1, 201)}     # 6000 pps from new addresses
        drops = monitor.simulate_drops({'10.0.0.1': 10, **flood})
        assert drops['10.0.0.1'] == 0                            # known client untouched
        assert sum(drops[ip] for ip in flood) > 0.9 * 6000       # unknown sources share a small budget

    def test_monitor_rate_limit_and_ttl_in_simulation(self, system):
        monitor = system.traffic_monitor
        monitor.sim_now = 50.0
        assert monitor.add_to_blacklist('203.0.113.7', ttl_s=10)
        assert monitor.rate_limit_ip('203.0.113.8', 20, ttl_s=10)
        drops = monitor.simulate_drops({'203.0.113.7': 100, '203.0.113.8': 100, '10.1.1.1': 5})
        assert drops == {'203.0.113.7': 100.0, '203.0.113.8': 80.0, '10.1.1.1': 0.0}
        assert monitor.get_rate_limited() == ['203.0.113.8']
        monitor.sim_now = 61.0
        assert monitor.simulate_drops({'203.0.113.7': 100, '203.0.113.8': 100}) == \
            {'203.0.113.7': 0.0, '203.0.113.8': 0.0}

    def test_kernel_map_stores_expiry(self, tmp_path):
        monitor = TrafficMonitor('veth1', blacklist_file=str(tmp_path / 'blacklist.json'))
        monitor.simulation_mode = False
        monitor.bpf = _FakeBPF()
        monitor.loaded = True
        assert monitor.add_to_blacklist('203.0.113.9', ttl_s=30)
        (value,) = monitor.bpf.maps['blacklist_map'].values()
        import time
        assert value > time.monotonic_ns()          # expiry in the bpf_ktime_get_ns() clock
        assert monitor.add_to_blacklist('203.0.113.10')
        assert monitor.bpf.maps['blacklist_map'][__import__('struct').unpack(
            'I', __import__('socket').inet_aton('203.0.113.10'))[0]] == 0   # 0 = permanent
        assert json.loads((tmp_path / 'blacklist.json').read_text()) == ['203.0.113.10']

    @staticmethod
    def _demo_outcome(system, policy):
        from src.config import MLConfig
        from src.core.anomaly_detector import MLEnhancedAnomalyDetector
        from src.dashboard.demo import DemoRunner
        system.anomaly_detector = MLEnhancedAnomalyDetector(model_path=MLConfig.DEFAULT_MODEL_PATH)
        system.set_policy(policy)
        clock = iter(range(10_000, 20_000))
        demo = system.demo = DemoRunner(system, step_seconds=0, clock=lambda: float(next(clock)))
        demo.play('udp_flood')
        for _ in range(45):
            demo.tick()
        return demo.status()

    def test_demo_p4_stops_flood_with_little_collateral(self, system):
        status = self._demo_outcome(system, 'p4')
        assert status['blocked_bots'] + status['limited_bots'] >= 90
        assert status['attack_drop_pct'] > 90
        assert status['legit_loss_pct'] < 5

    def test_demo_top5_costs_real_users_more_than_p4(self, tmp_path):
        outcomes = {}
        for policy in ('top5', 'p4'):
            sys_ = DDoSMitigationSystem('test0', ml_model_path=None, auto_detect_model=False, policy=policy)
            sys_.alert_system.alert_file = str(tmp_path / f'{policy}.log')
            sys_.traffic_profiler.profile_file = str(tmp_path / f'{policy}.json')
            monitor = TrafficMonitor('test0', blacklist_file=str(tmp_path / f'{policy}_bl.json'))
            monitor.simulation_mode, monitor.loaded = True, True
            sys_.traffic_monitor = monitor
            outcomes[policy] = self._demo_outcome(sys_, policy)
        assert outcomes['top5']['legit_loss_pct'] > 5 * outcomes['p4']['legit_loss_pct']
        assert outcomes['p4']['attack_drop_pct'] > outcomes['top5']['attack_drop_pct']
