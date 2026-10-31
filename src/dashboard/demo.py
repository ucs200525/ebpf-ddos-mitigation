"""
Dashboard demo engine (simulation mode only)

Plays scripted traffic scenarios through the real pipeline, one interval per
second: the same process_sample() call the live XDP loop makes, so detection,
mitigation and alerts are the system's own decisions. Packets are dropped by
the same rules the XDP program applies (blocks, rate limits, spoofing guard),
simulated by TrafficMonitor.simulate_drops().

Scenarios are fixed and predictable so they can be shown to an audience:
normal traffic, a legitimate flash crowd (must NOT be blocked), UDP / SYN /
multi-vector floods, and a guided tour that plays them in order.
"""

import threading
import time

import numpy as np

# name -> (title, what the audience should see, seconds)
SCENARIOS = {
    'normal':       ('Normal traffic', 'About 300 packets/s from 80 regular clients', 0),
    'flash_crowd':  ('Flash crowd (legitimate)',
                     'Traffic grows 5x from many new real users: no alert, nobody blocked', 30),
    'udp_flood':    ('UDP flood', '8,000 packets/s from 100 bots: detected, then blocked in the kernel', 30),
    'syn_flood':    ('TCP SYN flood', '6,000 SYN packets/s from 300 bots', 30),
    'multi_vector': ('Multi-vector attack', 'UDP + SYN + ICMP at once from 400 bots', 30),
}

TOUR = [('normal', 15), ('flash_crowd', 25), ('normal', 10), ('udp_flood', 30),
        ('normal', 15), ('syn_flood', 30), ('normal', 0)]

WARMUP_SECONDS = 15        # baseline needs normal samples before deviation rules apply
NORMAL_PPS = 300.0


def _zipf(rng, n, a):
    w = 1.0 / np.arange(1, n + 1) ** a
    rng.shuffle(w)
    return w / w.sum()


class DemoRunner:
    """Background thread that feeds one traffic interval per second"""

    def __init__(self, system, seed: int = 7, step_seconds: float = 1.0, clock=None):
        self.system = system
        self.clock = clock or time.time   # tests pass a simulated clock
        self.step = step_seconds
        self.rng = np.random.default_rng(seed)
        self._lock = threading.Lock()
        self._thread = None
        self._stop = threading.Event()
        self._queue = []           # [(scenario, seconds)], 0 = until changed
        self._current = None
        self._remaining = 0
        self._elapsed = 0
        self._tour = False
        self.cum = dict(total_packets=0, total_bytes=0, tcp_packets=0, udp_packets=0,
                        icmp_packets=0, dropped_packets=0, dropped_bytes=0,
                        passed_packets=0, passed_bytes=0, other_packets=0)
        # Ground truth the demo knows: what the policy cost real users (collateral damage)
        self.outcome = dict(legit=0, legit_dropped=0, attack=0, attack_dropped=0)

        # Fixed populations so blocked sources stay blocked across intervals
        self.clients = [f'172.16.{i // 250}.{i % 250 + 1}' for i in range(80)]
        self.client_w = _zipf(self.rng, 80, 0.9)
        self.crowd = [f'100.70.{i // 250}.{i % 250 + 1}' for i in range(1500)]
        self.crowd_w = _zipf(self.rng, 1500, 0.4)
        self.bots = [f'45.{33 + i // 250}.{(i * 7) % 250 + 1}.{i % 250 + 1}' for i in range(400)]

    # ---- control -------------------------------------------------------
    def play(self, scenario: str):
        """Start a scenario (or 'tour'); returns an error string or None"""
        if scenario != 'tour' and scenario not in SCENARIOS:
            return f'unknown scenario: {scenario}'
        with self._lock:
            plan = list(TOUR) if scenario == 'tour' else \
                ([(scenario, SCENARIOS[scenario][2]), ('normal', 0)] if scenario != 'normal' else [('normal', 0)])
            warm = WARMUP_SECONDS - self.system.anomaly_detector.baseline.samples
            if warm > 0 and plan[0][0] != 'normal':
                plan.insert(0, ('normal', warm))
            self._tour = scenario == 'tour'
            self._queue = plan
            self._next()
        if self.step:
            self._ensure_thread()
        return None

    def stop(self):
        with self._lock:
            self._queue, self._current, self._remaining, self._tour = [], None, 0, False

    def reset(self):
        self.stop()
        self.system.reset_demo_state()
        self.outcome = dict(legit=0, legit_dropped=0, attack=0, attack_dropped=0)

    def shutdown(self):
        self._stop.set()

    def status(self) -> dict:
        with self._lock:
            cur = self._current
            return {
                'running': cur is not None,
                'scenario': cur,
                'title': SCENARIOS[cur][0] if cur else None,
                'explain': SCENARIOS[cur][1] if cur else None,
                'seconds_left': self._remaining if cur and self._remaining > 0 else None,
                'tour': self._tour,
                'next': [SCENARIOS[s][0] for s, _ in self._queue[:3]],
                'learning': self.system.anomaly_detector.baseline.samples < WARMUP_SECONDS,
                **self._blocked_breakdown(),
                'scenarios': [{'id': k, 'title': v[0], 'explain': v[1]} for k, v in SCENARIOS.items()],
            }

    def _blocked_breakdown(self) -> dict:
        """The demo knows who is who: split blocked and slowed sources into bots and real users"""
        monitor = self.system.traffic_monitor
        blocked = set(monitor.get_blacklist())
        limited = set(monitor.get_rate_limited()) - blocked
        bots = set(self.bots)
        o = self.outcome
        pct = lambda a, b: round(100.0 * a / b, 2) if b else 0.0  # noqa: E731
        return {'blocked_bots': len(blocked & bots), 'blocked_legit': len(blocked - bots),
                'limited_bots': len(limited & bots), 'limited_legit': len(limited - bots),
                'policy': self.system.mitigation_policy,
                'legit_loss_pct': pct(o['legit_dropped'], o['legit']),
                'attack_drop_pct': pct(o['attack_dropped'], o['attack'])}

    # ---- engine --------------------------------------------------------
    def _next(self):
        if self._queue:
            self._current, self._remaining = self._queue.pop(0)
            self._elapsed = 0
        else:
            self._current, self._remaining = None, 0

    def _ensure_thread(self):
        if self._thread is None or not self._thread.is_alive():
            self._stop.clear()
            self._thread = threading.Thread(target=self._loop, name='demo-runner', daemon=True)
            self._thread.start()

    def _loop(self):
        while not self._stop.is_set():
            with self._lock:
                scenario, elapsed = self._current, self._elapsed
            if scenario is not None:
                self.tick()
            self._stop.wait(self.step)

    def tick(self):
        """Feed one interval of the current scenario and advance the plan"""
        with self._lock:
            scenario, elapsed = self._current, self._elapsed
        if scenario is None:
            return
        self._feed(scenario, elapsed)
        with self._lock:
            if self._current == scenario:
                self._elapsed += 1
                if self._remaining > 0:
                    self._remaining -= 1
                    if self._remaining == 0:
                        self._next()

    def _split(self, ips, weights, total):
        counts = self.rng.multinomial(int(total), weights)
        return [(ip, int(c)) for ip, c in zip(ips, counts) if c > 0]

    def _feed(self, scenario: str, t: int):
        rng = self.rng
        normal = NORMAL_PPS * rng.uniform(0.9, 1.1)
        entries = []   # (ip, packets, kind) kind: tcp/udp/icmp/syn
        for ip, c in self._split(self.clients, self.client_w, normal):
            entries.append((ip, c, 'mix'))

        if scenario == 'flash_crowd':
            # Real users arrive gradually (no sudden jump) with a normal protocol mix
            extra = NORMAL_PPS * min(4.0, 0.3 * (t + 1)) * rng.uniform(0.9, 1.1)
            for ip, c in self._split(self.crowd, self.crowd_w, extra):
                entries.append((ip, c, 'mix'))
        elif scenario in ('udp_flood', 'syn_flood', 'multi_vector'):
            ramp = min(1.0, 0.5 + 0.25 * t)
            if scenario == 'udp_flood':
                vectors = [(self.bots[:100], 8000, 'udp')]
            elif scenario == 'syn_flood':
                vectors = [(self.bots[100:400], 6000, 'syn')]
            else:
                vectors = [(self.bots[:200], 5000, 'udp'), (self.bots[200:400], 4000, 'syn'),
                           (self.bots[:400:4], 2000, 'icmp')]
            for bots, pps, kind in vectors:
                w = np.full(len(bots), 1.0 / len(bots))
                for ip, c in self._split(bots, w, pps * ramp * rng.uniform(0.95, 1.05)):
                    entries.append((ip, c, kind))

        self._emit(entries, SCENARIOS[scenario][0])

    def _emit(self, entries, label):
        monitor = self.system.traffic_monitor
        now = self.clock()
        monitor.sim_now = now
        totals = {}
        for ip, c, _ in entries:
            totals[ip] = totals.get(ip, 0) + c
        drop_share = {ip: d / totals[ip] for ip, d in monitor.simulate_drops(totals).items() if totals[ip]}
        per_ip = {}
        cum = self.cum
        for ip, c, kind in entries:
            if kind == 'mix':
                tcp, udp, icmp = (int(x) for x in self.rng.multinomial(c, (0.85, 0.12, 0.03)))
                syn = int(self.rng.binomial(tcp, 0.04))
            elif kind == 'udp':
                tcp, udp, icmp, syn = 0, c, 0, 0
            elif kind == 'syn':
                tcp, udp, icmp, syn = c, 0, 0, c
            else:
                tcp, udp, icmp, syn = 0, 0, c, 0
            size = 900 if kind == 'mix' else 80
            cum['total_packets'] += c; cum['total_bytes'] += c * size
            cum['tcp_packets'] += tcp; cum['udp_packets'] += udp; cum['icmp_packets'] += icmp
            # XDP counts every packet, then drops by its mitigation rules
            d = int(round(c * drop_share.get(ip, 0.0)))
            side = 'attack' if kind in ('udp', 'syn', 'icmp') else 'legit'
            self.outcome[side] += c
            self.outcome[side + '_dropped'] += d
            cum['dropped_packets'] += d; cum['dropped_bytes'] += d * size
            cum['passed_packets'] += c - d; cum['passed_bytes'] += (c - d) * size
            e = per_ip.setdefault(ip, {'ip': ip, 'packets': 0, 'bytes': 0, 'flow_count': 1,
                                       'syn_count': 0, 'udp_count': 0})
            e['packets'] += c; e['bytes'] += c * size; e['syn_count'] += syn; e['udp_count'] += udp
        self.system.process_sample(dict(cum), list(per_ip.values()), label=label, now=now)
