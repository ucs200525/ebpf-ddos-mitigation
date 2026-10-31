"""
Traffic Dataset - Labeled synthetic traffic runs for training the window classifier

Each run is a per-second sequence of the same counters the XDP program
exports (cumulative totals + per-interval per-source-IP counts). Runs are
fed through FeatureExtractor, so the model is trained on exactly the
features it will see at inference time.

Benign scenarios include hard negatives (bursty traffic, flash crowds);
attack scenarios include low-rate floods and spoofed-source floods so that
neither volume nor source concentration alone separates the classes.

This is SYNTHETIC data: it encodes assumptions about traffic shapes and
is not a substitute for evaluation on captured traffic.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .feature_extractor import FeatureExtractor

logger = logging.getLogger(__name__)

BENIGN_SCENARIOS = ['normal', 'normal_bursty', 'flash_crowd']
ATTACK_SCENARIOS = [
    'udp_flood', 'syn_flood', 'icmp_flood', 'mixed_flood',
    'udp_flood_spoofed', 'syn_flood_spoofed', 'low_rate_udp', 'low_rate_syn',
]
ALL_SCENARIOS = BENIGN_SCENARIOS + ATTACK_SCENARIOS

# Mitigation-safety scenarios: evaluation only, never used for training or in
# generate_dataset / generate_runs. They stress what legitimate traffic can look like:
#   heavy_legit                  1-5 existing clients start bulk transfers (3-15x the
#                                normal level, no attack)
#   udp_flood_with_crowd         botnet UDP flood while a flash crowd of new users arrives
#   udp_flood_spoofed_with_crowd spoofed UDP flood while a flash crowd of new users arrives
SAFETY_SCENARIOS = ['heavy_legit', 'udp_flood_with_crowd', 'udp_flood_spoofed_with_crowd']
_SAFETY_BASE = {'udp_flood_with_crowd': 'udp_flood',
                'udp_flood_spoofed_with_crowd': 'udp_flood_spoofed'}

# Upper bound on per-interval source entries emitted (runtime reports top 1000)
_MAX_SOURCES = 1500

# Packet-header signatures (for signature-based filtering, Dimolianis et al. 2021).
# Each source entry carries the dominant header signature of its packets:
# TCP (observed TTL, SYN window) or UDP/ICMP (observed TTL, packet size).
# The model is simple on purpose; real clients mix several signatures.
# Client operating systems use common stack defaults (initial TTL, SYN window);
# observed TTL = initial TTL - hop count.
_OS_PROFILES = [(128, 64240), (64, 64240), (64, 65535), (64, 29200), (255, 4128)]
_OS_WEIGHTS = [0.40, 0.15, 0.38, 0.05, 0.02]   # Windows, Linux, macOS/iOS/Android, old Linux, devices
_PROTO_TCP, _PROTO_UDP, _PROTO_ICMP = 1, 2, 3
_QUIC_SHARE = 0.3                               # clients whose packets are mostly QUIC/UDP


def signature_code(proto: int, ttl, value) -> np.ndarray:
    """Pack (protocol class, observed TTL, window or size) into one integer per entry"""
    return (np.int64(proto) << 40) | (np.asarray(ttl, dtype=np.int64) << 24) | np.asarray(value, dtype=np.int64)


class _HostSignatures:
    """Stable OS-stack signature per source id (TCP: observed TTL, SYN window)"""

    def __init__(self, rng: np.random.Generator):
        self.rng = rng
        self.codes: Dict[int, int] = {}

    def draw(self, n: int) -> np.ndarray:
        os_i = self.rng.choice(len(_OS_PROFILES), size=n, p=_OS_WEIGHTS)
        hops = self.rng.integers(4, 21, size=n)
        init = np.array([_OS_PROFILES[i][0] for i in os_i])
        win = np.array([_OS_PROFILES[i][1] for i in os_i])
        # Some clients mostly send UDP (QUIC fills packets to 1200-1350 bytes)
        quic = self.rng.random(n) < _QUIC_SHARE
        size = self.rng.integers(1200, 1351, size=n)
        return np.where(quic, signature_code(_PROTO_UDP, init - hops, size),
                        signature_code(_PROTO_TCP, init - hops, win))

    def of(self, ids: np.ndarray) -> np.ndarray:
        missing = [int(i) for i in ids if int(i) not in self.codes]
        if missing:
            self.codes.update(zip(missing, map(int, self.draw(len(missing)))))
        return np.array([self.codes[int(i)] for i in ids], dtype=np.int64)

# Source-identity ranges (ids are stable across a run so mitigation can track sources)
_ID_CROWD = 1_000_000
_ID_BOT = 2_000_000
_ID_SPOOF = 3_000_000
_ID_CHURN = 50_000_000


@dataclass
class TrafficStep:
    """One second of traffic, as the monitor would report it"""
    timestamp: float
    stats: Dict[str, int]          # cumulative counters
    ip_packets: np.ndarray         # per-source packets this interval
    ip_syn: np.ndarray             # per-source SYN packets this interval
    is_attack: bool                # attack traffic present this second
    ip_ids: Optional[np.ndarray] = None        # stable source identity per entry
    ip_is_attack: Optional[np.ndarray] = None  # ground truth: entry is an attack source
    ip_sig: Optional[np.ndarray] = None        # header signature per entry (signature_code)

    def ip_stats(self) -> List[Dict]:
        """Per-IP dicts in the format TrafficMonitor produces"""
        ids = self.ip_ids if self.ip_ids is not None else np.arange(len(self.ip_packets))
        return [
            {'ip': source_ip(int(i)), 'packets': int(p), 'syn_count': int(s),
             'bytes': int(p) * 500, 'flow_count': 1, 'udp_count': 0}
            for i, p, s in zip(ids, self.ip_packets, self.ip_syn) if p > 0
        ]


def source_ip(source_id: int) -> str:
    """Deterministic IPv4 address for a synthetic source id (10.0.0.0/8 space)"""
    return f'10.{(source_id >> 16) & 255}.{(source_id >> 8) & 255}.{source_id & 255}'


@dataclass
class TrafficRun:
    scenario: str
    steps: List[TrafficStep] = field(default_factory=list)
    tool_os_like: bool = False     # attack packets reuse common OS stack signatures


def _zipf_weights(rng: np.random.Generator, n: int, a: float) -> np.ndarray:
    w = 1.0 / np.arange(1, n + 1) ** a
    rng.shuffle(w)
    return w / w.sum()


def generate_run(rng: np.random.Generator, scenario: str, start_time: float = 1_000_000.0,
                 client_churn: float = 0.0, bot_pre_activity: float = 0.0) -> TrafficRun:
    """
    Generate one labeled traffic run of the given scenario.

    Optional realism (both off by default, which keeps the training data and
    the shipped model's evaluation reproducible):
      client_churn:     share of normal traffic carried by brand-new, short-lived
                        legitimate clients (new users keep arriving)
      bot_pre_activity: probability that each bot also sends low-rate,
                        normal-looking traffic before (and during) the attack

    Header signatures are drawn from a separate generator seeded from the
    current state of `rng` (reading the state does not advance it), so adding
    them leaves every packet count identical to earlier versions.
    """
    state = rng.bit_generator.state.get('state', {})
    sig_rng = np.random.default_rng(int(state.get('state', 0)) if isinstance(state, dict) else 0)
    hosts = _HostSignatures(sig_rng)
    duration = int(rng.integers(70, 121))
    onset = int(rng.integers(30, 56))
    # Some events end early so recovery windows are represented
    end = duration if rng.random() < 0.7 else min(duration, onset + int(rng.integers(15, 35)))

    base_pps = float(np.exp(rng.uniform(np.log(5), np.log(5000))))
    noise_sd = rng.uniform(0.08, 0.25)
    tcp_r = rng.uniform(0.70, 0.95)
    icmp_r = rng.uniform(0.0, 0.04)
    udp_r = max(0.0, 1.0 - tcp_r - icmp_r)
    normal_pkt_size = rng.uniform(300, 1100)
    syn_rate = rng.uniform(0.01, 0.08)

    n_clients = int(np.exp(rng.uniform(np.log(5), np.log(3000))))
    client_w = _zipf_weights(rng, n_clients, rng.uniform(0.6, 1.3))

    # Scenario-specific event parameters. Safety scenarios draw their extra
    # parameters only in their own branches, so other scenarios are unchanged.
    attack = _SAFETY_BASE.get(scenario, scenario)
    with_crowd = scenario.endswith('_with_crowd')
    is_attack_run = attack in ATTACK_SCENARIOS
    if attack.startswith('low_rate'):
        mult = rng.uniform(1.3, 2.5)
    elif scenario in ('flash_crowd', 'heavy_legit'):
        mult = rng.uniform(3, 15)
    else:
        mult = float(np.exp(rng.uniform(np.log(3), np.log(80))))
    ramp = int(rng.integers(1, 15))
    spoofed = attack.endswith('_spoofed')
    n_bots = int(np.exp(rng.uniform(np.log(1), np.log(2000))))
    bot_w = _zipf_weights(rng, n_bots, rng.uniform(0.0, 0.6))
    crowd_size = int(rng.uniform(500, 5000))
    crowd_w = _zipf_weights(rng, crowd_size, rng.uniform(0.2, 0.8))
    crowd_mult = rng.uniform(3, 15) if with_crowd else 0.0
    heavy = (rng.choice(n_clients, size=min(n_clients, int(rng.integers(1, 6))), replace=False)
             if scenario == 'heavy_legit' else None)
    burst_until = -1

    # Attack tool header profile: crafted values that no client stack uses by default,
    # or ordinary OS stacks (then attack and legitimate signatures can collide)
    tool_os_like = bool(sig_rng.random() < 0.5)
    tool_window = int(sig_rng.integers(1, 65536))
    tool_hops = int(sig_rng.integers(4, 21))      # the spoofing attacker's distance
    tool_size = int(sig_rng.integers(64, 1401))   # flood tools send one fixed packet size
    bot_hops = sig_rng.integers(4, 21, size=n_bots)
    bot_proto = None

    # Optional realism state (no random draws unless enabled)
    sessions = []          # churn sessions: [id, end_second, rate_pps]
    next_session_id = _ID_CHURN
    bot_pre_rate = np.zeros(n_bots)
    if bot_pre_activity > 0 and is_attack_run and not spoofed:
        active_bots = rng.random(n_bots) < bot_pre_activity
        bot_pre_rate = np.where(active_bots, rng.uniform(0.2, 2.0, n_bots), 0.0)

    cum = dict(total_packets=0, total_bytes=0, tcp_packets=0, udp_packets=0, icmp_packets=0)
    run = TrafficRun(scenario=scenario, tool_os_like=tool_os_like)

    for t in range(duration):
        # --- normal background ---
        level = base_pps * max(0.05, 1 + rng.normal(0, noise_sd))
        if scenario == 'normal_bursty':
            if t > burst_until and rng.random() < 0.04:
                burst_until = t + int(rng.integers(2, 7))
            if t <= burst_until:
                level *= rng.uniform(1.5, 3.5)
        n_norm = int(rng.poisson(level))
        norm_ip = rng.multinomial(n_norm, client_w).astype(np.float64)
        norm_tcp = rng.binomial(n_norm, tcp_r)
        norm_icmp = rng.binomial(n_norm - norm_tcp, icmp_r / max(1e-9, 1 - tcp_r))
        norm_udp = n_norm - norm_tcp - norm_icmp
        norm_syn = rng.binomial(norm_ip.astype(np.int64), tcp_r * syn_rate).astype(np.float64)
        norm_bytes = int(n_norm * normal_pkt_size * rng.uniform(0.9, 1.1))

        tcp, udp, icmp, nbytes = norm_tcp, norm_udp, norm_icmp, norm_bytes
        ip_pk, ip_syn = [norm_ip], [norm_syn]
        ip_ids = [np.arange(n_clients)]
        ip_atk = [np.zeros(n_clients, dtype=bool)]
        ip_sig = [hosts.of(ip_ids[0])]

        if client_churn > 0:
            mean_rate = 4.0
            mean_life = 8.0
            for _ in range(int(rng.poisson(level * client_churn / mean_rate / mean_life))):
                sessions.append([next_session_id, t + int(rng.integers(3, 15)),
                                 rng.uniform(1, 2 * mean_rate)])
                next_session_id += 1
            sessions = [ss for ss in sessions if ss[1] > t]
            if sessions:
                pk = rng.poisson([ss[2] for ss in sessions]).astype(np.float64)
                k = int(pk.sum())
                tcp_k = int(round(k * tcp_r))
                tcp += tcp_k
                udp += k - tcp_k
                nbytes += int(k * normal_pkt_size)
                ip_pk.append(pk)
                ip_syn.append(rng.binomial(pk.astype(np.int64), 0.1).astype(np.float64))
                ip_ids.append(np.array([ss[0] for ss in sessions]))
                ip_atk.append(np.zeros(len(pk), dtype=bool))
                ip_sig.append(hosts.of(ip_ids[-1]))

        active = onset <= t < end
        ramp_factor = min(1.0, (t - onset + 1) / ramp) if active else 0.0
        attack_now = False

        if active and heavy is not None:
            # Bulk transfers by known clients: large TCP data packets, no extra SYNs
            extra = rng.poisson(base_pps * (mult - 1) * ramp_factor / len(heavy), size=len(heavy))
            norm_ip[heavy] += extra
            tcp += int(extra.sum())
            nbytes += int(extra.sum()) * 1400

        if active and scenario == 'flash_crowd':
            n_extra = int(rng.poisson(base_pps * (mult - 1) * ramp_factor))
            extra_ip = rng.multinomial(n_extra, crowd_w).astype(np.float64)
            extra_syn = rng.binomial(extra_ip.astype(np.int64), rng.uniform(0.05, 0.15)).astype(np.float64)
            extra_tcp = rng.binomial(n_extra, rng.uniform(0.85, 0.98))
            tcp += extra_tcp
            udp += n_extra - extra_tcp
            nbytes += int(n_extra * rng.uniform(400, 1200))
            ip_pk.append(extra_ip)
            ip_syn.append(extra_syn)
            ip_ids.append(_ID_CROWD + np.arange(crowd_size))
            ip_atk.append(np.zeros(crowd_size, dtype=bool))
            ip_sig.append(hosts.of(ip_ids[-1]))

        elif active and is_attack_run:
            attack_now = True
            n_atk = int(rng.poisson(base_pps * mult * ramp_factor))
            kind = attack.replace('_spoofed', '').replace('low_rate_', '')
            if kind in ('udp', 'udp_flood'):
                shares = {'udp': 1.0}
                size = rng.uniform(64, 1400)
            elif kind in ('syn', 'syn_flood'):
                shares = {'syn': 1.0}
                size = rng.uniform(54, 74)
            elif kind == 'icmp_flood':
                shares = {'icmp': 1.0}
                size = rng.uniform(64, 1024)
            else:  # mixed
                u, s = rng.dirichlet([2, 2, 1])[:2]
                shares = {'udp': u, 'syn': s, 'icmp': 1 - u - s}
                size = rng.uniform(64, 1000)

            a_udp = int(n_atk * shares.get('udp', 0))
            a_syn = int(n_atk * shares.get('syn', 0))
            a_icmp = n_atk - a_udp - a_syn
            udp += a_udp
            tcp += a_syn
            icmp += a_icmp
            nbytes += int(n_atk * size)

            if spoofed:
                # Random source per packet: many sources with ~1 packet each
                n_src = min(n_atk, _MAX_SOURCES)
                atk_ip = np.full(n_src, n_atk / max(n_src, 1)) if n_src else np.zeros(0)
                atk_ip = np.floor(atk_ip + rng.random(n_src))
                atk_ids = _ID_SPOOF + t * _MAX_SOURCES + np.arange(n_src)  # fresh every second
            else:
                atk_ip = rng.multinomial(n_atk, bot_w).astype(np.float64)
                atk_ids = _ID_BOT + np.arange(n_bots)
            syn_frac = a_syn / n_atk if n_atk else 0.0
            atk_syn = np.floor(atk_ip * syn_frac)
            ip_pk.append(atk_ip)
            ip_syn.append(atk_syn)
            ip_ids.append(atk_ids)
            ip_atk.append(np.ones(len(atk_ip), dtype=bool))

            # Header signature of the flood packets
            if bot_proto is None:
                names = list(shares)
                pick = sig_rng.choice(len(names), size=n_bots, p=[shares[k] for k in names])
                code = {'udp': _PROTO_UDP, 'syn': _PROTO_TCP, 'icmp': _PROTO_ICMP}
                bot_proto = np.array([code[names[i]] for i in pick])
            if spoofed:
                hops = np.full(len(atk_ip), tool_hops)          # one attacking machine
                proto = np.full(len(atk_ip), bot_proto[0])
            else:
                hops, proto = bot_hops, bot_proto
            if tool_os_like and not spoofed:
                tcp_sig = hosts.of(atk_ids)                     # the bots' own stacks
            else:
                tcp_sig = signature_code(_PROTO_TCP, 64 - hops, 64240 if tool_os_like else tool_window)
            size_val = tool_size
            ip_sig.append(np.where(proto == _PROTO_TCP, tcp_sig,
                                   np.where(proto == _PROTO_UDP,
                                            signature_code(_PROTO_UDP, 64 - hops, size_val),
                                            signature_code(_PROTO_ICMP, 64 - hops, size_val))))

        if active and with_crowd:
            # New legitimate users arrive during the attack (same crowd model as flash_crowd)
            n_extra = int(rng.poisson(base_pps * (crowd_mult - 1) * ramp_factor))
            extra_ip = rng.multinomial(n_extra, crowd_w).astype(np.float64)
            extra_syn = rng.binomial(extra_ip.astype(np.int64), rng.uniform(0.05, 0.15)).astype(np.float64)
            extra_tcp = rng.binomial(n_extra, rng.uniform(0.85, 0.98))
            tcp += extra_tcp
            udp += n_extra - extra_tcp
            nbytes += int(n_extra * rng.uniform(400, 1200))
            ip_pk.append(extra_ip)
            ip_syn.append(extra_syn)
            ip_ids.append(_ID_CROWD + np.arange(crowd_size))
            ip_atk.append(np.zeros(crowd_size, dtype=bool))
            ip_sig.append(hosts.of(ip_ids[-1]))

        if bot_pre_rate.any():
            # Bots keep a low-rate, normal-looking background before and during the attack
            pre = rng.poisson(bot_pre_rate).astype(np.float64)
            k = int(pre.sum())
            tcp_k = int(round(k * tcp_r))
            tcp += tcp_k
            udp += k - tcp_k
            nbytes += int(k * normal_pkt_size)
            ip_pk.append(pre)
            ip_syn.append(np.zeros(n_bots))
            ip_ids.append(_ID_BOT + np.arange(n_bots))
            ip_atk.append(np.ones(n_bots, dtype=bool))
            ip_sig.append(hosts.of(ip_ids[-1]))             # normal-looking traffic: OS stack

        cum['total_packets'] += tcp + udp + icmp
        cum['total_bytes'] += nbytes
        cum['tcp_packets'] += tcp
        cum['udp_packets'] += udp
        cum['icmp_packets'] += icmp

        run.steps.append(TrafficStep(
            timestamp=start_time + t,
            stats=dict(cum),
            ip_packets=np.concatenate(ip_pk),
            ip_syn=np.concatenate(ip_syn),
            is_attack=attack_now,
            ip_ids=np.concatenate(ip_ids),
            ip_is_attack=np.concatenate(ip_atk),
            ip_sig=np.concatenate(ip_sig),
        ))

    return run


def run_to_windows(run: TrafficRun, window_size: float = 10.0,
                   min_samples: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """
    Feed a run through FeatureExtractor and return (features, labels).
    A window is labeled attack when most of its seconds contain attack traffic.
    """
    extractor = FeatureExtractor(window_size=window_size)
    w = int(window_size)
    X, y = [], []
    for i, step in enumerate(run.steps):
        extractor.update_arrays(step.stats, step.ip_packets, step.ip_syn, timestamp=step.timestamp)
        if i < min_samples:
            continue
        lo = max(1, i - w + 1)
        frac = np.mean([s.is_attack for s in run.steps[lo:i + 1]])
        X.append(extractor.extract_features(step.timestamp).features)
        y.append(int(frac >= 0.5))
    return np.array(X), np.array(y)


def generate_dataset(n_runs: int, seed: int, window_size: float = 10.0,
                     attack_fraction: float = 0.6) -> Dict:
    """
    Generate a labeled window dataset.

    Returns dict with X, y, groups (run id, for leakage-free splits),
    scenarios (per-window scenario name), feature_names.
    """
    rng = np.random.default_rng(seed)
    X_parts, y_parts, groups, scenarios = [], [], [], []
    for run_id in range(n_runs):
        if rng.random() < attack_fraction:
            scenario = str(rng.choice(ATTACK_SCENARIOS))
        else:
            scenario = str(rng.choice(BENIGN_SCENARIOS))
        run = generate_run(rng, scenario)
        X, y = run_to_windows(run, window_size)
        X_parts.append(X)
        y_parts.append(y)
        groups.extend([run_id] * len(y))
        scenarios.extend([scenario] * len(y))

    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)
    logger.info(f"Generated {n_runs} runs -> {len(y)} windows "
                f"({int(y.sum())} attack, {int(len(y) - y.sum())} benign), seed={seed}")
    return {
        'X': X,
        'y': y,
        'groups': np.array(groups),
        'scenarios': np.array(scenarios),
        'feature_names': list(FeatureExtractor().feature_names),
    }


def generate_scenario_runs(scenario: str, n_runs: int, seed: int, client_churn: float = 0.0,
                           bot_pre_activity: float = 0.0) -> List[TrafficRun]:
    """Generate n_runs of one scenario (used for the safety scenarios)"""
    rng = np.random.default_rng(seed)
    return [generate_run(rng, scenario, client_churn=client_churn, bot_pre_activity=bot_pre_activity)
            for _ in range(n_runs)]


def generate_runs(n_runs: int, seed: int, attack_fraction: float = 0.6,
                  client_churn: float = 0.0, bot_pre_activity: float = 0.0) -> List[TrafficRun]:
    """Generate raw runs (for end-to-end detector and mitigation evaluation)"""
    rng = np.random.default_rng(seed)
    runs = []
    for _ in range(n_runs):
        pool = ATTACK_SCENARIOS if rng.random() < attack_fraction else BENIGN_SCENARIOS
        runs.append(generate_run(rng, str(rng.choice(pool)),
                                 client_churn=client_churn, bot_pre_activity=bot_pre_activity))
    return runs
