"""
CIC-DDoS2019 (Sharafaldin et al., ICCST 2019) as seen by the XDP program.

scripts/cic_extract.py turns the packet captures into two tables per day
(data/external/cic/<day>.npz): per (second, source) counters and per
(10-second window, 5-tuple) flow counters, inbound traffic of the victim
network 192.168.50.0/24 only. This module labels them and turns them into
the inputs of the three levels:

  level 1  per-second TrafficSteps -> FeatureExtractor windows
  level 2  per-second TrafficSteps -> source scorer / mitigation policies
  level 3  flow rows -> FLOW_FEATURES

Ground truth, from the dataset's own CSV labels (scripts/cic_inspect.py ->
data/results/cic_inventory.json): every flow labelled as an attack has source
172.16.0.5 (the attack machine; the few others are the victim's replies), and
the attack labels tile each day without gaps. So:

  attack packet / flow   source address is 172.16.0.5 (exactly the CIC labelling)
  attack type            the CSV label whose time range contains the packet
  attack second          the attack machine sent >= ATTACK_MIN_PPS packets that second
                         (the labels also cover minutes in which the machine is
                         almost silent between attacks; those seconds are not
                         counted as "under attack" for detection)

Day 01-12 is the dataset's training day and 03-11 its testing day; nothing
from 03-11 is used for any fitting or threshold choice here.

Measured at the ingress (scripts/cic_experiment.py --stages data): outside its
attacks the attack machine sends a little background traffic (median 4 packets/s
on 01-12, 0 on 03-11). The 03-11 Portmap and UDP-lag attacks never reach flood
rates there (outside the flood episodes 1.8 and 4.3 packets/s on average, at most
432 and 104 in any second), so they create no load. The floods run at a median of
16,000-27,000 packets/s per episode on 03-11 and up to 42,000 on 01-12. Benign
traffic is light: 13-18 packets/s on average, bursts up to about 8,800.
"""

import ipaddress
import json
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Tuple

import numpy as np

from src.config import DATA_DIR
from .traffic_dataset import TrafficRun, TrafficStep, signature_code

CIC_DIR = DATA_DIR / 'external' / 'cic'
INVENTORY_PATH = DATA_DIR / 'results' / 'cic_inventory.json'
TRAIN_DAY, TEST_DAY = '01-12', '03-11'
ATTACKER_IP = int(ipaddress.IPv4Address('172.16.0.5'))
ATTACK_MIN_PPS = 100
# The CSV timestamps are local time in New Brunswick (Atlantic time). Checked against the
# captures: the first CSV flow of each day starts at the same microsecond as a packet
# (03-11: daylight time, UTC-3; 01-12: standard time, UTC-4).
UTC_OFFSET_H = {'01-12': -4, '03-11': -3}
_DATE = {'01-12': '2018-12-01', '03-11': '2018-11-03'}


def load_day(day: str) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """(per-second-per-source table, flow table) of one day, sorted by time"""
    path = CIC_DIR / f'{day}.npz'
    if not path.exists():
        raise FileNotFoundError(f'{path} missing: run scripts/cic_extract.py first')
    with np.load(path) as d:
        persec = {k[7:]: d[k] for k in d.files if k.startswith('persec_')}
        flows = {k[5:]: d[k] for k in d.files if k.startswith('flow_')}
    order = np.lexsort((persec['src'], persec['sec']))
    persec = {k: v[order] for k, v in persec.items()}
    return persec, flows


def load_flow_seconds(day: str, replay_windows: bool = False) -> Dict[str, np.ndarray]:
    """Flow counters per (second, 5-tuple), sorted by (source, second): a flow as it stood at
    any second of its window can be rebuilt from them with cic_extract.merge_flows.
    replay_windows=True falls back to the table of the live replay windows only
    (scripts/cic_replay_fsec.py), for use on those windows alone."""
    with np.load(CIC_DIR / f'{day}.npz') as d:
        f = {k[5:]: d[k] for k in d.files if k.startswith('fsec_')}
    replay = CIC_DIR / f'{day}_replay_fsec.npz'
    if not f and replay_windows and replay.exists():
        with np.load(replay) as d:
            f = {k[5:]: d[k] for k in d.files}
    if not f:
        raise FileNotFoundError(f'no per-second flow table for {day}: rerun scripts/cic_extract.py')
    order = np.lexsort((f['slot'], f['src']))
    return {k: v[order] for k, v in f.items()}


def _local_to_epoch(day: str, stamp: str) -> float:
    t = datetime.fromisoformat(stamp)
    return t.replace(tzinfo=timezone(timedelta(hours=UTC_OFFSET_H[day]))).timestamp()


def label_periods(day: str) -> List[Tuple[str, float, float]]:
    """(label, start, end) in epoch seconds of every attack label of a day, from the CSV inventory"""
    inv = json.loads(INVENTORY_PATH.read_text())
    spans: Dict[str, List[float]] = {}
    for name, labels in inv['files'].items():
        if f':{day}/' not in name:
            continue
        for label, s in labels.items():
            if label == 'BENIGN':
                continue
            lo, hi = _local_to_epoch(day, s['t_min']), _local_to_epoch(day, s['t_max'])
            cur = spans.setdefault(label, [lo, hi])
            cur[0], cur[1] = min(cur[0], lo), max(cur[1], hi)
    return sorted(((k, v[0], v[1]) for k, v in spans.items()), key=lambda p: p[1])


def attack_type_of(day: str, t: np.ndarray) -> np.ndarray:
    """Attack label whose time range contains t ('' outside every range).
    Ranges overlap only for WebDDoS (439 flows inside the UDP-lag range); the longer
    range wins there, so those seconds are named UDP-lag."""
    t = np.asarray(t, dtype=np.float64)
    out = np.full(t.shape, '', dtype=object)
    for label, lo, hi in sorted(label_periods(day), key=lambda p: -(p[2] - p[1])):
        m = (t >= lo) & (t <= hi) & (out == '')
        out[m] = label
    return out


def source_signatures(persec: Dict[str, np.ndarray]) -> np.ndarray:
    """Header signature (as in trace_experiment / SignatureFilter) of each row's source"""
    p = persec['sig_proto']
    cls = np.select([p == 6, p == 17, p == 1], [1, 2, 3], 2).astype(np.int64)
    return (cls << 40) | signature_code(0, persec['sig_ttl'], persec['sig_value'].astype(np.int64))


def second_index(persec: Dict[str, np.ndarray]):
    """Start and end of the capture (whole seconds) and row bounds per second"""
    t0, t1 = int(persec['sec'].min()), int(persec['sec'].max()) + 1
    bounds = np.searchsorted(persec['sec'], np.arange(t0, t1 + 1))
    return t0, t1, bounds


def attacker_pps(persec: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(seconds, attacker packets, other packets) for every second of the capture"""
    t0, t1, _ = second_index(persec)
    rel = persec['sec'] - t0
    atk = persec['src'] == ATTACKER_IP
    n = t1 - t0
    return (np.arange(t0, t1),
            np.bincount(rel[atk], weights=persec['packets'][atk], minlength=n),
            np.bincount(rel[~atk], weights=persec['packets'][~atk], minlength=n))


def attack_episodes(persec, min_gap: int = 30, min_len: int = 5) -> List[Tuple[int, int]]:
    """Maximal stretches [start, end) of attack seconds, merging gaps shorter than min_gap"""
    secs, atk, _ = attacker_pps(persec)
    on = np.flatnonzero(atk >= ATTACK_MIN_PPS)
    if not len(on):
        return []
    episodes, start, prev = [], on[0], on[0]
    for i in on[1:]:
        if i - prev > min_gap:
            episodes.append((start, prev + 1))
            start = i
        prev = i
    episodes.append((start, prev + 1))
    return [(int(secs[a]), int(secs[0] + b)) for a, b in episodes if b - a >= min_len]


def build_run(persec: Dict[str, np.ndarray], start: int, end: int, scenario: str,
              sigs: np.ndarray = None, bounds=None, t0=None) -> TrafficRun:
    """One TrafficRun of real traffic: seconds [start, end) of the capture"""
    if bounds is None:
        t0, _, bounds = second_index(persec)
    if sigs is None:
        sigs = source_signatures(persec)
    run = TrafficRun(scenario=scenario)
    cum = dict(total_packets=0, total_bytes=0, tcp_packets=0, udp_packets=0, icmp_packets=0)
    for t in range(start, end):
        i = t - t0
        lo, hi = (bounds[i], bounds[i + 1]) if 0 <= i < len(bounds) - 1 else (0, 0)
        sl = slice(lo, hi)
        src = persec['src'][sl].astype(np.int64)
        pk = persec['packets'][sl].astype(np.float64)
        atk = src == ATTACKER_IP
        cum['total_packets'] += int(pk.sum())
        cum['total_bytes'] += int(persec['bytes'][sl].sum())
        cum['tcp_packets'] += int(persec['tcp'][sl].sum())
        cum['udp_packets'] += int(persec['udp'][sl].sum())
        cum['icmp_packets'] += int(persec['icmp'][sl].sum())
        run.steps.append(TrafficStep(
            timestamp=float(t), stats=dict(cum), ip_packets=pk,
            ip_syn=persec['syn'][sl].astype(np.float64),
            is_attack=bool(pk[atk].sum() >= ATTACK_MIN_PPS),
            ip_ids=src, ip_is_attack=atk, ip_sig=sigs[sl]))
    return run


def flow_labels(day: str, flows: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """(is_attack, attack type) per flow row"""
    is_attack = flows['src'] == ATTACKER_IP
    kind = attack_type_of(day, flows['t_first'])
    kind = np.where(is_attack, kind, 'BENIGN')
    return is_attack, kind


def ip_str(ip: int) -> str:
    return str(ipaddress.IPv4Address(int(ip)))
