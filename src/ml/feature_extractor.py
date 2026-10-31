"""
Feature Extractor - Window-level traffic features for real-time ML classification

The XDP program exposes aggregate counters (total packets/bytes, protocol
counts) and per-source-IP counters (packets, bytes, SYN count, UDP count).
It does NOT expose per-flow timing, so CIC-style per-flow features (IAT,
active/idle times, header lengths, ...) cannot be computed honestly at
runtime. This extractor therefore computes features over a sliding time
window of per-interval samples, using only what eBPF actually provides.

The same extractor is used to build the training set
(src/ml/traffic_dataset.py), so training and inference see identical
feature definitions.
"""

import logging
import math
import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

# Bump when feature definitions change; stored with trained models
FEATURE_SET_VERSION = 2

FEATURE_NAMES = [
    'pps_mean',              # mean packets/s over the window
    'pps_max',               # peak packets/s in the window
    'pps_cv',                # std/mean of packets/s (burstiness)
    'bps_mean',              # mean bytes/s over the window
    'avg_packet_size',       # bytes per packet over the window
    'tcp_ratio',
    'udp_ratio',
    'icmp_ratio',
    'syn_share',             # SYN packets / packets, over listed source IPs
    'unique_src_ips',        # mean distinct source IPs per interval
    'src_ip_entropy_norm',   # Shannon entropy of sources / log2(n), 0..1
    'top1_share',            # share of packets from the heaviest source
    'top10_share',           # share of packets from the 10 heaviest sources
    'packets_per_ip',        # mean packets per source IP per interval
    'pps_ratio_long',        # window mean pps / long-term median pps
    'pps_growth',            # latest interval pps / window mean pps
]

# Cap on per-interval source IPs considered (mirrors TrafficMonitor limit)
MAX_IPS_PER_INTERVAL = 1000


@dataclass
class FlowFeatures:
    """Container for extracted features"""
    features: np.ndarray
    feature_names: List[str]
    timestamp: float
    flow_count: int

    def to_dict(self) -> Dict:
        return {name: float(value) for name, value in zip(self.feature_names, self.features)}


@dataclass
class _Sample:
    """One monitoring interval"""
    timestamp: float
    duration: float
    packets: int
    bytes: int
    tcp: int
    udp: int
    icmp: int
    syn: int
    ip_packets: int
    unique_ips: int
    entropy_norm: float
    top1_share: float
    top10_share: float


class FeatureExtractor:
    """Extract window-level traffic features from eBPF statistics"""

    def __init__(self, window_size: float = 10.0, long_window: float = 300.0):
        """
        Args:
            window_size: Seconds of history used for feature computation
            long_window: Seconds of history used for the long-term pps reference
        """
        self.window_size = window_size
        self.long_window = long_window
        self.feature_names = list(FEATURE_NAMES)
        self.reset()

    def reset(self) -> None:
        """Reset all state"""
        self.window: deque = deque()
        self.long_pps: deque = deque()
        self.last_stats: Dict = {}
        self.last_update_time: Optional[float] = None
        self.total_packets = 0
        self.total_bytes = 0
        self.samples_seen = 0

    def update(self, stats: Dict, ip_stats: Optional[List[Dict]] = None,
               timestamp: Optional[float] = None) -> None:
        """
        Add one monitoring interval.

        Args:
            stats: Cumulative counters from eBPF (total_packets, total_bytes,
                   tcp_packets, udp_packets, icmp_packets)
            ip_stats: Per-source-IP counts for THIS interval
                      (dicts with 'packets' and optionally 'syn_count')
            timestamp: Sample time (defaults to now); injectable for offline use
        """
        ip_list = ip_stats or []
        self.update_arrays(
            stats,
            np.array([ip.get('packets', 0) for ip in ip_list], dtype=np.float64),
            np.array([ip.get('syn_count', 0) for ip in ip_list], dtype=np.float64),
            timestamp,
        )

    def update_arrays(self, stats: Dict, ip_packets: np.ndarray, ip_syn: np.ndarray,
                      timestamp: Optional[float] = None) -> None:
        """Same as update(), with per-IP counts given as parallel arrays"""
        now = time.time() if timestamp is None else timestamp
        if not stats:
            return

        if self.last_update_time is None:
            # First sample: no interval to measure yet, just anchor counters
            self.last_stats = dict(stats)
            self.last_update_time = now
            return

        duration = now - self.last_update_time
        if duration <= 0:
            return

        deltas = {}
        for key in ('total_packets', 'total_bytes', 'tcp_packets', 'udp_packets', 'icmp_packets'):
            cur = stats.get(key, 0)
            prev = self.last_stats.get(key, 0)
            # Counters only go down if the source was reset; treat as fresh start
            deltas[key] = cur - prev if cur >= prev else cur

        order = np.argsort(-ip_packets, kind='stable')[:MAX_IPS_PER_INTERVAL]
        ip_counts = ip_packets[order]
        keep = ip_counts > 0
        ip_counts = ip_counts[keep]
        ip_total = float(ip_counts.sum())
        syn = int(ip_syn[order][keep].sum())

        if ip_total > 0:
            p = ip_counts / ip_total
            entropy = float(-(p * np.log2(p)).sum())
            n = len(ip_counts)
            entropy_norm = entropy / math.log2(n) if n > 1 else 0.0
            top1 = float(ip_counts[0] / ip_total)
            top10 = float(ip_counts[:10].sum() / ip_total)
        else:
            entropy_norm = top1 = top10 = 0.0

        self.window.append(_Sample(
            timestamp=now,
            duration=duration,
            packets=deltas['total_packets'],
            bytes=deltas['total_bytes'],
            tcp=deltas['tcp_packets'],
            udp=deltas['udp_packets'],
            icmp=deltas['icmp_packets'],
            syn=syn,
            ip_packets=int(ip_total),
            unique_ips=len(ip_counts),
            entropy_norm=entropy_norm,
            top1_share=top1,
            top10_share=top10,
        ))
        self.long_pps.append((now, deltas['total_packets'] / duration))

        while self.window and self.window[0].timestamp <= now - self.window_size:
            self.window.popleft()
        while self.long_pps and self.long_pps[0][0] <= now - self.long_window:
            self.long_pps.popleft()

        self.total_packets += deltas['total_packets']
        self.total_bytes += deltas['total_bytes']
        self.samples_seen += 1
        self.last_stats = dict(stats)
        self.last_update_time = now

    def extract_features(self, timestamp: Optional[float] = None) -> FlowFeatures:
        """Compute features over the current window"""
        now = time.time() if timestamp is None else timestamp
        features = np.zeros(len(self.feature_names), dtype=np.float64)
        samples = list(self.window)

        if samples:
            pps = np.array([s.packets / s.duration for s in samples])
            bps = np.array([s.bytes / s.duration for s in samples])
            packets = sum(s.packets for s in samples)
            proto_total = sum(s.tcp + s.udp + s.icmp for s in samples)
            ip_packets = sum(s.ip_packets for s in samples)
            pps_mean = float(pps.mean())

            features[0] = pps_mean
            features[1] = float(pps.max())
            features[2] = float(pps.std() / pps_mean) if pps_mean > 0 else 0.0
            features[3] = float(bps.mean())
            features[4] = sum(s.bytes for s in samples) / packets if packets > 0 else 0.0
            if proto_total > 0:
                features[5] = sum(s.tcp for s in samples) / proto_total
                features[6] = sum(s.udp for s in samples) / proto_total
                features[7] = sum(s.icmp for s in samples) / proto_total
            if ip_packets > 0:
                features[8] = sum(s.syn for s in samples) / ip_packets
            features[9] = float(np.mean([s.unique_ips for s in samples]))

            # Weight per-interval IP-distribution metrics by interval volume
            weights = np.array([s.ip_packets for s in samples], dtype=np.float64)
            if weights.sum() > 0:
                features[10] = float(np.average([s.entropy_norm for s in samples], weights=weights))
                features[11] = float(np.average([s.top1_share for s in samples], weights=weights))
                features[12] = float(np.average([s.top10_share for s in samples], weights=weights))

            unique_total = sum(s.unique_ips for s in samples)
            features[13] = ip_packets / unique_total if unique_total > 0 else 0.0

            long_ref = float(np.median([v for _, v in self.long_pps])) if self.long_pps else 0.0
            features[14] = pps_mean / (long_ref + 1.0)
            features[15] = float(pps[-1] / pps_mean) if pps_mean > 0 else 0.0

        features = np.nan_to_num(features, nan=0.0, posinf=1e10, neginf=0.0)

        return FlowFeatures(
            features=features,
            feature_names=self.feature_names,
            timestamp=now,
            flow_count=samples[-1].unique_ips if samples else 0,
        )

    def extract_features_for_prediction(self, scaler=None,
                                        feature_names: Optional[Sequence[str]] = None,
                                        timestamp: Optional[float] = None) -> np.ndarray:
        """
        Extract features as a (1, n_features) array ready for a model.

        Args:
            scaler: Optional fitted scaler applied after column selection
            feature_names: Column order the model was trained with; defaults
                           to this extractor's full feature list
        """
        flow_features = self.extract_features(timestamp)
        values = flow_features.to_dict()

        if feature_names:
            missing = [name for name in feature_names if name not in values]
            if missing:
                raise ValueError(
                    f"Model expects features the runtime extractor does not provide: {missing[:5]}"
                    f"{'...' if len(missing) > 5 else ''}"
                )
            row = np.array([[values[name] for name in feature_names]], dtype=np.float64)
        else:
            row = flow_features.features.reshape(1, -1)

        if scaler is not None:
            row = scaler.transform(row)
        return row

    def supports(self, feature_names: Sequence[str]) -> bool:
        """True if every requested feature can be produced by this extractor"""
        return all(name in self.feature_names for name in feature_names)

    def get_feature_summary(self) -> Dict:
        """Summary of current window state for the dashboard"""
        values = self.extract_features().to_dict()
        return {
            'total_packets': self.total_packets,
            'total_bytes': self.total_bytes,
            'window_samples': len(self.window),
            'samples_seen': self.samples_seen,
            'top_features': {
                'pps_mean': values['pps_mean'],
                'udp_ratio': values['udp_ratio'],
                'syn_share': values['syn_share'],
                'src_ip_entropy_norm': values['src_ip_entropy_norm'],
            },
        }


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    extractor = FeatureExtractor()
    t0 = time.time()
    for i in range(10):
        stats = {
            'total_packets': 1000 * (i + 1),
            'total_bytes': 100000 * (i + 1),
            'tcp_packets': 850 * (i + 1),
            'udp_packets': 100 * (i + 1),
            'icmp_packets': 50 * (i + 1),
        }
        ip_stats = [{'ip': f'10.0.0.{j}', 'packets': 100, 'syn_count': 2} for j in range(10)]
        extractor.update(stats, ip_stats, timestamp=t0 + i)

    for name, value in extractor.extract_features(t0 + 9).to_dict().items():
        print(f"  {name}: {value:.3f}")
