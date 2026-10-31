"""
Level 3: per-flow features and classifier.

A flow is a 5-tuple (source, destination, ports, protocol) observed during one
10-second window, in the INBOUND direction only. Every feature is computed
from a few counters per flow that the XDP program can keep in its flow map
(packets, bytes, length min/max/sum of squares, first/last time, sum of
squared and largest inter-arrival time, TCP flag counts), so the model sees
the same numbers offline (CIC-DDoS2019 packet captures, scripts/cic_extract.py)
and live (src/core/traffic_monitor.py reading flow_map).

Deliberately left out: addresses, raw port numbers, TTL and TCP window. In a
testbed with one attack machine those identify the machine rather than the
attack (a classifier would learn "TTL 63 = attack"). Ports are reduced to
"below 1024 or not", which separates requests to a service and responses
from one (e.g. reflected DNS/NTP/LDAP answers) without memorising services.
"""

import logging
import time
from pathlib import Path
from typing import Dict, Optional

import joblib
import numpy as np
import sklearn

logger = logging.getLogger(__name__)

FLOW_WINDOW_S = 10
FLOW_FEATURE_VERSION = 1

FLOW_FEATURES = [
    'protocol',        # IP protocol number (6 TCP, 17 UDP, 1 ICMP, ...)
    'src_port_low',    # source port < 1024 (a service answering, or reflecting)
    'dst_port_low',    # destination port < 1024 (a request to a service)
    'packets',         # packets in the window
    'duration',        # seconds from first to last packet in the window
    'len_mean',        # IP length, bytes
    'len_std',
    'len_min',
    'len_max',
    'iat_mean',        # inter-arrival time, seconds (0 for 1-packet flows)
    'iat_std',
    'iat_max',
    'syn_ratio',       # packets with SYN set / packets
    'ack_ratio',       # packets with ACK set / packets  ("only SYN" or "only ACK" flows stand out)
    'fin_rst_ratio',   # packets with FIN or RST / packets
    'psh_ratio',       # packets with PSH / packets
]

# Counter columns a flow record must provide
FLOW_COUNTERS = ('proto', 'sport', 'dport', 'packets', 'bytes', 'len_min', 'len_max', 'len_sumsq',
                 't_first', 't_last', 'iat_sumsq', 'iat_max', 'syn', 'ack', 'fin', 'rst', 'psh')


def flow_feature_matrix(f: Dict[str, np.ndarray]) -> np.ndarray:
    """(n_flows, len(FLOW_FEATURES)) from flow counter columns (see FLOW_COUNTERS)"""
    n = np.maximum(np.asarray(f['packets'], dtype=np.float64), 1.0)
    duration = np.maximum(np.asarray(f['t_last'], dtype=np.float64) - f['t_first'], 0.0)
    len_mean = np.asarray(f['bytes'], dtype=np.float64) / n
    len_std = np.sqrt(np.maximum(np.asarray(f['len_sumsq'], dtype=np.float64) / n - len_mean ** 2, 0.0))
    gaps = n - 1.0
    multi = gaps > 0
    iat_mean = np.divide(duration, gaps, out=np.zeros_like(duration), where=multi)
    iat_sq = np.divide(np.asarray(f['iat_sumsq'], dtype=np.float64), gaps, out=np.zeros_like(duration),
                       where=multi)
    iat_std = np.sqrt(np.maximum(iat_sq - iat_mean ** 2, 0.0))
    X = np.column_stack([
        np.asarray(f['proto'], dtype=np.float64),
        (np.asarray(f['sport']) < 1024).astype(np.float64),
        (np.asarray(f['dport']) < 1024).astype(np.float64),
        n,
        duration,
        len_mean,
        len_std,
        np.asarray(f['len_min'], dtype=np.float64),
        np.asarray(f['len_max'], dtype=np.float64),
        iat_mean,
        iat_std,
        np.asarray(f['iat_max'], dtype=np.float64),
        np.asarray(f['syn'], dtype=np.float64) / n,
        np.asarray(f['ack'], dtype=np.float64) / n,
        (np.asarray(f['fin'], dtype=np.float64) + np.asarray(f['rst'], dtype=np.float64)) / n,
        np.asarray(f['psh'], dtype=np.float64) / n,
    ])
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


class FlowClassifier:
    """Random forest giving P(flow is attack traffic) from FLOW_FEATURES"""

    def __init__(self, n_estimators: int = 100, max_depth: Optional[int] = 16,
                 min_samples_leaf: int = 5, random_state: int = 42):
        from sklearn.ensemble import RandomForestClassifier
        self.model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                            min_samples_leaf=min_samples_leaf, class_weight='balanced',
                                            n_jobs=-1, random_state=random_state)
        self.trained = False
        self.info: Dict = {}

    def fit(self, X: np.ndarray, y: np.ndarray, info: Optional[Dict] = None) -> 'FlowClassifier':
        self.model.fit(X, y)
        self.trained = True
        self.info = dict(info or {})
        self.info.update({'n_samples': int(len(y)), 'attack_share': float(np.mean(y)),
                          'trained_at': time.strftime('%Y-%m-%d %H:%M:%S'),
                          'sklearn_version': sklearn.__version__})
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if len(X) == 0:
            return np.zeros(0)
        # single-threaded for live per-window batches (thread start-up dominates), parallel offline
        self.model.set_params(n_jobs=1 if len(X) < 20_000 else -1)
        return self.model.predict_proba(X)[:, 1]

    def feature_importance(self) -> Dict[str, float]:
        return dict(sorted(zip(FLOW_FEATURES, map(float, self.model.feature_importances_)),
                           key=lambda kv: -kv[1]))

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({'model': self.model, 'features': FLOW_FEATURES, 'version': FLOW_FEATURE_VERSION,
                     'window_s': FLOW_WINDOW_S, 'info': self.info}, path)

    @classmethod
    def load(cls, path: str) -> 'FlowClassifier':
        data = joblib.load(path)
        if data.get('features') != FLOW_FEATURES or data.get('version') != FLOW_FEATURE_VERSION:
            raise ValueError('Flow classifier was trained with different flow features; retrain it')
        clf = cls()
        clf.model, clf.info, clf.trained = data['model'], data.get('info', {}), True
        return clf
