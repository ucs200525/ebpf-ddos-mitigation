"""
Per-source attacker scoring.

Every feature comes from counters the XDP program already keeps per source
(packets and SYNs in the current interval) plus a small user-space history
built from intervals judged normal. No payload or flow timing is needed.

Features are deliberately scale-free: rates are compared with the source's
own history and with a typical source on the same network, never used as
absolute numbers, so a model trained on one traffic volume transfers to
networks that are much smaller or larger. (Version 1 used absolute packet
counts and traffic shares; it learned "low-volume network = attack" and
blocked legitimate users of small networks.)
"""

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

SOURCE_FEATURES = [
    'known',       # source seen in >= 2 normal intervals before now (history-based filtering)
    'rate_jump',   # log(1 + packets now) - log(1 + own mean packets in normal intervals)
    'rel_rate',    # log(1 + packets now) - log(1 + median packets of a source in normal intervals)
    'syn_share',   # SYN packets / packets for this source this interval
    'age',         # seconds since the source first appeared, capped at 60, scaled to 0..1
]


def source_features(known: np.ndarray, pkts: np.ndarray, hist_mean: np.ndarray,
                    syn: np.ndarray, typical_rate: float, age_s: np.ndarray) -> np.ndarray:
    """Feature matrix (n_sources, len(SOURCE_FEATURES)) for sources with pkts > 0"""
    log_pkts = np.log1p(pkts)
    return np.column_stack([
        known.astype(np.float64),
        log_pkts - np.log1p(hist_mean),
        log_pkts - np.log1p(typical_rate),
        np.divide(syn, pkts, out=np.zeros_like(pkts, dtype=np.float64), where=pkts > 0),
        np.minimum(age_s, 60.0) / 60.0,
    ])


# Features that need this source's own history (the ablation "P4 without source history"
# trains the scorer without them; rel_rate only uses a network-wide typical rate)
HISTORY_FEATURES = ('known', 'rate_jump', 'age')


class SourceScorer:
    """
    Logistic-regression model giving P(source is an attacker).

    `features` selects a subset of SOURCE_FEATURES (for ablations); inputs are
    always full SOURCE_FEATURES matrices and the subset is taken here.
    """

    def __init__(self, features: Optional[List[str]] = None):
        self.features = list(features or SOURCE_FEATURES)
        unknown = [f for f in self.features if f not in SOURCE_FEATURES]
        if unknown:
            raise ValueError(f'unknown source features: {unknown}')
        self._cols = [SOURCE_FEATURES.index(f) for f in self.features]
        self.model = make_pipeline(StandardScaler(),
                                   LogisticRegression(class_weight='balanced', max_iter=1000))
        self.trained = False
        self.info: Dict = {}

    def fit(self, X: np.ndarray, y: np.ndarray, info: Optional[Dict] = None) -> 'SourceScorer':
        self.model.fit(X[:, self._cols], y)
        self.trained = True
        self.info = dict(info or {})
        self.info.update({'n_samples': int(len(y)), 'attack_share': float(np.mean(y)),
                          'trained_at': time.strftime('%Y-%m-%d %H:%M:%S'),
                          'sklearn_version': sklearn.__version__})
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if len(X) == 0:
            return np.zeros(0)
        return self.model.predict_proba(X[:, self._cols])[:, 1]

    def coefficients(self) -> Dict[str, float]:
        """Standardized coefficients (positive = more attacker-like)"""
        lr = self.model[-1]
        return dict(zip(self.features, map(float, lr.coef_[0])))

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({'model': self.model, 'features': self.features,
                     'all_features': SOURCE_FEATURES, 'info': self.info}, path)

    @classmethod
    def load(cls, path: str) -> 'SourceScorer':
        data = joblib.load(path)
        features = data.get('features')
        # Scorers saved before feature subsets store the full list and no 'all_features'
        if data.get('all_features', SOURCE_FEATURES) != SOURCE_FEATURES or \
                any(f not in SOURCE_FEATURES for f in features or []):
            raise ValueError('Source scorer was trained with different features; retrain it')
        scorer = cls(features)
        scorer.model, scorer.info, scorer.trained = data['model'], data.get('info', {}), True
        return scorer
