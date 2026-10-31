"""
Mitigation policies: deciding WHICH traffic to drop once an attack is detected.

Detection says "an attack is happening"; a mitigation policy decides which
sources to block or throttle. This package implements the system's current
rule, standard baselines, and the proposed source-scoring policy, together
with an evaluation harness that measures attack traffic stopped versus
legitimate traffic lost (collateral damage).
"""

from .policies import (
    MitigationPolicy, TopKBlacklist, HistoryFilter, AggregateRateLimit,
    PerSourceRateLimit, SourceScoringPolicy, AdaptiveHistoryFilter, SignatureFilter,
    FlowVerifiedScoring,
)
from .source_scorer import SourceScorer, SOURCE_FEATURES

__all__ = [
    'MitigationPolicy', 'TopKBlacklist', 'HistoryFilter', 'AggregateRateLimit',
    'PerSourceRateLimit', 'SourceScoringPolicy', 'AdaptiveHistoryFilter', 'SignatureFilter',
    'FlowVerifiedScoring',
    'SourceScorer', 'SOURCE_FEATURES',
]
