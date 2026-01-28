"""
Scoring modules for HEEP.

This package provides implementations for computing the composite
sample score S(x) that determines a sample's information value.

Components:
- SampleScorer: Combines all entropy dimensions into final score
- MutualInformationEstimator: Estimates sample-dataset MI contribution
"""

from .sample_score import SampleScorer
from .mutual_info import MutualInformationEstimator

__all__ = [
    "SampleScorer",
    "MutualInformationEstimator",
]
