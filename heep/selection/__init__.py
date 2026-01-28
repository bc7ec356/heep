"""
Selection modules for HEEP.

This package provides implementations for sample selection based
on information scores:

- ThresholdSelector: Simple threshold-based filtering
- ProgressiveFilter: Exponential decay filtering across training rounds
"""

from .threshold import ThresholdSelector
from .progressive import ProgressiveFilter

__all__ = [
    "ThresholdSelector",
    "ProgressiveFilter",
]
