"""
Entropy estimation modules for HEEP.

This package provides implementations for computing entropy across
four dimensions of speech data:

1. Acoustic Entropy (H_acoustic): Spectral and temporal diversity
2. Phonetic Entropy (H_phonetic): Phoneme distribution complexity  
3. Linguistic Entropy (H_linguistic): Vocabulary and syntax richness
4. Contextual Entropy (H_contextual): Domain and discourse diversity

Each entropy component contributes to the overall sample information score.
"""

from .acoustic import AcousticEntropyEstimator
from .phonetic import PhoneticEntropyEstimator
from .linguistic import LinguisticEntropyEstimator
from .contextual import ContextualEntropyEstimator

__all__ = [
    "AcousticEntropyEstimator",
    "PhoneticEntropyEstimator",
    "LinguisticEntropyEstimator",
    "ContextualEntropyEstimator",
]
