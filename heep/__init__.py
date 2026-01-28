"""
HEEP: High Entropy Exponential Pruning

A data curation framework for ASR that prioritizes information density over data quantity.
HEEP identifies high-entropy training samples that maximize informational diversity
while progressively filtering redundant data.

Paper: "HEEP: High Entropy Exponential Pruning for State-of-the-Art ASR 
        Through Strategic Data Curation"

Mathematical Foundation:
    Sample Score: S(x) = α₁·H_acoustic(x) + α₂·H_phonetic(x) + α₃·H_linguistic(x) 
                        + α₄·H_contextual(x) + α₅·MI(x, D)
    
    Selection: D' = {x ∈ D : S(x) > τ}
    
    Progressive Filtering: τₖ₊₁ = τₖ · decay_factor
"""

__version__ = "1.0.0"
__author__ = "Anonymous"

from .pipeline import HEEPPipeline
from .scoring.sample_score import SampleScorer
from .selection.progressive import ProgressiveFilter

__all__ = [
    "HEEPPipeline",
    "SampleScorer", 
    "ProgressiveFilter",
]
