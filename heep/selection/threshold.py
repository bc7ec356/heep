"""
Threshold-based Sample Selection for HEEP.

Implements the core selection criterion:
    D' = {x ∈ D : S(x) > τ}

Where:
    - D is the input dataset
    - D' is the selected subset
    - S(x) is the sample score
    - τ is the threshold

This is the fundamental building block of HEEP's data curation.
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass

from ..utils import AudioSample


@dataclass
class ThresholdSelectorConfig:
    """Configuration for threshold selection."""
    # Selection mode
    mode: str = "threshold"  # "threshold", "percentile", or "top_k"
    
    # For threshold mode
    threshold: float = 0.5
    
    # For percentile mode
    percentile: float = 50.0  # Keep top X%
    
    # For top_k mode
    top_k: int = 1000


class ThresholdSelector:
    """
    Selects samples based on their information scores.
    
    The selector filters samples using one of three modes:
    1. threshold: Keep samples with S(x) > τ
    2. percentile: Keep top X% of samples by score
    3. top_k: Keep exactly k highest-scoring samples
    
    Mathematical Formula:
        D' = {x ∈ D : S(x) > τ}
    
    Example:
        >>> selector = ThresholdSelector(threshold=0.7)
        >>> scores = scorer.compute_batch(samples)
        >>> selected = selector.select(samples, scores)
    """
    
    def __init__(
        self,
        config: Optional[ThresholdSelectorConfig] = None,
        threshold: Optional[float] = None
    ):
        """
        Initialize the threshold selector.
        
        Args:
            config: Configuration parameters
            threshold: Shorthand for setting threshold (overrides config)
        """
        self.config = config or ThresholdSelectorConfig()
        
        if threshold is not None:
            self.config.threshold = threshold
            self.config.mode = "threshold"
    
    def select(
        self,
        samples: List[AudioSample],
        scores: np.ndarray
    ) -> Tuple[List[AudioSample], np.ndarray]:
        """
        Select high-information samples.
        
        Args:
            samples: List of AudioSample objects
            scores: Array of scores (same length as samples)
        
        Returns:
            Tuple of (selected_samples, selected_scores)
        """
        scores = np.asarray(scores)
        
        if len(samples) != len(scores):
            raise ValueError(f"Mismatch: {len(samples)} samples vs {len(scores)} scores")
        
        if len(samples) == 0:
            return [], np.array([])
        
        # Determine selection mask based on mode
        if self.config.mode == "threshold":
            mask = scores > self.config.threshold
        
        elif self.config.mode == "percentile":
            threshold = np.percentile(scores, 100 - self.config.percentile)
            mask = scores >= threshold
        
        elif self.config.mode == "top_k":
            k = min(self.config.top_k, len(samples))
            indices = np.argsort(scores)[-k:]
            mask = np.zeros(len(samples), dtype=bool)
            mask[indices] = True
        
        else:
            raise ValueError(f"Unknown selection mode: {self.config.mode}")
        
        # Apply mask
        selected_samples = [s for s, m in zip(samples, mask) if m]
        selected_scores = scores[mask]
        
        return selected_samples, selected_scores
    
    def select_indices(self, scores: np.ndarray) -> np.ndarray:
        """
        Get indices of selected samples.
        
        Args:
            scores: Array of scores
        
        Returns:
            Array of selected indices
        """
        scores = np.asarray(scores)
        
        if self.config.mode == "threshold":
            return np.where(scores > self.config.threshold)[0]
        
        elif self.config.mode == "percentile":
            threshold = np.percentile(scores, 100 - self.config.percentile)
            return np.where(scores >= threshold)[0]
        
        elif self.config.mode == "top_k":
            k = min(self.config.top_k, len(scores))
            return np.argsort(scores)[-k:]
        
        else:
            raise ValueError(f"Unknown selection mode: {self.config.mode}")
    
    def get_threshold(self) -> float:
        """Get current threshold value."""
        return self.config.threshold
    
    def set_threshold(self, threshold: float):
        """Set threshold value."""
        self.config.threshold = threshold
        self.config.mode = "threshold"
    
    def compute_retention_rate(self, scores: np.ndarray) -> float:
        """
        Compute what fraction of samples would be retained.
        
        Args:
            scores: Array of scores
        
        Returns:
            Fraction of samples retained (0 to 1)
        """
        indices = self.select_indices(scores)
        return len(indices) / len(scores) if len(scores) > 0 else 0.0
