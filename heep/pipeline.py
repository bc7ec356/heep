"""
HEEP Pipeline - Main entry point for data curation.

This module provides the high-level HEEPPipeline class that orchestrates
the complete data curation process.

Usage:
    >>> from heep import HEEPPipeline
    >>> pipeline = HEEPPipeline()
    >>> curated_samples, stats = pipeline.run(dataset)
"""

import numpy as np
from typing import List, Tuple, Optional, Callable, Dict, Any
from dataclasses import dataclass
import logging

from .utils import AudioSample
from .scoring import SampleScorer, SampleScorerConfig
from .selection import ProgressiveFilter, ProgressiveFilterConfig

logger = logging.getLogger(__name__)


@dataclass
class HEEPConfig:
    """
    Complete configuration for HEEP pipeline.
    
    This combines configurations for all components into a single
    convenient configuration object.
    """
    # Scorer configuration
    weight_acoustic: float = 0.25
    weight_phonetic: float = 0.20
    weight_linguistic: float = 0.25
    weight_contextual: float = 0.15
    weight_mutual_info: float = 0.15
    
    # Progressive filtering configuration
    initial_threshold: float = 0.3
    growth_factor: float = 1.1
    max_rounds: int = 10
    min_samples: int = 1000
    min_fraction: float = 0.1
    diversity_perturbation: float = 0.05
    
    # Normalization
    normalize_components: bool = True
    
    def to_scorer_config(self) -> SampleScorerConfig:
        """Convert to SampleScorerConfig."""
        return SampleScorerConfig(
            weight_acoustic=self.weight_acoustic,
            weight_phonetic=self.weight_phonetic,
            weight_linguistic=self.weight_linguistic,
            weight_contextual=self.weight_contextual,
            weight_mutual_info=self.weight_mutual_info,
            normalize_components=self.normalize_components
        )
    
    def to_filter_config(self) -> ProgressiveFilterConfig:
        """Convert to ProgressiveFilterConfig."""
        return ProgressiveFilterConfig(
            initial_threshold=self.initial_threshold,
            growth_factor=self.growth_factor,
            max_rounds=self.max_rounds,
            min_samples=self.min_samples,
            min_fraction=self.min_fraction,
            diversity_perturbation=self.diversity_perturbation
        )


class HEEPPipeline:
    """
    High Entropy Exponential Pruning Pipeline.
    
    This is the main interface for HEEP data curation. It combines:
    - Multi-dimensional entropy estimation
    - Composite sample scoring
    - Progressive threshold filtering
    
    The pipeline implements Algorithm 1 from the paper:
    
    ```
    Input: Dataset D, threshold τ, decay factor
    Output: Curated dataset D*
    
    1. Initialize scorer with entropy estimators
    2. For each round k:
        a. Compute S(x) for all x ∈ D
        b. D ← {x ∈ D : S(x) > τ}
        c. τ ← τ × growth_factor
        d. If stopping criterion, break
    3. Return D
    ```
    
    Example:
        >>> # Basic usage
        >>> from heep import HEEPPipeline
        >>> from heep.utils import AudioSample
        >>> 
        >>> # Load your data
        >>> samples = [AudioSample(audio=audio, transcription=text) for audio, text in data]
        >>> 
        >>> # Run HEEP curation
        >>> pipeline = HEEPPipeline()
        >>> curated_samples, stats = pipeline.run(samples)
        >>> 
        >>> print(f"Curated {len(curated_samples)} from {len(samples)} samples")
        >>> print(f"Retention rate: {stats['reduction_ratio']:.1%}")
    
    Example with custom configuration:
        >>> config = HEEPConfig(
        ...     weight_acoustic=0.3,
        ...     weight_linguistic=0.3,
        ...     initial_threshold=0.4,
        ...     max_rounds=5
        ... )
        >>> pipeline = HEEPPipeline(config)
        >>> curated, stats = pipeline.run(samples)
    """
    
    def __init__(self, config: Optional[HEEPConfig] = None):
        """
        Initialize the HEEP pipeline.
        
        Args:
            config: Pipeline configuration. Uses defaults if None.
        """
        self.config = config or HEEPConfig()
        
        # Initialize components
        self.scorer = SampleScorer(self.config.to_scorer_config())
        self.filter = ProgressiveFilter(self.config.to_filter_config())
        
        # State
        self._is_fitted = False
    
    def fit(self, samples: List[AudioSample]):
        """
        Fit the pipeline to a dataset.
        
        This computes normalization statistics and fits the MI estimator.
        Call this before run() if you want to pre-fit, otherwise run()
        will fit automatically.
        
        Args:
            samples: Training dataset
        """
        logger.info(f"Fitting HEEP pipeline on {len(samples)} samples")
        self.scorer.fit(samples)
        self._is_fitted = True
    
    def score(self, samples: List[AudioSample]) -> np.ndarray:
        """
        Compute information scores for samples.
        
        Args:
            samples: List of AudioSample objects
        
        Returns:
            Array of scores
        """
        if not self._is_fitted:
            self.fit(samples)
        
        return self.scorer.compute_batch(samples)
    
    def score_detailed(self, sample: AudioSample) -> Dict[str, float]:
        """
        Get detailed score breakdown for a single sample.
        
        Args:
            sample: AudioSample to score
        
        Returns:
            Dictionary with component scores
        """
        return self.scorer.compute_detailed(sample)
    
    def run(
        self,
        samples: List[AudioSample],
        train_callback: Optional[Callable] = None
    ) -> Tuple[List[AudioSample], Dict[str, Any]]:
        """
        Run the complete HEEP curation pipeline.
        
        This is the main entry point for data curation.
        
        Algorithm (from paper):
            1. Fit scorer to dataset (entropy estimators + MI)
            2. Progressive filtering loop:
               a. Score all samples: S(x) = Σᵢ αᵢHᵢ(x) + MI(x,D)
               b. Filter: D' = {x ∈ D : S(x) > τ}
               c. Update threshold: τ = τ × growth_factor
               d. Optionally train model on D'
               e. Repeat until stopping criterion
        
        Args:
            samples: Input dataset
            train_callback: Optional function to call after each round
                           Signature: callback(samples, round_num) -> None
        
        Returns:
            Tuple of:
                - curated_samples: High-information samples
                - stats: Dictionary with curation statistics
        """
        if len(samples) == 0:
            logger.warning("Empty dataset provided")
            return [], {'total_rounds': 0, 'original_size': 0, 'final_size': 0}
        
        logger.info(f"Starting HEEP pipeline with {len(samples)} samples")
        
        # Fit scorer if not already fitted
        if not self._is_fitted:
            self.fit(samples)
        
        # Run progressive filtering
        curated_samples, stats = self.filter.run(
            samples,
            self.scorer,
            train_callback
        )
        
        logger.info(f"HEEP complete: {len(samples)} → {len(curated_samples)} samples")
        
        return curated_samples, stats
    
    def select_single_round(
        self,
        samples: List[AudioSample],
        threshold: Optional[float] = None,
        percentile: Optional[float] = None,
        top_k: Optional[int] = None
    ) -> Tuple[List[AudioSample], np.ndarray]:
        """
        Perform single-round selection without progressive filtering.
        
        Useful for quick filtering or when you want direct control
        over the selection process.
        
        Args:
            samples: Input samples
            threshold: Score threshold (mutually exclusive with percentile/top_k)
            percentile: Keep top X% (mutually exclusive with threshold/top_k)
            top_k: Keep exactly k samples (mutually exclusive with threshold/percentile)
        
        Returns:
            Tuple of (selected_samples, selected_scores)
        """
        from .selection import ThresholdSelector, ThresholdSelectorConfig
        
        # Ensure fitted
        if not self._is_fitted:
            self.fit(samples)
        
        # Score samples
        scores = self.scorer.compute_batch(samples)
        
        # Configure selector
        if threshold is not None:
            config = ThresholdSelectorConfig(mode="threshold", threshold=threshold)
        elif percentile is not None:
            config = ThresholdSelectorConfig(mode="percentile", percentile=percentile)
        elif top_k is not None:
            config = ThresholdSelectorConfig(mode="top_k", top_k=top_k)
        else:
            config = ThresholdSelectorConfig(mode="threshold", threshold=self.config.initial_threshold)
        
        selector = ThresholdSelector(config)
        
        return selector.select(samples, scores)
    
    def analyze_dataset(self, samples: List[AudioSample]) -> Dict[str, Any]:
        """
        Analyze a dataset and return statistics about entropy distributions.
        
        Useful for understanding your data before running HEEP.
        
        Args:
            samples: Dataset to analyze
        
        Returns:
            Dictionary with statistics for each entropy component
        """
        if not self._is_fitted:
            self.fit(samples)
        
        # Score all samples with detailed breakdown
        detailed_scores = [self.scorer.compute_detailed(s) for s in samples]
        
        # Aggregate statistics
        components = ['acoustic', 'phonetic', 'linguistic', 'contextual', 'mutual_info']
        stats = {}
        
        for comp in components:
            raw_key = f'{comp}_raw'
            if raw_key in detailed_scores[0]:
                values = [d[raw_key] for d in detailed_scores]
                arr = np.array(values)
                stats[comp] = {
                    'mean': float(arr.mean()),
                    'std': float(arr.std()),
                    'min': float(arr.min()),
                    'max': float(arr.max()),
                    'median': float(np.median(arr))
                }
        
        # Overall score statistics
        total_scores = np.array([d['total_score'] for d in detailed_scores])
        stats['total_score'] = {
            'mean': float(total_scores.mean()),
            'std': float(total_scores.std()),
            'min': float(total_scores.min()),
            'max': float(total_scores.max()),
            'median': float(np.median(total_scores))
        }
        
        # Percentile thresholds for reference
        percentiles = [10, 25, 50, 75, 90, 95, 99]
        stats['score_percentiles'] = {
            p: float(np.percentile(total_scores, p)) for p in percentiles
        }
        
        return stats
    
    def get_config(self) -> HEEPConfig:
        """Get current configuration."""
        return self.config
    
    def set_weights(self, **weights):
        """
        Update entropy component weights.
        
        Args:
            **weights: Keyword arguments for weights
                      (acoustic, phonetic, linguistic, contextual, mutual_info)
        """
        self.scorer.set_weights(**weights)
