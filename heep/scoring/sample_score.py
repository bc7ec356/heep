"""
Composite Sample Scoring for HEEP.

Computes S(x) - the overall information score for a sample by combining
all entropy dimensions and mutual information.

Mathematical Formulation (Paper Equation 1):
    S(x) = α₁·H_acoustic(x) + α₂·H_phonetic(x) + α₃·H_linguistic(x) 
         + α₄·H_contextual(x) + β·MI(x, D)

Where:
    - H_* are entropy components from different dimensions
    - MI(x, D) is mutual information with the dataset
    - α₁...α₄, β are configurable weights (default: 0.25, 0.20, 0.25, 0.15, 0.15)

The sample score determines which samples are retained during HEEP filtering.
Higher scores indicate more informative samples.
"""

import numpy as np
from typing import Optional, List, Dict
from dataclasses import dataclass, field

from ..utils import AudioSample, normalize_scores
from ..entropy import (
    AcousticEntropyEstimator,
    PhoneticEntropyEstimator,
    LinguisticEntropyEstimator,
    ContextualEntropyEstimator
)
from .mutual_info import MutualInformationEstimator


@dataclass
class SampleScorerConfig:
    """Configuration for sample scoring."""
    # Component weights (α₁...α₅)
    weight_acoustic: float = 0.25      # α₁
    weight_phonetic: float = 0.20      # α₂
    weight_linguistic: float = 0.25    # α₃
    weight_contextual: float = 0.15    # α₄
    weight_mutual_info: float = 0.15   # α₅
    
    # Normalization
    normalize_components: bool = True
    normalization_method: str = "minmax"
    
    # Component enable/disable
    use_acoustic: bool = True
    use_phonetic: bool = True
    use_linguistic: bool = True
    use_contextual: bool = True
    use_mutual_info: bool = True


class SampleScorer:
    """
    Computes composite information scores for audio samples.
    
    The sample scorer combines entropy from multiple dimensions
    into a single score that represents the sample's information value.
    
    Mathematical Formula:
        S(x) = Σᵢ αᵢ · normalize(Hᵢ(x))
    
    Where:
        - αᵢ are the component weights
        - Hᵢ are the entropy components (acoustic, phonetic, linguistic, contextual, MI)
        - normalize() maps each component to [0, 1] for fair combination
    
    The weights can be tuned based on the target application:
        - For noisy audio: increase weight_acoustic
        - For domain-specific ASR: increase weight_contextual
        - For low-resource languages: increase weight_phonetic
    
    Example:
        >>> scorer = SampleScorer()
        >>> scorer.fit(training_samples)  # For MI estimation
        >>> scores = scorer.compute_batch(samples)
        >>> high_info_samples = [s for s, score in zip(samples, scores) if score > 0.7]
    """
    
    def __init__(self, config: Optional[SampleScorerConfig] = None):
        """
        Initialize the sample scorer.
        
        Args:
            config: Configuration parameters. Uses defaults if None.
        """
        self.config = config or SampleScorerConfig()
        
        # Initialize entropy estimators
        self._acoustic = AcousticEntropyEstimator() if self.config.use_acoustic else None
        self._phonetic = PhoneticEntropyEstimator() if self.config.use_phonetic else None
        self._linguistic = LinguisticEntropyEstimator() if self.config.use_linguistic else None
        self._contextual = ContextualEntropyEstimator() if self.config.use_contextual else None
        self._mutual_info = MutualInformationEstimator() if self.config.use_mutual_info else None
        
        # Normalization statistics (populated during batch processing)
        self._component_stats: Dict[str, Dict[str, float]] = {}
        self._fitted = False
    
    def fit(self, samples: List[AudioSample]):
        """
        Fit the scorer to a dataset.
        
        This computes normalization statistics and fits the MI estimator.
        
        Args:
            samples: List of AudioSample objects
        """
        if len(samples) == 0:
            raise ValueError("Cannot fit on empty dataset")
        
        # Fit MI estimator
        if self._mutual_info is not None:
            self._mutual_info.fit(samples)
        
        # Compute component values for normalization
        if self.config.normalize_components:
            component_values = {
                'acoustic': [],
                'phonetic': [],
                'linguistic': [],
                'contextual': [],
                'mutual_info': []
            }
            
            for sample in samples:
                if self._acoustic is not None:
                    component_values['acoustic'].append(self._acoustic.compute(sample))
                if self._phonetic is not None:
                    component_values['phonetic'].append(self._phonetic.compute(sample))
                if self._linguistic is not None:
                    component_values['linguistic'].append(self._linguistic.compute(sample))
                if self._contextual is not None:
                    component_values['contextual'].append(self._contextual.compute(sample))
                if self._mutual_info is not None:
                    component_values['mutual_info'].append(self._mutual_info.compute(sample))
            
            # Store statistics
            for name, values in component_values.items():
                if len(values) > 0:
                    arr = np.array(values)
                    self._component_stats[name] = {
                        'min': float(arr.min()),
                        'max': float(arr.max()),
                        'mean': float(arr.mean()),
                        'std': float(arr.std())
                    }
        
        self._fitted = True
    
    def _normalize_component(self, value: float, component_name: str) -> float:
        """
        Normalize a component value to [0, 1] range.
        
        Args:
            value: Raw component value
            component_name: Name of the component
        
        Returns:
            Normalized value in [0, 1]
        """
        if not self.config.normalize_components:
            return value
        
        if component_name not in self._component_stats:
            return value
        
        stats = self._component_stats[component_name]
        
        if self.config.normalization_method == "minmax":
            range_val = stats['max'] - stats['min']
            if range_val < 1e-10:
                return 0.5
            return (value - stats['min']) / range_val
        
        elif self.config.normalization_method == "zscore":
            if stats['std'] < 1e-10:
                return 0.5
            z = (value - stats['mean']) / stats['std']
            # Clip to reasonable range and scale to [0, 1]
            return (np.clip(z, -3, 3) + 3) / 6
        
        return value
    
    def compute(self, sample: AudioSample) -> float:
        """
        Compute the composite information score for a single sample.
        
        Mathematical Formula:
            S(x) = α₁·H_acoustic(x) + α₂·H_phonetic(x) + α₃·H_linguistic(x) 
                 + α₄·H_contextual(x) + α₅·MI(x, D)
        
        Args:
            sample: AudioSample to score
        
        Returns:
            Composite information score
        """
        score = 0.0
        
        # Acoustic entropy
        if self._acoustic is not None:
            h_acoustic = self._acoustic.compute(sample)
            h_acoustic = self._normalize_component(h_acoustic, 'acoustic')
            score += self.config.weight_acoustic * h_acoustic
        
        # Phonetic entropy
        if self._phonetic is not None:
            h_phonetic = self._phonetic.compute(sample)
            h_phonetic = self._normalize_component(h_phonetic, 'phonetic')
            score += self.config.weight_phonetic * h_phonetic
        
        # Linguistic entropy
        if self._linguistic is not None:
            h_linguistic = self._linguistic.compute(sample)
            h_linguistic = self._normalize_component(h_linguistic, 'linguistic')
            score += self.config.weight_linguistic * h_linguistic
        
        # Contextual entropy
        if self._contextual is not None:
            h_contextual = self._contextual.compute(sample)
            h_contextual = self._normalize_component(h_contextual, 'contextual')
            score += self.config.weight_contextual * h_contextual
        
        # Mutual information
        if self._mutual_info is not None:
            mi = self._mutual_info.compute(sample)
            mi = self._normalize_component(mi, 'mutual_info')
            score += self.config.weight_mutual_info * mi
        
        return float(score)
    
    def compute_detailed(self, sample: AudioSample) -> Dict[str, float]:
        """
        Compute detailed breakdown of score components.
        
        Args:
            sample: AudioSample to score
        
        Returns:
            Dictionary with individual component values and total score
        """
        result = {}
        
        if self._acoustic is not None:
            h = self._acoustic.compute(sample)
            result['acoustic_raw'] = h
            result['acoustic_normalized'] = self._normalize_component(h, 'acoustic')
            result['acoustic_weighted'] = self.config.weight_acoustic * result['acoustic_normalized']
        
        if self._phonetic is not None:
            h = self._phonetic.compute(sample)
            result['phonetic_raw'] = h
            result['phonetic_normalized'] = self._normalize_component(h, 'phonetic')
            result['phonetic_weighted'] = self.config.weight_phonetic * result['phonetic_normalized']
        
        if self._linguistic is not None:
            h = self._linguistic.compute(sample)
            result['linguistic_raw'] = h
            result['linguistic_normalized'] = self._normalize_component(h, 'linguistic')
            result['linguistic_weighted'] = self.config.weight_linguistic * result['linguistic_normalized']
        
        if self._contextual is not None:
            h = self._contextual.compute(sample)
            result['contextual_raw'] = h
            result['contextual_normalized'] = self._normalize_component(h, 'contextual')
            result['contextual_weighted'] = self.config.weight_contextual * result['contextual_normalized']
        
        if self._mutual_info is not None:
            mi = self._mutual_info.compute(sample)
            result['mutual_info_raw'] = mi
            result['mutual_info_normalized'] = self._normalize_component(mi, 'mutual_info')
            result['mutual_info_weighted'] = self.config.weight_mutual_info * result['mutual_info_normalized']
        
        # Total score
        result['total_score'] = sum(v for k, v in result.items() if k.endswith('_weighted'))
        
        return result
    
    def compute_batch(self, samples: List[AudioSample]) -> np.ndarray:
        """
        Compute scores for multiple samples.
        
        Args:
            samples: List of AudioSample objects
        
        Returns:
            Array of scores
        """
        return np.array([self.compute(s) for s in samples])
    
    def get_weights(self) -> Dict[str, float]:
        """Get current component weights."""
        return {
            'acoustic': self.config.weight_acoustic,
            'phonetic': self.config.weight_phonetic,
            'linguistic': self.config.weight_linguistic,
            'contextual': self.config.weight_contextual,
            'mutual_info': self.config.weight_mutual_info
        }
    
    def set_weights(
        self,
        acoustic: Optional[float] = None,
        phonetic: Optional[float] = None,
        linguistic: Optional[float] = None,
        contextual: Optional[float] = None,
        mutual_info: Optional[float] = None
    ):
        """
        Update component weights.
        
        Args:
            acoustic: New weight for acoustic entropy
            phonetic: New weight for phonetic entropy
            linguistic: New weight for linguistic entropy
            contextual: New weight for contextual entropy
            mutual_info: New weight for mutual information
        """
        if acoustic is not None:
            self.config.weight_acoustic = acoustic
        if phonetic is not None:
            self.config.weight_phonetic = phonetic
        if linguistic is not None:
            self.config.weight_linguistic = linguistic
        if contextual is not None:
            self.config.weight_contextual = contextual
        if mutual_info is not None:
            self.config.weight_mutual_info = mutual_info
