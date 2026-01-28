"""
Mutual Information Estimation for HEEP.

Estimates MI(x, D) - the mutual information between a sample and the dataset,
quantifying how much unique information a sample contributes.

Mathematical Formulation (Paper Equation 6):
    MI(x, D) = H(x) - H(x|D)

In practice, we estimate this as:
    MI(x, D) ≈ divergence between sample features and dataset feature distribution

High MI indicates a sample contains information not well-represented
in the rest of the dataset, making it valuable for training.
"""

import numpy as np
from typing import Optional, List, Dict
from dataclasses import dataclass

from ..utils import AudioSample, discretize_features


@dataclass
class MutualInformationConfig:
    """Configuration for mutual information estimation."""
    # Feature representation
    feature_type: str = "combined"  # "acoustic", "linguistic", or "combined"
    n_bins: int = 32  # Discretization bins
    
    # Estimation method
    method: str = "kl_divergence"  # "kl_divergence", "binning", or "knn"
    
    # KNN settings (if method="knn")
    k_neighbors: int = 5
    
    # Smoothing
    smoothing: float = 1e-10


class MutualInformationEstimator:
    """
    Estimates mutual information between samples and the dataset.
    
    The MI(x, D) measures how much unique information sample x provides
    relative to the dataset D. High MI samples are:
    - Underrepresented in the dataset
    - Contain rare patterns or combinations
    - Valuable for improving model coverage
    
    Implementation:
        1. Build feature representation of dataset distribution
        2. For each sample, compute divergence from dataset distribution
        3. Higher divergence = higher MI = more valuable sample
    
    Example:
        >>> estimator = MutualInformationEstimator()
        >>> estimator.fit(dataset_samples)
        >>> mi = estimator.compute(new_sample)
        >>> print(f"Sample MI: {mi:.3f}")
    """
    
    def __init__(self, config: Optional[MutualInformationConfig] = None):
        """
        Initialize the mutual information estimator.
        
        Args:
            config: Configuration parameters. Uses defaults if None.
        """
        self.config = config or MutualInformationConfig()
        
        # Dataset statistics (populated by fit())
        self._dataset_distribution: Optional[np.ndarray] = None
        self._feature_bins: Optional[np.ndarray] = None
        self._fitted = False
    
    def _extract_features(self, sample: AudioSample) -> np.ndarray:
        """
        Extract features from a sample for MI estimation.
        
        Args:
            sample: AudioSample object
        
        Returns:
            Feature vector
        """
        features = []
        
        if self.config.feature_type in ["linguistic", "combined"]:
            # Simple linguistic features from transcription
            text = sample.transcription.lower()
            words = text.split()
            
            # Word count (normalized)
            features.append(min(len(words) / 100, 1.0))
            
            # Unique word ratio
            if len(words) > 0:
                features.append(len(set(words)) / len(words))
            else:
                features.append(0.0)
            
            # Average word length
            if len(words) > 0:
                features.append(np.mean([len(w) for w in words]) / 10)
            else:
                features.append(0.0)
            
            # Character count (normalized)
            features.append(min(len(text) / 500, 1.0))
        
        if self.config.feature_type in ["acoustic", "combined"]:
            # Simple acoustic features
            audio = sample.audio
            
            # Audio duration (normalized)
            duration = len(audio) / sample.sample_rate
            features.append(min(duration / 30, 1.0))
            
            # RMS energy
            rms = np.sqrt(np.mean(audio.astype(np.float64) ** 2))
            features.append(min(rms / 0.1, 1.0))
            
            # Zero crossing rate
            zcr = np.mean(np.abs(np.diff(np.sign(audio))) / 2)
            features.append(zcr)
            
            # Simple spectral centroid approximation
            fft = np.abs(np.fft.rfft(audio[:min(len(audio), 16000)]))
            if fft.sum() > 0:
                centroid = np.sum(np.arange(len(fft)) * fft) / fft.sum()
                features.append(centroid / len(fft))
            else:
                features.append(0.5)
        
        return np.array(features, dtype=np.float64)
    
    def fit(self, samples: List[AudioSample]):
        """
        Fit the estimator to a dataset.
        
        Builds the dataset distribution for MI computation.
        
        Args:
            samples: List of AudioSample objects representing the dataset
        """
        if len(samples) == 0:
            raise ValueError("Cannot fit on empty dataset")
        
        # Extract features from all samples
        all_features = np.array([self._extract_features(s) for s in samples])
        
        # Discretize features
        discrete_features = discretize_features(
            all_features,
            n_bins=self.config.n_bins,
            method="quantile"
        )
        
        # Store bin edges for later use
        self._feature_bins = []
        for i in range(all_features.shape[1]):
            percentiles = np.linspace(0, 100, self.config.n_bins + 1)
            bins = np.percentile(all_features[:, i], percentiles)
            self._feature_bins.append(bins)
        
        # Build joint distribution over discretized features
        # Flatten to 1D bin indices
        n_features = discrete_features.shape[1]
        flat_indices = np.zeros(len(samples), dtype=np.int64)
        
        for i in range(n_features):
            flat_indices = flat_indices * self.config.n_bins + discrete_features[:, i]
        
        # Estimate distribution
        n_total_bins = self.config.n_bins ** n_features
        counts = np.bincount(flat_indices, minlength=n_total_bins)
        self._dataset_distribution = (counts + self.config.smoothing) / (counts.sum() + self.config.smoothing * n_total_bins)
        
        self._fitted = True
    
    def compute(self, sample: AudioSample) -> float:
        """
        Compute mutual information for a single sample.
        
        Mathematical Formula:
            MI(x, D) ≈ -log₂ p(x|D)
        
        Where p(x|D) is the probability of sample x under the dataset distribution.
        Lower probability = higher MI = more unique sample.
        
        Args:
            sample: AudioSample to evaluate
        
        Returns:
            Estimated mutual information value
        """
        if not self._fitted:
            # Return neutral value if not fitted
            return 0.0
        
        features = self._extract_features(sample)
        
        # Discretize using stored bins
        discrete = np.zeros(len(features), dtype=np.int32)
        for i, (feat, bins) in enumerate(zip(features, self._feature_bins)):
            discrete[i] = np.clip(np.digitize(feat, bins[1:-1]), 0, self.config.n_bins - 1)
        
        # Compute flat index
        flat_idx = 0
        for i, d in enumerate(discrete):
            flat_idx = flat_idx * self.config.n_bins + d
        
        # Get probability and convert to information content
        if flat_idx < len(self._dataset_distribution):
            prob = self._dataset_distribution[flat_idx]
        else:
            prob = self.config.smoothing
        
        # MI ≈ -log₂(p(x|D)) = information content
        mi = -np.log2(prob + self.config.smoothing)
        
        return float(mi)
    
    def compute_batch(self, samples: List[AudioSample]) -> np.ndarray:
        """
        Compute mutual information for multiple samples.
        
        Args:
            samples: List of AudioSample objects
        
        Returns:
            Array of MI values
        """
        return np.array([self.compute(s) for s in samples])
