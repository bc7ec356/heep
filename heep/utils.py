"""
Utility functions for HEEP.

Provides common operations for entropy computation, audio processing,
and data handling.
"""

import numpy as np
from typing import Union, List, Optional
from dataclasses import dataclass


@dataclass
class AudioSample:
    """
    Represents a single audio sample with its transcription.
    
    Attributes:
        audio: Audio waveform as numpy array (shape: [num_samples])
        transcription: Text transcription of the audio
        sample_rate: Audio sample rate in Hz (default: 16000)
        sample_id: Unique identifier for the sample
        metadata: Optional dictionary of additional metadata
    """
    audio: np.ndarray
    transcription: str
    sample_rate: int = 16000
    sample_id: Optional[str] = None
    metadata: Optional[dict] = None


def compute_entropy(probabilities: np.ndarray, epsilon: float = 1e-10) -> float:
    """
    Compute Shannon entropy from a probability distribution.
    
    Mathematical Formula (Equation 1 in paper):
        H(X) = -Σᵢ p(xᵢ) log₂ p(xᵢ)
    
    Args:
        probabilities: Probability distribution (must sum to 1)
        epsilon: Small value to avoid log(0)
    
    Returns:
        Entropy value in bits
    
    Example:
        >>> probs = np.array([0.5, 0.5])
        >>> compute_entropy(probs)
        1.0  # Maximum entropy for 2 outcomes
    """
    # Ensure probabilities are valid
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities + epsilon
    probabilities = probabilities / probabilities.sum()
    
    # Compute entropy: H = -Σ p(x) log p(x)
    entropy = -np.sum(probabilities * np.log2(probabilities))
    
    return float(entropy)


def compute_conditional_entropy(
    joint_probs: np.ndarray, 
    marginal_probs: np.ndarray,
    epsilon: float = 1e-10
) -> float:
    """
    Compute conditional entropy H(Y|X).
    
    Mathematical Formula (Equation 3 in paper):
        H(Y|X) = -Σₓ p(x) Σᵧ p(y|x) log₂ p(y|x)
               = -Σₓ,ᵧ p(x,y) log₂ p(y|x)
    
    Args:
        joint_probs: Joint probability matrix p(x,y) of shape [|X|, |Y|]
        marginal_probs: Marginal probabilities p(x) of shape [|X|]
        epsilon: Small value for numerical stability
    
    Returns:
        Conditional entropy H(Y|X) in bits
    """
    joint_probs = np.asarray(joint_probs, dtype=np.float64) + epsilon
    marginal_probs = np.asarray(marginal_probs, dtype=np.float64) + epsilon
    
    # Normalize
    joint_probs = joint_probs / joint_probs.sum()
    marginal_probs = marginal_probs / marginal_probs.sum()
    
    # Compute conditional probabilities p(y|x) = p(x,y) / p(x)
    # Shape: [|X|, |Y|]
    conditional_probs = joint_probs / marginal_probs[:, np.newaxis]
    conditional_probs = np.clip(conditional_probs, epsilon, 1.0)
    
    # H(Y|X) = -Σ p(x,y) log p(y|x)
    conditional_entropy = -np.sum(joint_probs * np.log2(conditional_probs))
    
    return float(conditional_entropy)


def normalize_scores(scores: np.ndarray, method: str = "minmax") -> np.ndarray:
    """
    Normalize entropy scores to [0, 1] range.
    
    Args:
        scores: Array of entropy scores
        method: Normalization method ("minmax", "zscore", "softmax")
    
    Returns:
        Normalized scores
    """
    scores = np.asarray(scores, dtype=np.float64)
    
    if method == "minmax":
        min_val = scores.min()
        max_val = scores.max()
        if max_val - min_val < 1e-10:
            return np.ones_like(scores) * 0.5
        return (scores - min_val) / (max_val - min_val)
    
    elif method == "zscore":
        mean = scores.mean()
        std = scores.std()
        if std < 1e-10:
            return np.zeros_like(scores)
        return (scores - mean) / std
    
    elif method == "softmax":
        exp_scores = np.exp(scores - scores.max())
        return exp_scores / exp_scores.sum()
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def discretize_features(
    features: np.ndarray, 
    n_bins: int = 64,
    method: str = "uniform"
) -> np.ndarray:
    """
    Discretize continuous features into bins for entropy computation.
    
    This is essential for computing entropy over continuous features like MFCCs.
    
    Args:
        features: Continuous features of shape [n_samples, n_features]
        n_bins: Number of discretization bins
        method: "uniform" for equal-width bins, "quantile" for equal-frequency
    
    Returns:
        Discretized features (integer bin indices)
    """
    features = np.asarray(features, dtype=np.float64)
    
    if features.ndim == 1:
        features = features.reshape(-1, 1)
    
    n_samples, n_features = features.shape
    discretized = np.zeros((n_samples, n_features), dtype=np.int32)
    
    for i in range(n_features):
        col = features[:, i]
        
        if method == "uniform":
            min_val, max_val = col.min(), col.max()
            if max_val - min_val < 1e-10:
                discretized[:, i] = 0
            else:
                bins = np.linspace(min_val, max_val, n_bins + 1)
                discretized[:, i] = np.digitize(col, bins[1:-1])
        
        elif method == "quantile":
            percentiles = np.linspace(0, 100, n_bins + 1)
            bins = np.percentile(col, percentiles)
            bins = np.unique(bins)  # Remove duplicates
            discretized[:, i] = np.digitize(col, bins[1:-1])
        
        else:
            raise ValueError(f"Unknown discretization method: {method}")
    
    return discretized


def estimate_probability_distribution(
    discrete_values: np.ndarray,
    n_categories: Optional[int] = None
) -> np.ndarray:
    """
    Estimate probability distribution from discrete values.
    
    Args:
        discrete_values: Array of discrete category indices
        n_categories: Total number of categories (inferred if None)
    
    Returns:
        Probability distribution over categories
    """
    discrete_values = np.asarray(discrete_values).flatten()
    
    if n_categories is None:
        n_categories = int(discrete_values.max()) + 1
    
    counts = np.bincount(discrete_values.astype(int), minlength=n_categories)
    probabilities = counts / counts.sum()
    
    return probabilities
