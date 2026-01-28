"""
Acoustic Entropy Estimation for HEEP.

Computes H_acoustic(x) - the entropy of acoustic features (MFCCs, mel-spectrograms)
to measure spectral diversity and acoustic complexity of audio samples.

Mathematical Formulation (Paper Equation 2):
    H_acoustic(x) = -Σᵢ p(bᵢ|x) log₂ p(bᵢ|x)

Where:
    - x is the audio sample
    - bᵢ are discretized acoustic feature bins
    - p(bᵢ|x) is the probability of bin bᵢ given sample x

The acoustic entropy measures how diverse the spectral content is across
the audio sample. High entropy indicates varied acoustic patterns.
"""

import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass, field

from ..utils import (
    compute_entropy,
    discretize_features,
    estimate_probability_distribution,
    AudioSample
)


@dataclass
class AcousticEntropyConfig:
    """Configuration for acoustic entropy estimation."""
    # MFCC parameters
    n_mfcc: int = 13
    n_mels: int = 80
    n_fft: int = 512
    hop_length: int = 160  # 10ms at 16kHz
    
    # Discretization parameters
    n_bins: int = 64
    discretization_method: str = "quantile"
    
    # Entropy computation
    use_delta: bool = True  # Include delta features
    use_delta_delta: bool = True  # Include delta-delta features


class AcousticEntropyEstimator:
    """
    Estimates acoustic entropy from audio samples.
    
    The acoustic entropy H_acoustic(x) quantifies the spectral diversity
    of an audio sample. Samples with higher acoustic entropy contain
    more varied spectral patterns, which typically correspond to:
    - Multiple speakers or acoustic conditions
    - Diverse phonetic content
    - Background acoustic variability
    
    Implementation:
        1. Extract MFCC features from audio
        2. Optionally compute delta and delta-delta features
        3. Discretize continuous features into bins
        4. Estimate probability distribution over bins
        5. Compute Shannon entropy
    
    Example:
        >>> estimator = AcousticEntropyEstimator()
        >>> sample = AudioSample(audio=audio_array, transcription="hello world")
        >>> entropy = estimator.compute(sample)
        >>> print(f"Acoustic entropy: {entropy:.3f} bits")
    """
    
    def __init__(self, config: Optional[AcousticEntropyConfig] = None):
        """
        Initialize the acoustic entropy estimator.
        
        Args:
            config: Configuration parameters. Uses defaults if None.
        """
        self.config = config or AcousticEntropyConfig()
        self._fitted = False
        self._global_min = None
        self._global_max = None
    
    def extract_mfcc(
        self, 
        audio: np.ndarray, 
        sample_rate: int = 16000
    ) -> np.ndarray:
        """
        Extract MFCC features from audio.
        
        Args:
            audio: Audio waveform of shape [num_samples]
            sample_rate: Sample rate in Hz
        
        Returns:
            MFCC features of shape [n_frames, n_mfcc * (1 + use_delta + use_delta_delta)]
        """
        try:
            import librosa
        except ImportError:
            # Fallback to simple spectral features if librosa not available
            return self._extract_spectral_features(audio, sample_rate)
        
        # Extract MFCCs
        mfccs = librosa.feature.mfcc(
            y=audio.astype(np.float32),
            sr=sample_rate,
            n_mfcc=self.config.n_mfcc,
            n_mels=self.config.n_mels,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length
        )  # Shape: [n_mfcc, n_frames]
        
        features = [mfccs.T]  # Transpose to [n_frames, n_mfcc]
        
        # Add delta features (first derivative)
        if self.config.use_delta:
            delta = librosa.feature.delta(mfccs)
            features.append(delta.T)
        
        # Add delta-delta features (second derivative)
        if self.config.use_delta_delta:
            delta2 = librosa.feature.delta(mfccs, order=2)
            features.append(delta2.T)
        
        return np.concatenate(features, axis=1)
    
    def _extract_spectral_features(
        self,
        audio: np.ndarray,
        sample_rate: int
    ) -> np.ndarray:
        """
        Fallback spectral feature extraction without librosa.
        
        Uses simple FFT-based features.
        """
        # Frame the audio
        frame_length = self.config.n_fft
        hop_length = self.config.hop_length
        
        # Pad audio
        audio = np.pad(audio, (0, frame_length), mode='constant')
        
        # Create frames
        n_frames = 1 + (len(audio) - frame_length) // hop_length
        frames = np.zeros((n_frames, frame_length))
        
        for i in range(n_frames):
            start = i * hop_length
            frames[i] = audio[start:start + frame_length]
        
        # Apply Hann window
        window = np.hanning(frame_length)
        frames = frames * window
        
        # Compute FFT magnitude
        fft_mag = np.abs(np.fft.rfft(frames, axis=1))
        
        # Simple mel-scale approximation using log-spaced bins
        n_features = self.config.n_mfcc
        features = np.zeros((n_frames, n_features))
        
        fft_bins = fft_mag.shape[1]
        bin_edges = np.logspace(0, np.log10(fft_bins), n_features + 1).astype(int)
        
        for i in range(n_features):
            start_bin = bin_edges[i]
            end_bin = min(bin_edges[i + 1], fft_bins)
            if end_bin > start_bin:
                features[:, i] = np.mean(fft_mag[:, start_bin:end_bin], axis=1)
        
        # Apply log compression
        features = np.log1p(features)
        
        return features
    
    def compute(self, sample: AudioSample) -> float:
        """
        Compute acoustic entropy for a single audio sample.
        
        Mathematical Formula:
            H_acoustic(x) = -Σᵢ p(bᵢ|x) log₂ p(bᵢ|x)
        
        Where bᵢ are discretized MFCC bins.
        
        Args:
            sample: AudioSample containing audio waveform
        
        Returns:
            Acoustic entropy value in bits
        """
        # Extract MFCC features
        features = self.extract_mfcc(sample.audio, sample.sample_rate)
        
        if features.shape[0] == 0:
            return 0.0
        
        # Discretize features
        discrete_features = discretize_features(
            features,
            n_bins=self.config.n_bins,
            method=self.config.discretization_method
        )
        
        # Compute entropy for each feature dimension and average
        entropies = []
        for dim in range(discrete_features.shape[1]):
            probs = estimate_probability_distribution(
                discrete_features[:, dim],
                n_categories=self.config.n_bins
            )
            h = compute_entropy(probs)
            entropies.append(h)
        
        # Return average entropy across dimensions
        return float(np.mean(entropies))
    
    def compute_batch(self, samples: list) -> np.ndarray:
        """
        Compute acoustic entropy for multiple samples.
        
        Args:
            samples: List of AudioSample objects
        
        Returns:
            Array of entropy values
        """
        return np.array([self.compute(s) for s in samples])
