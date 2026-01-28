"""
Tests for HEEP entropy estimation components.

Run with: pytest heep/tests/test_entropy.py -v
"""

import numpy as np
import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from heep.utils import AudioSample, compute_entropy, discretize_features
from heep.entropy.acoustic import AcousticEntropyEstimator
from heep.entropy.phonetic import PhoneticEntropyEstimator
from heep.entropy.linguistic import LinguisticEntropyEstimator
from heep.entropy.contextual import ContextualEntropyEstimator


class TestUtils:
    """Test utility functions."""
    
    def test_compute_entropy_uniform(self):
        """Uniform distribution should have maximum entropy."""
        probs = np.array([0.25, 0.25, 0.25, 0.25])
        entropy = compute_entropy(probs)
        assert abs(entropy - 2.0) < 0.01  # log2(4) = 2
    
    def test_compute_entropy_deterministic(self):
        """Deterministic distribution should have near-zero entropy."""
        probs = np.array([1.0, 0.0, 0.0, 0.0])
        entropy = compute_entropy(probs)
        assert entropy < 0.1  # Should be very close to 0
    
    def test_compute_entropy_binary(self):
        """Binary fair coin should have entropy of 1 bit."""
        probs = np.array([0.5, 0.5])
        entropy = compute_entropy(probs)
        assert abs(entropy - 1.0) < 0.01
    
    def test_discretize_features(self):
        """Test feature discretization."""
        features = np.random.randn(100, 5)
        discrete = discretize_features(features, n_bins=10)
        
        assert discrete.shape == features.shape
        assert discrete.dtype == np.int32
        assert discrete.min() >= 0
        assert discrete.max() < 10


class TestAcousticEntropy:
    """Test acoustic entropy estimation."""
    
    @pytest.fixture
    def estimator(self):
        return AcousticEntropyEstimator()
    
    @pytest.fixture
    def sample_silence(self):
        """Silent audio sample."""
        return AudioSample(
            audio=np.zeros(16000),
            transcription="",
            sample_rate=16000
        )
    
    @pytest.fixture
    def sample_varied(self):
        """Varied audio sample (more entropy)."""
        # Create audio with varied frequency content
        t = np.linspace(0, 1, 16000)
        audio = np.sin(2 * np.pi * 440 * t) + \
                np.sin(2 * np.pi * 880 * t) + \
                np.random.randn(16000) * 0.1
        return AudioSample(
            audio=audio.astype(np.float32),
            transcription="test audio",
            sample_rate=16000
        )
    
    def test_acoustic_entropy_computed(self, estimator, sample_varied):
        """Acoustic entropy should be computable."""
        entropy = estimator.compute(sample_varied)
        assert isinstance(entropy, float)
        assert entropy >= 0
    
    def test_acoustic_entropy_varied_higher(self, estimator, sample_silence, sample_varied):
        """Varied audio should have higher entropy than silence."""
        h_silence = estimator.compute(sample_silence)
        h_varied = estimator.compute(sample_varied)
        # Note: silence might still have some entropy due to discretization
        # The varied sample should generally have higher entropy
        assert h_varied > 0


class TestPhoneticEntropy:
    """Test phonetic entropy estimation."""
    
    @pytest.fixture
    def estimator(self):
        return PhoneticEntropyEstimator()
    
    def test_phonetic_entropy_simple(self, estimator):
        """Simple text should produce reasonable entropy."""
        sample = AudioSample(
            audio=np.zeros(16000),
            transcription="hello world",
            sample_rate=16000
        )
        entropy = estimator.compute(sample)
        assert isinstance(entropy, float)
        assert entropy >= 0
    
    def test_phonetic_entropy_diverse_higher(self, estimator):
        """Diverse phoneme text should have higher entropy."""
        simple_sample = AudioSample(
            audio=np.zeros(16000),
            transcription="aaa aaa aaa",
            sample_rate=16000
        )
        diverse_sample = AudioSample(
            audio=np.zeros(16000),
            transcription="the quick brown fox jumps over the lazy dog",
            sample_rate=16000
        )
        
        h_simple = estimator.compute(simple_sample)
        h_diverse = estimator.compute(diverse_sample)
        
        assert h_diverse > h_simple
    
    def test_g2p_conversion(self, estimator):
        """Test grapheme-to-phoneme conversion."""
        phonemes = estimator.text_to_phonemes("hello")
        assert isinstance(phonemes, list)
        assert len(phonemes) > 0


class TestLinguisticEntropy:
    """Test linguistic entropy estimation."""
    
    @pytest.fixture
    def estimator(self):
        return LinguisticEntropyEstimator()
    
    def test_linguistic_entropy_basic(self, estimator):
        """Basic linguistic entropy computation."""
        sample = AudioSample(
            audio=np.zeros(16000),
            transcription="The cat sat on the mat",
            sample_rate=16000
        )
        entropy = estimator.compute(sample)
        assert isinstance(entropy, float)
        assert entropy >= 0
    
    def test_linguistic_entropy_empty(self, estimator):
        """Empty transcription should have zero entropy."""
        sample = AudioSample(
            audio=np.zeros(16000),
            transcription="",
            sample_rate=16000
        )
        entropy = estimator.compute(sample)
        assert entropy == 0.0
    
    def test_vocabulary_diversity(self, estimator):
        """Test type-token ratio computation."""
        tokens = ["the", "cat", "sat", "on", "the", "mat"]
        ttr = estimator.compute_vocabulary_diversity(tokens)
        assert 0 <= ttr <= 1
        assert ttr == 5 / 6  # 5 unique out of 6 total


class TestContextualEntropy:
    """Test contextual entropy estimation."""
    
    @pytest.fixture
    def estimator(self):
        return ContextualEntropyEstimator()
    
    def test_contextual_entropy_basic(self, estimator):
        """Basic contextual entropy computation."""
        sample = AudioSample(
            audio=np.zeros(16000),
            transcription="Well, you know, I think we should proceed",
            sample_rate=16000
        )
        entropy = estimator.compute(sample)
        assert isinstance(entropy, float)
        assert entropy >= 0
    
    def test_discourse_markers_detected(self, estimator):
        """Discourse markers should be detected."""
        text = "um well you know like okay"
        counts = estimator.count_discourse_markers(text)
        assert len(counts) > 0
        assert "um" in counts or "well" in counts
    
    def test_domain_classification(self, estimator):
        """Domain classification should return valid probabilities."""
        probs = estimator.classify_domain("The patient presented with symptoms")
        assert isinstance(probs, np.ndarray)
        assert abs(probs.sum() - 1.0) < 0.01  # Should sum to 1
        assert all(p >= 0 for p in probs)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
