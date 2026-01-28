"""
Tests for HEEP pipeline and scoring components.

Run with: pytest heep/tests/test_pipeline.py -v
"""

import numpy as np
import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from heep.utils import AudioSample
from heep.scoring import SampleScorer, SampleScorerConfig
from heep.scoring import MutualInformationEstimator
from heep.selection import ThresholdSelector, ProgressiveFilter
from heep.pipeline import HEEPPipeline, HEEPConfig


def create_sample_dataset(n_samples: int = 100) -> list:
    """Create a synthetic dataset for testing."""
    samples = []
    texts = [
        "Hello world this is a test",
        "The quick brown fox jumps over the lazy dog",
        "Machine learning is transforming speech recognition",
        "Um well you know like basically",
        "Patient presents with acute symptoms requiring immediate attention",
        "Market earnings exceeded quarterly projections significantly",
        "The algorithm processes data efficiently",
    ]
    
    for i in range(n_samples):
        # Create varied audio
        duration = np.random.uniform(0.5, 3.0)
        n_samples_audio = int(duration * 16000)
        
        # Mix of sine waves and noise
        t = np.linspace(0, duration, n_samples_audio)
        freq = np.random.uniform(200, 800)
        audio = np.sin(2 * np.pi * freq * t) + np.random.randn(n_samples_audio) * 0.1
        
        sample = AudioSample(
            audio=audio.astype(np.float32),
            transcription=texts[i % len(texts)],
            sample_rate=16000,
            sample_id=f"sample_{i}"
        )
        samples.append(sample)
    
    return samples


class TestSampleScorer:
    """Test sample scoring functionality."""
    
    @pytest.fixture
    def samples(self):
        return create_sample_dataset(20)
    
    @pytest.fixture
    def scorer(self):
        return SampleScorer()
    
    def test_scorer_initialization(self):
        """Scorer should initialize with default config."""
        scorer = SampleScorer()
        weights = scorer.get_weights()
        assert 'acoustic' in weights
        assert 'linguistic' in weights
    
    def test_scorer_fit(self, scorer, samples):
        """Scorer should fit without error."""
        scorer.fit(samples)
        assert scorer._fitted
    
    def test_scorer_compute(self, scorer, samples):
        """Scorer should compute scores."""
        scorer.fit(samples)
        score = scorer.compute(samples[0])
        assert isinstance(score, float)
        assert score >= 0
    
    def test_scorer_batch(self, scorer, samples):
        """Scorer should compute batch scores."""
        scorer.fit(samples)
        scores = scorer.compute_batch(samples)
        assert len(scores) == len(samples)
        assert all(s >= 0 for s in scores)
    
    def test_scorer_detailed(self, scorer, samples):
        """Scorer should provide detailed breakdown."""
        scorer.fit(samples)
        details = scorer.compute_detailed(samples[0])
        assert 'total_score' in details
        assert 'acoustic_raw' in details or 'linguistic_raw' in details


class TestMutualInformation:
    """Test mutual information estimation."""
    
    @pytest.fixture
    def samples(self):
        return create_sample_dataset(50)
    
    @pytest.fixture
    def estimator(self):
        return MutualInformationEstimator()
    
    def test_mi_fit(self, estimator, samples):
        """MI estimator should fit."""
        estimator.fit(samples)
        assert estimator._fitted
    
    def test_mi_compute(self, estimator, samples):
        """MI estimator should compute values."""
        estimator.fit(samples)
        mi = estimator.compute(samples[0])
        assert isinstance(mi, float)
        assert mi >= 0


class TestThresholdSelector:
    """Test threshold-based selection."""
    
    def test_threshold_selection(self):
        """Basic threshold selection."""
        samples = create_sample_dataset(10)
        scores = np.array([0.2, 0.4, 0.6, 0.8, 1.0, 0.3, 0.5, 0.7, 0.9, 0.1])
        
        selector = ThresholdSelector(threshold=0.5)
        selected, selected_scores = selector.select(samples, scores)
        
        assert len(selected) == 5  # 0.6, 0.8, 1.0, 0.7, 0.9
        assert all(s > 0.5 for s in selected_scores)
    
    def test_percentile_selection(self):
        """Percentile-based selection."""
        from heep.selection.threshold import ThresholdSelectorConfig
        
        samples = create_sample_dataset(10)
        scores = np.arange(10) / 10  # 0.0 to 0.9
        
        config = ThresholdSelectorConfig(mode="percentile", percentile=50)
        selector = ThresholdSelector(config)
        selected, _ = selector.select(samples, scores)
        
        assert len(selected) == 5  # Top 50%
    
    def test_top_k_selection(self):
        """Top-k selection."""
        from heep.selection.threshold import ThresholdSelectorConfig
        
        samples = create_sample_dataset(10)
        scores = np.arange(10) / 10
        
        config = ThresholdSelectorConfig(mode="top_k", top_k=3)
        selector = ThresholdSelector(config)
        selected, _ = selector.select(samples, scores)
        
        assert len(selected) == 3


class TestProgressiveFilter:
    """Test progressive filtering."""
    
    @pytest.fixture
    def samples(self):
        return create_sample_dataset(100)
    
    def test_progressive_filter_run(self, samples):
        """Progressive filter should reduce dataset."""
        scorer = SampleScorer()
        scorer.fit(samples)
        
        from heep.selection.progressive import ProgressiveFilterConfig
        config = ProgressiveFilterConfig(
            initial_threshold=0.3,
            growth_factor=1.2,
            max_rounds=3,
            min_samples=10
        )
        
        filter = ProgressiveFilter(config)
        curated, stats = filter.run(samples, scorer)
        
        assert len(curated) < len(samples)
        assert len(curated) >= config.min_samples
        assert 'total_rounds' in stats
    
    def test_progressive_filter_history(self, samples):
        """Progressive filter should record history."""
        scorer = SampleScorer()
        scorer.fit(samples)
        
        filter = ProgressiveFilter()
        filter.run(samples, scorer)
        
        assert len(filter.history) > 0
        assert 'round' in filter.history[0]
        assert 'retention_rate' in filter.history[0]


class TestHEEPPipeline:
    """Test complete HEEP pipeline."""
    
    @pytest.fixture
    def samples(self):
        return create_sample_dataset(100)
    
    def test_pipeline_basic(self, samples):
        """Pipeline should run end-to-end."""
        pipeline = HEEPPipeline()
        curated, stats = pipeline.run(samples)
        
        assert len(curated) <= len(samples)
        assert 'original_size' in stats
        assert 'final_size' in stats
    
    def test_pipeline_with_config(self, samples):
        """Pipeline should accept custom config."""
        config = HEEPConfig(
            weight_acoustic=0.4,
            weight_linguistic=0.4,
            weight_phonetic=0.1,
            weight_contextual=0.05,
            weight_mutual_info=0.05,
            initial_threshold=0.2,
            max_rounds=2
        )
        
        pipeline = HEEPPipeline(config)
        curated, stats = pipeline.run(samples)
        
        assert len(curated) > 0
    
    def test_pipeline_analyze(self, samples):
        """Pipeline should analyze dataset."""
        pipeline = HEEPPipeline()
        stats = pipeline.analyze_dataset(samples)
        
        assert 'total_score' in stats
        assert 'score_percentiles' in stats
    
    def test_pipeline_single_round(self, samples):
        """Pipeline should support single-round selection."""
        pipeline = HEEPPipeline()
        
        selected, scores = pipeline.select_single_round(samples, percentile=50)
        
        assert len(selected) == len(samples) // 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
