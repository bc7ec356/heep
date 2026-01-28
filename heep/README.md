# HEEP: High Entropy Exponential Pruning

A Python implementation of the HEEP data curation methodology for Automatic Speech Recognition (ASR).

## Overview

HEEP is an entropy-based data curation framework that prioritizes information density over data quantity. It identifies high-information training samples while progressively filtering redundant data.

**Core Insight**: Training on strategically selected high-entropy samples leads to better ASR models than training on larger but redundant datasets.

## Mathematical Foundation

### Sample Score (Equation 7)

```
S(x) = α₁·H_acoustic(x) + α₂·H_phonetic(x) + α₃·H_linguistic(x) + α₄·H_contextual(x) + α₅·MI(x, D)
```

Where:
- `H_acoustic(x)`: Spectral/MFCC entropy measuring acoustic diversity
- `H_phonetic(x)`: Phoneme distribution entropy capturing phonetic complexity
- `H_linguistic(x)`: Vocabulary and syntax entropy measuring linguistic richness
- `H_contextual(x)`: Domain and discourse entropy
- `MI(x, D)`: Mutual information contribution relative to dataset
- `α₁...α₅`: Configurable weights (default: 0.25, 0.20, 0.25, 0.15, 0.15)

### Selection Criterion (Equation 8)

```
D' = {x ∈ D : S(x) > τ}
```

### Progressive Filtering (Equation 9)

```
τₖ₊₁ = τₖ × growth_factor
```

## Installation

```bash
# Install dependencies
pip install numpy

# Optional: for full acoustic features
pip install librosa

# Optional: for G2P conversion
pip install g2p_en
```

## Quick Start

```python
from heep import HEEPPipeline
from heep.utils import AudioSample
import numpy as np

# Prepare your data
samples = [
    AudioSample(
        audio=np.random.randn(16000),  # 1 second of audio
        transcription="Hello world, this is a test."
    )
    for _ in range(1000)
]

# Run HEEP curation
pipeline = HEEPPipeline()
curated_samples, stats = pipeline.run(samples)

print(f"Original: {len(samples)} samples")
print(f"Curated: {len(curated_samples)} samples")
print(f"Retention: {stats['reduction_ratio']:.1%}")
```

## Configuration

### Custom Weights

```python
from heep import HEEPPipeline, HEEPConfig

# Emphasize acoustic features (e.g., for noisy data)
config = HEEPConfig(
    weight_acoustic=0.35,
    weight_phonetic=0.15,
    weight_linguistic=0.25,
    weight_contextual=0.15,
    weight_mutual_info=0.10
)

pipeline = HEEPPipeline(config)
```

### Progressive Filtering Settings

```python
config = HEEPConfig(
    initial_threshold=0.3,    # Starting threshold
    growth_factor=1.1,        # Threshold multiplier each round
    max_rounds=10,            # Maximum filtering rounds
    min_samples=1000,         # Stop if below this
    min_fraction=0.1,         # Stop if below 10% of original
    diversity_perturbation=0.05  # Random retention for diversity
)
```

## Detailed Usage

### Score Individual Samples

```python
from heep import HEEPPipeline
from heep.utils import AudioSample

pipeline = HEEPPipeline()
pipeline.fit(samples)  # Fit normalization statistics

# Get detailed score breakdown
sample = samples[0]
details = pipeline.score_detailed(sample)

print(f"Acoustic entropy: {details['acoustic_raw']:.3f}")
print(f"Phonetic entropy: {details['phonetic_raw']:.3f}")
print(f"Linguistic entropy: {details['linguistic_raw']:.3f}")
print(f"Contextual entropy: {details['contextual_raw']:.3f}")
print(f"Mutual information: {details['mutual_info_raw']:.3f}")
print(f"Total score: {details['total_score']:.3f}")
```

### Single-Round Selection

```python
# Select top 50% by score
selected, scores = pipeline.select_single_round(samples, percentile=50)

# Select with fixed threshold
selected, scores = pipeline.select_single_round(samples, threshold=0.6)

# Select exactly top 500 samples
selected, scores = pipeline.select_single_round(samples, top_k=500)
```

### Analyze Dataset

```python
stats = pipeline.analyze_dataset(samples)

print("Score distribution:")
for percentile, value in stats['score_percentiles'].items():
    print(f"  {percentile}th percentile: {value:.3f}")
```

### With Training Callback

```python
def train_model(samples, round_num):
    """Called after each filtering round."""
    print(f"Round {round_num}: Training on {len(samples)} samples...")
    # Your training code here

curated, stats = pipeline.run(samples, train_callback=train_model)
```

## Components

### Entropy Estimators

```python
from heep.entropy import (
    AcousticEntropyEstimator,
    PhoneticEntropyEstimator,
    LinguisticEntropyEstimator,
    ContextualEntropyEstimator
)

# Use individual estimators
acoustic = AcousticEntropyEstimator()
h_acoustic = acoustic.compute(sample)

phonetic = PhoneticEntropyEstimator()
h_phonetic = phonetic.compute(sample)
```

### Sample Scorer

```python
from heep.scoring import SampleScorer, SampleScorerConfig

config = SampleScorerConfig(
    weight_acoustic=0.3,
    weight_linguistic=0.3,
    use_mutual_info=False  # Disable MI computation
)

scorer = SampleScorer(config)
scorer.fit(samples)
scores = scorer.compute_batch(samples)
```

### Progressive Filter

```python
from heep.selection import ProgressiveFilter, ProgressiveFilterConfig

config = ProgressiveFilterConfig(
    initial_threshold=0.3,
    growth_factor=1.15,
    max_rounds=8
)

filter = ProgressiveFilter(config)
curated, stats = filter.run(samples, scorer)
print(filter.get_history_summary())
```

## Algorithm Pseudocode

```
Algorithm: HEEP Data Curation

Input: Dataset D, initial threshold τ₀, growth factor g
Output: Curated dataset D*

1. Initialize scorer with entropy estimators
2. Fit scorer to D (compute normalization stats, fit MI estimator)
3. D* ← D
4. k ← 0
5. While |D*| > min_samples AND k < max_rounds:
    a. For each x ∈ D*:
        Compute S(x) = Σᵢ αᵢ·Hᵢ(x) + MI(x, D)
    b. D* ← {x ∈ D* : S(x) > τₖ}
    c. τₖ₊₁ ← τₖ × g
    d. k ← k + 1
6. Return D*
```

## Paper Reference

This implementation accompanies the paper:

> **HEEP: High Entropy Exponential Pruning for State-of-the-Art ASR Through Strategic Data Curation**
>
> We challenge the "more data is better" paradigm with HEEP, demonstrating that strategic entropy-based data curation outperforms brute-force scaling while requiring orders of magnitude fewer resources.

## Code-to-Paper Mapping

| Paper Equation | Code Location |
|----------------|---------------|
| Eq. 1: H(X) = -Σ p(x) log p(x) | `heep/utils.py:compute_entropy()` |
| Eq. 2: H_acoustic | `heep/entropy/acoustic.py` |
| Eq. 3: H_phonetic | `heep/entropy/phonetic.py` |
| Eq. 4: H_linguistic | `heep/entropy/linguistic.py` |
| Eq. 5: H_contextual | `heep/entropy/contextual.py` |
| Eq. 6: MI(x, D) | `heep/scoring/mutual_info.py` |
| Eq. 7: S(x) = Σ αᵢHᵢ | `heep/scoring/sample_score.py` |
| Eq. 8: D' = {x : S(x) > τ} | `heep/selection/threshold.py` |
| Eq. 9: τₖ₊₁ = τₖ × g | `heep/selection/progressive.py` |
| Algorithm 1 | `heep/pipeline.py:HEEPPipeline.run()` |

## License

[License details here]
