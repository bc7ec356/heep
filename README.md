# HEEP

**High Entropy Exponential Pruning for Speech Recognition**

HEEP is a comprehensive framework for training state-of-the-art multilingual Automatic Speech Recognition (ASR) models through strategic entropy-based data curation. It challenges the "more data is better" paradigm by demonstrating that carefully selected high-information samples outperform brute-force data scaling.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Mathematical Foundation](#mathematical-foundation)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Data Curation with HEEP](#data-curation-with-heep)
- [Training](#training)
- [Inference](#inference)
- [Configuration Reference](#configuration-reference)
- [API Reference](#api-reference)
- [Testing](#testing)
- [Paper Reference](#paper-reference)

---

## Overview

HEEP (High Entropy Exponential Pruning) is an entropy-based data curation methodology that prioritizes information density over data quantity. The framework identifies high-information training samples while progressively filtering redundant data, enabling:

- Training on 10-20% of data while matching or exceeding full-dataset performance
- Efficient multilingual model development with cross-lingual transfer
- Error-aware adaptive sample selection across training rounds
- Significant reduction in computational resources and training time

**Core Insight**: Strategic selection of high-entropy samples leads to better ASR models than training on larger but redundant datasets.

---

## Key Features

### Data Curation (HEEP Library)

- **Multi-dimensional Entropy Estimation**: Acoustic, phonetic, linguistic, and contextual entropy
- **Mutual Information Scoring**: Measures sample contribution relative to the dataset
- **Progressive Filtering**: Iteratively increases selectivity with exponential thresholds
- **Error-Aware Adaptation**: Adjusts selection based on model errors after each round
- **Cross-lingual Transfer**: Leverages phonetic overlap between related languages

### Training Pipeline

- **Multilingual Support**: Configure multiple language groups with shared base tags
- **Distributed Training**: Full support for multi-GPU training via HuggingFace Accelerate
- **Mixed Precision**: FP16 training for faster computation and lower memory usage
- **Checkpoint Management**: Automatic saving at configurable intervals
- **TensorBoard Integration**: Real-time training metrics visualization

### Inference Pipeline

- **Single File and Batch Processing**: Transcribe individual files or entire directories
- **Multiple Output Formats**: Plain text, JSON, SRT subtitles, WebVTT
- **GPU Acceleration**: Automatic device selection with FP16 inference
- **Timestamp Support**: Word and segment-level timestamp generation

---

## Mathematical Foundation

### Sample Score (Equation 7)

The information score for each sample combines multiple entropy dimensions:

```
S(x) = alpha_1 * H_acoustic(x) + alpha_2 * H_phonetic(x) + alpha_3 * H_linguistic(x) 
     + alpha_4 * H_contextual(x) + alpha_5 * MI(x, D)
```

Where:
- `H_acoustic(x)`: Spectral/MFCC entropy measuring acoustic diversity
- `H_phonetic(x)`: Phoneme distribution entropy capturing phonetic complexity
- `H_linguistic(x)`: Vocabulary and syntax entropy measuring linguistic richness
- `H_contextual(x)`: Domain and discourse entropy
- `MI(x, D)`: Mutual information contribution relative to dataset
- `alpha_1...alpha_5`: Configurable weights (default: 0.25, 0.20, 0.25, 0.15, 0.15)

### Selection Criterion (Equation 8)

Samples are selected based on a threshold:

```
D' = {x in D : S(x) > tau}
```

### Progressive Filtering (Equation 9)

The threshold increases exponentially across rounds:

```
tau_{k+1} = tau_k * growth_factor
```

### Error-Aware Adaptation

After each training round, sample scores are adjusted based on model errors:

```
S'(x) = S(x) + lambda_err * ErrorRelevance(x, errors_k) + lambda_cross * CrossLingualOverlap(x)
```

---

## Project Structure

```
heep/
|-- README.md                 # This file
|-- training.py               # Multilingual training script
|-- inference.py              # Inference and transcription script
|-- heep/                     # HEEP data curation library
    |-- __init__.py           # Package exports
    |-- pipeline.py           # Main HEEPPipeline class
    |-- utils.py              # Utility functions and AudioSample class
    |-- requirements.txt      # Python dependencies
    |-- README.md             # HEEP library documentation
    |-- entropy/              # Entropy estimation modules
    |   |-- __init__.py
    |   |-- acoustic.py       # H_acoustic: MFCC/spectral entropy
    |   |-- phonetic.py       # H_phonetic: Phoneme distribution entropy
    |   |-- linguistic.py     # H_linguistic: Vocabulary/syntax entropy
    |   |-- contextual.py     # H_contextual: Domain/discourse entropy
    |-- scoring/              # Sample scoring modules
    |   |-- __init__.py
    |   |-- sample_score.py   # Composite score computation (Eq. 7)
    |   |-- mutual_info.py    # Mutual information estimation (Eq. 6)
    |-- selection/            # Sample selection modules
    |   |-- __init__.py
    |   |-- threshold.py      # Threshold-based selection (Eq. 8)
    |   |-- progressive.py    # Progressive filtering (Eq. 9)
    |-- tests/                # Unit tests
        |-- __init__.py
        |-- test_entropy.py
        |-- test_pipeline.py
```

---

## Installation

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (for GPU acceleration)

### Install Dependencies

```bash
# Clone the repository
git clone https://github.com/your-username/heep.git
cd heep

# Install HEEP library dependencies
pip install -r heep/requirements.txt

# Install training/inference dependencies
pip install torch torchaudio transformers accelerate datasets soundfile

# Optional: Install Flash Attention 2 for faster inference
pip install flash-attn --no-build-isolation
```

### Full Installation (Recommended)

```bash
pip install numpy>=1.20.0
pip install librosa>=0.9.0        # For full acoustic features
pip install g2p_en>=2.1.0         # For grapheme-to-phoneme conversion
pip install torch>=2.0.0
pip install torchaudio>=2.0.0
pip install transformers>=4.36.0
pip install accelerate>=0.25.0
pip install datasets>=2.14.0
pip install soundfile>=0.12.0
pip install tensorboard>=2.14.0
```

---

## Quick Start

### Data Curation with HEEP

```python
from heep import HEEPPipeline
from heep.utils import AudioSample
import numpy as np

# Prepare your data
samples = [
    AudioSample(
        audio=np.random.randn(16000),  # 1 second of audio at 16kHz
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

### Training

```bash
# Configure training.py with your dataset paths and hyperparameters
# Then run:
python training.py

# For distributed training on multiple GPUs:
accelerate launch training.py
```

### Inference

```bash
# Single file transcription
python inference.py --audio path/to/audio.wav

# Batch processing
python inference.py --audio_dir path/to/audio_folder --benchmark

# With specific language and output format
python inference.py --audio file.wav --language hi --output_format srt --timestamps
```

---

## Data Curation with HEEP

### Basic Usage

```python
from heep import HEEPPipeline, HEEPConfig
from heep.utils import AudioSample

# Custom configuration
config = HEEPConfig(
    # Entropy weights
    weight_acoustic=0.25,
    weight_phonetic=0.20,
    weight_linguistic=0.25,
    weight_contextual=0.15,
    weight_mutual_info=0.15,
    
    # Progressive filtering
    initial_threshold=0.3,
    growth_factor=1.1,
    max_rounds=10,
    min_samples=1000,
    min_fraction=0.1,
    
    # Diversity preservation
    diversity_perturbation=0.05
)

pipeline = HEEPPipeline(config)
curated, stats = pipeline.run(samples)
```

### Score Individual Samples

```python
pipeline = HEEPPipeline()
pipeline.fit(samples)

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

### With Training Callback

```python
def train_model(samples, round_num):
    """Called after each filtering round."""
    print(f"Round {round_num}: Training on {len(samples)} samples...")
    # Your training code here
    return model

def eval_model(model, samples):
    """Evaluate model and return predictions/references."""
    predictions = [model.transcribe(s.audio) for s in samples]
    references = [s.transcription for s in samples]
    return predictions, references

# Run with callbacks for error-aware adaptation
curated, stats = pipeline.run(
    samples,
    train_callback=train_model,
    eval_callback=eval_model
)
```

### Analyze Dataset

```python
stats = pipeline.analyze_dataset(samples)

print("Score distribution:")
for percentile, value in stats['score_percentiles'].items():
    print(f"  {percentile}th percentile: {value:.3f}")

print("\nComponent statistics:")
for component in ['acoustic', 'phonetic', 'linguistic', 'contextual']:
    if component in stats:
        print(f"  {component}: mean={stats[component]['mean']:.3f}, std={stats[component]['std']:.3f}")
```

---

## Training

### Configuration

Edit `training.py` to configure your training:

```python
# Language groups for multilingual training
LANGUAGE_GROUPS = {
    "lang_group_1": {
        "base_lang": "xx",  # Replace with language code (e.g., "hi", "en")
        "languages": ["Language1", "Language2", "Language3"],
    },
    "lang_group_2": {
        "base_lang": "yy",  # Replace with language code
        "languages": ["Language4", "Language5"],
    },
}

# Dataset paths (replace with your HuggingFace dataset paths)
PRIMARY_DATASET_PATH = "your-username/your-multilingual-dataset"
SECONDARY_DATASET_PATH = "your-username/your-secondary-dataset"

# Hyperparameters
BATCH_SIZE = 4
LEARNING_RATE = 1e-5
NUM_TRAIN_EPOCHS = 3
WARMUP_STEPS = 500
GRADIENT_ACCUMULATION_STEPS = 8

# Model (replace with your pretrained model path)
PRETRAINED_MODEL_NAME = "your-username/your-pretrained-model"
OUTPUT_DIR = "./finetuned-model"
```

### Running Training

```bash
# Single GPU
python training.py

# Multi-GPU with Accelerate
accelerate config  # Configure distributed settings
accelerate launch training.py

# With specific GPU
CUDA_VISIBLE_DEVICES=0,1 accelerate launch training.py
```

### Monitoring

Training logs are saved to TensorBoard:

```bash
tensorboard --logdir ./finetuned-model
```

---

## Inference

### Command Line Options

```
Input Options:
  --audio, -a           Path to single audio file
  --audio_dir, -d       Directory containing audio files

Model Options:
  --model, -m           HuggingFace model ID (default: your-username/your-model)
  --language, -l        Language code (auto-detect if not specified)
  --task, -t            Task: transcribe or translate (default: transcribe)

Output Options:
  --output, -o          Output file path (default: stdout)
  --output_format, -f   Format: txt, json, srt, vtt (default: txt)
  --timestamps          Include word-level timestamps

Performance Options:
  --benchmark           Print timing statistics
```

### Examples

```bash
# Basic transcription
python inference.py --audio recording.wav

# Transcription with SRT output
python inference.py --audio speech.wav --language hi --output_format srt --timestamps --output subtitles.srt

# Batch processing with benchmarking
python inference.py --audio_dir ./audio_files --benchmark

# Translation to English
python inference.py --audio foreign_speech.wav --task translate

# Using a specific model
python inference.py --audio file.wav --model your-username/your-model
```

### Programmatic Usage

```python
from inference import load_model, transcribe_file, setup_device

# Setup
device, torch_dtype = setup_device()
model, processor, pipe = load_model("your-username/your-model", device, torch_dtype)

# Transcribe
result = transcribe_file(
    pipe,
    "audio.wav",
    language="en",
    return_timestamps=True
)

print(f"Transcription: {result['text']}")
print(f"Inference time: {result['inference_time']:.2f}s")
print(f"Real-time factor: {result['rtf']:.3f}x")
```

---

## Configuration Reference

### HEEPConfig

| Parameter | Default | Description |
|-----------|---------|-------------|
| `weight_acoustic` | 0.25 | Weight for acoustic entropy (alpha_1) |
| `weight_phonetic` | 0.20 | Weight for phonetic entropy (alpha_2) |
| `weight_linguistic` | 0.25 | Weight for linguistic entropy (alpha_3) |
| `weight_contextual` | 0.15 | Weight for contextual entropy (alpha_4) |
| `weight_mutual_info` | 0.15 | Weight for mutual information (alpha_5) |
| `initial_threshold` | 0.3 | Starting threshold (tau_0) |
| `growth_factor` | 1.1 | Threshold multiplier each round |
| `max_rounds` | 10 | Maximum filtering rounds |
| `min_samples` | 1000 | Stop if below this count |
| `min_fraction` | 0.1 | Stop if below this fraction |
| `diversity_perturbation` | 0.05 | Random retention probability |

### ProgressiveFilterConfig (Advanced)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `error_boost_weight` | 0.3 | Weight for error-relevant samples |
| `error_analysis_enabled` | True | Enable error-aware adaptation |
| `cross_lingual_boost` | 0.2 | Boost for cross-lingual transfer |

---

## API Reference

### Core Classes

#### HEEPPipeline

Main interface for HEEP data curation.

```python
class HEEPPipeline:
    def __init__(self, config: Optional[HEEPConfig] = None)
    def fit(self, samples: List[AudioSample])
    def score(self, samples: List[AudioSample]) -> np.ndarray
    def score_detailed(self, sample: AudioSample) -> Dict[str, float]
    def run(self, samples, train_callback=None) -> Tuple[List[AudioSample], Dict]
    def select_single_round(self, samples, threshold=None, percentile=None, top_k=None)
    def analyze_dataset(self, samples) -> Dict[str, Any]
```

#### AudioSample

Data class representing an audio sample.

```python
@dataclass
class AudioSample:
    audio: np.ndarray          # Audio waveform [num_samples]
    transcription: str         # Text transcription
    sample_rate: int = 16000   # Sample rate in Hz
    sample_id: Optional[str]   # Unique identifier
    metadata: Optional[dict]   # Additional metadata
```

#### SampleScorer

Computes composite information scores.

```python
class SampleScorer:
    def __init__(self, config: Optional[SampleScorerConfig] = None)
    def fit(self, samples: List[AudioSample])
    def compute(self, sample: AudioSample) -> float
    def compute_batch(self, samples: List[AudioSample]) -> np.ndarray
    def compute_detailed(self, sample: AudioSample) -> Dict[str, float]
```

#### ProgressiveFilter

Implements iterative filtering with error-aware adaptation.

```python
class ProgressiveFilter:
    def __init__(self, config: Optional[ProgressiveFilterConfig] = None)
    def run(self, samples, scorer, train_callback=None, eval_callback=None)
    def filter_round(self, samples, scores, preserve_diversity=True)
    def analyze_errors(self, predictions, references, samples) -> ErrorPattern
    def get_history_summary() -> str
```

---

## Testing

Run the test suite:

```bash
# Run all tests
cd heep
pytest tests/ -v

# Run specific test file
pytest tests/test_entropy.py -v

# Run with coverage
pytest tests/ --cov=heep --cov-report=html
```

---

## Paper Reference

This implementation accompanies the paper:

> **HEEP: High Entropy Exponential Pruning for State-of-the-Art ASR Through Strategic Data Curation**
>
> We challenge the "more data is better" paradigm with HEEP, demonstrating that strategic entropy-based data curation outperforms brute-force scaling while requiring orders of magnitude fewer resources.

### Code-to-Paper Mapping

| Paper Equation | Code Location |
|----------------|---------------|
| Eq. 1: H(X) = -sum p(x) log p(x) | `heep/utils.py:compute_entropy()` |
| Eq. 2: H_acoustic | `heep/entropy/acoustic.py` |
| Eq. 3: H_phonetic | `heep/entropy/phonetic.py` |
| Eq. 4: H_linguistic | `heep/entropy/linguistic.py` |
| Eq. 5: H_contextual | `heep/entropy/contextual.py` |
| Eq. 6: MI(x, D) | `heep/scoring/mutual_info.py` |
| Eq. 7: S(x) = sum alpha_i * H_i | `heep/scoring/sample_score.py` |
| Eq. 8: D' = {x : S(x) > tau} | `heep/selection/threshold.py` |
| Eq. 9: tau_{k+1} = tau_k * g | `heep/selection/progressive.py` |
| Algorithm 1 | `heep/pipeline.py:HEEPPipeline.run()` |

---

## Algorithm Overview

```
Algorithm: HEEP Data Curation with Error-Aware Adaptation

Input: Dataset D, initial threshold tau_0, growth factor g
Output: Curated dataset D*

1. Initialize scorer with entropy estimators
2. Fit scorer to D (compute normalization stats, fit MI estimator)
3. D* <- D
4. k <- 0
5. While |D*| > min_samples AND k < max_rounds:
    a. For each x in D*:
        Compute S(x) = sum_i alpha_i * H_i(x) + MI(x, D)
    b. If error_patterns available:
        Adjust S'(x) = S(x) + lambda_err * ErrorRelevance(x)
    c. D* <- {x in D* : S'(x) > tau_k}
    d. If train_callback: Train model on D*
    e. If eval_callback: Analyze errors, update error_patterns
    f. tau_{k+1} <- tau_k * g
    g. k <- k + 1
6. Return D*
```

---

## Acknowledgments

This work builds upon research in information theory, data curation, and multilingual speech recognition. We thank the open-source community for the foundational tools that made this project possible.
