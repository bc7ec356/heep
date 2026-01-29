"""
Phonetic Entropy Estimation for HEEP.

Computes H_phonetic(x) - the entropy of phoneme distributions to measure
phonetic diversity and complexity of audio samples.

Mathematical Formulation (Paper Equation 4):
    H_p(x) = -Σ_c p(c) Σ_f p(f|c) log p(f|c)

Where:
    - c represents phonetic contexts (preceding phonemes)
    - f represents following phonemes
    - p(c) is the probability of context c
    - p(f|c) is the conditional probability of phoneme f given context c

This is the weighted conditional entropy. Phoneme sequences are obtained via
G2P conversion (not forced alignment) to avoid circular dependency with ASR.
This means H_p captures *expected phonetic complexity* from the transcription
text, complementing (not duplicating) acoustic entropy H_a.
"""

import numpy as np
from typing import Optional, List, Dict
from dataclasses import dataclass
from collections import Counter

from ..utils import (
    compute_entropy,
    compute_conditional_entropy,
    AudioSample
)


# Standard phoneme inventory (IPA-based, simplified)
PHONEME_INVENTORY = [
    # Vowels
    'AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'EH', 'ER', 'EY', 'IH', 
    'IY', 'OW', 'OY', 'UH', 'UW',
    # Consonants
    'B', 'CH', 'D', 'DH', 'F', 'G', 'HH', 'JH', 'K', 'L', 
    'M', 'N', 'NG', 'P', 'R', 'S', 'SH', 'T', 'TH', 'V', 
    'W', 'Y', 'Z', 'ZH',
    # Special
    'SIL', 'SPN'  # Silence, spoken noise
]


@dataclass
class PhoneticEntropyConfig:
    """Configuration for phonetic entropy estimation."""
    # Phoneme recognition settings
    use_g2p: bool = True  # Use grapheme-to-phoneme conversion
    phoneme_inventory: List[str] = None
    
    # Context settings for conditional entropy
    context_size: int = 1  # Number of preceding phonemes for context
    
    # Smoothing
    smoothing: float = 0.01  # Laplace smoothing factor
    
    def __post_init__(self):
        if self.phoneme_inventory is None:
            self.phoneme_inventory = PHONEME_INVENTORY.copy()


class PhoneticEntropyEstimator:
    """
    Estimates phonetic entropy from audio samples.
    
    The phonetic entropy H_phonetic(x) quantifies the diversity and
    complexity of phoneme patterns in an audio sample. High phonetic
    entropy indicates:
    - Diverse phoneme usage
    - Complex phonotactic patterns
    - Rare phoneme sequences
    
    Implementation:
        1. Convert transcription to phoneme sequence (G2P)
        2. Build phoneme context-next pairs
        3. Estimate conditional probability p(next|context)
        4. Compute weighted conditional entropy
    
    Example:
        >>> estimator = PhoneticEntropyEstimator()
        >>> sample = AudioSample(audio=audio_array, transcription="hello world")
        >>> entropy = estimator.compute(sample)
        >>> print(f"Phonetic entropy: {entropy:.3f} bits")
    """
    
    def __init__(self, config: Optional[PhoneticEntropyConfig] = None):
        """
        Initialize the phonetic entropy estimator.
        
        Args:
            config: Configuration parameters. Uses defaults if None.
        """
        self.config = config or PhoneticEntropyConfig()
        self._g2p = None
        self._phoneme_to_idx = {
            p: i for i, p in enumerate(self.config.phoneme_inventory)
        }
    
    def _init_g2p(self):
        """Initialize grapheme-to-phoneme converter."""
        if self._g2p is not None:
            return
        
        try:
            from g2p_en import G2p
            self._g2p = G2p()
        except ImportError:
            # Fallback: simple rule-based approximation
            self._g2p = self._simple_g2p
    
    def _simple_g2p(self, text: str) -> List[str]:
        """
        Simple rule-based grapheme-to-phoneme conversion.
        
        This is a fallback when g2p_en is not available.
        Maps characters to approximate phonemes.
        """
        # Very simplified G2P mapping
        char_to_phoneme = {
            'a': 'AH', 'b': 'B', 'c': 'K', 'd': 'D', 'e': 'EH',
            'f': 'F', 'g': 'G', 'h': 'HH', 'i': 'IH', 'j': 'JH',
            'k': 'K', 'l': 'L', 'm': 'M', 'n': 'N', 'o': 'OW',
            'p': 'P', 'q': 'K', 'r': 'R', 's': 'S', 't': 'T',
            'u': 'UW', 'v': 'V', 'w': 'W', 'x': 'K', 'y': 'Y',
            'z': 'Z', ' ': 'SIL'
        }
        
        text = text.lower()
        phonemes = []
        for char in text:
            if char in char_to_phoneme:
                phonemes.append(char_to_phoneme[char])
        
        return phonemes
    
    def text_to_phonemes(self, text: str) -> List[str]:
        """
        Convert text to phoneme sequence.
        
        Args:
            text: Input text string
        
        Returns:
            List of phoneme symbols
        """
        if not self.config.use_g2p:
            return self._simple_g2p(text)
        
        self._init_g2p()
        
        if callable(self._g2p) and self._g2p == self._simple_g2p:
            return self._simple_g2p(text)
        
        # Use g2p_en
        phonemes = self._g2p(text)
        
        # Filter to known phonemes and remove stress markers
        filtered = []
        for p in phonemes:
            # Remove stress markers (numbers at end)
            p_clean = ''.join(c for c in p if not c.isdigit())
            if p_clean in self._phoneme_to_idx:
                filtered.append(p_clean)
            elif p == ' ':
                filtered.append('SIL')
        
        return filtered
    
    def compute(self, sample: AudioSample) -> float:
        """
        Compute phonetic entropy for a single audio sample.
        
        Mathematical Formula:
            H_phonetic(x) = -Σ_c p(c) Σ_f p(f|c) log₂ p(f|c)
        
        This is the weighted conditional entropy where:
        - c is the context (preceding phoneme(s))
        - f is the following phoneme
        - The weighting by p(c) ensures proper entropy computation
        
        Args:
            sample: AudioSample containing transcription
        
        Returns:
            Phonetic entropy value in bits
        """
        # Convert transcription to phonemes
        phonemes = self.text_to_phonemes(sample.transcription)
        
        if len(phonemes) < self.config.context_size + 1:
            return 0.0
        
        n_phonemes = len(self.config.phoneme_inventory)
        context_size = self.config.context_size
        
        # Build context-next pairs
        context_counts = Counter()
        joint_counts = Counter()
        
        for i in range(context_size, len(phonemes)):
            context = tuple(phonemes[i - context_size:i])
            next_phoneme = phonemes[i]
            
            context_counts[context] += 1
            joint_counts[(context, next_phoneme)] += 1
        
        total = sum(context_counts.values())
        if total == 0:
            return 0.0
        
        # Compute weighted conditional entropy
        # H(F|C) = -Σ_c p(c) Σ_f p(f|c) log p(f|c)
        entropy = 0.0
        smoothing = self.config.smoothing
        
        for context, context_count in context_counts.items():
            p_context = context_count / total
            
            # Get all next phonemes for this context
            context_next_total = context_count + smoothing * n_phonemes
            
            for next_phoneme in self.config.phoneme_inventory:
                joint_count = joint_counts.get((context, next_phoneme), 0)
                p_next_given_context = (joint_count + smoothing) / context_next_total
                
                if p_next_given_context > 0:
                    entropy -= p_context * p_next_given_context * np.log2(p_next_given_context)
        
        return float(entropy)
    
    def compute_unigram_entropy(self, sample: AudioSample) -> float:
        """
        Compute simple unigram phoneme entropy (without context).
        
        Formula: H(P) = -Σ_p p(p) log₂ p(p)
        
        Args:
            sample: AudioSample containing transcription
        
        Returns:
            Unigram phonetic entropy in bits
        """
        phonemes = self.text_to_phonemes(sample.transcription)
        
        if len(phonemes) == 0:
            return 0.0
        
        # Count phonemes
        counts = Counter(phonemes)
        total = sum(counts.values())
        
        # Compute probabilities
        probs = np.array([counts.get(p, 0) / total 
                         for p in self.config.phoneme_inventory])
        
        return compute_entropy(probs)
    
    def compute_batch(self, samples: list) -> np.ndarray:
        """
        Compute phonetic entropy for multiple samples.
        
        Args:
            samples: List of AudioSample objects
        
        Returns:
            Array of entropy values
        """
        return np.array([self.compute(s) for s in samples])
