"""
Linguistic Entropy Estimation for HEEP.

Computes H_linguistic(x) - the entropy of linguistic features to measure
vocabulary richness and syntactic complexity of transcriptions.

Mathematical Formulation (Paper Equation 5):
    H_l(x) = H_uni(y) + H_bi(y)
    H_uni(y) = -Σ_t p(t|y) log p(t|y)      [unigram token entropy]
    H_bi(y) = -Σ_{t1,t2} p(t2|t1) log p(t2|t1)  [bigram transition entropy]

Where:
    - t represents tokens (words or subwords)
    - p(t|y) is the probability of token t in transcription y

The linguistic entropy captures:
- Vocabulary diversity (unique words used)
- Token distribution (uniform vs. skewed usage)
- Syntactic patterns via n-gram analysis
"""

import numpy as np
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass
from collections import Counter
import re

from ..utils import compute_entropy, compute_conditional_entropy, AudioSample


@dataclass
class LinguisticEntropyConfig:
    """Configuration for linguistic entropy estimation."""
    # Tokenization
    tokenization: str = "word"  # "word", "subword", or "character"
    lowercase: bool = True
    remove_punctuation: bool = True
    
    # N-gram settings
    use_bigrams: bool = True
    bigram_weight: float = 0.5  # Weight for bigram entropy in final score
    
    # Vocabulary
    min_token_freq: int = 1  # Minimum frequency to include token
    
    # Rare word bonus
    rare_word_threshold: float = 0.001  # Tokens below this frequency get bonus
    rare_word_bonus: float = 0.1  # Bonus entropy for rare words


class LinguisticEntropyEstimator:
    """
    Estimates linguistic entropy from text transcriptions.
    
    The linguistic entropy H_linguistic(x) quantifies the vocabulary
    richness and syntactic complexity of a transcription. High linguistic
    entropy indicates:
    - Diverse vocabulary usage
    - Complex sentence structures
    - Rare or domain-specific terminology
    
    Implementation:
        1. Tokenize the transcription
        2. Compute unigram token distribution
        3. Optionally compute bigram distribution
        4. Calculate entropy over distributions
        5. Apply rare word bonuses
    
    Example:
        >>> estimator = LinguisticEntropyEstimator()
        >>> sample = AudioSample(audio=audio_array, transcription="The quick brown fox")
        >>> entropy = estimator.compute(sample)
        >>> print(f"Linguistic entropy: {entropy:.3f} bits")
    """
    
    def __init__(self, config: Optional[LinguisticEntropyConfig] = None):
        """
        Initialize the linguistic entropy estimator.
        
        Args:
            config: Configuration parameters. Uses defaults if None.
        """
        self.config = config or LinguisticEntropyConfig()
        
        # Reference corpus statistics for rare word detection
        # These would ideally be loaded from a large corpus
        self._corpus_freqs: Optional[Dict[str, float]] = None
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text according to configuration.
        
        Args:
            text: Input text string
        
        Returns:
            List of tokens
        """
        if self.config.lowercase:
            text = text.lower()
        
        if self.config.remove_punctuation:
            text = re.sub(r'[^\w\s]', '', text)
        
        if self.config.tokenization == "word":
            tokens = text.split()
        elif self.config.tokenization == "character":
            tokens = list(text.replace(" ", "_"))
        elif self.config.tokenization == "subword":
            # Simple subword: split into character n-grams
            tokens = []
            for word in text.split():
                if len(word) <= 3:
                    tokens.append(word)
                else:
                    # Split into trigrams
                    for i in range(0, len(word), 3):
                        tokens.append(word[i:i+3])
        else:
            tokens = text.split()
        
        return [t for t in tokens if len(t) > 0]
    
    def compute_unigram_entropy(self, tokens: List[str]) -> float:
        """
        Compute unigram entropy over tokens.
        
        Formula: H(T) = -Σ_t p(t) log₂ p(t)
        
        Args:
            tokens: List of tokens
        
        Returns:
            Unigram entropy in bits
        """
        if len(tokens) == 0:
            return 0.0
        
        counts = Counter(tokens)
        total = len(tokens)
        
        probs = np.array([count / total for count in counts.values()])
        
        return compute_entropy(probs)
    
    def compute_bigram_entropy(self, tokens: List[str]) -> float:
        """
        Compute bigram conditional entropy.
        
        Formula: H(T2|T1) = -Σ_{t1} p(t1) Σ_{t2} p(t2|t1) log₂ p(t2|t1)
        
        Args:
            tokens: List of tokens
        
        Returns:
            Bigram conditional entropy in bits
        """
        if len(tokens) < 2:
            return 0.0
        
        # Count unigrams and bigrams
        unigram_counts = Counter(tokens[:-1])  # All tokens except last
        bigram_counts = Counter(zip(tokens[:-1], tokens[1:]))
        
        total_bigrams = len(tokens) - 1
        
        # Compute conditional entropy
        entropy = 0.0
        smoothing = 1e-10
        
        for (t1, t2), count in bigram_counts.items():
            p_bigram = count / total_bigrams
            p_t1 = unigram_counts[t1] / total_bigrams
            p_t2_given_t1 = count / unigram_counts[t1]
            
            if p_t2_given_t1 > smoothing:
                entropy -= p_bigram * np.log2(p_t2_given_t1 + smoothing)
        
        return float(entropy)
    
    def compute_vocabulary_diversity(self, tokens: List[str]) -> float:
        """
        Compute type-token ratio as a diversity measure.
        
        Formula: TTR = |unique_tokens| / |total_tokens|
        
        Higher TTR indicates more diverse vocabulary.
        
        Args:
            tokens: List of tokens
        
        Returns:
            Type-token ratio in [0, 1]
        """
        if len(tokens) == 0:
            return 0.0
        
        unique = len(set(tokens))
        total = len(tokens)
        
        return unique / total
    
    def compute_rare_word_bonus(self, tokens: List[str]) -> float:
        """
        Compute bonus entropy for rare/unusual words.
        
        Words that are rare in a reference corpus contribute
        additional information value.
        
        Args:
            tokens: List of tokens
        
        Returns:
            Bonus entropy value
        """
        if self._corpus_freqs is None or len(tokens) == 0:
            return 0.0
        
        bonus = 0.0
        for token in set(tokens):
            freq = self._corpus_freqs.get(token, 0.0)
            if freq < self.config.rare_word_threshold:
                bonus += self.config.rare_word_bonus
        
        return bonus
    
    def compute(self, sample: AudioSample) -> float:
        """
        Compute linguistic entropy for a single audio sample.
        
        Mathematical Formula:
            H_linguistic(x) = H_unigram(x) + w_bigram * H_bigram(x) + bonus
        
        Args:
            sample: AudioSample containing transcription
        
        Returns:
            Linguistic entropy value in bits
        """
        tokens = self.tokenize(sample.transcription)
        
        if len(tokens) == 0:
            return 0.0
        
        # Unigram entropy
        unigram_h = self.compute_unigram_entropy(tokens)
        
        # Bigram entropy (optional)
        bigram_h = 0.0
        if self.config.use_bigrams and len(tokens) >= 2:
            bigram_h = self.compute_bigram_entropy(tokens)
        
        # Combine entropies
        total_entropy = unigram_h + self.config.bigram_weight * bigram_h
        
        # Add rare word bonus
        total_entropy += self.compute_rare_word_bonus(tokens)
        
        return float(total_entropy)
    
    def set_corpus_frequencies(self, corpus_freqs: Dict[str, float]):
        """
        Set reference corpus frequencies for rare word detection.
        
        Args:
            corpus_freqs: Dictionary mapping tokens to their corpus frequency
        """
        self._corpus_freqs = corpus_freqs
    
    def compute_batch(self, samples: list) -> np.ndarray:
        """
        Compute linguistic entropy for multiple samples.
        
        Args:
            samples: List of AudioSample objects
        
        Returns:
            Array of entropy values
        """
        return np.array([self.compute(s) for s in samples])
