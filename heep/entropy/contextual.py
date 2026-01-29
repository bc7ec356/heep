"""
Contextual Entropy Estimation for HEEP.

Computes H_contextual(x) - the entropy of contextual/domain features to measure
domain specificity and discourse diversity of audio samples.

Mathematical Formulation (Paper Equation 6):
    H_c(x) = -Σ_d p(d|x) log p(d|x) + H_disc

Where:
    - d represents domain categories (conversational, broadcast, meeting, etc.)
    - p(d|x) is domain classification probability from a text classifier
    - H_disc measures discourse marker diversity

The contextual entropy captures:
- Domain specificity (medical, legal, conversational, etc.)
- Discourse markers and speech patterns
- Semantic diversity via embedding space coverage
"""

import numpy as np
from typing import Optional, List, Dict
from dataclasses import dataclass
from collections import Counter

from ..utils import compute_entropy, AudioSample


# Domain categories for ASR
DOMAIN_CATEGORIES = [
    "conversational",      # Casual speech, meetings
    "broadcast",           # News, radio
    "telephony",           # Phone calls
    "lecture",             # Educational content
    "medical",             # Healthcare domain
    "legal",               # Legal/court proceedings
    "financial",           # Business, earnings calls
    "technical",           # Technical discussions
    "spontaneous",         # Unscripted speech
    "read_speech",         # Read aloud content
    "multilingual",        # Code-switching, multiple languages
    "noisy",               # High background noise
    "other"
]

# Discourse markers that indicate speech complexity
DISCOURSE_MARKERS = [
    # Fillers
    "um", "uh", "er", "ah", "like", "you know", "i mean",
    # Connectives
    "so", "well", "now", "then", "anyway", "however", "therefore",
    # Hedges
    "maybe", "perhaps", "kind of", "sort of", "probably",
    # Engagement
    "right", "okay", "yeah", "yes", "no", "actually", "basically"
]


@dataclass
class ContextualEntropyConfig:
    """Configuration for contextual entropy estimation."""
    # Domain classification
    domain_categories: List[str] = None
    use_domain_classifier: bool = True
    
    # Discourse markers
    discourse_markers: List[str] = None
    discourse_weight: float = 0.3
    
    # Semantic embeddings
    use_semantic_embeddings: bool = False
    embedding_dim: int = 768  # BERT-like dimension
    
    def __post_init__(self):
        if self.domain_categories is None:
            self.domain_categories = DOMAIN_CATEGORIES.copy()
        if self.discourse_markers is None:
            self.discourse_markers = DISCOURSE_MARKERS.copy()


class ContextualEntropyEstimator:
    """
    Estimates contextual entropy from audio samples.
    
    The contextual entropy H_contextual(x) quantifies the domain
    specificity and discourse complexity of a sample. High contextual
    entropy indicates:
    - Samples that span multiple domains
    - Rich discourse structure (markers, hedges, etc.)
    - Semantic diversity in content
    
    Implementation:
        1. Classify sample into domain categories
        2. Detect discourse markers
        3. Optionally compute semantic embedding entropy
        4. Combine into final contextual entropy
    
    Example:
        >>> estimator = ContextualEntropyEstimator()
        >>> sample = AudioSample(audio=audio_array, transcription="Well, you know...")
        >>> entropy = estimator.compute(sample)
        >>> print(f"Contextual entropy: {entropy:.3f} bits")
    """
    
    def __init__(self, config: Optional[ContextualEntropyConfig] = None):
        """
        Initialize the contextual entropy estimator.
        
        Args:
            config: Configuration parameters. Uses defaults if None.
        """
        self.config = config or ContextualEntropyConfig()
        self._domain_classifier = None
        self._embedding_model = None
    
    def classify_domain(self, text: str) -> np.ndarray:
        """
        Classify text into domain categories.
        
        Returns probability distribution over domains.
        
        Args:
            text: Input text string
        
        Returns:
            Probability distribution over domain categories
        """
        text_lower = text.lower()
        n_domains = len(self.config.domain_categories)
        
        # Simple keyword-based classification
        # In production, use a trained classifier
        domain_keywords = {
            "conversational": ["yeah", "okay", "like", "um", "uh", "gonna"],
            "broadcast": ["news", "report", "today", "announced", "according"],
            "telephony": ["hello", "speaking", "call", "phone"],
            "lecture": ["chapter", "example", "equation", "concept", "theory"],
            "medical": ["patient", "diagnosis", "treatment", "symptoms", "doctor"],
            "legal": ["court", "defendant", "plaintiff", "verdict", "judge"],
            "financial": ["market", "earnings", "revenue", "quarter", "growth"],
            "technical": ["system", "data", "algorithm", "process", "function"],
            "spontaneous": ["well", "so", "actually", "basically"],
            "read_speech": [],  # Hard to detect without acoustic features
            "multilingual": [],  # Would need language detection
            "noisy": [],  # Would need acoustic analysis
            "other": []
        }
        
        # Count keyword matches
        scores = np.zeros(n_domains)
        for i, domain in enumerate(self.config.domain_categories):
            keywords = domain_keywords.get(domain, [])
            for keyword in keywords:
                if keyword in text_lower:
                    scores[i] += 1
        
        # Convert to probabilities with smoothing
        scores = scores + 0.1  # Smoothing
        probs = scores / scores.sum()
        
        return probs
    
    def count_discourse_markers(self, text: str) -> Dict[str, int]:
        """
        Count discourse markers in text.
        
        Args:
            text: Input text string
        
        Returns:
            Dictionary of marker counts
        """
        text_lower = text.lower()
        counts = {}
        
        for marker in self.config.discourse_markers:
            # Count occurrences (word boundary aware for single words)
            if ' ' in marker:
                count = text_lower.count(marker)
            else:
                # Word boundary matching
                import re
                pattern = r'\b' + re.escape(marker) + r'\b'
                count = len(re.findall(pattern, text_lower))
            
            if count > 0:
                counts[marker] = count
        
        return counts
    
    def compute_discourse_entropy(self, text: str) -> float:
        """
        Compute entropy over discourse marker distribution.
        
        Formula: H_discourse(x) = -Σ_m p(m|x) log₂ p(m|x)
        
        Args:
            text: Input text string
        
        Returns:
            Discourse marker entropy in bits
        """
        counts = self.count_discourse_markers(text)
        
        if len(counts) == 0:
            return 0.0
        
        total = sum(counts.values())
        probs = np.array([c / total for c in counts.values()])
        
        return compute_entropy(probs)
    
    def compute_domain_entropy(self, text: str) -> float:
        """
        Compute entropy over domain classification.
        
        Formula: H_domain(x) = -Σ_d p(d|x) log₂ p(d|x)
        
        Args:
            text: Input text string
        
        Returns:
            Domain entropy in bits
        """
        domain_probs = self.classify_domain(text)
        return compute_entropy(domain_probs)
    
    def compute_semantic_entropy(self, text: str) -> float:
        """
        Compute entropy based on semantic embedding diversity.
        
        This measures how spread out the content is in embedding space.
        
        Args:
            text: Input text string
        
        Returns:
            Semantic entropy estimate
        """
        if not self.config.use_semantic_embeddings:
            return 0.0
        
        # Simple proxy: unique content words / total words
        words = text.lower().split()
        if len(words) == 0:
            return 0.0
        
        # Filter out common words
        common_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 
                       'been', 'being', 'have', 'has', 'had', 'do', 'does',
                       'did', 'will', 'would', 'could', 'should', 'may',
                       'might', 'must', 'shall', 'can', 'need', 'dare',
                       'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by',
                       'from', 'as', 'into', 'through', 'during', 'before',
                       'after', 'above', 'below', 'between', 'under', 'again',
                       'further', 'then', 'once', 'and', 'but', 'or', 'nor',
                       'so', 'yet', 'both', 'either', 'neither', 'not', 'only',
                       'own', 'same', 'than', 'too', 'very', 'just'}
        
        content_words = [w for w in words if w not in common_words and len(w) > 2]
        
        if len(content_words) == 0:
            return 0.0
        
        unique_ratio = len(set(content_words)) / len(content_words)
        
        # Convert ratio to entropy-like scale
        return -np.log2(1 - unique_ratio + 1e-10) if unique_ratio < 1 else 0.0
    
    def compute(self, sample: AudioSample) -> float:
        """
        Compute contextual entropy for a single audio sample.
        
        Mathematical Formula:
            H_contextual(x) = H_domain(x) + w_discourse * H_discourse(x) + H_semantic(x)
        
        Args:
            sample: AudioSample containing transcription
        
        Returns:
            Contextual entropy value in bits
        """
        text = sample.transcription
        
        if not text or len(text.strip()) == 0:
            return 0.0
        
        # Domain entropy
        domain_h = self.compute_domain_entropy(text)
        
        # Discourse entropy
        discourse_h = self.compute_discourse_entropy(text)
        
        # Semantic entropy (optional)
        semantic_h = 0.0
        if self.config.use_semantic_embeddings:
            semantic_h = self.compute_semantic_entropy(text)
        
        # Combine entropies
        total_entropy = (
            domain_h + 
            self.config.discourse_weight * discourse_h +
            semantic_h
        )
        
        return float(total_entropy)
    
    def compute_batch(self, samples: list) -> np.ndarray:
        """
        Compute contextual entropy for multiple samples.
        
        Args:
            samples: List of AudioSample objects
        
        Returns:
            Array of entropy values
        """
        return np.array([self.compute(s) for s in samples])
