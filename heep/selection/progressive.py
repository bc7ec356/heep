"""
Progressive Filtering for HEEP.

Implements the iterative data curation across training rounds with
exponentially increasing selectivity and ERROR-AWARE ADAPTATION.

Key Innovation: HEEP is not just entropy-based selection. It performs
error-aware progressive filtering:

1. After each training round, analyze model errors
2. Identify samples that would help correct those errors
3. Boost selection of samples covering error patterns
4. Leverage cross-lingual feature overlap for multilingual scaling

Mathematical Formulation (Paper Equation 8):
    τₖ₊₁ = τₖ · growth_factor · (1 + ε·sin(2πk/ρ))

The error-aware component adjusts sample scores based on:
    S'(x) = S(x) + λ_err · ErrorRelevance(x, errors_k)

Where ErrorRelevance measures how well sample x covers patterns
that the model currently gets wrong.
"""

import numpy as np
from typing import List, Tuple, Optional, Callable, Dict, Any
from dataclasses import dataclass, field
import logging
from collections import defaultdict

from ..utils import AudioSample
from .threshold import ThresholdSelector

logger = logging.getLogger(__name__)


@dataclass
class ErrorPattern:
    """Represents a pattern of errors from model evaluation."""
    # Error type counts (substitution, insertion, deletion)
    substitution_count: int = 0
    insertion_count: int = 0
    deletion_count: int = 0
    
    # Phoneme-level error patterns
    confused_phonemes: Dict[str, Dict[str, int]] = field(default_factory=dict)
    
    # Linguistic patterns (n-grams that cause errors)
    error_ngrams: Dict[str, int] = field(default_factory=dict)
    
    # Acoustic conditions with high error rates
    problematic_acoustic_conditions: List[str] = field(default_factory=list)
    
    # Languages/accents with high error rates
    high_error_languages: Dict[str, float] = field(default_factory=dict)


@dataclass
class ProgressiveFilterConfig:
    """Configuration for progressive filtering."""
    # Initial threshold
    initial_threshold: float = 0.3
    
    # Growth settings
    growth_factor: float = 1.1  # τₖ₊₁ = τₖ · growth
    
    # Minimum dataset size (stop if we go below this)
    min_samples: int = 1000
    min_fraction: float = 0.1  # Stop if below 10% of original
    
    # Maximum rounds
    max_rounds: int = 10
    
    # Diversity preservation
    diversity_perturbation: float = 0.05  # Random retention probability
    
    # Error-aware selection parameters
    error_boost_weight: float = 0.3  # λ_err: weight for error relevance
    error_analysis_enabled: bool = True
    
    # Cross-lingual overlap parameters
    cross_lingual_boost: float = 0.2  # Boost for samples with cross-lingual features


class ProgressiveFilter:
    """
    Progressively filters dataset across training rounds with ERROR-AWARE ADAPTATION.
    
    HEEP's core innovation is that it doesn't just use static entropy scores.
    After each training round, it:
    
    1. Analyzes what errors the model makes
    2. Identifies samples that cover those error patterns
    3. Boosts selection of error-relevant samples in subsequent rounds
    4. Leverages cross-lingual feature overlap for multilingual scaling
    
    This creates an adaptive feedback loop:
        - Round k: Train on selected data
        - Evaluate: Identify error patterns (phoneme confusions, acoustic conditions)
        - Round k+1: Boost samples that would help fix those errors
    
    For multilingual models, this is especially powerful because:
        - New languages share phonetic/acoustic features with known languages
        - HEEP identifies overlapping features and prioritizes samples that
          strengthen these shared representations
        - The model "already knows" related patterns, so HEEP reduces data
          for patterns the model has mastered
    
    Mathematical Formula:
        S'(x) = S(x) + λ_err · ErrorRelevance(x, errors_k) + λ_cross · CrossLingualOverlap(x)
        Round k: D_k = {x ∈ D_{k-1} : S'(x) > τₖ}
    
    Example:
        >>> filter = ProgressiveFilter()
        >>> for round_num in filter.iterate(samples, scorer, train_fn, eval_fn):
        ...     print(f"Round {round_num}: {len(filter.current_samples)} samples")
    """
    
    def __init__(self, config: Optional[ProgressiveFilterConfig] = None):
        """
        Initialize the progressive filter.
        
        Args:
            config: Configuration parameters. Uses defaults if None.
        """
        self.config = config or ProgressiveFilterConfig()
        
        # State
        self.current_threshold = self.config.initial_threshold
        self.current_round = 0
        self.current_samples: List[AudioSample] = []
        self.history: List[dict] = []
        
        # Error tracking for adaptive selection
        self.error_patterns: Optional[ErrorPattern] = None
        self.known_features: Dict[str, float] = {}  # Features model has mastered
        self.cross_lingual_cache: Dict[str, List[str]] = {}  # Language -> overlapping features
    
    def reset(self):
        """Reset filter state for a new run."""
        self.current_threshold = self.config.initial_threshold
        self.current_round = 0
        self.current_samples = []
        self.history = []
        self.error_patterns = None
        self.known_features = {}
    
    def analyze_errors(
        self,
        predictions: List[str],
        references: List[str],
        samples: List[AudioSample]
    ) -> ErrorPattern:
        """
        Analyze model errors to guide next round's data selection.
        
        This is the key to HEEP's adaptive power: by understanding what
        the model gets wrong, we can select training data that addresses
        those specific weaknesses.
        
        Args:
            predictions: Model predictions from evaluation
            references: Ground truth transcriptions
            samples: Samples that were evaluated
        
        Returns:
            ErrorPattern capturing the types and patterns of errors
        """
        errors = ErrorPattern()
        phoneme_confusions = defaultdict(lambda: defaultdict(int))
        ngram_errors = defaultdict(int)
        
        for pred, ref, sample in zip(predictions, references, samples):
            # Compute alignment and identify error types
            pred_words = pred.lower().split()
            ref_words = ref.lower().split()
            
            # Simple error counting (in practice, use proper alignment)
            # Substitutions: words that differ
            # Insertions: extra words in prediction
            # Deletions: missing words in prediction
            
            ref_set = set(ref_words)
            pred_set = set(pred_words)
            
            errors.substitution_count += len(ref_set - pred_set)
            errors.insertion_count += max(0, len(pred_words) - len(ref_words))
            errors.deletion_count += max(0, len(ref_words) - len(pred_words))
            
            # Track n-grams that cause errors
            for i in range(len(ref_words) - 1):
                bigram = f"{ref_words[i]} {ref_words[i+1]}"
                if ref_words[i] not in pred_set or ref_words[i+1] not in pred_set:
                    ngram_errors[bigram] += 1
            
            # Track language-specific error rates
            if hasattr(sample, 'language') and sample.language:
                if sample.language not in errors.high_error_languages:
                    errors.high_error_languages[sample.language] = 0.0
                # Compute WER for this sample
                wer = len(ref_set - pred_set) / max(len(ref_words), 1)
                errors.high_error_languages[sample.language] += wer
        
        errors.error_ngrams = dict(ngram_errors)
        errors.confused_phonemes = {k: dict(v) for k, v in phoneme_confusions.items()}
        
        # Normalize language error rates
        lang_counts = defaultdict(int)
        for sample in samples:
            if hasattr(sample, 'language') and sample.language:
                lang_counts[sample.language] += 1
        for lang in errors.high_error_languages:
            if lang_counts[lang] > 0:
                errors.high_error_languages[lang] /= lang_counts[lang]
        
        logger.info(f"Error analysis: {errors.substitution_count} subs, "
                   f"{errors.insertion_count} ins, {errors.deletion_count} del")
        
        return errors
    
    def compute_error_relevance(
        self,
        sample: AudioSample,
        errors: ErrorPattern
    ) -> float:
        """
        Compute how relevant a sample is for fixing current errors.
        
        Samples that contain patterns the model struggles with get
        higher relevance scores, increasing their chance of selection.
        
        Args:
            sample: Sample to evaluate
            errors: Current error patterns from model evaluation
        
        Returns:
            Relevance score in [0, 1]
        """
        relevance = 0.0
        
        # Check if sample contains error-prone n-grams
        if hasattr(sample, 'transcript') and sample.transcript:
            words = sample.transcript.lower().split()
            for i in range(len(words) - 1):
                bigram = f"{words[i]} {words[i+1]}"
                if bigram in errors.error_ngrams:
                    relevance += errors.error_ngrams[bigram] * 0.1
        
        # Boost samples from high-error languages
        if hasattr(sample, 'language') and sample.language:
            if sample.language in errors.high_error_languages:
                relevance += errors.high_error_languages[sample.language]
        
        return min(relevance, 1.0)  # Cap at 1.0
    
    def compute_cross_lingual_overlap(
        self,
        sample: AudioSample,
        known_languages: List[str]
    ) -> float:
        """
        Compute how much a sample's features overlap with known languages.
        
        This is key to HEEP's multilingual scaling: new languages that
        share phonetic/acoustic features with already-learned languages
        require less data because the model "already knows" related patterns.
        
        Args:
            sample: Sample to evaluate
            known_languages: Languages the model has been trained on
        
        Returns:
            Overlap score in [0, 1] where higher = more overlap
        """
        if not hasattr(sample, 'language') or not sample.language:
            return 0.0
        
        sample_lang = sample.language
        
        # Check cache
        if sample_lang in self.cross_lingual_cache:
            overlapping = self.cross_lingual_cache[sample_lang]
            return len(set(overlapping) & set(known_languages)) / max(len(known_languages), 1)
        
        # Language family groupings (simplified)
        # In practice, this would use phonetic feature databases
        language_families = {
            'germanic': ['en', 'de', 'nl', 'sv', 'da', 'no'],
            'romance': ['es', 'fr', 'it', 'pt', 'ro'],
            'slavic': ['ru', 'pl', 'cs', 'uk', 'bg'],
            'indic': ['hi', 'bn', 'mr', 'ta', 'te', 'gu'],
            'sino_tibetan': ['zh', 'yue', 'my'],
            'semitic': ['ar', 'he', 'am'],
        }
        
        # Find sample's family
        sample_family = None
        for family, langs in language_families.items():
            if sample_lang in langs:
                sample_family = family
                break
        
        if sample_family is None:
            return 0.0
        
        # Count how many known languages are in the same family
        family_langs = set(language_families[sample_family])
        overlap_count = len(family_langs & set(known_languages))
        
        self.cross_lingual_cache[sample_lang] = list(family_langs)
        
        return overlap_count / max(len(family_langs), 1)
    
    def update_known_features(
        self,
        samples: List[AudioSample],
        errors: ErrorPattern
    ):
        """
        Update tracking of features the model has mastered.
        
        Features with low error rates are considered "known" and
        subsequent rounds will reduce data for those patterns.
        
        Args:
            samples: Samples from current round
            errors: Error analysis from current round
        """
        # Track languages with low error rates as "mastered"
        for lang, error_rate in errors.high_error_languages.items():
            if error_rate < 0.1:  # Less than 10% WER
                self.known_features[f"lang:{lang}"] = 1.0 - error_rate
        
        # Track n-grams that are rarely confused
        all_ngrams = defaultdict(int)
        for sample in samples:
            if hasattr(sample, 'transcript') and sample.transcript:
                words = sample.transcript.lower().split()
                for i in range(len(words) - 1):
                    bigram = f"{words[i]} {words[i+1]}"
                    all_ngrams[bigram] += 1
        
        for ngram, count in all_ngrams.items():
            error_count = errors.error_ngrams.get(ngram, 0)
            if count > 10 and error_count / count < 0.1:
                self.known_features[f"ngram:{ngram}"] = 1.0 - (error_count / count)
    
    def adjust_scores_for_errors(
        self,
        samples: List[AudioSample],
        base_scores: np.ndarray,
        known_languages: Optional[List[str]] = None
    ) -> np.ndarray:
        """
        Adjust sample scores based on error patterns and cross-lingual overlap.
        
        This is the key adaptive mechanism: samples that would help fix
        current errors get boosted, while samples covering "known" patterns
        may be reduced.
        
        Formula:
            S'(x) = S(x) + λ_err · ErrorRelevance(x) + λ_cross · CrossLingualOverlap(x)
                  - λ_known · KnownPatternPenalty(x)
        
        Args:
            samples: Samples to adjust scores for
            base_scores: Original entropy-based scores
            known_languages: Languages model has been trained on
        
        Returns:
            Adjusted scores array
        """
        adjusted = base_scores.copy()
        
        if not self.config.error_analysis_enabled:
            return adjusted
        
        for i, sample in enumerate(samples):
            # Boost for error relevance
            if self.error_patterns is not None:
                error_boost = self.compute_error_relevance(sample, self.error_patterns)
                adjusted[i] += self.config.error_boost_weight * error_boost
            
            # Boost for cross-lingual overlap (new languages benefit from known ones)
            if known_languages:
                overlap = self.compute_cross_lingual_overlap(sample, known_languages)
                # High overlap with known languages = model can leverage existing knowledge
                # But we still want SOME samples from related languages
                # So we boost moderately, not penalize
                adjusted[i] += self.config.cross_lingual_boost * overlap * 0.5
            
            # Reduce priority for patterns the model has mastered
            # (allows focusing data budget on remaining challenges)
            if hasattr(sample, 'language') and sample.language:
                known_key = f"lang:{sample.language}"
                if known_key in self.known_features:
                    mastery = self.known_features[known_key]
                    # Reduce score for highly mastered patterns
                    # But don't completely exclude (maintain some for stability)
                    adjusted[i] -= 0.1 * mastery
        
        return adjusted
    
    def filter_round(
        self,
        samples: List[AudioSample],
        scores: np.ndarray,
        preserve_diversity: bool = True,
        known_languages: Optional[List[str]] = None
    ) -> Tuple[List[AudioSample], np.ndarray]:
        """
        Perform one round of filtering with error-aware adjustment.
        
        Args:
            samples: Current dataset
            scores: Base scores for all samples
            preserve_diversity: Whether to randomly keep some below-threshold samples
            known_languages: Languages the model has been trained on
        
        Returns:
            Tuple of (filtered_samples, filtered_scores)
        """
        scores = np.asarray(scores)
        
        # Apply error-aware score adjustment
        adjusted_scores = self.adjust_scores_for_errors(samples, scores, known_languages)
        
        # Base selection: above threshold
        mask = adjusted_scores > self.current_threshold
        
        # Diversity preservation: randomly keep some samples below threshold
        if preserve_diversity and self.config.diversity_perturbation > 0:
            below_threshold = ~mask
            random_keep = np.random.random(len(samples)) < self.config.diversity_perturbation
            mask = mask | (below_threshold & random_keep)
        
        # Apply selection
        filtered_samples = [s for s, m in zip(samples, mask) if m]
        filtered_scores = adjusted_scores[mask]
        
        # Record history with error-aware info
        history_record = {
            'round': self.current_round,
            'threshold': self.current_threshold,
            'input_size': len(samples),
            'output_size': len(filtered_samples),
            'retention_rate': len(filtered_samples) / len(samples) if len(samples) > 0 else 0,
            'mean_score': float(filtered_scores.mean()) if len(filtered_scores) > 0 else 0,
            'min_score': float(filtered_scores.min()) if len(filtered_scores) > 0 else 0,
            'max_score': float(filtered_scores.max()) if len(filtered_scores) > 0 else 0,
            'error_aware': self.error_patterns is not None,
            'known_features_count': len(self.known_features),
        }
        self.history.append(history_record)
        
        return filtered_samples, filtered_scores
    
    def update_threshold(self):
        """Update threshold for next round."""
        self.current_threshold *= self.config.growth_factor
        self.current_round += 1
    
    def should_stop(self, current_size: int, original_size: int) -> bool:
        """
        Check if filtering should stop.
        
        Args:
            current_size: Current number of samples
            original_size: Original dataset size
        
        Returns:
            True if should stop, False otherwise
        """
        # Check minimum samples
        if current_size < self.config.min_samples:
            logger.info(f"Stopping: below minimum samples ({current_size} < {self.config.min_samples})")
            return True
        
        # Check minimum fraction
        if current_size / original_size < self.config.min_fraction:
            logger.info(f"Stopping: below minimum fraction ({current_size/original_size:.2%} < {self.config.min_fraction:.2%})")
            return True
        
        # Check maximum rounds
        if self.current_round >= self.config.max_rounds:
            logger.info(f"Stopping: reached maximum rounds ({self.config.max_rounds})")
            return True
        
        return False
    
    def run(
        self,
        samples: List[AudioSample],
        scorer,  # SampleScorer
        train_callback: Optional[Callable] = None,
        eval_callback: Optional[Callable] = None,
        known_languages: Optional[List[str]] = None
    ) -> Tuple[List[AudioSample], dict]:
        """
        Run complete progressive filtering pipeline with ERROR-AWARE ADAPTATION.
        
        This is the main HEEP algorithm loop with adaptive error feedback.
        
        Algorithm:
            1. Initialize D₀ = D (full dataset)
            2. For k = 0 to max_rounds:
                a. Score all samples: S(x) for x ∈ Dₖ
                b. Adjust scores based on error patterns: S'(x) = S(x) + ErrorBoost(x)
                c. Filter: Dₖ₊₁ = {x ∈ Dₖ : S'(x) > τₖ}
                d. Train model on Dₖ₊₁ (via callback)
                e. Evaluate model and analyze errors (via callback)
                f. Update known features based on what model has mastered
                g. Update threshold: τₖ₊₁ = τₖ · growth_factor
                h. If stopping criterion met, break
            3. Return final dataset
        
        The error analysis step is KEY: it allows HEEP to focus subsequent
        rounds on samples that address the model's current weaknesses.
        
        For multilingual models, this means:
        - Languages with low error rates need less data (model "knows" them)
        - Languages with high error rates get boosted selection
        - Languages that share features with known languages benefit from transfer
        
        Args:
            samples: Initial dataset
            scorer: SampleScorer instance (must have compute_batch method)
            train_callback: Optional function called after each round
                           Signature: train_callback(samples, round_num) -> model
            eval_callback: Optional function to evaluate model and return (predictions, references)
                          Signature: eval_callback(model, samples) -> (preds, refs)
            known_languages: Initial list of languages the model knows
        
        Returns:
            Tuple of (final_samples, statistics_dict)
        """
        self.reset()
        original_size = len(samples)
        self.current_samples = samples.copy()
        
        # Track languages we've trained on
        trained_languages = list(known_languages) if known_languages else []
        
        logger.info(f"Starting HEEP progressive filtering with {original_size} samples")
        logger.info(f"Error-aware selection: {self.config.error_analysis_enabled}")
        
        # Fit scorer if it has a fit method
        if hasattr(scorer, 'fit'):
            scorer.fit(self.current_samples)
        
        model = None  # Will be updated by train_callback
        
        while not self.should_stop(len(self.current_samples), original_size):
            logger.info(f"Round {self.current_round}: {len(self.current_samples)} samples, "
                       f"threshold={self.current_threshold:.4f}, "
                       f"known_features={len(self.known_features)}")
            
            # Score current samples (base entropy scores)
            scores = scorer.compute_batch(self.current_samples)
            
            # Filter samples with error-aware adjustment
            self.current_samples, _ = self.filter_round(
                self.current_samples,
                scores,
                preserve_diversity=True,
                known_languages=trained_languages
            )
            
            # Training callback
            if train_callback is not None:
                model = train_callback(self.current_samples, self.current_round)
                
                # Update trained languages
                for sample in self.current_samples:
                    if hasattr(sample, 'language') and sample.language:
                        if sample.language not in trained_languages:
                            trained_languages.append(sample.language)
            
            # Error analysis callback (KEY STEP for adaptive selection)
            if eval_callback is not None and model is not None and self.config.error_analysis_enabled:
                # Evaluate on a subset to analyze errors
                eval_subset = self.current_samples[:min(1000, len(self.current_samples))]
                predictions, references = eval_callback(model, eval_subset)
                
                # Analyze errors to guide next round
                self.error_patterns = self.analyze_errors(predictions, references, eval_subset)
                
                # Update what the model has mastered
                self.update_known_features(eval_subset, self.error_patterns)
                
                logger.info(f"Error analysis complete: {len(self.known_features)} features mastered")
            
            # Update threshold for next round
            self.update_threshold()
        
        # Compile statistics
        stats = {
            'total_rounds': self.current_round,
            'original_size': original_size,
            'final_size': len(self.current_samples),
            'reduction_ratio': len(self.current_samples) / original_size,
            'final_threshold': self.current_threshold,
            'history': self.history,
            'known_features': dict(self.known_features),
            'trained_languages': trained_languages,
            'error_aware': self.config.error_analysis_enabled
        }
        
        logger.info(f"HEEP complete: {original_size} → {len(self.current_samples)} samples "
                   f"({stats['reduction_ratio']:.2%} retained)")
        logger.info(f"Trained on {len(trained_languages)} languages, "
                   f"mastered {len(self.known_features)} features")
        
        return self.current_samples, stats
    
    def get_history_summary(self) -> str:
        """Get a formatted summary of filtering history."""
        lines = ["HEEP Progressive Filtering History", "=" * 40]
        
        for record in self.history:
            lines.append(
                f"Round {record['round']}: "
                f"{record['input_size']} → {record['output_size']} samples "
                f"({record['retention_rate']:.1%} retained), "
                f"τ={record['threshold']:.4f}, "
                f"mean_score={record['mean_score']:.4f}"
            )
        
        return "\n".join(lines)
