"""
Multilingual Speech-to-Text Model Training Script
==================================================

This script provides a complete training pipeline for fine-tuning speech-to-text
models on multilingual audio datasets. It supports:
- Multiple language groups with configurable language tags
- Combined training from multiple dataset sources
- Distributed training using Hugging Face Accelerate
- Mixed precision training (FP16)
- Checkpoint saving and TensorBoard logging

Usage:
    python training.py

Configuration:
    Modify the hyperparameters and dataset configurations below before running.
"""

import os
import math
import logging
from typing import Any
from dataclasses import dataclass

import torch
import torchaudio
import soundfile as sf
from datasets import load_dataset, concatenate_datasets, Audio
from transformers import (
    AutoProcessor,
    AutoModelForSpeechSeq2Seq,
)
from transformers.optimization import get_scheduler
from torch.optim import AdamW
from accelerate import Accelerator
from torch.utils.data import DataLoader

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# LANGUAGE CONFIGURATION
# =============================================================================
# Define language groups for multilingual training.
# Each group maps a base language code to related languages/dialects.
# The base language code is used as the language tag during training.
#
# Structure:
#   "group_name": {
#       "base_lang": "language_code",     # ISO 639-1 code for the language tag
#       "languages": ["lang1", "lang2"],  # List of languages/dialects in this group
#   }
# =============================================================================

LANGUAGE_GROUPS = {
    "lang_group_1": {
        "base_lang": "xx",  # Replace with your base language code (e.g., "hi", "en")
        "languages": [
            "Language1",
            "Language2",
            "Dialect1",
            "Dialect2",
            # Add more languages/dialects as needed
        ],
    },
    "lang_group_2": {
        "base_lang": "yy",  # Replace with your base language code
        "languages": [
            "Language3",
            "Language4",
        ],
    },
    # Add more language groups as needed for your multilingual training
}

# =============================================================================
# DATASET CONFIGURATION
# =============================================================================
# Configure your dataset sources here.
# 
# PRIMARY_DATASET_PATH: HuggingFace dataset path for your main multilingual dataset
# SECONDARY_DATASETS: List of additional datasets to include in training
#                     Format: (dataset_name, split_name)
# =============================================================================

# Primary multilingual dataset (e.g., from HuggingFace Hub)
PRIMARY_DATASET_PATH = "your-username/your-multilingual-dataset"

# Secondary datasets for additional training data
SECONDARY_DATASET_PATH = "your-username/your-secondary-dataset"
SECONDARY_DATASETS = [
    ("dataset_subset_1", "train"),
    ("dataset_subset_2", "train"),
    ("dataset_subset_3", "test"),
    # Add more (name, split) tuples as needed
]

# =============================================================================
# TRAINING HYPERPARAMETERS
# =============================================================================
# Adjust these based on your hardware and dataset size.
# =============================================================================

# Batch size per device (effective batch = BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * num_gpus)
BATCH_SIZE = 4

# Learning rate for AdamW optimizer
LEARNING_RATE = 1e-5

# Number of complete passes through the training data
NUM_TRAIN_EPOCHS = 1

# Number of warmup steps for learning rate scheduler
WARMUP_STEPS = 500

# Accumulate gradients over this many steps before updating weights
# Useful for simulating larger batch sizes on limited GPU memory
GRADIENT_ACCUMULATION_STEPS = 8

# Directory to save model checkpoints and final model
OUTPUT_DIR = "./finetuned-speech-model-multilingual"

# Base path for audio files (if audio paths in dataset are relative)
AUDIO_BASE_PATH = ""

# Pretrained model identifier (from HuggingFace Hub)
PRETRAINED_MODEL_NAME = "your-username/your-pretrained-model"

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def add_language_tag(example: dict, lang_code: str) -> dict:
    """
    Add a language tag to the transcript for multilingual training.
    
    This allows the model to learn language-specific patterns and predict
    the appropriate language during inference.
    
    Args:
        example: Dictionary containing the audio sample with 'transcript' key
        lang_code: ISO 639-1 language code (e.g., 'en', 'hi', 'es')
    
    Returns:
        Modified example with language tag prepended to transcript
    
    Example:
        Input transcript: "Hello world"
        Output transcript: "<|en|><|transcribe|>Hello world"
    """
    example["transcript"] = f"<|{lang_code}|><|transcribe|>{example['transcript']}"
    return example


def safe_rename_columns(dataset):
    """
    Safely rename the 'text' column to 'transcript' if it exists.
    
    Different datasets may use different column names for transcriptions.
    This function standardizes the column name to 'transcript'.
    
    Args:
        dataset: HuggingFace dataset object
    
    Returns:
        Dataset with standardized column name
    """
    if "text" in dataset.column_names:
        return dataset.rename_columns({"text": "transcript"})
    return dataset


def load_primary_datasets():
    """
    Load and prepare the primary multilingual dataset.
    
    This function:
    1. Iterates through all configured language groups
    2. Loads each language subset from the primary dataset
    3. Adds appropriate language tags to each sample
    4. Combines all subsets into a single shuffled dataset
    
    Returns:
        Concatenated and shuffled HuggingFace dataset
    
    Raises:
        Logs warnings for any languages that fail to load
    """
    all_datasets = []
    
    for group_key, group_info in LANGUAGE_GROUPS.items():
        base_lang = group_info["base_lang"]
        
        for language in group_info["languages"]:
            try:
                # Load language-specific subset from the dataset
                dataset = load_dataset(
                    PRIMARY_DATASET_PATH,
                    language,
                    split="train"
                )
                
                # Standardize column names
                dataset = safe_rename_columns(dataset)
                
                # Add language tag for multilingual training
                dataset = dataset.map(lambda x: add_language_tag(x, base_lang))
                
                all_datasets.append(dataset)
                logger.info(f"Loaded primary dataset: {language} with tag <|{base_lang}|>")
                
            except Exception as e:
                logger.warning(f"Could not load primary dataset for {language}: {e}")
    
    # Combine all datasets and shuffle for better training
    return concatenate_datasets(all_datasets).shuffle(seed=42)


def load_secondary_datasets():
    """
    Load and prepare secondary datasets for additional training data.
    
    This function loads datasets from the secondary source (e.g., English-only
    benchmark datasets) and adds appropriate language tags.
    
    Returns:
        Concatenated and shuffled HuggingFace dataset
    
    Raises:
        Logs warnings for any datasets that fail to load
    """
    all_datasets = []
    
    for dataset_name, split in SECONDARY_DATASETS:
        try:
            # Load specific dataset subset
            dataset = load_dataset(
                SECONDARY_DATASET_PATH,
                name=dataset_name,
                split=split
            )
            
            # Standardize column names
            dataset = safe_rename_columns(dataset)
            
            # Add language tag (modify "en" to your secondary dataset's language)
            dataset = dataset.map(lambda x: add_language_tag(x, "en"))
            
            all_datasets.append(dataset)
            logger.info(f"Loaded secondary dataset: {dataset_name} ({split}) with tag <|en|>")
            
        except Exception as e:
            logger.warning(f"Could not load secondary dataset {dataset_name}: {e}")
    
    # Combine all datasets and shuffle
    return concatenate_datasets(all_datasets).shuffle(seed=42)


def load_and_prepare_datasets():
    """
    Main function to load, prepare, and combine all training datasets.
    
    This function:
    1. Loads primary multilingual datasets
    2. Loads secondary datasets (if configured)
    3. Casts audio to consistent sampling rate (16kHz)
    4. Combines all datasets into a single training set
    
    Returns:
        Combined HuggingFace dataset ready for preprocessing
    """
    logger.info("Loading primary multilingual datasets...")
    primary_dataset = load_primary_datasets()
    
    # Cast audio column to ensure consistent sampling rate (16kHz)
    primary_dataset = primary_dataset.cast_column(
        "audio",
        Audio(sampling_rate=16000, decode=True)
    )

    logger.info("Loading secondary datasets...")
    secondary_dataset = load_secondary_datasets()
    
    # Cast audio column to ensure consistent sampling rate (16kHz)
    secondary_dataset = secondary_dataset.cast_column(
        "audio",
        Audio(sampling_rate=16000, decode=True)
    )
    
    # Combine all datasets into a single training set
    combined_dataset = concatenate_datasets([primary_dataset, secondary_dataset]).shuffle(
        seed=42
    )
    
    logger.info(f"Total combined dataset size: {len(combined_dataset)}")
    return combined_dataset


def prepare_dataset(batch: dict, processor) -> dict:
    """
    Prepare a batch of audio samples for model training.
    
    This function processes raw audio waveforms into model-ready features:
    1. Extracts audio arrays from the batch
    2. Computes acoustic features (e.g., mel spectrograms)
    3. Tokenizes transcripts into label IDs
    
    Note: Padding is NOT applied here - it's handled by the data collator
    to ensure efficient batching with variable-length sequences.
    
    Args:
        batch: Dictionary containing 'audio' and 'transcript' keys
        processor: Model processor with feature_extractor and tokenizer
    
    Returns:
        Dictionary with 'input_features' and 'labels' ready for training
    """
    # Extract raw audio arrays from the batch
    audio_arrays = [item["array"] for item in batch["audio"]]
    
    # Compute acoustic features (mel spectrograms) without padding
    inputs = processor.feature_extractor(
        audio_arrays,
        sampling_rate=16000
    )
    batch["input_features"] = inputs.input_features

    # Tokenize transcripts into label IDs
    batch["labels"] = processor.tokenizer(
        text=batch["transcript"],
        truncation=True,
        max_length=448,  # Maximum sequence length for labels
    ).input_ids
    
    return batch


# =============================================================================
# DATA COLLATOR
# =============================================================================


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """
    Custom data collator for speech-to-text sequence-to-sequence models.
    
    This collator handles the unique padding requirements of speech models:
    - Input features (audio) and labels (text) have different lengths
    - They require different padding strategies
    - Label padding tokens must be replaced with -100 for loss masking
    
    Attributes:
        feature_extractor: Processor's feature extractor for padding audio features
        tokenizer: Processor's tokenizer for padding text labels
        padding: Whether to apply dynamic padding (default: True)
    """
    feature_extractor: Any
    tokenizer: Any
    padding: bool = True

    def __call__(self, features: list) -> dict:
        """
        Collate a list of samples into a padded batch.
        
        Args:
            features: List of dictionaries with 'input_features' and 'labels'
        
        Returns:
            Batched and padded dictionary ready for model forward pass
        """
        # Separate input features and labels for different padding strategies
        input_features = [
            {"input_features": feature["input_features"]}
            for feature in features
        ]
        label_features = [
            {"input_ids": feature["labels"]}
            for feature in features
        ]

        # Pad audio features
        batch = self.feature_extractor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )

        # Pad label sequences
        labels_batch = self.tokenizer.pad(
            label_features,
            padding=self.padding,
            return_tensors="pt",
        )

        # Replace padding tokens with -100 to ignore them in loss computation
        # This ensures the model only learns from actual transcript tokens
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1),
            -100
        )

        # For multilingual training, we want the model to learn to predict
        # the language tokens, so we DON'T mask the BOS token
        batch["labels"] = labels

        return batch


# =============================================================================
# MAIN TRAINING LOOP
# =============================================================================


def main():
    """
    Main training function that orchestrates the entire training pipeline.
    
    This function:
    1. Initializes the Accelerator for distributed training
    2. Loads the pretrained model and processor
    3. Prepares datasets and dataloaders
    4. Sets up optimizer and learning rate scheduler
    5. Runs the training loop with gradient accumulation
    6. Saves checkpoints and final model
    """
    
    # -------------------------------------------------------------------------
    # STEP 1: Initialize Accelerator for distributed training
    # -------------------------------------------------------------------------
    accelerator = Accelerator(
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        mixed_precision="fp16",  # Use FP16 for faster training and lower memory
        log_with="tensorboard",   # Enable TensorBoard logging
        project_dir=OUTPUT_DIR
    )
    
    # Configure logging (only on main process to avoid duplicate logs)
    if accelerator.is_main_process:
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
    logger.info(accelerator.state)

    # -------------------------------------------------------------------------
    # STEP 2: Load pretrained model and processor
    # -------------------------------------------------------------------------
    logger.info(f"Loading pretrained model: {PRETRAINED_MODEL_NAME}")
    
    processor = AutoProcessor.from_pretrained(PRETRAINED_MODEL_NAME)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(PRETRAINED_MODEL_NAME)
    
    # Log model configuration
    if hasattr(model.config, "num_mel_bins"):
        logger.info(f"Model expects {model.config.num_mel_bins} mel bins")
    
    # -------------------------------------------------------------------------
    # STEP 3: Configure model for multilingual training
    # -------------------------------------------------------------------------
    # Disable forced decoder IDs to let the model learn language prediction
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.config.use_cache = False  # Disable KV cache during training
    
    # Update generation config for inference
    if hasattr(model, "generation_config"):
        model.generation_config.forced_decoder_ids = None
        model.generation_config.suppress_tokens = []
    
    # -------------------------------------------------------------------------
    # STEP 4: Load and prepare datasets
    # -------------------------------------------------------------------------
    # Use main_process_first to avoid race conditions in distributed setting
    with accelerator.main_process_first():
        combined_dataset = load_and_prepare_datasets()

    # Process dataset: extract audio features and tokenize transcripts
    logger.info("Processing dataset (audio feature extraction and tokenization)...")
    combined_dataset = combined_dataset.map(
        lambda batch: prepare_dataset(batch, processor),
        remove_columns=combined_dataset.column_names,
        num_proc=4,          # Number of parallel processes for preprocessing
        batched=True,        # Process in batches for efficiency
        batch_size=32,       # Batch size for preprocessing
    )

    # Set dataset format for PyTorch
    combined_dataset.set_format(
        type="torch",
        columns=["input_features", "labels"]
    )

    # -------------------------------------------------------------------------
    # STEP 5: Create DataLoader with custom collator
    # -------------------------------------------------------------------------
    feature_extractor = getattr(processor, "feature_extractor")
    tokenizer = getattr(processor, "tokenizer")
    
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        feature_extractor=feature_extractor,
        tokenizer=tokenizer
    )
    
    train_dataloader = DataLoader(
        combined_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,              # Shuffle for better training
        collate_fn=data_collator,
        drop_last=True,            # Drop incomplete batches
        pin_memory=True,           # Faster data transfer to GPU
        num_workers=4,             # Parallel data loading
    )

    # -------------------------------------------------------------------------
    # STEP 6: Set up optimizer with weight decay
    # -------------------------------------------------------------------------
    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.01,  # L2 regularization
        eps=1e-6            # Numerical stability
    )

    # -------------------------------------------------------------------------
    # STEP 7: Calculate training steps and create LR scheduler
    # -------------------------------------------------------------------------
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / GRADIENT_ACCUMULATION_STEPS
    )
    num_training_steps = NUM_TRAIN_EPOCHS * num_update_steps_per_epoch
    
    lr_scheduler = get_scheduler(
        "linear",                          # Linear decay schedule
        optimizer=optimizer,
        num_warmup_steps=WARMUP_STEPS,     # Gradual warmup
        num_training_steps=num_training_steps,
    )

    # -------------------------------------------------------------------------
    # STEP 8: Prepare for distributed training
    # -------------------------------------------------------------------------
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    # Initialize experiment tracking
    if accelerator.is_main_process:
        accelerator.init_trackers(
            "multilingual-speech-training",
            config=accelerator.unwrap_model(model).config.to_dict()
        )

    # -------------------------------------------------------------------------
    # STEP 9: Training loop
    # -------------------------------------------------------------------------
    model.train()
    global_step = 0
    total_loss = 0.0
    
    # Calculate checkpoint save intervals (save every 20% of training)
    total_steps = NUM_TRAIN_EPOCHS * len(train_dataloader)
    checkpoint_steps = max(1000, total_steps // 5)
    
    logger.info(f"Starting training for {NUM_TRAIN_EPOCHS} epochs")
    logger.info(f"Total training steps: {total_steps}")
    logger.info(f"Checkpoint interval: {checkpoint_steps} steps")
    
    for epoch in range(NUM_TRAIN_EPOCHS):
        epoch_loss = 0.0
        
        for step, batch in enumerate(train_dataloader):
            # Gradient accumulation context
            with accelerator.accumulate(model):
                # Forward pass
                outputs = model(**batch)
                loss = outputs.loss
                
                # Backward pass
                accelerator.backward(loss)
                
                # Gradient clipping (prevents exploding gradients)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # Optimizer step
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Accumulate loss for logging
            total_loss += loss.detach().float()
            epoch_loss += loss.detach().float()

            # Update global step counter
            if accelerator.sync_gradients:
                global_step += 1

            # Log progress every 100 steps
            if global_step % 100 == 0 and accelerator.is_main_process:
                avg_loss = total_loss / global_step
                current_lr = lr_scheduler.get_last_lr()[0]
                logger.info(
                    f"Epoch {epoch}, Step {global_step} - "
                    f"Loss: {avg_loss:.4f}, LR: {current_lr:.2e}"
                )
                accelerator.log(
                    {"train_loss": avg_loss, "learning_rate": current_lr},
                    step=global_step
                )

            # Save periodic checkpoints
            if global_step > 0 and global_step % checkpoint_steps == 0:
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    checkpoint_dir = f"{OUTPUT_DIR}/checkpoint-{global_step}"
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    
                    # Save model
                    unwrapped_model = accelerator.unwrap_model(model)
                    unwrapped_model.save_pretrained(checkpoint_dir)
                    
                    # Save processor
                    processor.save_pretrained(checkpoint_dir)
                    
                    logger.info(f"Checkpoint saved at {checkpoint_dir}")

        # Log epoch completion
        if accelerator.is_main_process:
            avg_epoch_loss = epoch_loss / len(train_dataloader)
            logger.info(
                f"Epoch {epoch} completed - Average loss: {avg_epoch_loss:.4f}"
            )
            
            # Save checkpoint at end of each epoch
            accelerator.wait_for_everyone()
            epoch_checkpoint_dir = f"{OUTPUT_DIR}/checkpoint-epoch-{epoch}"
            os.makedirs(epoch_checkpoint_dir, exist_ok=True)
            
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(epoch_checkpoint_dir)
            processor.save_pretrained(epoch_checkpoint_dir)
            
            logger.info(f"Epoch checkpoint saved at {epoch_checkpoint_dir}")

    # -------------------------------------------------------------------------
    # STEP 10: Save final model
    # -------------------------------------------------------------------------
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    
    if accelerator.is_main_process:
        unwrapped_model.save_pretrained(OUTPUT_DIR)
        processor.save_pretrained(OUTPUT_DIR)
        logger.info(f"Final model saved at {OUTPUT_DIR}")
    
    # Clean up experiment tracking
    accelerator.end_training()
    logger.info("Training completed successfully!")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()
