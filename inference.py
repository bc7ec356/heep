"""
Speech-to-Text Inference Script
===============================

This script provides inference capabilities for speech-to-text models
using HuggingFace Transformers. It supports:

- Single file transcription
- Batch processing of audio directories
- Multiple output formats (txt, json, srt, vtt)
- GPU acceleration with FP16 precision

Usage:
    # Single file transcription
    python inference.py --audio path/to/audio.wav
    
    # Batch processing
    python inference.py --audio_dir path/to/audio_folder
    
    # With specific language
    python inference.py --audio file.wav --language hi

Requirements:
    - PyTorch 2.0+
    - transformers >= 4.36.0
    - torchaudio
"""

import os
import sys
import time
import json
import argparse
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any

import torch
import torchaudio
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

# Default model to use for inference (replace with your model path)
DEFAULT_MODEL_ID = "bc7ec356/heep-universal"

# Audio processing settings
CHUNK_LENGTH_S = 30  # Process audio in 30-second chunks
DEFAULT_BATCH_SIZE = 16  # Batch size for inference

# Supported audio file extensions
AUDIO_EXTENSIONS = [".wav", ".mp3", ".flac", ".m4a", ".ogg", ".webm"]


# =============================================================================
# HUGGING FACE AUTHENTICATION
# =============================================================================

def setup_hf_authentication():
    """
    Setup Hugging Face Hub authentication using environment variable or default token.
    
    Checks for HF_TOKEN or HUGGINGFACE_TOKEN environment variables.
    """
    from huggingface_hub import login
    
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    
    if hf_token:
        try:
            login(token=hf_token, add_to_git_credential=False)
            logger.info("Logged in to Hugging Face Hub with provided token.")
        except Exception as e:
            logger.warning(f"Failed to log in to Hugging Face Hub: {e}")


# =============================================================================
# GPU SETUP
# =============================================================================

def setup_device():
    """
    Configure the optimal device and dtype for inference.
    
    Returns:
        tuple: (device string, torch dtype)
    """
    if torch.cuda.is_available():
        device = "cuda:0"
        torch_dtype = torch.float16
        
        # Log GPU info
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logger.info(f"Using GPU: {gpu_name} ({gpu_memory:.1f} GB)")
    else:
        device = "cpu"
        torch_dtype = torch.float32
        logger.info("Using CPU for inference")
    
    return device, torch_dtype


# =============================================================================
# MODEL LOADING
# =============================================================================

def load_model(model_id: str, device: str, torch_dtype: torch.dtype):
    """
    Load the speech-to-text model and processor.
    
    Args:
        model_id: HuggingFace model identifier
        device: Target device (cuda:0 or cpu)
        torch_dtype: Data type for model weights
    
    Returns:
        tuple: (model, processor, pipeline)
    """
    logger.info(f"Loading model: {model_id}")
    start_time = time.time()
    
    # Load model with optimizations
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    )
    model.to(device)
    
    # Load processor
    processor = AutoProcessor.from_pretrained(model_id)
    
    # Create pipeline
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        chunk_length_s=CHUNK_LENGTH_S,
        batch_size=DEFAULT_BATCH_SIZE,
        torch_dtype=torch_dtype,
        device=device,
    )
    
    load_time = time.time() - start_time
    logger.info(f"Model loaded in {load_time:.2f}s")
    
    return model, processor, pipe


# =============================================================================
# TRANSCRIPTION FUNCTIONS
# =============================================================================

def transcribe_file(
    pipe,
    audio_path: str,
    language: Optional[str] = None,
    task: str = "transcribe",
    return_timestamps: bool = False,
) -> Dict[str, Any]:
    """
    Transcribe a single audio file.
    
    Args:
        pipe: HuggingFace ASR pipeline
        audio_path: Path to audio file
        language: Optional language code (e.g., 'en', 'hi')
        task: 'transcribe' or 'translate'
        return_timestamps: Whether to return timestamps
    
    Returns:
        Dictionary with transcription results
    """
    start_time = time.time()
    
    # Build generation kwargs
    generate_kwargs = {"task": task}
    if language:
        generate_kwargs["language"] = language
    
    # Run transcription
    result = pipe(
        audio_path,
        generate_kwargs=generate_kwargs,
        return_timestamps=return_timestamps,
    )
    
    # Add timing info
    inference_time = time.time() - start_time
    result["inference_time"] = inference_time
    result["file"] = audio_path
    
    # Calculate audio duration
    try:
        waveform, sample_rate = torchaudio.load(audio_path)
        audio_duration = waveform.shape[1] / sample_rate
        result["audio_duration"] = audio_duration
        result["rtf"] = inference_time / audio_duration  # Real-time factor
    except Exception:
        pass
    
    return result


def transcribe_directory(
    pipe,
    audio_dir: str,
    language: Optional[str] = None,
    task: str = "transcribe",
    return_timestamps: bool = False,
) -> List[Dict[str, Any]]:
    """
    Transcribe all audio files in a directory.
    
    Args:
        pipe: HuggingFace ASR pipeline
        audio_dir: Directory containing audio files
        language: Optional language code
        task: 'transcribe' or 'translate'
        return_timestamps: Whether to return timestamps
    
    Returns:
        List of transcription results
    """
    audio_dir = Path(audio_dir)
    audio_files = []
    
    # Find all audio files
    for ext in AUDIO_EXTENSIONS:
        audio_files.extend(audio_dir.glob(f"*{ext}"))
        audio_files.extend(audio_dir.glob(f"*{ext.upper()}"))
    
    if not audio_files:
        logger.warning(f"No audio files found in {audio_dir}")
        return []
    
    logger.info(f"Found {len(audio_files)} audio files")
    
    # Transcribe each file
    results = []
    for i, audio_file in enumerate(sorted(audio_files)):
        logger.info(f"Processing [{i+1}/{len(audio_files)}]: {audio_file.name}")
        
        result = transcribe_file(
            pipe,
            str(audio_file),
            language=language,
            task=task,
            return_timestamps=return_timestamps,
        )
        results.append(result)
    
    return results


# =============================================================================
# OUTPUT FORMATTING
# =============================================================================

def format_output(result: Dict[str, Any], output_format: str) -> str:
    """
    Format transcription result based on output type.
    
    Args:
        result: Transcription result dictionary
        output_format: One of 'txt', 'json', 'srt', 'vtt'
    
    Returns:
        Formatted string
    """
    if output_format == "txt":
        return result["text"]
    
    elif output_format == "json":
        return json.dumps(result, indent=2, ensure_ascii=False)
    
    elif output_format == "srt":
        if "chunks" not in result:
            return result["text"]
        
        srt_output = []
        for i, chunk in enumerate(result["chunks"], 1):
            start = chunk["timestamp"][0] or 0
            end = chunk["timestamp"][1] or start + 1
            
            start_str = f"{int(start//3600):02d}:{int((start%3600)//60):02d}:{int(start%60):02d},{int((start%1)*1000):03d}"
            end_str = f"{int(end//3600):02d}:{int((end%3600)//60):02d}:{int(end%60):02d},{int((end%1)*1000):03d}"
            
            srt_output.append(f"{i}\n{start_str} --> {end_str}\n{chunk['text'].strip()}\n")
        
        return "\n".join(srt_output)
    
    elif output_format == "vtt":
        if "chunks" not in result:
            return f"WEBVTT\n\n{result['text']}"
        
        vtt_output = ["WEBVTT\n"]
        for chunk in result["chunks"]:
            start = chunk["timestamp"][0] or 0
            end = chunk["timestamp"][1] or start + 1
            
            start_str = f"{int(start//3600):02d}:{int((start%3600)//60):02d}:{start%60:06.3f}"
            end_str = f"{int(end//3600):02d}:{int((end%3600)//60):02d}:{end%60:06.3f}"
            
            vtt_output.append(f"{start_str} --> {end_str}\n{chunk['text'].strip()}\n")
        
        return "\n".join(vtt_output)
    
    return result["text"]


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Speech-to-Text Inference Script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input arguments
    input_group = parser.add_argument_group("Input Options")
    input_group.add_argument(
        "--audio", "-a",
        type=str,
        help="Path to single audio file for transcription"
    )
    input_group.add_argument(
        "--audio_dir", "-d",
        type=str,
        help="Directory containing audio files for batch processing"
    )
    
    # Model arguments
    model_group = parser.add_argument_group("Model Options")
    model_group.add_argument(
        "--model", "-m",
        type=str,
        default=DEFAULT_MODEL_ID,
        help="HuggingFace model identifier (e.g., your-username/your-model)"
    )
    model_group.add_argument(
        "--language", "-l",
        type=str,
        default=None,
        help="Language code (e.g., 'en', 'hi'). Auto-detect if not specified"
    )
    model_group.add_argument(
        "--task", "-t",
        type=str,
        choices=["transcribe", "translate"],
        default="transcribe",
        help="Task: transcribe or translate to English"
    )
    
    # Output arguments
    output_group = parser.add_argument_group("Output Options")
    output_group.add_argument(
        "--output", "-o",
        type=str,
        help="Output file path (default: print to stdout)"
    )
    output_group.add_argument(
        "--output_format", "-f",
        type=str,
        choices=["txt", "json", "srt", "vtt"],
        default="txt",
        help="Output format"
    )
    output_group.add_argument(
        "--timestamps",
        action="store_true",
        help="Return word-level timestamps (for srt/vtt output)"
    )
    
    # Performance arguments
    perf_group = parser.add_argument_group("Performance Options")
    perf_group.add_argument(
        "--benchmark",
        action="store_true",
        help="Print timing statistics"
    )
    
    return parser.parse_args()


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """Main entry point for the script."""
    args = parse_arguments()
    
    # Validate input
    if not args.audio and not args.audio_dir:
        logger.error("Please provide --audio or --audio_dir")
        sys.exit(1)
    
    # Setup authentication
    setup_hf_authentication()
    
    # Setup device
    device, torch_dtype = setup_device()
    
    # Load model
    model, processor, pipe = load_model(args.model, device, torch_dtype)
    
    # Process audio
    results = []
    
    if args.audio:
        # Single file transcription
        logger.info(f"Processing: {args.audio}")
        result = transcribe_file(
            pipe,
            args.audio,
            language=args.language,
            task=args.task,
            return_timestamps=args.timestamps,
        )
        results.append(result)
        
    elif args.audio_dir:
        # Batch processing
        results = transcribe_directory(
            pipe,
            args.audio_dir,
            language=args.language,
            task=args.task,
            return_timestamps=args.timestamps,
        )
    
    # Output results
    for result in results:
        output_text = format_output(result, args.output_format)
        
        if args.output:
            with open(args.output, "a", encoding="utf-8") as f:
                f.write(output_text + "\n")
        else:
            print(output_text)
        
        # Print benchmark info if requested
        if args.benchmark:
            if "inference_time" in result:
                logger.info(f"File: {result.get('file', 'N/A')}")
                logger.info(f"Inference time: {result['inference_time']:.2f}s")
                if "audio_duration" in result:
                    logger.info(f"Audio duration: {result['audio_duration']:.2f}s")
                    logger.info(f"Real-time factor: {result['rtf']:.3f}x")
    
    # Summary for batch processing
    if len(results) > 1 and args.benchmark:
        total_audio = sum(r.get("audio_duration", 0) for r in results)
        total_inference = sum(r.get("inference_time", 0) for r in results)
        if total_audio > 0:
            logger.info(f"\n{'='*50}")
            logger.info(f"Total files processed: {len(results)}")
            logger.info(f"Total audio duration: {total_audio:.1f}s")
            logger.info(f"Total inference time: {total_inference:.1f}s")
            logger.info(f"Average RTF: {total_inference/total_audio:.3f}x")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()
