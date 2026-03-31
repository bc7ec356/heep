import torch
from qwen_asr import Qwen3ASRModel

# Load model with Transformers backend
asr = Qwen3ASRModel.from_pretrained(
    "bc7ec356/heep-indic",
    dtype=torch.bfloat16,
    device_map="cuda:0",
)

# Transcribe
results = asr.transcribe(
    audio="path/to/audio.wav",
    language="Hindi",
)
print(results[0].text)
