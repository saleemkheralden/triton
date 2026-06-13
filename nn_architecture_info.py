import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration

model_id = "raw_models/whisper-large-v3-turbo"

processor = WhisperProcessor.from_pretrained(model_id)
model = WhisperForConditionalGeneration.from_pretrained(model_id)

print(f"Model: {model_id}")
print(f"Config: {model.config}")
print("\n=== Architecture ===")
print(model)
print(f"\n=== Total Parameters ===")
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total: {total_params:,}")
print(f"Trainable: {trainable_params:,}")
print(f"Frozen: {total_params - trainable_params:,}")
