from transformers import WhisperProcessor
from tritonclient.grpc import InferInput
import numpy as np
import torch
import os

def whisper_embed(
	audio: list[bytes],
	processor: WhisperProcessor,
	sample_rate: int = 16000
):	
	# Now `audio` is a 1D or 2D numpy array of float32 samples in the range [-1.0, 1.0).
	# Next step is to convert this "raw" audio array into tokenized vector that the model can understand.
	inputs = processor(
		audio, 
		sampling_rate=sample_rate, 
		return_tensors="pt", 
		return_attention_mask=True,
		truncation=False,  # default value is True, this truncates the audio into 30s window (adds padding if shorter than 30s)
		# padding="longest"
	)
	input_features, attention_mask = inputs.input_features, inputs.get("attention_mask")
	
	return input_features, attention_mask

def build_triton_input(
	input_features: torch.tensor,
	input_layer_name: str = "mel",
	input_data_type: str = "FP32"
):
	input_features = input_features.numpy().astype(np.float32)
	infer_input = InferInput(input_layer_name, input_features.shape, input_data_type)
	infer_input.set_data_from_numpy(input_features)

	return infer_input