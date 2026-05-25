from transformers import WhisperProcessor

import os

def whisper_tokinze(
	audio: list[bytes],
	sample_rate: int = 16000
):
	processor = WhisperProcessor.from_pretrained(
		"./models/whisper-large-v3-turbo"
	)
	
	# Now `audio` is a 1D numpy array of float32 samples in the range [-1.0, 1.0).
	# Next step is to convert this "raw" audio array into tokenized vector that the model can understand.
	inputs = processor(audio, sampling_rate=sample_rate, return_tensors="pt", return_attention_mask=True)
	input_features, attention_mask = inputs.input_features, inputs.get("attention_mask")
	
	return input_features, attention_mask