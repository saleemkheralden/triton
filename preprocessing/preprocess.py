from transformers import WhisperProcessor
from .audio import normalize_audio
from .embedder import whisper_embed, build_triton_input
import numpy as np

def build_input(
	audio_bytes: list[bytes] | bytes,
	processor: WhisperProcessor,
	channels: int = 1,
	sample_width: int = 2,
	sample_rate: int = 16000,
	input_layer_name: str = "mel",
	input_data_type: str = "FP32",
	triton_input: bool = True,
	vad_mask: np.ndarray = None
):
	norm_audio = normalize_audio(
		audio_bytes=audio_bytes, 
		channels=channels, 
		sample_width=sample_width
	)

	if vad_mask is not None:
		norm_audio = norm_audio[np.array(vad_mask, dtype=bool)]
	

	input_features, attention_mask = whisper_embed(
		audio=norm_audio,
		processor=processor,
		sample_rate=sample_rate
	)

	if not triton_input:
		return input_features, attention_mask

	triton_input = build_triton_input(
		input_features=input_features,
		input_layer_name=input_layer_name,
		input_data_type=input_data_type
	)

	return triton_input, attention_mask

	