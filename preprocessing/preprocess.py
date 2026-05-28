from transformers import WhisperProcessor
from .audio import normalize_audio
from .tokenizer import whisper_tokinze, build_triton_input

def build_input(
	audio_bytes: list[bytes] | bytes,
	processor: WhisperProcessor,
	channels: int = 1,
	sample_width: int = 2,
	sample_rate: int = 16000,
	input_layer_name: str = "mel",
	input_data_type: str = "FP32",
	triton_input: bool = True
):
	norm_audio = normalize_audio(
		audio_bytes=audio_bytes, 
		channels=channels, 
		sample_width=sample_width
	)

	input_features, attention_mask = whisper_tokinze(
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

	