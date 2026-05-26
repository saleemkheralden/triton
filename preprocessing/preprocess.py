from .audio import normalize_audio
from .tokenizer import whisper_tokinze, build_triton_input

def build_input(
	audio_bytes: list[bytes] | bytes,
	channels: int = 1,
	sample_width: int = 2,
	sample_rate: int = 16000,
	model_path: str = "./models/whisper-large-v3-turbo",
	input_layer_name: str = "mel",
	input_data_type: str = "FP32"	
):
	norm_audio = normalize_audio(
		audio_bytes=audio_bytes, 
		channels=channels, 
		sample_width=sample_width
	)

	input_features, attention_mask = whisper_tokinze(
		audio=norm_audio,
		sample_rate=sample_rate,
		model_path=model_path
	)

	triton_input = build_triton_input(
		input_features=input_features,
		input_layer_name=input_layer_name,
		input_data_type=input_data_type
	)

	return triton_input, attention_mask

	