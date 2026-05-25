import numpy as np

dtype_mapping = {
	1: np.int8,
	2: np.int16,
	4: np.int32
}

def normalize_audio(
	audio_bytes: list[bytes] | bytes,
	channels: int = 1,
	sample_width: int = 2
) -> list[np.array] | np.array:
	list_flag = True
	if not isinstance(audio_bytes, list):
		list_flag = False
		audio_bytes = [audio_bytes]

	ret = []

	for e in audio_bytes:
		# Assuming 16-bit PCM, adjust if audio is different
		# How to know which dtype to use? check the sample_width from the incoming audio
		# sample_width is in bytes, and the dtype in bits.
		# 16-bit PCM is calculated using sample_width x 8 (8 bits per byte)
		if sample_width not in dtype_mapping:
			raise ValueError(f"Unsupported sample width: {sample_width} (supported values are {','.join(dtype_mapping.keys())})")
		audio_type = dtype_mapping[sample_width]

		# audio is a 1D array of 16-bit signed integers.
		audio = np.frombuffer(e, dtype=audio_type)

		# to normalize to [-1.0, 1.0) range, divide by the max absolute value of the min and max of the dtype.
		info = np.iinfo(audio_type)
		abs_max = max(info.max, abs(info.min))
		audio = audio.astype(np.float32) / abs_max

		# If the audio has multiple channels, reshape and average to mono
		if channels > 1:
			audio = audio.reshape(-1, channels).mean(axis=1)
		
		ret.append(audio)

	if not list_flag:
		return ret[0]
		
	return ret