import numpy as np
import webrtcvad

VAD_MS = [10, 20, 30]
TEN_MS = 10 / 1000

dtype_mapping = {
	1: np.int8,
	2: np.int16,
	4: np.int32
}

def normalize_audio(
	audio_bytes: list[bytes] | bytes,
	channels: int = 1,
	sample_width: int = 2
) -> list[np.ndarray] | np.ndarray:
	list_flag = True
	if not isinstance(audio_bytes, list):
		list_flag = False
		audio_bytes = [audio_bytes]

	
	# Assuming 16-bit PCM, adjust if audio is different
	# How to know which dtype to use? check the sample_width from the incoming audio
	# sample_width is in bytes, and the dtype in bits.
	# 16-bit PCM is calculated using sample_width x 8 (8 bits per byte)
	if sample_width not in dtype_mapping:
		raise ValueError(f"Unsupported sample width: {sample_width} (supported values are {','.join(dtype_mapping.keys())})")
	audio_type = dtype_mapping[sample_width]

	info = np.iinfo(audio_type)
	abs_max = max(info.max, abs(info.min))

	ret = []

	for e in audio_bytes:

		# audio is a 1D array of 16-bit signed integers.
		audio = np.frombuffer(e, dtype=audio_type)

		# to normalize to [-1.0, 1.0) range, divide by the max absolute value of the min and max of the dtype.
		audio = audio.astype(np.float32) / abs_max

		# If the audio has multiple channels, reshape and average to mono
		if channels > 1:
			audio = audio.reshape(-1, channels).mean(axis=1)
		
		ret.append(audio)

	if not list_flag:
		return ret[0]
		
	return ret

def calc_bytes_per_ms(
	sample_rate: int,
	sample_width: int,
	channels: int,
	ms: float = TEN_MS
) -> float:
	r"""
	According to the forumal of duration (in seconds) as a function of the audio bytes.
	sample rate, sample width and channels we get
	
	.. math:: duration = \frac{bytes}{sr \cdot sw \cdot ch}

	s.t. 
		sr - sample rate
		sw - sample width
		ch - channels
	"""

	return ms * sample_rate * sample_width * channels



def vad_collector(
	chunk: bytes,
	bytes_per_ten_ms: int, 
	vad: webrtcvad.Vad, 
	sample_rate: int = 16000
):
	frame_samples = int(TEN_MS * sample_rate)
	ret = []

	for frame_idx in range(0, len(chunk), bytes_per_ten_ms):
		frame = chunk[frame_idx:min(frame_idx + bytes_per_ten_ms, len(chunk))]
		frame = frame + b'\x00' * (bytes_per_ten_ms - len(frame))

		# each approval is for 10ms frame, i.e., sample_rate x 10ms samples.
		frame_approval = vad.is_speech(frame, 16000)
		ret.extend([frame_approval] * frame_samples)
	return np.array(ret, dtype=np.float32)