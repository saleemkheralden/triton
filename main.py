import os
import wave
from preprocessing import normalize_audio, whisper_tokinze, build_input

import numpy as np
import matplotlib.pyplot as plt

from transformers import WhisperProcessor

processor = WhisperProcessor.from_pretrained(
	"./models/whisper-large-v3-turbo"
)

# /audio_20260513_174839.wav
AUDIO_DIR = 'saved_audio'
# TEST_AUDIO_PATH = list(map(lambda x: f"{AUDIO_DIR}/{x}", os.listdir(AUDIO_DIR)))
TEST_AUDIO_PATH = ['saved_audio/bnm-1-ys.wav']
print(TEST_AUDIO_PATH)
WINDOW_SIZE = int(16000 * 2)

PCM_DTYPE_BY_WIDTH = {
    1: np.int8,
    2: np.int16,
    4: np.int32,
}

def stream_audio_chunks(wav_data, chunk_size=4096):
    for i in range(0, len(wav_data), chunk_size):
        yield wav_data[i:i+chunk_size]

def read_audio(audio_path: str):
    with wave.open(audio_path, "rb") as wav_file:
        sample_rate = wav_file.getframerate()
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        n_frames = wav_file.getnframes()

        print(f"Audio file info: sample_rate={sample_rate}, channels={channels}, sample_width={sample_width}, n_frames={n_frames}")
        wav_data = wav_file.readframes(n_frames)
        print(f"Read {len(wav_data)} bytes of audio data from {audio_path}")

        if sample_width not in PCM_DTYPE_BY_WIDTH:
            raise ValueError(f"Unsupported sample width: {sample_width}")

        # Convert raw PCM bytes to float32 in [-1, 1] for safe processing.
        pcm_dtype = PCM_DTYPE_BY_WIDTH[sample_width]
        raw_audio = np.frombuffer(wav_data, dtype=pcm_dtype)
        if channels > 1:
            raw_audio = raw_audio.reshape(-1, channels).mean(axis=1)
            channels = 1

        raw_audio = raw_audio[int(sample_rate * 60 * 8):int(sample_rate * 60 * 8.5)]

        info = np.iinfo(pcm_dtype)
        abs_max = float(max(info.max, abs(info.min)))
        audio = raw_audio.astype(np.float32) / abs_max

        if sample_rate != 16_000:
            print("Resampling audio to 16kHz")
            old_indices = np.arange(audio.shape[0], dtype=np.float32)
            new_length = int(round(audio.shape[0] * 16_000 / sample_rate))
            new_indices = np.linspace(0, audio.shape[0] - 1, new_length, dtype=np.float32)
            audio = np.interp(new_indices, old_indices, audio).astype(np.float32)
            sample_rate = 16_000

        # Re-encode as int16 PCM so downstream frombuffer-based normalization still works.
        wav_data = np.clip(audio * 32768.0, -32768, 32767).astype(np.int16).tobytes()
        sample_width = 2

    return { 'data': wav_data, 'sr': sample_rate, 'ch': channels, 'sw': sample_width }

audio_info = read_audio(TEST_AUDIO_PATH[0])
data = audio_info['data']
sr = audio_info['sr']
ch = audio_info['ch']
sw = audio_info['sw']
# batch = list(map(read_audio, TEST_AUDIO_PATH))
# normalized_audio = normalize_audio(list([ e['data'] for e in batch ]))
# x, attention_mask = whisper_tokinze(normalized_audio)

x, attention_mask = build_input(
    audio_bytes=data,
    processor=processor,
    channels=ch,
    sample_width=sw,
    sample_rate=sr,
    triton_input=False
)

# import matplotlib.pyplot as plt

# data = x[-1].detach().numpy()
# plt.figure(figsize=(12, 6))
# plt.imshow(data, aspect='auto', cmap='viridis')
# plt.colorbar(label='Intensity')
# plt.xlabel('Time Steps')
# plt.ylabel('Feature Dimension')
# plt.title('Audio Feature Heatmap')
# plt.tight_layout()
# plt.show()

# print(x.shape)
# exit(0)

from transformers import WhisperForConditionalGeneration, WhisperProcessor

triton_model = WhisperForConditionalGeneration.from_pretrained(
    "./models/whisper-large-v3-turbo"
)

triton_model.generation_config.forced_decoder_ids = None
triton_model.generation_config.suppress_tokens = None
triton_model.generation_config.begin_suppress_tokens = None
triton_model.eval()
# triton_model.device()

hidden_state = triton_model.generate(
    x[-1:],
    attention_mask=attention_mask[-1:],
    language="he",
    task="transcribe"
)
print(hidden_state)
print(f"hidden state shape: {hidden_state.shape}")
output = processor.batch_decode(hidden_state, skip_special_tokens=False)[0].strip()
print(output[::-1])






# print(list([ e.shape for e in normalized_audio ]))
# print(x.shape)
# print(x[-1].std(dim=0).shape)
# print(x[-1].std(dim=0))
# print(attention_mask)


# import pyaudio


# p = pyaudio.PyAudio()
# stream = p.open(
# 	format=pyaudio.paInt16, 
# 	channels=1, 
# 	rate=16000, 
# 	input=True, 
# 	frames_per_buffer=1024
# )


# print("Recording...")

# # data = stream.read(1024)
# # print(data)

