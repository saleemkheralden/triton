import os
import wave
from preprocessing import normalize_audio, whisper_tokinze, build_input

import numpy as np

# /audio_20260513_174839.wav
AUDIO_DIR = 'saved_audio'
TEST_AUDIO_PATH = list(map(lambda x: f"{AUDIO_DIR}/{x}", os.listdir(AUDIO_DIR)))
print(TEST_AUDIO_PATH)
WINDOW_SIZE = int(16000 * 2)

def read_audio(audio_path: str):
    with wave.open(audio_path, "rb") as wav_file:
        sample_rate = wav_file.getframerate()
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        n_frames = wav_file.getnframes()

        print(f"Audio file info: sample_rate={sample_rate}, channels={channels}, sample_width={sample_width}, n_frames={n_frames}")
        wav_data = wav_file.readframes(n_frames)
        print(f"Read {len(wav_data)} bytes of audio data from {audio_path}")
    
    return { 'data': wav_data, 'sr': sample_rate, 'ch': channels }

batch = list(map(read_audio, TEST_AUDIO_PATH))
# normalized_audio = normalize_audio(list([ e['data'] for e in batch ]))
# x, attention_mask = whisper_tokinze(normalized_audio)

x, attention_mask = build_input(list([ e['data'] for e in batch ]))

print(x.shape)
exit(0)

from transformers import WhisperForConditionalGeneration, WhisperProcessor

triton_model = WhisperForConditionalGeneration.from_pretrained(
    "./models/whisper-large-v3-turbo"
)
processor = WhisperProcessor.from_pretrained(
	"./models/whisper-large-v3-turbo"
)

triton_model.generation_config.forced_decoder_ids = None
triton_model.generation_config.suppress_tokens = None
triton_model.generation_config.begin_suppress_tokens = None
triton_model.eval()

hidden_state = triton_model.generate(
    x[-1:],
    attention_mask=attention_mask[-1:],
    language="he",
    task="transcribe"
)
print(hidden_state)
print(f"hidden state shape: {hidden_state.shape}")
output = processor.batch_decode(hidden_state, skip_special_tokens=False)  #[0].strip()
print(output)






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

