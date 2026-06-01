from .triton import TritonAdapter
from preprocessing import 
import numpy as np
from collections import deque

class WhisperAdapter:

	def __init__(self):
		# self.audio_buffer = np.array([],dtype=np.float32)
		self.audio_chunks = deque()		
		self.triton_adapter = TritonAdapter()


	def insert_audio_chunk(self, audio):
		# self.audio_buffer = np.append(self.audio_buffer, audio)
		self.audio_chunks.append(audio)

	def process_iter(self):
		audio = np.concatenate(self.audio_chunks)
		
		self.triton_adapter()





