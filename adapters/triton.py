import torch
import asyncio
import tritonclient.grpc.aio as triton_grpc_aio

class TritonAdapter:
	def __init__(
		self,
		triton_url
	):
		self.triton_client = triton_grpc_aio.InferenceServerClient(url=triton_url)
		triton_live = asyncio.run(self.triton_client.is_server_live())

		if not triton_live:
			raise Exception(f"Triton server isn't live at url: {triton_url}")
		

	async def __call__(
		self,
		infer_input: torch.tensor,
		model_name: str = "whisper",
		model_output_layer_name: str = "encoder_hidden_state"
	) -> str:
		return self.infer(
			infer_input=infer_input,
			model_name=model_name,
			model_output_layer_name=model_output_layer_name
		)

	async def infer(
		self,
		infer_input: torch.tensor,
		model_name: str = "whisper",
		model_output_layer_name: str = "encoder_hidden_state"
	) -> str:
		hidden_states = await self.triton_client.infer(
			model_name=model_name,
			input=[infer_input],
			outputs=[model_output_layer_name]
		)

		return hidden_states
	

 