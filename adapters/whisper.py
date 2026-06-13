import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration

model_id = "../raw_models/whisper-large-v3-turbo"

processor = WhisperProcessor.from_pretrained(model_id)
model = WhisperForConditionalGeneration.from_pretrained(model_id)

model.eval()

dummy_input = torch.randn(1, 128, 3000)

torch.onnx.export(
    model.model.encoder,
    (dummy_input,),
    "whisper_dynamic_encoder.onnx",
    input_names=["input_features"],
    output_names=["last_hidden_state"],
    dynamic_axes={
        "input_features": {0: "batch_size"},
        "last_hidden_state": {0: "batch_size"},
    }
)

print("Encoder exported")