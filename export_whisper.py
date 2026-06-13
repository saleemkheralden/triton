import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration

model_id = "models/whisper-large-v3-turbo"

processor = WhisperProcessor.from_pretrained(model_id)
model = WhisperForConditionalGeneration.from_pretrained(model_id)

model.eval()

dummy_input = torch.randn(1, 128, 3000)  # mel spectrogram shape

torch.onnx.export(
    model.model.encoder,
    (dummy_input,),
    "whisper_dynamic_encoder.onnx",
    input_names=["mel_input"],
    output_names=["encoder_hidden_features"],
    dynamic_axes={
        "mel_input": {2: "time"}
    },
    opset_version=17
)

print("Encoder exported")