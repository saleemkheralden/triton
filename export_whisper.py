import torch
from pyannote.audio import Model

# 1. Load your local pyannote model weights
model = Model.from_pretrained("raw_models/pyannote-segmentation")
model.eval()

# 2. Create dummy input (Batch=1, Channels=1, Samples=160000 -> 10 seconds of 16kHz audio)
dummy_input = torch.zeros(1, 1, 160000)

# 3. Export directly to your Triton target directory
torch.onnx.export(
    model, 
    dummy_input, 
    "triton_models/pyannote-segmentation/1/model.onnx",
    do_constant_folding=True,
    input_names=["input_values"],
    output_names=["logits"],
    dynamic_axes={
        "input_values": {0: "batch_size", 2: "num_samples"},
        "logits": {0: "batch_size", 1: "num_frames"}
    },
    # opset_version=14
)
print("ONNX export complete!")