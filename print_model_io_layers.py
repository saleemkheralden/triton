import argparse
from pathlib import Path

try:
    import onnx
except ModuleNotFoundError:
    onnx = None


def print_model_io(model_path: Path) -> None:
    if onnx is None:
        raise RuntimeError(
            "Missing dependency 'onnx'. Install it with: pip install onnx"
        )

    model = onnx.load(str(model_path))

    print(f"Model: {model_path}")

    print("\nInputs:")
    for value_info in model.graph.input:
        print(f"- {value_info.name}")

    print("\nOutputs:")
    for value_info in model.graph.output:
        print(f"- {value_info.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Print input and output layer names from an ONNX model"
    )
    parser.add_argument(
        "--model",
        default="models/onnx_whisper/1/model.onnx",
        help="Path to ONNX model file",
    )
    args = parser.parse_args()

    print_model_io(Path(args.model))
