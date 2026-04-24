import argparse
import torch
from models.change_classifier import ChangeClassifier


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Export ChangeClassifier model to ONNX format."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to the .pth checkpoint to load (optional).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="change_classifier.onnx",
        help="Output path for the exported ONNX file.",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=256,
        help="Spatial size (height and width) of the input images (default: 256).",
    )
    return parser.parse_args()


def export_onnx(modelpath, output, input_size):
    # Initialise model
    model = ChangeClassifier(pretrained=False)
    if modelpath is not None:
        model.load_state_dict(torch.load(modelpath, map_location="cpu"))
        print(f"Loaded weights from {modelpath}")

    model.eval()

    # Create dummy inputs: (batch=1, channels=3, H, W)
    dummy_ref = torch.randn(1, 3, input_size, input_size)
    dummy_test = torch.randn(1, 3, input_size, input_size)

    # Export to ONNX
    torch.onnx.export(
        model,
        (dummy_ref, dummy_test),
        output,
        input_names=["reference", "test"],
        output_names=["change_mask"],
        dynamic_axes={
            "reference": {0: "batch_size"},
            "test": {0: "batch_size"},
            "change_mask": {0: "batch_size"},
        },
        opset_version=18,
    )
    print(f"ONNX model saved to {output}")


if __name__ == "__main__":
    args = parse_arguments()
    export_onnx(args.model_path, args.output, args.input_size)
