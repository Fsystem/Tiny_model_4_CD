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
    parser.add_argument(
        "--single-file",
        action="store_true",
        help=(
            "Export ONNX in a single .onnx file (disable external tensor data sidecar)."
        ),
    )
    return parser.parse_args()


def export_onnx(modelpath, output, input_size, single_file=False):
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
        external_data=not single_file,
        dynamic_axes={
            "reference": {0: "batch_size"},
            "test": {0: "batch_size"},
            "change_mask": {0: "batch_size"},
        },
        dynamo=False,  # 使用传统 TorchScript 方式
        opset_version=18,
    )
    print(f"ONNX model saved to {output}")
    if single_file:
        print("Export mode: single-file ONNX")
    else:
        print("Export mode: ONNX + external data sidecar")


if __name__ == "__main__":
    args = parse_arguments()
    export_onnx(args.model_path, args.output, args.input_size, args.single_file)
