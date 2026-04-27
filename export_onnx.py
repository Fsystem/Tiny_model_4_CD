import argparse
import torch
from models.change_classifier import ChangeClassifier


def _load_state_dict_with_compat(model, modelpath):
    state_dict = torch.load(modelpath, map_location="cpu")

    try:
        model.load_state_dict(state_dict)
        return
    except RuntimeError as load_error:
        # Backward-compat for checkpoints where _mixing_mask[2] was wrapped in
        # an extra "_mixing" module.
        old_prefix = "_mixing_mask.2._mixing._convmix."
        new_prefix = "_mixing_mask.2._convmix."

        remapped_state_dict = {}
        remapped_any_key = False
        for key, value in state_dict.items():
            if key.startswith(old_prefix):
                remapped_state_dict[new_prefix + key[len(old_prefix) :]] = value
                remapped_any_key = True
            else:
                remapped_state_dict[key] = value

        if not remapped_any_key:
            raise load_error

        model.load_state_dict(remapped_state_dict)
        print(
            "Loaded checkpoint with legacy key remapping for _mixing_mask.2."  # noqa: E501
        )


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
        _load_state_dict_with_compat(model, modelpath)
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
