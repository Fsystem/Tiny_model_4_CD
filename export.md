# Export without pre-trained weights (random weights)
python export_onnx.py --output change_classifier.onnx

# Export from a saved checkpoint
python export_onnx.py --model-path path/to/model.pth --output change_classifier.onnx

# Custom input spatial size (default 256)
python export_onnx.py --model-path pretrained_models/whu_best.pth --output pretrained_models/whu_best_change_classifier.onnx --input-size 512