from ultralytics import YOLO

def export_optimized_model(weights_path='runs/detect/train/weights/best.pt'):
    # Load the best trained model
    model = YOLO(weights_path)

    # Export to OpenVINO (Best for Intel CPUs)
    print("Exporting to OpenVINO...")
    model.export(format='openvino', imgsz=640, half=True) # half=True for FP16 precision

    # Export to ONNX (Good general-purpose format)
    print("Exporting to ONNX...")
    model.export(format='onnx', imgsz=640)

    print("Exports complete. Check the weights folder for .xml and .onnx files.")

if __name__ == "__main__":
    export_optimized_model()
