import numpy as np
from ultralytics import YOLO
import os

def test_model():
    model_path = "best_openvino_model"
    if not os.path.exists(model_path):
        print(f"Error: Model path {model_path} does not exist.")
        return

    print(f"Loading model from {model_path}...")
    try:
        model = YOLO(model_path, task='detect')
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # Create a dummy image (1024x1024 as suggested by README)
    dummy_img = np.zeros((1024, 1024, 3), dtype=np.uint8)
    
    print("Running inference on dummy image...")
    try:
        results = model.predict(dummy_img, imgsz=1024, verbose=False)
        print("Inference successful.")
        # Check if we got any results (even if 0 detections)
        if len(results) > 0:
            print(f"Number of detected boxes: {len(results[0].boxes)}")
    except Exception as e:
        print(f"Inference failed: {e}")

if __name__ == "__main__":
    test_model()
