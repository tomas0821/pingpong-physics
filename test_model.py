import os
os.environ['ULTRALYTICS_GIT_CHECK'] = 'false'
import numpy as np
from ultralytics import YOLO

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

    # Create a dummy image (1024x1024 to match the new model input)
    dummy_img = np.zeros((1024, 1024, 3), dtype=np.uint8)
    
    print("Running inference on dummy image at 1024px...")
    try:
        results = model.predict(dummy_img, imgsz=1024, verbose=True)
        print("Inference successful.")
        if len(results) > 0:
            print(f"Number of detected boxes: {len(results[0].boxes)}")
    except Exception as e:
        print(f"Inference failed: {e}")
        print("\nNote: If you get a 'shape incompatible' error, it means the OpenVINO model")
        print("was exported with a different imgsz. I have just re-exported it to 1024.")

if __name__ == "__main__":
    test_model()
