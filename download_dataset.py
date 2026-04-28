import os
from roboflow import Roboflow

def download_dataset():
    # Read API key from file
    with open("ROBOFLOW_API_KEY", "r") as f:
        api_key = f.read().strip()

    rf = Roboflow(api_key=api_key)
    project = rf.workspace("pingpong-ojuhj").project("ping-pong-detection-0guzq")
    version = project.version(3)
    
    # Download the dataset in YOLO11 format
    dataset = version.download("yolov11")
    print(f"Dataset downloaded to: {dataset.location}")

if __name__ == "__main__":
    download_dataset()
