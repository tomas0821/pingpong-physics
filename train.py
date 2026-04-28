from ultralytics import YOLO
import os

def train_model():
    # Load a pretrained YOLO11n model
    model = YOLO("yolo11n.pt")

    # Define the path to the data.yaml file
    # Roboflow usually downloads it to a folder named 'Ping-Pong-Detection-3'
    data_path = os.path.join(os.getcwd(), "Ping-Pong-Detection-3", "data.yaml")
    
    if not os.path.exists(data_path):
        print(f"Error: data.yaml not found at {data_path}")
        return

    # Train the model
    # Optimized for small objects (imgsz=1024) and high-speed movement
    results = model.train(
        data=data_path,
        epochs=150,      # Increased epochs for better convergence at higher res
        imgsz=1024,      # Higher resolution = better small object detection
        batch=-1,        # Auto-batch to maximize A100 GPU utilization
        patience=30,     # More patience for the higher resolution training
        device=0,        
        plots=True,
        save=True,
        cache=True,
        # Augmentation tweaks for small/fast objects:
        mosaic=1.0,      # Mix images to see objects in different contexts
        mixup=0.1,       # Helps with motion blur robustness
        scale=0.7,       # Significant scaling to simulate distance
        fliplr=0.5,      # Horizontal flips
        overlap_mask=True
    )
    
    print("Training complete. Results saved to 'runs/detect/train'")

if __name__ == "__main__":
    train_model()
