import cv2
import time
import csv
from ultralytics import YOLO

def run_tracking(model_path='runs/detect/train/weights/best.pt', output_csv='pingpong_data.csv'):
    # Load the trained model
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Make sure you have trained the model first or provide a valid path.")
        return

    # Initialize webcam
    cap = cv2.VideoCapture(0) # 0 is usually the default camera

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Prepare CSV file
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['timestamp', 'x', 'y', 'conf'])

        print("Starting real-time tracking... Press 'q' to quit.")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Run YOLO inference
            # stream=True for memory efficiency in long videos/real-time
            results = model.track(frame, persist=True, verbose=False)

            current_time = time.time()

            # Process detections
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # Get coordinates (center x, center y)
                    x, y, w, h = box.xywh[0]
                    conf = box.conf[0]
                    
                    # Log to CSV
                    writer.writerow([current_time, x.item(), y.item(), conf.item()])

                    # Draw on frame (optional, for visualization)
                    cv2.rectangle(frame, (int(x - w/2), int(y - h/2)), (int(x + w/2), int(y + h/2)), (0, 255, 0), 2)
                    cv2.putText(frame, f"Pingpong {conf:.2f}", (int(x - w/2), int(y - h/2) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Display the frame
            cv2.imshow('Pingpong Tracking', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Tracking finished. Data saved to {output_csv}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Pingpong Ball Tracking')
    parser.add_argument('--model', type=str, default='runs/detect/train/weights/best.pt', help='Path to model weights (.pt or openvino folder)')
    parser.add_argument('--output', type=str, default='pingpong_data.csv', help='Path to output CSV file')
    args = parser.parse_args()
    
    run_tracking(model_path=args.model, output_csv=args.output)
