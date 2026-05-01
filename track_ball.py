import cv2
import time
import csv
import numpy as np
from ultralytics import YOLO
import argparse
import os

# --- Configuration ---
SCALE_LENGTH_CM = 10.0
# Use OpenVINO folder if it exists, otherwise fallback to .pt
DEFAULT_MODEL = "best_openvino_model" if os.path.exists("best_openvino_model") else "best.pt"

# --- State Variables ---
calibration_points = []
cm_per_pixel = None
tracking_active = False

def onMouse(event, x, y, flags, param):
    global calibration_points, cm_per_pixel
    if cm_per_pixel is None and event == cv2.EVENT_LBUTTONDOWN:
        if len(calibration_points) < 2:
            calibration_points.append((x, y))
            if len(calibration_points) == 2:
                pixel_dist = np.linalg.norm(np.array(calibration_points[0]) - np.array(calibration_points[1]))
                if pixel_dist > 0:
                    cm_per_pixel = SCALE_LENGTH_CM / pixel_dist
                    print(f"Calibration Complete! Scale: {cm_per_pixel:.4f} cm/pixel")

def run_tracking(model_path, output_csv):
    global tracking_active, cm_per_pixel, calibration_points
    
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    cap = cv2.VideoCapture(0)
    # Set higher resolution for 1024px model
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    cv2.namedWindow('Pingpong Tracking')
    cv2.setMouseCallback('Pingpong Tracking', onMouse)

    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['timestamp', 'x_px', 'y_px', 'x_cm', 'y_cm', 'conf'])

        print("Controls: 'S' to start/stop tracking, 'R' to reset calibration, 'Q' to quit.")
        
        while True:
            ret, frame = cap.read()
            if not ret: break

            disp = frame.copy()
            current_time = time.time()

            if tracking_active:
                results = model.track(frame, persist=True, verbose=False, imgsz=640)
                if results and results[0].boxes:
                    for box in results[0].boxes:
                        x, y, w, h = box.xywh[0]
                        conf = box.conf[0]
                        
                        # Convert to CM if calibrated
                        x_cm = x.item() * cm_per_pixel if cm_per_pixel else 0
                        y_cm = y.item() * cm_per_pixel if cm_per_pixel else 0
                        
                        writer.writerow([current_time, x.item(), y.item(), x_cm, y_cm, conf.item()])

                        # Visuals
                        cv2.rectangle(disp, (int(x - w/2), int(y - h/2)), (int(x + w/2), int(y + h/2)), (0, 255, 0), 2)
                        label = f"Pos: {x_cm:.1f}, {y_cm:.1f} cm" if cm_per_pixel else f"Conf: {conf:.2f}"
                        cv2.putText(disp, label, (int(x - w/2), int(y - h/2) - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Draw Calibration visuals
            if cm_per_pixel is None:
                for pt in calibration_points:
                    cv2.circle(disp, pt, 5, (0, 0, 255), -1)
                if len(calibration_points) == 1:
                    cv2.putText(disp, "Click second point for 10cm scale", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                else:
                    cv2.putText(disp, "CALIBRATION REQUIRED: Click two points 10cm apart", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # Menu
            status = "TRACKING" if tracking_active else "IDLE"
            cv2.putText(disp, f"S:{status} | Scale: {'OK' if cm_per_pixel else 'NO'} | S:Toggle R:Reset Q:Quit", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.imshow('Pingpong Tracking', disp)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            elif key == ord('s'): tracking_active = not tracking_active
            elif key == ord('r'):
                cm_per_pixel = None
                calibration_points.clear()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL)
    parser.add_argument('--output', type=str, default='pingpong_data.csv')
    args = parser.parse_args()
    run_tracking(args.model, args.output)
