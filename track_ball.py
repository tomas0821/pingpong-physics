import cv2
import time
import csv
import numpy as np
from ultralytics import YOLO
import argparse
import os
from utils import PerspectiveCalibration

# --- Configuration ---
DEFAULT_CALIB_WIDTH = 40.0
DEFAULT_CALIB_HEIGHT = 20.0
DEFAULT_MODEL = "best_openvino_model" if os.path.exists("best_openvino_model") else "best.pt"

# --- State Variables ---
calib = PerspectiveCalibration(DEFAULT_CALIB_WIDTH, DEFAULT_CALIB_HEIGHT)
tracking_active = False

def onMouse(event, x, y, flags, param):
    global calib
    if not calib.is_calibrated() and event == cv2.EVENT_LBUTTONDOWN:
        calib.add_point(x, y)

def run_tracking(model_path, output_csv):
    global tracking_active, calib
    
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    cap = cv2.VideoCapture(0)
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
                # Using 1024 as per latest model re-export
                results = model.track(frame, persist=True, verbose=False, imgsz=1024)
                if results and results[0].boxes:
                    for box in results[0].boxes:
                        x, y, w, h = box.xywh[0]
                        conf = box.conf[0]
                        
                        pos_cm = calib.map_point(x.item(), y.item())
                        x_cm, y_cm = (pos_cm[0], pos_cm[1]) if pos_cm is not None else (0, 0)
                        
                        writer.writerow([current_time, x.item(), y.item(), x_cm, y_cm, conf.item()])

                        cv2.rectangle(disp, (int(x - w/2), int(y - h/2)), (int(x + w/2), int(y + h/2)), (0, 255, 0), 2)
                        label = f"Pos: {x_cm:.1f}, {y_cm:.1f} cm" if calib.is_calibrated() else f"Conf: {conf:.2f}"
                        cv2.putText(disp, label, (int(x - w/2), int(y - h/2) - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            calib.draw_info(disp)
            status = "TRACKING" if tracking_active else "IDLE"
            cv2.putText(disp, f"S:{status} | Scale: {'OK' if calib.is_calibrated() else 'NO'} | S:Toggle R:Reset Q:Quit", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.imshow('Pingpong Tracking', disp)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            elif key == ord('s'): tracking_active = not tracking_active
            elif key == ord('r'): calib.reset()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL)
    parser.add_argument('--output', type=str, default='pingpong_data.csv')
    args = parser.parse_args()
    run_tracking(args.model, args.output)
