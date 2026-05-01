import cv2
from ultralytics import YOLO
import numpy as np
import time
import math
from collections import deque
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import argparse
import os
from utils import PerspectiveCalibration

# ---------------- Configuration ----------------
DEFAULT_CALIB_WIDTH = 30.0
DEFAULT_CALIB_HEIGHT = 30.0
DEFAULT_MODEL = "best_openvino_model" if os.path.exists("best_openvino_model") else "best.pt"
CAMERA_INDEX = 0
TARGET_WIDTH, TARGET_HEIGHT = 1280, 720
CONFIDENCE_THRESHOLD = 0.6
POINT_HISTORY_LENGTH = 500 

# --- State Variables ---
calib = PerspectiveCalibration(DEFAULT_CALIB_WIDTH, DEFAULT_CALIB_HEIGHT)
pivot_point_px = None
pivot_point_cm = None
tracking_active = False
is_paused = False
pendulum_history = deque(maxlen=POINT_HISTORY_LENGTH) 
period_measurements = []
last_pass_time = None
last_pass_side = None 

def onMouse(event, x, y, flags, param):
    global calib, pivot_point_px, pivot_point_cm
    if not calib.is_calibrated():
        if event == cv2.EVENT_LBUTTONDOWN:
            calib.add_point(x, y)
        return
    
    if pivot_point_px is None and event == cv2.EVENT_LBUTTONDOWN:
        pivot_point_px = (x, y)
        pivot_point_cm = calib.map_point(x, y)
        print(f"Pivot point set at PX:{pivot_point_px}, CM:{pivot_point_cm}")

def calculate_pendulum_properties(pos_cm, time_val, history, pivot_cm):
    if not history or pivot_cm is None: return {'angle': 0, 'angular_velocity': 0}
    dx = pos_cm[0] - pivot_cm[0]
    dy = pos_cm[1] - pivot_cm[1]
    angle = math.atan2(dx, dy)
    
    prev_pos_cm, prev_time, prev_angle = history[-1]
    dt = time_val - prev_time
    av = (angle - prev_angle) / dt if dt > 1e-6 else 0
    return {'angle': angle, 'angular_velocity': av}

def detect_period(current_angle, current_time):
    global last_pass_time, last_pass_side, period_measurements
    side = -1 if current_angle < 0 else 1
    if last_pass_side is not None and side != last_pass_side and abs(current_angle) < 0.1:
        if last_pass_time is not None:
            T = 2 * (current_time - last_pass_time)
            if 0.2 < T < 10.0:
                period_measurements.append(T)
                if len(period_measurements) > 10: period_measurements.pop(0)
        last_pass_time = current_time
    last_pass_side = side

def damped_oscillation(t, A, beta, omega, phi):
    return A * np.exp(-beta * t) * np.cos(omega * t + phi)

def plot_pendulum_data(history, avg_period):
    if len(history) < 20: return
    data = list(history)
    t = np.array([pt[1] - data[0][1] for pt in data])
    theta = np.array([pt[2] for pt in data])
    
    try:
        p0 = [np.max(np.abs(theta)), 0.05, 2*np.pi/avg_period if avg_period > 0 else 5, 0]
        popt, _ = curve_fit(damped_oscillation, t, theta, p0=p0, maxfev=5000)
        t_smooth = np.linspace(t.min(), t.max(), 500)
        theta_fit = damped_oscillation(t_smooth, *popt)
        plt.plot(t_smooth, np.degrees(theta_fit), 'g-', label='Damped Fit')
    except: pass

    plt.plot(t, np.degrees(theta), 'b.', label='Data', markersize=2)
    plt.title('Pendulum: Angle vs Time'); plt.grid(); plt.legend()
    plt.savefig('pendulum_fit.pdf', bbox_inches='tight')
    plt.show()

def run_pendulum(model_path):
    global tracking_active, is_paused, pivot_point_px, pivot_point_cm, calib
    model = YOLO(model_path)
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, TARGET_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_HEIGHT)
    
    cv2.namedWindow("Pendulum Analyzer")
    cv2.setMouseCallback("Pendulum Analyzer", onMouse)
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        curr_t = time.time()
        det_info = None
        avg_T = np.mean(period_measurements) if period_measurements else 0.0

        if tracking_active and not is_paused and pivot_point_cm is not None:
            results = model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False, imgsz=1024)
            if results and results[0].boxes:
                best = max(results[0].boxes, key=lambda b: b.conf[0])
                box = best.xyxy[0]
                center_px = (int((box[0]+box[2])/2), int((box[1]+box[3])/2))
                pos_cm = calib.map_point(center_px[0], center_px[1])
                if pos_cm is not None:
                    props = calculate_pendulum_properties(pos_cm, curr_t, pendulum_history, pivot_point_cm)
                    pendulum_history.append((pos_cm, curr_t, props['angle']))
                    detect_period(props['angle'], curr_t)
                    det_info = (center_px, pos_cm, int(box[2]-box[0]))

        disp = frame.copy()
        if pivot_point_px: cv2.drawMarker(disp, pivot_point_px, (0, 255, 255), cv2.MARKER_CROSS, 20, 2)
        if det_info:
            c_px, _, w_px = det_info
            cv2.circle(disp, c_px, max(5, int(w_px/2)), (0, 255, 0), 2)
        
        status = "CALIBRATE" if not calib.is_calibrated() else ("SET PIVOT" if pivot_point_px is None else "READY")
        cv2.putText(disp, f"S:{status} | S:Start G:Graph R:Reset Q:Quit", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        calib.draw_info(disp)
        cv2.imshow("Pendulum Analyzer", disp)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        elif key == ord('s') and pivot_point_cm is not None: 
            tracking_active = not tracking_active
            if tracking_active: pendulum_history.clear(); period_measurements.clear()
        elif key == ord('p'): is_paused = not is_paused
        elif key == ord('g') and is_paused: plot_pendulum_data(pendulum_history, avg_T)
        elif key == ord('r'): 
            pivot_point_px = None; pivot_point_cm = None; calib.reset(); pendulum_history.clear()

    cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL)
    args = parser.parse_args()
    run_pendulum(args.model)
