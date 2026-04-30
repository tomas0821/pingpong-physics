import cv2
from ultralytics import YOLO
import numpy as np
import time
import math
from collections import deque
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import argparse

# ---------------- Configuration ----------------
SCALE_LENGTH_CM = 10.0
# Use OpenVINO folder if it exists, otherwise fallback to .pt
DEFAULT_MODEL = "best_openvino_model" if os.path.exists("best_openvino_model") else "best.pt"
CAMERA_INDEX = 0
TARGET_WIDTH, TARGET_HEIGHT = 1280, 720
TARGET_FPS = 60
CONFIDENCE_THRESHOLD = 0.6
POINT_HISTORY_LENGTH = 500 

# --- State Variables ---
calibration_points = []
cm_per_pixel = None
pivot_point = None
tracking_active = False
is_paused = False
pendulum_history = deque(maxlen=POINT_HISTORY_LENGTH) 
period_measurements = []
last_pass_time = None
last_pass_side = None 

def onMouse(event, x, y, flags, param):
    global calibration_points, cm_per_pixel, pivot_point
    if cm_per_pixel is None and event == cv2.EVENT_LBUTTONDOWN:
        if len(calibration_points) < 2:
            calibration_points.append((x, y))
            if len(calibration_points) == 2:
                pixel_dist = np.linalg.norm(np.array(calibration_points[0]) - np.array(calibration_points[1]))
                if pixel_dist > 0:
                    cm_per_pixel = SCALE_LENGTH_CM / pixel_dist
                    print(f"Calibration Complete! Scale: {cm_per_pixel:.4f} cm/pixel")
        return
    if cm_per_pixel is not None and pivot_point is None and event == cv2.EVENT_LBUTTONDOWN:
        pivot_point = (x, y)
        print(f"Pivot point set at {pivot_point}.")

def calculate_pendulum_properties(pos, time, history, pivot):
    if not history or pivot is None: return {'angle': 0, 'angular_velocity': 0}
    angle = math.atan2(pos[0] - pivot[0], pos[1] - pivot[1])
    prev_pos, prev_time, prev_angle = history[-1]
    dt = time - prev_time
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
    
    # Fit
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
    print("Manuscript figure saved as pendulum_fit.pdf")
    plt.show()

def run_pendulum(model_path):
    global tracking_active, is_paused, pivot_point, cm_per_pixel, calibration_points
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
        det_center = None
        avg_T = np.mean(period_measurements) if period_measurements else 0.0

        if tracking_active and not is_paused and pivot_point:
            # Inference at 1024px for tiny ball accuracy
            results = model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False, imgsz=1024)
            if results and results[0].boxes:
                best = max(results[0].boxes, key=lambda b: b.conf[0])
                box = best.xyxy[0]
                center = (int((box[0]+box[2])/2), int((box[1]+box[3])/2))
                props = calculate_pendulum_properties(center, curr_t, pendulum_history, pivot_point)
                pendulum_history.append((center, curr_t, props['angle']))
                detect_period(props['angle'], curr_t)
                det_center = (center[0], center[1], int(box[2]-box[0]))

        disp = frame.copy()
        if pivot_point: cv2.drawMarker(disp, pivot_point, (0, 255, 255), cv2.MARKER_CROSS, 20, 2)
        if det_center: cv2.circle(disp, det_center[:2], max(5, int(det_center[2]/2)), (0, 255, 0), 2)
        if tracking_active and len(pendulum_history) > 1:
            for i in range(1, len(pendulum_history)): cv2.line(disp, pendulum_history[i-1][0], pendulum_history[i][0], (0, 0, 255), 2)
            cv2.line(disp, pivot_point, pendulum_history[-1][0], (255, 255, 255), 1)

        status = "CALIBRATE" if cm_per_pixel is None else ("SET PIVOT" if pivot_point is None else "READY")
        cv2.putText(disp, f"S:{status} | Scale: {'OK' if cm_per_pixel else 'NO'} | S:Start G:Graph R:Reset Q:Quit", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw Calibration visuals if needed
        if cm_per_pixel is None:
            for pt in calibration_points:
                cv2.circle(disp, pt, 5, (0, 0, 255), -1)
            if len(calibration_points) == 1:
                cv2.putText(disp, "Click second point for 10cm scale", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            else:
                cv2.putText(disp, "CALIBRATION REQUIRED: Click two points 10cm apart", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        cv2.imshow("Pendulum Analyzer", disp)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        elif key == ord('s') and pivot_point: 
            tracking_active = not tracking_active
            if tracking_active: pendulum_history.clear(); period_measurements.clear()
        elif key == ord('p'): is_paused = not is_paused
        elif key == ord('g') and is_paused: plot_pendulum_data(pendulum_history, avg_T)
        elif key == ord('r'): 
            pivot_point = None; cm_per_pixel = None; calibration_points.clear(); pendulum_history.clear()

    cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL)
    args = parser.parse_args()
    run_pendulum(args.model)
