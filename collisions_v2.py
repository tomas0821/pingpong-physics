import cv2
from ultralytics import YOLO
import numpy as np
import time
from collections import deque
import matplotlib.pyplot as plt
from scipy.stats import linregress 
import argparse
import os
from utils import PerspectiveCalibration, calculate_restitution

# ---------------- Configuration ----------------
DEFAULT_CALIB_WIDTH = 40.0
DEFAULT_CALIB_HEIGHT = 20.0
DEFAULT_MODEL = "best_openvino_model" if os.path.exists("best_openvino_model") else "best.pt"
CAMERA_INDEX = 0
TARGET_WIDTH, TARGET_HEIGHT = 1280, 720
CONFIDENCE_THRESHOLD = 0.6
POINT_HISTORY_LENGTH = 200 
MATCHING_MAX_DISTANCE_PX = 150
CLICK_SELECTION_THRESHOLD_PX = 50 

# --- State Variables ---
calib = PerspectiveCalibration(DEFAULT_CALIB_WIDTH, DEFAULT_CALIB_HEIGHT)
tracking_active = False
is_paused = False
measurement_mode = False 
selection_points = [] 
velocity_results = [] 

# --- Ball States ---
point_history_1 = deque(maxlen=POINT_HISTORY_LENGTH)
last_pos_px_1 = None
point_history_2 = deque(maxlen=POINT_HISTORY_LENGTH)
last_pos_px_2 = None

def onMouse(event, x, y, flags, param):
    global calib, selection_points, velocity_results
    if not calib.is_calibrated():
        if event == cv2.EVENT_LBUTTONDOWN:
            calib.add_point(x, y)
        return

    if measurement_mode and event == cv2.EVENT_LBUTTONDOWN:
        click_pos = (x, y)
        closest = find_closest_point(click_pos, point_history_1, point_history_2)
        if closest:
            # If it's the second point and different ball, replace the first
            if len(selection_points) == 1 and closest['ball_id'] != selection_points[0]['ball_id']:
                selection_points = [closest]
                print(f"Switched to Ball {closest['ball_id']}")
            else:
                selection_points.append(closest)
                print(f"Point {len(selection_points)} selected for Ball {closest['ball_id']}")

            if len(selection_points) == 2:
                s, e = (selection_points[0], selection_points[1]) if selection_points[0]['index'] < selection_points[1]['index'] else (selection_points[1], selection_points[0])
                history = point_history_1 if s['ball_id'] == 1 else point_history_2
                v = calculate_average_velocity(history, s['index'], e['index'])
                if v is not None:
                    res = {'vel': v, 'pos_px': e['pos_px'], 'start_px': s['pos_px'], 'end_px': e['pos_px'], 'ball_id': s['ball_id']}
                    velocity_results.append(res)
                    plot_trajectory_data(history, s['index'], e['index'], s['ball_id'])
                    
                    if len(velocity_results) >= 2:
                        v1 = velocity_results[-2]
                        v2 = velocity_results[-1]
                        if v1['ball_id'] == v2['ball_id']:
                            e_val = np.linalg.norm(v2['vel']) / np.linalg.norm(v1['vel'])
                            print(f"Coefficient of Restitution (Estimated): {e_val:.3f}")
                selection_points = []

def distance_px(p1, p2):
    if p1 is None or p2 is None: return float('inf')
    return np.linalg.norm(np.array(p1) - np.array(p2))

def find_closest_point(click_pos, h1, h2):
    min_dist, closest = float('inf'), None
    for ball_id, history in [(1, h1), (2, h2)]:
        for i, (pos_px, pos_cm, t) in enumerate(history):
            d = distance_px(click_pos, pos_px)
            if d < min_dist:
                min_dist, closest = d, {'pos_px': pos_px, 'index': i, 'time': t, 'ball_id': ball_id}
    return closest if min_dist < CLICK_SELECTION_THRESHOLD_PX else None

def calculate_average_velocity(history, start_idx, end_idx):
    if start_idx >= end_idx: return None
    (p1_px, p1_cm, t1), (p2_px, p2_cm, t2) = history[start_idx], history[end_idx]
    dt = t2 - t1
    if dt <= 1e-6: return None
    v = (p2_cm - p1_cm) / dt
    return v 

def plot_trajectory_data(history, start_idx, end_idx, ball_id):
    segment = list(history)[start_idx : end_idx + 1]
    t = np.array([pt[2] - segment[0][2] for pt in segment])
    x = np.array([pt[1][0] for pt in segment])
    y = np.array([pt[1][1] for pt in segment])
    sx, ix, rx, _, _ = linregress(t, x)
    sy, iy, ry, _, _ = linregress(t, y)
    plt.figure(figsize=(8, 6))
    plt.subplot(211); plt.scatter(t, x); plt.plot(t, sx*t+ix, 'r', label=f'Vx={sx:.2f} (R2={rx**2:.2f})'); plt.legend(); plt.grid()
    plt.subplot(212); plt.scatter(t, y); plt.plot(t, sy*t+iy, 'g', label=f'Vy={sy:.2f} (R2={ry**2:.2f})'); plt.legend(); plt.grid()
    plt.savefig('collision_vectors.pdf', bbox_inches='tight')
    plt.show()

def match_detections(detections, last1_px, last2_px):
    d1, d2 = None, None
    if not detections: return None, None
    dets = sorted(detections, key=lambda x: x[2], reverse=True)[:2]
    
    if len(dets) == 1:
        dist1, dist2 = distance_px(dets[0][:2], last1_px), distance_px(dets[0][:2], last2_px)
        if last1_px is None and last2_px is None: d1 = dets[0]
        elif dist1 < dist2 and dist1 < MATCHING_MAX_DISTANCE_PX: d1 = dets[0]
        elif dist2 <= dist1 and dist2 < MATCHING_MAX_DISTANCE_PX: d2 = dets[0]
    elif len(dets) == 2:
        pA, pB = dets[0][:2], dets[1][:2]
        if last1_px is None and last2_px is None: d1, d2 = (dets[0], dets[1]) if pA[0] < pB[0] else (dets[1], dets[0])
        else:
            if (distance_px(pA, last1_px) + distance_px(pB, last2_px)) <= (distance_px(pA, last2_px) + distance_px(pB, last1_px)):
                if distance_px(pA, last1_px) < MATCHING_MAX_DISTANCE_PX: d1 = dets[0]
                if distance_px(pB, last2_px) < MATCHING_MAX_DISTANCE_PX: d2 = dets[1]
            else:
                if distance_px(pA, last2_px) < MATCHING_MAX_DISTANCE_PX: d2 = dets[0]
                if distance_px(pB, last1_px) < MATCHING_MAX_DISTANCE_PX: d1 = dets[1]
    return d1, d2

def run_collisions(model_path):
    global tracking_active, is_paused, measurement_mode, selection_points, velocity_results, calib, last_pos_px_1, last_pos_px_2
    model = YOLO(model_path)
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, TARGET_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_HEIGHT)
    
    cv2.namedWindow("Collision Analyzer")
    cv2.setMouseCallback("Collision Analyzer", onMouse)
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        curr_t = time.time()
        draw_centers = {}

        if tracking_active and not is_paused and calib.is_calibrated():
            results = model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False, imgsz=1024)
            dets = [(int((b.xyxy[0][0]+b.xyxy[0][2])/2), int((b.xyxy[0][1]+b.xyxy[0][3])/2), int(b.xyxy[0][2]-b.xyxy[0][0])) for b in results[0].boxes]
            m1, m2 = match_detections(dets, last_pos_px_1, last_pos_px_2)
            
            if m1: 
                p_cm = calib.map_point(m1[0], m1[1])
                if p_cm is not None:
                    point_history_1.append((m1[:2], p_cm, curr_t)); last_pos_px_1 = m1[:2]; draw_centers[1] = m1
            else: last_pos_px_1 = None
            if m2: 
                p_cm = calib.map_point(m2[0], m2[1])
                if p_cm is not None:
                    point_history_2.append((m2[:2], p_cm, curr_t)); last_pos_px_2 = m2[:2]; draw_centers[2] = m2
            else: last_pos_px_2 = None

        disp = frame.copy()
        for i, (pts, color) in enumerate([(point_history_1, (0,0,255)), (point_history_2, (0,255,0))], 1):
            for j in range(1, len(pts)): cv2.line(disp, pts[j-1][0], pts[j][0], color, 2)
            if i in draw_centers: 
                x,y,w = draw_centers[i]
                cv2.circle(disp, (x,y), max(5, int(w/2)), color, 2)
                cv2.putText(disp, str(i), (x+10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        for res in velocity_results:
            cv2.line(disp, res['start_px'], res['end_px'], (255, 255, 0), 2)
            speed = np.linalg.norm(res['vel'])
            cv2.putText(disp, f"V:{speed:.1f}cm/s", (res['pos_px'][0]+15, res['pos_px'][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        # Draw pending selection points
        for sel in selection_points:
            cv2.circle(disp, sel['pos_px'], 7, (0, 255, 255), -1)

        status = "CALIBRATE" if not calib.is_calibrated() else ("MEASUREMENT MODE" if measurement_mode else "TRACKING")
        cv2.putText(disp, f"S:{status} | S:Start P:Pause M:Measure C:Clear Meas R:Reset Q:Quit", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        calib.draw_info(disp)
        if calib.is_calibrated() and not tracking_active:
            cv2.putText(disp, "Check the green grid. If distorted, press 'R' to re-calibrate.", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("Collision Analyzer", disp)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        elif key == ord('s'): tracking_active = not tracking_active
        elif key == ord('m'): measurement_mode = not measurement_mode if is_paused else False
        elif key == ord('p'): is_paused = not is_paused
        elif key == ord('c'): # New key to clear measurements only
            velocity_results = []
            selection_points = []
            print("Measurements cleared.")
        elif key == ord('r'): 
            point_history_1.clear(); point_history_2.clear(); last_pos_px_1 = None; last_pos_px_2 = None; calib.reset(); velocity_results = []

    cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL)
    args = parser.parse_args()
    run_collisions(args.model)
