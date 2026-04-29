import cv2
from ultralytics import YOLO
import numpy as np
import time
from collections import deque
import matplotlib.pyplot as plt
from scipy.stats import linregress 
import argparse
import os

# ---------------- Configuration ----------------
SCALE_LENGTH_CM = 10.0
# Use OpenVINO folder if it exists, otherwise fallback to .pt
DEFAULT_MODEL = "best_openvino_model" if os.path.exists("best_openvino_model") else "best.pt"
CAMERA_INDEX = 0
TARGET_WIDTH, TARGET_HEIGHT = 1280, 720 # Increased resolution for 1024px model
TARGET_FPS = 60
CONFIDENCE_THRESHOLD = 0.6 # Our new model is very confident
POINT_HISTORY_LENGTH = 200 
MATCHING_MAX_DISTANCE_PX = 150
CLICK_SELECTION_THRESHOLD_PX = 20 

# --- State Variables ---
calibration_points = []
cm_per_pixel = None
tracking_active = False
is_paused = False
measurement_mode = False 
selection_points = [] 
measurement_result = None 

# --- Ball States ---
point_history_1 = deque(maxlen=POINT_HISTORY_LENGTH)
last_pos_1 = None
point_history_2 = deque(maxlen=POINT_HISTORY_LENGTH)
last_pos_2 = None

def onMouse(event, x, y, flags, param):
    global calibration_points, cm_per_pixel, selection_points, measurement_result
    if cm_per_pixel is None and event == cv2.EVENT_LBUTTONDOWN:
        if len(calibration_points) < 2:
            calibration_points.append((x, y))
            if len(calibration_points) == 2:
                pixel_dist = np.linalg.norm(np.array(calibration_points[0]) - np.array(calibration_points[1]))
                if pixel_dist > 0:
                    cm_per_pixel = SCALE_LENGTH_CM / pixel_dist
                    print(f"Calibración Completa! Escala: {cm_per_pixel:.4f} cm/pixel")
        return

    if measurement_mode and event == cv2.EVENT_LBUTTONDOWN:
        click_pos = (x, y)
        closest = find_closest_point(click_pos, point_history_1, point_history_2)
        if closest:
            selection_points.append(closest)
            if len(selection_points) == 2:
                if selection_points[0]['ball_id'] == selection_points[1]['ball_id']:
                    s, e = (selection_points[0], selection_points[1]) if selection_points[0]['index'] < selection_points[1]['index'] else (selection_points[1], selection_points[0])
                    history = point_history_1 if s['ball_id'] == 1 else point_history_2
                    v = calculate_average_velocity(history, s['index'], e['index'])
                    if v:
                        measurement_result = {'speed': v['speed'], 'vx': v['vx'], 'vy': v['vy'], 'pos': e['pos'], 'start_sel': s['pos'], 'end_sel': e['pos']}
                        plot_trajectory_data(history, s['index'], e['index'], s['ball_id'])
                selection_points = []

def distance(p1, p2):
    if p1 is None or p2 is None: return float('inf')
    return np.linalg.norm(np.array(p1) - np.array(p2))

def find_closest_point(click_pos, h1, h2):
    min_dist, closest = float('inf'), None
    for ball_id, history in [(1, h1), (2, h2)]:
        for i, (pos, t) in enumerate(history):
            d = distance(click_pos, pos)
            if d < min_dist:
                min_dist, closest = d, {'pos': pos, 'index': i, 'time': t, 'ball_id': ball_id}
    return closest if min_dist < CLICK_SELECTION_THRESHOLD_PX else None

def calculate_average_velocity(history, start_idx, end_idx):
    if cm_per_pixel is None or start_idx >= end_idx: return None
    (p1, t1), (p2, t2) = history[start_idx], history[end_idx]
    dt = t2 - t1
    if dt <= 1e-6: return None
    v = (np.array(p2) - np.array(p1)) * cm_per_pixel / dt
    return {'speed': np.linalg.norm(v), 'vx': v[0], 'vy': v[1]}

def plot_trajectory_data(history, start_idx, end_idx, ball_id):
    segment = list(history)[start_idx : end_idx + 1]
    t = np.array([pt[1] - segment[0][1] for pt in segment])
    x = np.array([pt[0][0] * cm_per_pixel for pt in segment])
    y = np.array([pt[0][1] * cm_per_pixel for pt in segment])
    sx, ix, rx, _, _ = linregress(t, x)
    sy, iy, ry, _, _ = linregress(t, y)
    plt.figure(figsize=(8, 6))
    plt.subplot(211); plt.scatter(t, x); plt.plot(t, sx*t+ix, 'r', label=f'Vx={sx:.2f} (R2={rx**2:.2f})'); plt.legend(); plt.grid()
    plt.subplot(212); plt.scatter(t, y); plt.plot(t, sy*t+iy, 'g', label=f'Vy={sy:.2f} (R2={ry**2:.2f})'); plt.legend(); plt.grid()
    plt.show()

def match_detections(detections, last1, last2):
    d1, d2 = None, None
    if not detections: return None, None
    detections.sort(key=lambda x: x[2], reverse=True) # Sort by width/confidence
    dets = detections[:2]
    if len(dets) == 1:
        dist1, dist2 = distance(dets[0][:2], last1), distance(dets[0][:2], last2)
        if last1 is None and last2 is None: d1 = dets[0]
        elif dist1 < dist2 and dist1 < MATCHING_MAX_DISTANCE_PX: d1 = dets[0]
        elif dist2 <= dist1 and dist2 < MATCHING_MAX_DISTANCE_PX: d2 = dets[0]
    elif len(dets) == 2:
        pA, pB = dets[0][:2], dets[1][:2]
        if last1 is None and last2 is None: d1, d2 = (dets[0], dets[1]) if pA[0] < pB[0] else (dets[1], dets[0])
        else:
            if (distance(pA, last1) + distance(pB, last2)) <= (distance(pA, last2) + distance(pB, last1)):
                if distance(pA, last1) < MATCHING_MAX_DISTANCE_PX: d1 = dets[0]
                if distance(pB, last2) < MATCHING_MAX_DISTANCE_PX: d2 = dets[1]
            else:
                if distance(pA, last2) < MATCHING_MAX_DISTANCE_PX: d2 = dets[0]
                if distance(pB, last1) < MATCHING_MAX_DISTANCE_PX: d1 = dets[1]
    return d1, d2

def run_collisions(model_path):
    global tracking_active, is_paused, measurement_mode, selection_points, measurement_result, cm_per_pixel, last_pos_1, last_pos_2
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

        if tracking_active and not is_paused:
            # Inference at 1024 for small ball precision
            results = model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False, imgsz=1024)
            dets = [(int((b.xyxy[0][0]+b.xyxy[0][2])/2), int((b.xyxy[0][1]+b.xyxy[0][3])/2), int(b.xyxy[0][2]-b.xyxy[0][0])) for b in results[0].boxes]
            m1, m2 = match_detections(dets, last_pos_1, last_pos_2)
            
            if m1: 
                point_history_1.append((m1[:2], curr_t)); last_pos_1 = m1[:2]; draw_centers[1] = m1
            else: last_pos_1 = None
            if m2: 
                point_history_2.append((m2[:2], curr_t)); last_pos_2 = m2[:2]; draw_centers[2] = m2
            else: last_pos_2 = None

        disp = frame.copy()
        for i, (pts, color) in enumerate([(point_history_1, (0,0,255)), (point_history_2, (0,255,0))], 1):
            for j in range(1, len(pts)): cv2.line(disp, pts[j-1][0], pts[j][0], color, 2)
            if i in draw_centers: 
                x,y,w = draw_centers[i]
                cv2.circle(disp, (x,y), max(5, int(w/2)), color, 2)
                cv2.putText(disp, str(i), (x+10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        if measurement_result:
            r = measurement_result
            cv2.line(disp, r['start_sel'], r['end_sel'], (255, 255, 0), 2)
            cv2.putText(disp, f"V:{r['speed']:.1f}cm/s", (r['pos'][0]+15, r['pos'][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        status = "CALIBRATE" if cm_per_pixel is None else ("MEASUREMENT MODE" if measurement_mode else "TRACKING")
        cv2.putText(disp, f"S:{status} | Scale: {'OK' if cm_per_pixel else 'NO'} | S:Start P:Pause M:Measure R:Reset Q:Quit", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw Calibration visuals if needed
        if cm_per_pixel is None:
            for pt in calibration_points:
                cv2.circle(disp, pt, 5, (0, 0, 255), -1)
            if len(calibration_points) == 1:
                cv2.putText(disp, "Click second point for 10cm scale", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            else:
                cv2.putText(disp, "CALIBRATION REQUIRED: Click two points 10cm apart", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        cv2.imshow("Collision Analyzer", disp)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        elif key == ord('s'): tracking_active = not tracking_active
        elif key == ord('m'): measurement_mode = not measurement_mode if is_paused else False
        elif key == ord('p'): is_paused = not is_paused
        elif key == ord('r'): 
            point_history_1.clear(); point_history_2.clear(); last_pos_1 = None; last_pos_2 = None; cm_per_pixel = None; calibration_points.clear()

    cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL)
    args = parser.parse_args()
    run_collisions(args.model)
