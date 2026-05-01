import cv2
import numpy as np

class PerspectiveCalibration:
    def __init__(self, real_width_cm=10.0, real_height_cm=10.0):
        self.points = []
        self.real_width = real_width_cm
        self.real_height = real_height_cm
        self.matrix = None

    def add_point(self, x, y):
        if len(self.points) < 4:
            self.points.append((x, y))
            if len(self.points) == 4:
                self._compute_matrix()
                return True
        return False

    def _compute_matrix(self):
        # Sort points: top-left, top-right, bottom-right, bottom-left
        pts = np.array(self.points, dtype="float32")
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)
        
        rect = np.zeros((4, 2), dtype="float32")
        rect[0] = pts[np.argmin(s)]       # top-left
        rect[2] = pts[np.argmax(s)]       # bottom-right
        rect[1] = pts[np.argmin(diff)]    # top-right
        rect[3] = pts[np.argmax(diff)]    # bottom-left
        
        dst = np.array([
            [0, 0],
            [self.real_width, 0],
            [self.real_width, self.real_height],
            [0, self.real_height]], dtype="float32")
        
        self.matrix = cv2.getPerspectiveTransform(rect, dst)

    def map_point(self, x, y):
        if self.matrix is None:
            return None
        p = np.array([[[x, y]]], dtype="float32")
        p_transformed = cv2.perspectiveTransform(p, self.matrix)
        return p_transformed[0][0] # returns np.array([x_cm, y_cm])

    def is_calibrated(self):
        return self.matrix is not None

    def reset(self):
        self.points = []
        self.matrix = None

    def draw_info(self, frame):
        for pt in self.points:
            cv2.circle(frame, pt, 5, (0, 0, 255), -1)
        
        if len(self.points) < 4:
            cv2.putText(frame, f"Calibration: Click point {len(self.points)+1}/4 (Define {self.real_width}x{self.real_height}cm rectangle)", 
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        else:
            pts = np.array(self.points, np.int32).reshape((-1, 1, 2))
            s = pts.sum(axis=2).flatten()
            diff = np.diff(pts, axis=2).flatten()
            rect = [pts[np.argmin(s)][0], pts[np.argmin(diff)][0], pts[np.argmax(s)][0], pts[np.argmax(diff)][0]]
            rect = np.array(rect, np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [rect], True, (0, 255, 0), 2)

def calculate_velocity_cm_s(p1_cm, t1, p2_cm, t2):
    dt = t2 - t1
    if dt <= 1e-6: return np.array([0.0, 0.0])
    return (p2_cm - p1_cm) / dt

def calculate_restitution(v1_i, v2_i, v1_f, v2_f):
    rel_v_i = np.linalg.norm(v1_i - v2_i)
    rel_v_f = np.linalg.norm(v1_f - v2_f)
    if rel_v_i < 1e-6: return 0
    return rel_v_f / rel_v_i
