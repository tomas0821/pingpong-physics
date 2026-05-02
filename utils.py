import cv2
import numpy as np

class PerspectiveCalibration:
    def __init__(self, real_width_cm=10.0, real_height_cm=10.0):
        self.points = []
        self.real_width = real_width_cm
        self.real_height = real_height_cm
        self.matrix = None
        self.inv_matrix = None

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
        self.inv_matrix = cv2.getPerspectiveTransform(dst, rect)

    def map_point(self, x, y):
        if self.matrix is None:
            return None
        p = np.array([[[x, y]]], dtype="float32")
        p_transformed = cv2.perspectiveTransform(p, self.matrix)
        return p_transformed[0][0]

    def map_back(self, x_cm, y_cm):
        if self.inv_matrix is None:
            return None
        p = np.array([[[x_cm, y_cm]]], dtype="float32")
        p_px = cv2.perspectiveTransform(p, self.inv_matrix)
        return tuple(p_px[0][0].astype(int))

    def is_calibrated(self):
        return self.matrix is not None

    def reset(self):
        self.points = []
        self.matrix = None
        self.inv_matrix = None

    def draw_info(self, frame):
        for pt in self.points:
            cv2.circle(frame, pt, 5, (0, 0, 255), -1)
        
        if len(self.points) < 4:
            cv2.putText(frame, f"Calibration: Click point {len(self.points)+1}/4 (Define {self.real_width}x{self.real_height}cm area)", 
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        else:
            self.draw_grid(frame)

    def draw_grid(self, frame):
        if self.inv_matrix is None: return
        # Draw a 5x5 grid within the calibrated area
        steps = 5
        for i in range(steps + 1):
            # Vertical lines
            x_cm = (self.real_width / steps) * i
            p1 = self.map_back(x_cm, 0)
            p2 = self.map_back(x_cm, self.real_height)
            cv2.line(frame, p1, p2, (0, 255, 0), 1)
            
            # Horizontal lines
            y_cm = (self.real_height / steps) * i
            p3 = self.map_back(0, y_cm)
            p4 = self.map_back(self.real_width, y_cm)
            cv2.line(frame, p3, p4, (0, 255, 0), 1)

def calculate_restitution(v1, v2):
    rel_v_i = np.linalg.norm(v1)
    rel_v_f = np.linalg.norm(v2)
    if rel_v_i < 1e-6: return 0
    return rel_v_f / rel_v_i
