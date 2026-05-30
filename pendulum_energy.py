import cv2
from ultralytics import YOLO
import numpy as np
import time
import math
from collections import deque
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
import argparse
import os
from utils import LineCalibration

# ---------------- Configuration ----------------
DEFAULT_SCALE_CM = 30.0
DEFAULT_MODEL = "best_openvino_model" if os.path.exists("best_openvino_model") else "best.pt"
CAMERA_INDEX = 0
TARGET_WIDTH, TARGET_HEIGHT = 1280, 720
CONFIDENCE_THRESHOLD = 0.6
POINT_HISTORY_LENGTH = 500
G_CM_S2 = 981.0  # cm/s²
VELOCITY_SMOOTH_WINDOW = 5  # frames used to smooth velocity estimate

# --- State Variables ---
calib = LineCalibration(DEFAULT_SCALE_CM)
pivot_point_px = None
pivot_point_cm = None
tracking_active = False
is_paused = False
# Each entry: (pos_cm: np.array, t: float, v: float, h_cm: float, ke: float, pe: float, e: float)
energy_history = deque(maxlen=POINT_HISTORY_LENGTH)
pendulum_length_cm = None  # locked after first few detections
E0 = None  # reference energy (first valid frame after tracking starts)
angle_history = deque(maxlen=POINT_HISTORY_LENGTH)


def damped_oscillation(t, A, beta, omega, phi):
    return A * np.exp(-beta * t) * np.cos(omega * t + phi)


def fit_g_from_omega():
    """Fit θ(t)=A·e^{-βt}·cos(ωt+φ) to the full angle history; return (g_m_s2, g_sigma) or None."""
    if pendulum_length_cm is None or len(angle_history) < 30:
        return None
    data = list(angle_history)
    t_arr = np.array([d[0] for d in data])
    theta  = np.array([d[1] for d in data])
    t_arr  = t_arr - t_arr[0]

    A0 = float(np.max(np.abs(theta)))
    if A0 < 0.05:
        return None

    crossings = [t_arr[i] for i in range(1, len(theta)) if theta[i-1] * theta[i] < 0]
    if len(crossings) >= 2:
        T_est  = 2.0 * float(np.median(np.diff(crossings)))
        omega0 = 2.0 * np.pi / T_est
    else:
        omega0 = 5.0

    try:
        popt, pcov = curve_fit(
            damped_oscillation, t_arr, theta,
            p0=[A0, 0.05, omega0, 0.0],
            bounds=([0, 0, 0.3, -np.pi], [np.pi, 5.0, 50.0, np.pi]),
            maxfev=10000,
        )
        omega       = float(popt[2])
        sigma_omega = float(np.sqrt(np.diag(pcov)[2]))
        g       = omega**2 * pendulum_length_cm / 100.0
        g_sigma = 2.0 * omega * sigma_omega * pendulum_length_cm / 100.0
        return g, g_sigma
    except Exception:
        return None


def onMouse(event, x, y, flags, param):
    global calib, pivot_point_px, pivot_point_cm
    if not calib.is_calibrated():
        if event == cv2.EVENT_LBUTTONDOWN:
            calib.add_point(x, y)
        return
    if pivot_point_px is None and event == cv2.EVENT_LBUTTONDOWN:
        pivot_point_px = (x, y)
        pivot_point_cm = calib.map_point(x, y)
        print(f"Pivot set at PX:{pivot_point_px}  CM:{pivot_point_cm}")


def compute_energy(pos_cm, t):
    """
    Returns (v_cm_s, h_cm, ke_specific, pe_specific, e_specific) or None.
    All energy quantities are specific (per unit mass), in cm²/s².
    h is height above the lowest point of the swing (directly below pivot).
    v is computed from the last VELOCITY_SMOOTH_WINDOW frames.
    """
    global pendulum_length_cm

    if pivot_point_cm is None or len(energy_history) < 2:
        return None

    # Lock pendulum length from distance pivot→ball (average over first 10 frames)
    dist = float(np.linalg.norm(pos_cm - np.array(pivot_point_cm)))
    if pendulum_length_cm is None:
        if len(energy_history) < 10:
            return None
        L_samples = [
            float(np.linalg.norm(entry[0] - np.array(pivot_point_cm)))
            for entry in energy_history
        ]
        pendulum_length_cm = float(np.mean(L_samples))
        print(f"Pendulum length locked: {pendulum_length_cm:.2f} cm")

    # Height above lowest point of swing
    # Lowest point y_cm = pivot_y + L  (y increases downward in calibrated space)
    lowest_y = pivot_point_cm[1] + pendulum_length_cm
    h_cm = max(0.0, lowest_y - pos_cm[1])
    angle_history.append((t, math.atan2(float(pos_cm[0] - pivot_point_cm[0]),
                                        float(pos_cm[1] - pivot_point_cm[1]))))

    # Linear velocity via finite differences over a smoothing window
    # Average frame-to-frame speeds (not net displacement) so direction reversals
    # inside the window don't cancel out velocity near the turning points.
    window = list(energy_history)[max(0, len(energy_history) - VELOCITY_SMOOTH_WINDOW):]
    if len(window) < 2:
        return None
    speeds = []
    for i in range(1, len(window)):
        p_prev, t_prev = window[i-1][0], window[i-1][1]
        p_curr, t_curr = window[i][0], window[i][1]
        dt_i = t_curr - t_prev
        if dt_i > 1e-6:
            speeds.append(float(np.linalg.norm(p_curr - p_prev) / dt_i))
    if not speeds:
        return None
    v = float(np.mean(speeds))  # cm/s

    ke = 0.5 * v**2
    pe = G_CM_S2 * h_cm
    e = ke + pe
    return v, h_cm, ke, pe, e


def plot_energy(history):
    if len(history) < 20:
        print("Not enough data to plot.")
        return

    data = list(history)
    t0 = data[0][1]
    t  = np.array([d[1] - t0 for d in data])
    v  = np.array([d[2]       for d in data])
    h  = np.array([d[3]       for d in data])
    ke = np.array([d[4]       for d in data])
    pe = np.array([d[5]       for d in data])
    e  = np.array([d[6]       for d in data])

    # --- Peak detection ---
    # Bottom crossings: local maxima of linear speed
    v_peak_idx, _ = find_peaks(v, prominence=np.std(v) * 0.4, distance=5)
    # Turning points: local maxima of height
    h_peak_idx, _ = find_peaks(h, prominence=np.std(h) * 0.3, distance=5)

    # --- Extract g from paired (v_bottom, h_top) adjacent events ---
    # For each height peak find the nearest speed peak; keep only close-in-time pairs
    g_estimates, g_times = [], []
    for hi in h_peak_idx:
        if h[hi] < 0.5:          # ignore turning points < 0.5 cm high
            continue
        if len(v_peak_idx) == 0:
            continue
        vi = v_peak_idx[np.argmin(np.abs(v_peak_idx - hi))]
        if abs(t[hi] - t[vi]) > 2.0:   # more than 2 s apart → skip
            continue
        if v[vi] < 1.0:                 # ball barely moving → skip
            continue
        g_cm = v[vi] ** 2 / (2.0 * h[hi])   # cm/s²
        g_estimates.append(g_cm / 100.0)     # → m/s²
        g_times.append((t[hi] + t[vi]) / 2.0)

    # --- Energy at each bottom crossing (for loss-per-cycle) ---
    e_bottom = e[v_peak_idx] if len(v_peak_idx) else np.array([])
    t_bottom = t[v_peak_idx] if len(v_peak_idx) else np.array([])

    # --- Plot ---
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(11, 11), sharex=True)

    # Panel 1 — energy components
    ax1.plot(t, ke, 'r-',  label='KE/m  (½v²)',  linewidth=1.2)
    ax1.plot(t, pe, 'b-',  label='PE/m  (gh)',    linewidth=1.2)
    ax1.plot(t, e,  'k-',  label='Total E/m',     linewidth=1.5)
    ax1.set_ylabel('Specific energy (cm²/s²)')
    ax1.set_title('Energy Conservation — Pendulum')
    ax1.legend(); ax1.grid()

    # Panel 2 — energy retention
    if e[0] > 0:
        ax2.plot(t, e / e[0] * 100, 'g-', linewidth=1.2, label='E(t)/E₀')
    if len(e_bottom) >= 2:
        ax2.scatter(t_bottom, e_bottom / e_bottom[0] * 100,
                    color='k', zorder=5, s=30, label='E at bottom crossings')
    ax2.axhline(100, color='k', linestyle='--', linewidth=0.8)
    ax2.set_ylabel('E(t) / E₀  (%)')
    ax2.set_title('Energy Retention')
    ax2.legend(); ax2.grid()

    # Panel 3 — g extraction: ω²·L curve fit + v²/2h scatter
    g_fit  = fit_g_from_omega()
    g_arr  = np.array(g_estimates) if g_estimates else None
    g_mean = float(np.mean(g_arr)) if g_arr is not None else None
    g_std  = float(np.std(g_arr))  if g_arr is not None else None

    if g_fit is not None or g_arr is not None:
        ax3.axhline(9.81, color='r', linestyle='--', linewidth=1.5, label='g theoretical = 9.81 m/s²')

        if g_arr is not None:
            pct_scatter = abs(g_mean - 9.81) / 9.81 * 100
            ax3.scatter(g_times, g_arr, color='purple', zorder=5, s=40, alpha=0.6,
                        label='v²/2h per half-swing')
            ax3.axhline(g_mean, color='purple', linestyle=':', linewidth=1.2,
                        label=f'v²/2h mean = {g_mean:.2f} ± {g_std:.2f} m/s²')
            ax3.fill_between([t[0], t[-1]], g_mean - g_std, g_mean + g_std,
                             alpha=0.10, color='purple')

        if g_fit is not None:
            g_val, g_sig = g_fit
            pct_fit = abs(g_val - 9.81) / 9.81 * 100
            ax3.axhline(g_val, color='steelblue', linestyle='-', linewidth=2.0,
                        label=f'ω²·L fit = {g_val:.3f} ± {g_sig:.3f} m/s²  (err {pct_fit:.1f}%)')
            ax3.fill_between([t[0], t[-1]], g_val - g_sig, g_val + g_sig,
                             alpha=0.20, color='steelblue')
            ax3.set_title(f'g (ω²·L fit): {g_val:.3f} ± {g_sig:.3f} m/s²  |  Error: {pct_fit:.1f}%')
        else:
            ax3.set_title(f'g (v²/2h): {g_mean:.2f} ± {g_std:.2f} m/s²  |  Error: {pct_scatter:.1f}%')

        ax3.set_ylabel('g  (m/s²)')
        ax3.legend(); ax3.grid()

        if g_fit is not None:
            g_val, g_sig = g_fit
            pct_fit = abs(g_val - 9.81) / 9.81 * 100
            print("\n--- Gravity Extraction (ω²·L fit) ---")
            print(f"  g = {g_val:.4f} ± {g_sig:.4f} m/s²")
            print(f"  Theoretical: 9.8100 m/s²")
            print(f"  % Error    : {pct_fit:.2f}%")
        if g_arr is not None:
            pct_scatter = abs(g_mean - 9.81) / 9.81 * 100
            print("\n--- Gravity Extraction (v²/2h method) ---")
            print(f"  g = {g_mean:.3f} ± {g_std:.3f} m/s²")
            print(f"  % Error    : {pct_scatter:.2f}%")
            print(f"  N samples  : {len(g_estimates)}")
    else:
        ax3.text(0.5, 0.5, 'Not enough swing cycles detected.\nLet the pendulum complete at least 3 full swings.',
                 ha='center', va='center', transform=ax3.transAxes, fontsize=11)
        ax3.set_ylabel('g  (m/s²)')

    ax3.set_xlabel('Time (s)')

    # Console summary — energy loss
    if len(e_bottom) >= 2:
        loss = np.diff(e_bottom) / e_bottom[:-1] * 100
        print(f"\n--- Energy Loss ---")
        print(f"  Mean loss per bottom crossing : {np.mean(np.abs(loss)):.1f}%")
        print(f"  Total E retained (first→last) : {e_bottom[-1] / e_bottom[0] * 100:.1f}%")

    plt.tight_layout()
    plt.savefig('energy_conservation.pdf', bbox_inches='tight')
    print("\nPlot saved to energy_conservation.pdf")
    plt.show()


def run_energy(model_path):
    global tracking_active, is_paused, pivot_point_px, pivot_point_cm
    global calib, energy_history, angle_history, pendulum_length_cm, E0

    model = YOLO(model_path)
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, TARGET_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_HEIGHT)

    cv2.namedWindow("Energy Conservation")
    cv2.setMouseCallback("Energy Conservation", onMouse)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        curr_t = time.time()
        det_px = None

        if tracking_active and not is_paused and pivot_point_cm is not None:
            results = model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False, imgsz=1024)
            if results and results[0].boxes:
                best = max(results[0].boxes, key=lambda b: b.conf[0])
                box = best.xyxy[0]
                center_px = (int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2))
                pos_cm = calib.map_point(center_px[0], center_px[1])
                if pos_cm is not None:
                    pos_cm = np.array(pos_cm)
                    result = compute_energy(pos_cm, curr_t)
                    if result is not None:
                        v, h_cm, ke, pe, e = result
                        if E0 is None:
                            E0 = e
                        energy_history.append((pos_cm, curr_t, v, h_cm, ke, pe, e))
                        det_px = (center_px, int(box[2] - box[0]))
                    else:
                        # Still accumulate position before length is locked
                        energy_history.append((pos_cm, curr_t, 0, 0, 0, 0, 0))
                        det_px = (center_px, int(box[2] - box[0]))

        disp = frame.copy()

        # Draw trajectory
        pts = list(energy_history)
        for i in range(1, len(pts)):
            cv2.line(disp, tuple(pts[i-1][0].astype(int)), tuple(pts[i][0].astype(int)), (0, 200, 255), 2)

        # Draw pivot
        if pivot_point_px:
            cv2.drawMarker(disp, pivot_point_px, (0, 255, 255), cv2.MARKER_CROSS, 20, 2)

        # Draw detected ball
        if det_px:
            c_px, w_px = det_px
            cv2.circle(disp, c_px, max(5, int(w_px / 2)), (0, 255, 0), 2)

        # Live energy readout
        if energy_history and energy_history[-1][6] > 0:
            last = energy_history[-1]
            v, h_cm, ke, pe, e = last[2], last[3], last[4], last[5], last[6]
            retention = (e / E0 * 100) if E0 and E0 > 0 else 0.0
            cv2.putText(disp, f"v={v:.1f} cm/s  h={h_cm:.2f} cm", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(disp, f"KE/m={ke:.0f}  PE/m={pe:.0f}  E/m={e:.0f} cm2/s2", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(disp, f"E retained: {retention:.1f}%", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        status = ("CALIBRATE" if not calib.is_calibrated()
                  else "SET PIVOT" if pivot_point_px is None
                  else "TRACKING" if (tracking_active and not is_paused)
                  else "PAUSED" if is_paused
                  else "READY")
        cv2.putText(disp, f"{status} | S:Start  P:Pause  G:Plot  R:Reset  Q:Quit",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        calib.draw_info(disp)

        cv2.imshow("Energy Conservation", disp)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s') and pivot_point_cm is not None:
            tracking_active = not tracking_active
            if tracking_active:
                energy_history.clear()
                angle_history.clear()
                pendulum_length_cm = None
                E0 = None
        elif key == ord('p'):
            is_paused = not is_paused
        elif key == ord('g') and is_paused:
            plot_energy(energy_history)
        elif key == ord('r'):
            pivot_point_px = None
            pivot_point_cm = None
            calib.reset()
            energy_history.clear()
            angle_history.clear()
            pendulum_length_cm = None
            E0 = None

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL)
    args = parser.parse_args()
    run_energy(args.model)
