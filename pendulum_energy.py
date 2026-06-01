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
from utils import LineCalibration

# ---------------- Configuration ----------------
DEFAULT_SCALE_CM = 20.0
DEFAULT_MODEL = "best_openvino_model" if os.path.exists("best_openvino_model") else "best.pt"
CAMERA_INDEX = 0
TARGET_WIDTH, TARGET_HEIGHT = 1280, 720
CONFIDENCE_THRESHOLD = 0.6
POINT_HISTORY_LENGTH = 500
G_CM_S2 = 981.0  # cm/s²
ANGLE_FIT_WINDOW = 15  # frames used for polynomial angular-velocity fit

# --- State variables ---
calib = LineCalibration(DEFAULT_SCALE_CM)
pivot_point_px = None
pivot_point_cm = None
tracking_active = False
is_paused = False
# Each entry: (pos_cm, t, v, h_cm, ke, pe, e)
energy_history = deque(maxlen=POINT_HISTORY_LENGTH)
# Angle history for polynomial fit: (t, angle_rad)
angle_history = deque(maxlen=POINT_HISTORY_LENGTH)
pendulum_length_cm = None
manual_length_cm   = None   # set via --length; bypasses pixel-based estimation
E0 = None


def damped_oscillation(t, A, beta, omega, phi):
    return A * np.exp(-beta * t) * np.cos(omega * t + phi)


def _g_from_energy_variance(popt, L_cm):
    """
    Estimate g by minimising the variance of E(t) = K + g·H over the fit curves.

    For a damped pendulum E(t) = E₀·e^{-2βt}, so K + g·H multiplied by e^{2βt}
    should be constant. Find g that minimises that variance:

        g = -Cov[K·e^{2βt}, H·e^{2βt}] / Var[H·e^{2βt}]

    Uses all points of the smooth curves — not just peaks.
    Returns (g_m_s2, g_sigma) or None.
    """
    if L_cm is None or len(angle_history) < 30:
        return None

    A, beta, omega, phi = popt
    ang_data = list(angle_history)
    t_arr    = np.array([d[0] for d in ang_data]) - ang_data[0][0]
    t_sm     = np.linspace(0, t_arr[-1], 2000)

    theta_sm  = damped_oscillation(t_sm, *popt)
    dtheta_dt = (-A * beta  * np.exp(-beta * t_sm) * np.cos(omega * t_sm + phi)
                 - A * omega * np.exp(-beta * t_sm) * np.sin(omega * t_sm + phi))

    K = 0.5 * (L_cm * np.abs(dtheta_dt)) ** 2   # cm²/s²
    H = L_cm * (1.0 - np.cos(theta_sm))          # cm

    detrend = np.exp(2.0 * beta * t_sm)
    K_dt = K * detrend
    H_dt = H * detrend

    var_H = float(np.var(H_dt))
    if var_H < 1e-10:
        return None

    g_cm_s2 = float(-np.cov(K_dt, H_dt)[0, 1] / var_H)
    g_m_s2  = g_cm_s2 / 100.0

    # Uncertainty from OLS residuals of K_dt = -g·H_dt + E₀
    E0_hat    = float(np.mean(K_dt + g_cm_s2 * H_dt))
    residuals = K_dt + g_cm_s2 * H_dt - E0_hat
    sigma_g_m = float(np.std(residuals) / (100.0 * np.sqrt((len(H_dt) - 1) * var_H)))

    return g_m_s2, sigma_g_m


def fit_g_from_omega():
    """Fit θ(t)=A·e^{-βt}·cos(ωt+φ) to full angle history; return (g_m_s2, g_sigma, popt, t_start) or None."""
    if pendulum_length_cm is None or len(angle_history) < 30:
        return None
    data = list(angle_history)
    t_arr = np.array([d[0] for d in data])
    theta  = np.array([d[1] for d in data])
    t_start_abs = float(t_arr[0])
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
        A_fit       = float(popt[0])
        sigma_omega = float(np.sqrt(np.diag(pcov)[2]))

        # L: manual if --length was given, else median over all frames
        if manual_length_cm is not None:
            L_cm = manual_length_cm
        elif len(energy_history) >= 10 and pivot_point_cm is not None:
            dists = [float(np.linalg.norm(np.array(e[0]) - np.array(pivot_point_cm)))
                     for e in energy_history]
            L_cm = float(np.median(dists))
        else:
            L_cm = pendulum_length_cm

        # Large-angle correction: T_actual ≈ T_small·(1 + A²/16)
        # → ω_small = ω_fit·(1 + A²/16)
        # → g = ω_small²·L = ω_fit²·L·(1 + A_fit²/16)²
        corr    = (1.0 + A_fit**2 / 16.0) ** 2
        g       = omega**2 * L_cm / 100.0 * corr
        g_sigma = 2.0 * omega * sigma_omega * L_cm / 100.0 * corr

        return g, g_sigma, list(popt), t_start_abs
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
    Returns (v, h_cm, ke, pe, e) or None.

    Velocity is obtained by fitting a polynomial to the recent angle history
    and evaluating its analytical derivative. This forces v→0 at the turning
    points correctly even at low frame rates.
    All energy quantities are specific (per unit mass), in cm²/s².
    """
    global pendulum_length_cm

    if pivot_point_cm is None or len(energy_history) < 2:
        return None

    # Lock pendulum length
    if pendulum_length_cm is None:
        if manual_length_cm is not None:
            pendulum_length_cm = manual_length_cm
            print(f"Pendulum length (manual): {pendulum_length_cm:.2f} cm")
        else:
            if len(energy_history) < 10:
                return None
            L_samples = [
                float(np.linalg.norm(entry[0] - np.array(pivot_point_cm)))
                for entry in energy_history
            ]
            pendulum_length_cm = float(np.mean(L_samples))
            print(f"\n⚠  L estimated from pixels: {pendulum_length_cm:.2f} cm")
            print(f"   If the calibration ruler is NOT in the same plane as the pendulum,")
            print(f"   L will be wrong and g will carry the same proportional error.")
            print(f"   → Measure L physically (pivot → ball centre) and run with:")
            print(f"     python pendulum_energy.py --length <L_cm>\n")

    # Height above lowest point of swing (y increases downward in image space)
    lowest_y = pivot_point_cm[1] + pendulum_length_cm
    h_cm = max(0.0, lowest_y - pos_cm[1])

    # Angle relative to pivot
    dx = float(pos_cm[0] - pivot_point_cm[0])
    dy = float(pos_cm[1] - pivot_point_cm[1])
    angle = math.atan2(dx, dy)
    angle_history.append((t, angle))

    if len(angle_history) < 4:
        return None

    # Use the current half-arc segment: points since the last zero crossing.
    # Fall back to the last ANGLE_FIT_WINDOW points if no crossing yet.
    all_pts = list(angle_history)
    segment_start = 0
    for i in range(len(all_pts) - 1, 0, -1):
        if (all_pts[i][1] >= 0) != (all_pts[i - 1][1] >= 0):
            segment_start = i
            break
    window = all_pts[segment_start:] if (len(all_pts) - segment_start) >= 4 \
        else all_pts[max(0, len(all_pts) - ANGLE_FIT_WINDOW):]

    # Polynomial fit to segment → analytical derivative = ω(t)
    t_win  = np.array([p[0] for p in window])
    a_win  = np.array([p[1] for p in window])
    t0_win = t_win[0]
    t_norm = t_win - t0_win

    try:
        deg    = min(3, len(window) - 1)
        coeffs = np.polyfit(t_norm, a_win, deg=deg)
        omega  = float(np.poly1d(np.polyder(coeffs))(t - t0_win))  # rad/s
    except Exception:
        return None

    v  = pendulum_length_cm * abs(omega)   # cm/s
    ke = 0.5 * v**2
    pe = G_CM_S2 * h_cm
    e  = ke + pe
    return v, h_cm, ke, pe, e


def plot_energy(history):
    if len(history) < 20:
        print("Not enough data to plot.")
        return

    data = list(history)
    t0   = data[0][1]

    fit_result = fit_g_from_omega()
    ang_data   = list(angle_history)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    # --- Panel 1: damped oscillation fit to θ(t) ---
    if ang_data:
        t_ang = np.array([d[0] for d in ang_data]) - t0
        theta  = np.degrees(np.array([d[1] for d in ang_data]))
        ax1.plot(t_ang, theta, 'b.', markersize=2, alpha=0.6, label='θ data')
        if fit_result is not None:
            _, _, popt, t_ang_start = fit_result
            A, beta, omega_fit, phi = popt
            T_fit     = 2 * np.pi / omega_fit
            t_norm_sm = np.linspace(0, t_ang[-1] - t_ang[0], 500)
            theta_sm  = np.degrees(damped_oscillation(t_norm_sm, *popt))
            ax1.plot(t_norm_sm + (t_ang_start - t0), theta_sm, 'g-', linewidth=1.5,
                     label=f'Fit: A={np.degrees(A):.1f}°  β={beta:.3f} s⁻¹'
                           f'  ω={omega_fit:.3f} rad/s  T={T_fit:.3f} s')
    ax1.set_ylabel('Angle θ  (degrees)')
    ax1.set_title('Damped Oscillation Fit  θ(t) = A·e^{−βt}·cos(ωt+φ)')
    ax1.legend(fontsize=8); ax1.grid()

    # --- Panel 2: KE and PE from the fit ---
    g_ev = None
    if fit_result is not None and ang_data and pendulum_length_cm is not None:
        _, _, popt, t_ang_start = fit_result
        A, beta, omega_fit, phi = popt
        t_ang_loc = np.array([d[0] for d in ang_data]) - t0
        t_norm_sm = np.linspace(0, t_ang_loc[-1] - t_ang_loc[0], 500)
        t_main_sm = t_norm_sm + (t_ang_start - t0)

        # L consistent with fit_g_from_omega
        if manual_length_cm is not None:
            L_cm = manual_length_cm
        elif len(energy_history) >= 10 and pivot_point_cm is not None:
            dists = [float(np.linalg.norm(np.array(e[0]) - np.array(pivot_point_cm)))
                     for e in energy_history]
            L_cm = float(np.median(dists))
        else:
            L_cm = pendulum_length_cm

        # g from energy variance minimisation (uses all curve points)
        g_ev     = _g_from_energy_variance(popt, L_cm)
        g_ev_val = g_ev[0] if g_ev is not None else fit_result[0]
        g_cm_use = g_ev_val * 100.0

        theta_sm  = damped_oscillation(t_norm_sm, *popt)
        dtheta_dt = (-A * beta      * np.exp(-beta * t_norm_sm) * np.cos(omega_fit * t_norm_sm + phi)
                     - A * omega_fit * np.exp(-beta * t_norm_sm) * np.sin(omega_fit * t_norm_sm + phi))

        v_sm  = L_cm * np.abs(dtheta_dt)
        h_sm  = L_cm * (1.0 - np.cos(theta_sm))   # exact formula
        ke_sm = 0.5 * v_sm**2
        pe_sm = g_cm_use * h_sm
        e_sm  = ke_sm + pe_sm

        ax2.plot(t_main_sm, ke_sm, 'r-', linewidth=1.5, label='KE/m  (½v²)')
        ax2.plot(t_main_sm, pe_sm, 'b-', linewidth=1.5, label='PE/m  (g·h)')
        ax2.plot(t_main_sm, e_sm,  'k-', linewidth=2.0, label='Total E/m')
        ax2.set_ylabel('Specific energy  (cm²/s²)')

        g_om_val, g_om_sig = fit_result[0], fit_result[1]
        pct_om = abs(g_om_val - 9.81) / 9.81 * 100
        if g_ev is not None:
            g_ev_sig = g_ev[1]
            pct_ev   = abs(g_ev_val - 9.81) / 9.81 * 100
            ax2.set_title(
                f'g (E variance) = {g_ev_val:.3f} ± {g_ev_sig:.3f} m/s²  (err {pct_ev:.1f}%)  '
                f'|  g (ω²·L) = {g_om_val:.3f} ± {g_om_sig:.3f} m/s²  (err {pct_om:.1f}%)'
            )
        else:
            ax2.set_title(
                f'g (ω²·L) = {g_om_val:.3f} ± {g_om_sig:.3f} m/s²  (err {pct_om:.1f}%)'
            )
        ax2.legend(fontsize=8); ax2.grid()
    else:
        ax2.text(0.5, 0.5, 'Fit not available.\nLet the pendulum complete at least 3 full cycles.',
                 ha='center', va='center', transform=ax2.transAxes, fontsize=11)
        ax2.set_ylabel('Specific energy  (cm²/s²)')

    ax2.set_xlabel('Time (s)')

    # Console summary
    if fit_result is not None:
        g_om_val, g_om_sig = fit_result[0], fit_result[1]
        A_fit = fit_result[2][0]
        corr  = (1.0 + A_fit**2 / 16.0) ** 2
        print("\n--- Gravity Extraction ---")
        print(f"  ω²·L (corrected) = {g_om_val:.4f} ± {g_om_sig:.4f} m/s²"
              f"  (err {abs(g_om_val-9.81)/9.81*100:.1f}%)")
        print(f"  ω²·L (raw)       = {g_om_val/corr:.4f} m/s²"
              f"  | correction ×{corr:.4f}  (A₀={np.degrees(A_fit):.1f}°)")
        if g_ev is not None:
            g_ev_val, g_ev_sig = g_ev
            print(f"  E variance       = {g_ev_val:.4f} ± {g_ev_sig:.4f} m/s²"
                  f"  (err {abs(g_ev_val-9.81)/9.81*100:.1f}%)")
        print(f"  Theoretical      = 9.8100 m/s²")

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
                        energy_history.append((pos_cm, curr_t, 0, 0, 0, 0, 0))
                        det_px = (center_px, int(box[2] - box[0]))

        disp = frame.copy()

        pts = list(energy_history)
        for i in range(1, len(pts)):
            cv2.line(disp, tuple(pts[i-1][0].astype(int)), tuple(pts[i][0].astype(int)), (0, 200, 255), 2)

        if pivot_point_px:
            cv2.drawMarker(disp, pivot_point_px, (0, 255, 255), cv2.MARKER_CROSS, 20, 2)

        if det_px:
            c_px, w_px = det_px
            cv2.circle(disp, c_px, max(5, int(w_px / 2)), (0, 255, 0), 2)

        if pendulum_length_cm is not None:
            if manual_length_cm is not None:
                l_color = (0, 255, 0)    # green — reliable
                l_label = f"L={pendulum_length_cm:.1f} cm (manual)"
            else:
                l_color = (0, 0, 255)    # red — warning
                l_label = f"L={pendulum_length_cm:.1f} cm (pixels - use --length!)"
            cv2.putText(disp, l_label, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, l_color, 2)
        if energy_history and energy_history[-1][6] > 0:
            last = energy_history[-1]
            v, h_cm = last[2], last[3]
            cv2.putText(disp, f"v={v:.1f} cm/s  h={h_cm:.2f} cm", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        status = ("CALIBRATE" if not calib.is_calibrated()
                  else "SET PIVOT" if pivot_point_px is None
                  else "TRACKING" if (tracking_active and not is_paused)
                  else "PAUSED"   if is_paused
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
    parser.add_argument('--model',  type=str,   default=DEFAULT_MODEL)
    parser.add_argument('--length', type=float, default=None,
                        help='Pendulum length in cm (pivot to ball centre). '
                             'Recommended when the calibration ruler is not in '
                             'the same plane as the pendulum.')
    args = parser.parse_args()
    manual_length_cm = args.length
    if manual_length_cm is not None:
        print(f"Manual length: {manual_length_cm:.1f} cm")
    run_energy(args.model)
