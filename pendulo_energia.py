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

# ---------------- Configuración ----------------
DEFAULT_SCALE_CM = 30.0
DEFAULT_MODEL = "best_openvino_model" if os.path.exists("best_openvino_model") else "best.pt"
CAMERA_INDEX = 0
TARGET_WIDTH, TARGET_HEIGHT = 1280, 720
CONFIDENCE_THRESHOLD = 0.6
POINT_HISTORY_LENGTH = 500
G_CM_S2 = 981.0  # cm/s²
ANGLE_FIT_WINDOW = 15  # detections used for polynomial angular-velocity fit

# --- Variables de estado ---
calib = LineCalibration(DEFAULT_SCALE_CM)
pivot_point_px = None
pivot_point_cm = None
tracking_active = False
is_paused = False
# Cada entrada: (pos_cm, t, v, h_cm, ke, pe, e)
energy_history = deque(maxlen=POINT_HISTORY_LENGTH)
# Historial de ángulos para el ajuste polinomial: (t, angle_rad)
angle_history = deque(maxlen=POINT_HISTORY_LENGTH)
pendulum_length_cm = None
E0 = None


def damped_oscillation(t, A, beta, omega, phi):
    return A * np.exp(-beta * t) * np.cos(omega * t + phi)


def fit_g_from_omega():
    """Ajusta θ(t)=A·e^{-βt}·cos(ωt+φ) al historial completo; devuelve (g_m_s2, g_sigma) o None."""
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
        sigma_omega = float(np.sqrt(np.diag(pcov)[2]))
        g       = omega**2 * pendulum_length_cm / 100.0
        g_sigma = 2.0 * omega * sigma_omega * pendulum_length_cm / 100.0
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
        print(f"Pivote fijado en PX:{pivot_point_px}  CM:{pivot_point_cm}")


def compute_energy(pos_cm, t):
    """
    Devuelve (v, h_cm, ec, ep, e) o None.

    La velocidad se obtiene ajustando un polinomio al historial de ángulos
    recientes y evaluando su derivada analítica. Esto permite que v→0 en los
    extremos del péndulo aunque el FPS sea bajo, a diferencia de calcular
    desplazamiento frame-a-frame.
    Energías específicas (por unidad de masa) en cm²/s².
    """
    global pendulum_length_cm

    if pivot_point_cm is None or len(energy_history) < 2:
        return None

    # Fijar la longitud del péndulo con las primeras 10 detecciones
    if pendulum_length_cm is None:
        if len(energy_history) < 10:
            return None
        L_samples = [
            float(np.linalg.norm(entry[0] - np.array(pivot_point_cm)))
            for entry in energy_history
        ]
        pendulum_length_cm = float(np.mean(L_samples))
        print(f"Longitud del péndulo fijada: {pendulum_length_cm:.2f} cm")

    # Altura sobre el punto más bajo de la oscilación
    lowest_y = pivot_point_cm[1] + pendulum_length_cm
    h_cm = max(0.0, lowest_y - pos_cm[1])

    # Ángulo respecto al pivote
    dx = float(pos_cm[0] - pivot_point_cm[0])
    dy = float(pos_cm[1] - pivot_point_cm[1])
    angle = math.atan2(dx, dy)
    angle_history.append((t, angle))

    if len(angle_history) < 4:
        return None

    # Usar el segmento del medio arco actual: puntos desde el último cruce por
    # cero del ángulo hasta ahora. Si aún no hay cruce, usar los últimos
    # ANGLE_FIT_WINDOW puntos como respaldo.
    all_pts = list(angle_history)
    segment_start = 0
    for i in range(len(all_pts) - 1, 0, -1):
        if (all_pts[i][1] >= 0) != (all_pts[i - 1][1] >= 0):
            segment_start = i
            break
    window = all_pts[segment_start:] if (len(all_pts) - segment_start) >= 4 \
        else all_pts[max(0, len(all_pts) - ANGLE_FIT_WINDOW):]

    # Ajuste polinomial al segmento → derivada analítica = ω(t)
    t_win = np.array([p[0] for p in window])
    a_win = np.array([p[1] for p in window])
    t0 = t_win[0]
    t_norm = t_win - t0  # normalizar para estabilidad numérica

    try:
        deg = min(3, len(window) - 1)
        coeffs = np.polyfit(t_norm, a_win, deg=deg)
        omega = float(np.poly1d(np.polyder(coeffs))(t - t0))  # rad/s
    except Exception:
        return None

    v = pendulum_length_cm * abs(omega)  # cm/s

    ec = 0.5 * v**2
    ep = G_CM_S2 * h_cm
    e = ec + ep
    return v, h_cm, ec, ep, e


def plot_energy(history):
    if len(history) < 20:
        print("Datos insuficientes para graficar.")
        return

    data = list(history)
    t0 = data[0][1]
    t  = np.array([d[1] - t0 for d in data])
    v  = np.array([d[2]       for d in data])
    h  = np.array([d[3]       for d in data])
    ec = np.array([d[4]       for d in data])
    ep = np.array([d[5]       for d in data])
    e  = np.array([d[6]       for d in data])

    # --- Detección de picos ---
    v_peak_idx, _ = find_peaks(v, prominence=np.std(v) * 0.4, distance=5)
    h_peak_idx, _ = find_peaks(h, prominence=np.std(h) * 0.3, distance=5)

    # --- Extracción de g: g = v²_fondo / (2 · h_cima) ---
    g_estimates, g_times = [], []
    for hi in h_peak_idx:
        if h[hi] < 0.5:
            continue
        if len(v_peak_idx) == 0:
            continue
        vi = v_peak_idx[np.argmin(np.abs(v_peak_idx - hi))]
        if abs(t[hi] - t[vi]) > 2.0:
            continue
        if v[vi] < 1.0:
            continue
        g_cm = v[vi] ** 2 / (2.0 * h[hi])   # cm/s²
        g_estimates.append(g_cm / 100.0)     # → m/s²
        g_times.append((t[hi] + t[vi]) / 2.0)

    # --- Energía en cada cruce inferior ---
    e_fondo = e[v_peak_idx] if len(v_peak_idx) else np.array([])
    t_fondo = t[v_peak_idx] if len(v_peak_idx) else np.array([])

    # --- Gráfica ---
    fit_result = fit_g_from_omega()
    ang_data   = list(angle_history)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(11, 11), sharex=True)

    # Panel 1 — ajuste de oscilación amortiguada θ(t)
    if ang_data:
        t_ang_main = np.array([d[0] for d in ang_data]) - t0
        theta_ang  = np.degrees(np.array([d[1] for d in ang_data]))
        ax1.plot(t_ang_main, theta_ang, 'b.', markersize=2, alpha=0.6, label='datos θ')
        if fit_result is not None:
            _, _, popt, t_ang_start = fit_result
            A, beta, omega_fit, phi = popt
            T_fit = 2 * np.pi / omega_fit
            t_norm_dur = t_ang_main[-1] - t_ang_main[0]
            t_norm_sm  = np.linspace(0, t_norm_dur, 500)
            theta_sm   = np.degrees(damped_oscillation(t_norm_sm, *popt))
            ax1.plot(t_norm_sm + (t_ang_start - t0), theta_sm, 'g-', linewidth=1.5,
                     label=f'Ajuste: A={np.degrees(A):.1f}°  β={beta:.3f} s⁻¹  ω={omega_fit:.3f} rad/s  T={T_fit:.3f} s')
    ax1.set_ylabel('Ángulo θ  (grados)')
    ax1.set_title('Ajuste de Oscilación Amortiguada  θ(t) = A·e^{−βt}·cos(ωt+φ)')
    ax1.legend(); ax1.grid()

    # Panel 2 — retención de energía
    if e[0] > 0:
        ax2.plot(t, e / e[0] * 100, 'g-', linewidth=1.2, label='E(t)/E₀')
    if len(e_fondo) >= 2:
        ax2.scatter(t_fondo, e_fondo / e_fondo[0] * 100,
                    color='k', zorder=5, s=30, label='E en cruces inferiores')
    ax2.axhline(100, color='k', linestyle='--', linewidth=0.8)
    ax2.set_ylabel('E(t) / E₀  (%)')
    ax2.set_title('Retención de Energía')
    ax2.legend(); ax2.grid()

    # Panel 3 — extracción de g: ajuste ω²·L + dispersión v²/2h
    g_arr  = np.array(g_estimates) if g_estimates else None
    g_mean = float(np.mean(g_arr)) if g_arr is not None else None
    g_std  = float(np.std(g_arr))  if g_arr is not None else None

    if fit_result is not None or g_arr is not None:
        ax3.axhline(9.81, color='r', linestyle='--', linewidth=1.5, label='g teórica = 9.81 m/s²')

        if g_arr is not None:
            pct_scatter = abs(g_mean - 9.81) / 9.81 * 100
            ax3.scatter(g_times, g_arr, color='purple', zorder=5, s=40, alpha=0.6,
                        label='v²/2h por semiciclo')
            ax3.axhline(g_mean, color='purple', linestyle=':', linewidth=1.2,
                        label=f'media v²/2h = {g_mean:.2f} ± {g_std:.2f} m/s²')
            ax3.fill_between([t[0], t[-1]], g_mean - g_std, g_mean + g_std,
                             alpha=0.10, color='purple')

        if fit_result is not None:
            g_val, g_sig = fit_result[0], fit_result[1]
            pct_fit = abs(g_val - 9.81) / 9.81 * 100
            ax3.axhline(g_val, color='steelblue', linestyle='-', linewidth=2.0,
                        label=f'ajuste ω²·L = {g_val:.3f} ± {g_sig:.3f} m/s²  (err {pct_fit:.1f}%)')
            ax3.fill_between([t[0], t[-1]], g_val - g_sig, g_val + g_sig,
                             alpha=0.20, color='steelblue')
            ax3.set_title(f'g (ajuste ω²·L): {g_val:.3f} ± {g_sig:.3f} m/s²  |  Error: {pct_fit:.1f}%')
        else:
            ax3.set_title(f'g (v²/2h): {g_mean:.2f} ± {g_std:.2f} m/s²  |  Error: {pct_scatter:.1f}%')

        ax3.set_ylabel('g  (m/s²)')
        ax3.legend(); ax3.grid()

        if fit_result is not None:
            g_val, g_sig = fit_result[0], fit_result[1]
            pct_fit = abs(g_val - 9.81) / 9.81 * 100
            print("\n--- Extracción de g (ajuste ω²·L) ---")
            print(f"  g = {g_val:.4f} ± {g_sig:.4f} m/s²")
            print(f"  g teórica  : 9.8100 m/s²")
            print(f"  % Error    : {pct_fit:.2f}%")
        if g_arr is not None:
            pct_scatter = abs(g_mean - 9.81) / 9.81 * 100
            print("\n--- Extracción de g (método v²/2h) ---")
            print(f"  g = {g_mean:.3f} ± {g_std:.3f} m/s²")
            print(f"  % Error    : {pct_scatter:.2f}%")
            print(f"  N muestras : {len(g_estimates)}")
    else:
        ax3.text(0.5, 0.5,
                 'Ciclos insuficientes.\nDeje el péndulo completar al menos 3 oscilaciones completas.',
                 ha='center', va='center', transform=ax3.transAxes, fontsize=11)
        ax3.set_ylabel('g  (m/s²)')

    ax3.set_xlabel('Tiempo (s)')

    if len(e_fondo) >= 2:
        loss = np.diff(e_fondo) / e_fondo[:-1] * 100
        print(f"\n--- Pérdida de Energía ---")
        print(f"  Pérdida media por cruce inferior : {np.mean(np.abs(loss)):.1f}%")
        print(f"  E total retenida (inicio→fin)    : {e_fondo[-1] / e_fondo[0] * 100:.1f}%")

    plt.tight_layout()
    plt.savefig('conservacion_energia.pdf', bbox_inches='tight')
    print("\nGráfica guardada en conservacion_energia.pdf")
    plt.show()


def run_energy(model_path):
    global tracking_active, is_paused, pivot_point_px, pivot_point_cm
    global calib, energy_history, angle_history, pendulum_length_cm, E0

    model = YOLO(model_path)
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, TARGET_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_HEIGHT)

    cv2.namedWindow("Conservacion de Energia")
    cv2.setMouseCallback("Conservacion de Energia", onMouse)

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
                        v, h_cm, ec, ep, e = result
                        if E0 is None:
                            E0 = e
                        energy_history.append((pos_cm, curr_t, v, h_cm, ec, ep, e))
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

        if energy_history and energy_history[-1][6] > 0:
            last = energy_history[-1]
            v, h_cm, ec, ep, e = last[2], last[3], last[4], last[5], last[6]
            retenida = (e / E0 * 100) if E0 and E0 > 0 else 0.0
            cv2.putText(disp, f"v={v:.1f} cm/s  h={h_cm:.2f} cm", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(disp, f"EC/m={ec:.0f}  EP/m={ep:.0f}  E/m={e:.0f} cm2/s2", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(disp, f"E retenida: {retenida:.1f}%", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        if not calib.is_calibrated():
            estado = "CALIBRAR"
        elif pivot_point_px is None:
            estado = "FIJAR PIVOTE"
        elif tracking_active and not is_paused:
            estado = "RASTREANDO"
        elif is_paused:
            estado = "PAUSADO"
        else:
            estado = "LISTO"

        cv2.putText(disp, f"{estado} | S:Iniciar  P:Pausar  G:Grafica  R:Reiniciar  Q:Salir",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        calib.draw_info(disp, lang='es')

        cv2.imshow("Conservacion de Energia", disp)

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
