import cv2
from ultralytics import YOLO
import numpy as np
import time
import math
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
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
        A_fit       = float(popt[0])
        sigma_omega = float(np.sqrt(np.diag(pcov)[2]))

        # Mejora 2: L robusto — mediana sobre todos los fotogramas capturados
        if len(energy_history) >= 10 and pivot_point_cm is not None:
            dists = [float(np.linalg.norm(np.array(e[0]) - np.array(pivot_point_cm)))
                     for e in energy_history]
            L_cm = float(np.median(dists))
        else:
            L_cm = pendulum_length_cm

        # Mejora 1: corrección de ángulo grande
        # T_real ≈ T_pequeño·(1 + A²/16)  →  ω_pequeño = ω_ajuste·(1 + A²/16)
        # g = ω_pequeño²·L = ω_ajuste²·L·(1 + A_ajuste²/16)²
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
    t0   = data[0][1]

    fit_result = fit_g_from_omega()
    ang_data   = list(angle_history)

    fig = plt.figure(figsize=(11, 11))
    ax1 = fig.add_subplot(3, 1, 1)
    ax2 = fig.add_subplot(3, 1, 2)                 # phase portrait — own x-axis
    ax3 = fig.add_subplot(3, 1, 3, sharex=ax1)     # time axis shared with panel 1

    # --- Panel 1: ajuste de oscilación amortiguada θ(t) ---
    if ang_data:
        t_ang = np.array([d[0] for d in ang_data]) - t0
        theta  = np.degrees(np.array([d[1] for d in ang_data]))
        ax1.plot(t_ang, theta, 'b.', markersize=2, alpha=0.6, label='datos θ')
        if fit_result is not None:
            _, _, popt, t_ang_start = fit_result
            A, beta, omega_fit, phi = popt
            T_fit      = 2 * np.pi / omega_fit
            t_norm_sm  = np.linspace(0, t_ang[-1] - t_ang[0], 500)
            theta_sm   = np.degrees(damped_oscillation(t_norm_sm, *popt))
            ax1.plot(t_norm_sm + (t_ang_start - t0), theta_sm, 'g-', linewidth=1.5,
                     label=f'Ajuste: A={np.degrees(A):.1f}°  β={beta:.3f} s⁻¹'
                           f'  ω={omega_fit:.3f} rad/s  T={T_fit:.3f} s')
    ax1.set_ylabel('Ángulo θ  (grados)')
    ax1.set_title('Ajuste de Oscilación Amortiguada  θ(t) = A·e^{−βt}·cos(ωt+φ)')
    ax1.legend(fontsize=8); ax1.grid()

    # --- Panel 2: Retrato de fase (θ, v) — espiral amortiguada ---
    if fit_result is not None and ang_data and pendulum_length_cm is not None:
        _, _, popt, t_ang_start = fit_result
        A, beta, omega_fit, phi = popt
        t_ang_loc  = np.array([d[0] for d in ang_data]) - t0
        t_norm_sm  = np.linspace(0, t_ang_loc[-1] - t_ang_loc[0], 2000)

        theta_ph   = np.degrees(damped_oscillation(t_norm_sm, *popt))
        dtheta_dt  = (-A * beta      * np.exp(-beta * t_norm_sm) * np.cos(omega_fit * t_norm_sm + phi)
                      - A * omega_fit * np.exp(-beta * t_norm_sm) * np.sin(omega_fit * t_norm_sm + phi))
        v_signed   = pendulum_length_cm * dtheta_dt   # signed cm/s

        # Gradient colour along the curve to show time progression
        pts  = np.array([theta_ph, v_signed]).T.reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        lc   = LineCollection(segs, cmap='plasma', linewidth=1.8, alpha=0.9)
        lc.set_array(t_norm_sm[:-1])
        ax2.add_collection(lc)
        fig.colorbar(lc, ax=ax2, label='Tiempo (s)', shrink=0.8)

        ax2.plot(theta_ph[0],  v_signed[0],  'go', markersize=8, zorder=5, label='inicio')
        ax2.plot(theta_ph[-1], v_signed[-1], 'rs', markersize=8, zorder=5, label='fin')
        ax2.autoscale()
        ax2.axhline(0, color='k', linewidth=0.5, alpha=0.4)
        ax2.axvline(0, color='k', linewidth=0.5, alpha=0.4)
        ax2.legend(fontsize=8, loc='upper right')
    else:
        ax2.text(0.5, 0.5, 'Ajuste no disponible.\nOscilar al menos 3 ciclos completos.',
                 ha='center', va='center', transform=ax2.transAxes, fontsize=11)

    ax2.set_xlabel('Ángulo θ  (grados)')
    ax2.set_ylabel('Velocidad v  (cm/s)')
    ax2.set_title('Retrato de Fase  (θ, v)')
    ax2.grid(alpha=0.3)

    # --- Panel 3: EC y EP calculadas del ajuste ---
    if fit_result is not None and ang_data and pendulum_length_cm is not None:
        _, _, popt, t_ang_start = fit_result
        A, beta, omega_fit, phi = popt
        t_ang_loc  = np.array([d[0] for d in ang_data]) - t0
        t_norm_sm  = np.linspace(0, t_ang_loc[-1] - t_ang_loc[0], 500)
        t_main_sm  = t_norm_sm + (t_ang_start - t0)

        theta_sm   = damped_oscillation(t_norm_sm, *popt)
        dtheta_dt  = (-A * beta      * np.exp(-beta * t_norm_sm) * np.cos(omega_fit * t_norm_sm + phi)
                      - A * omega_fit * np.exp(-beta * t_norm_sm) * np.sin(omega_fit * t_norm_sm + phi))
        v_sm  = pendulum_length_cm * np.abs(dtheta_dt)        # cm/s
        h_sm  = pendulum_length_cm * (1.0 - np.cos(theta_sm)) # cm  (fórmula exacta)
        ke_sm = 0.5 * v_sm**2                                 # cm²/s²
        pe_sm = G_CM_S2 * h_sm                                # cm²/s²
        e_sm  = ke_sm + pe_sm

        ax3.plot(t_main_sm, ke_sm, 'r-',  linewidth=1.5, label='EC/m  (½v²)')
        ax3.plot(t_main_sm, pe_sm, 'b-',  linewidth=1.5, label='EP/m  (g·h)')
        ax3.plot(t_main_sm, e_sm,  'k-',  linewidth=2.0, label='E total/m')
        ax3.set_ylabel('Energía específica  (cm²/s²)')
        ax3.set_title('Energía Cinética y Potencial (del ajuste)')
        ax3.legend(fontsize=8); ax3.grid()
    else:
        ax3.text(0.5, 0.5, 'Ajuste no disponible.\nOscilar el péndulo al menos 3 ciclos completos.',
                 ha='center', va='center', transform=ax3.transAxes, fontsize=11)
        ax3.set_ylabel('Energía específica  (cm²/s²)')

    ax3.set_xlabel('Tiempo (s)')

    # Consola: g extraída por ajuste ω²·L
    if fit_result is not None:
        g_val, g_sig = fit_result[0], fit_result[1]
        pct_fit = abs(g_val - 9.81) / 9.81 * 100
        A_fit   = fit_result[2][0]
        corr    = (1.0 + A_fit**2 / 16.0) ** 2
        print("\n--- Extracción de g (ajuste ω²·L) ---")
        print(f"  g (corregida)   = {g_val:.4f} ± {g_sig:.4f} m/s²")
        print(f"  g (ω²·L bruta)  = {g_val/corr:.4f} m/s²")
        print(f"  Corrección ángulo grande: ×{corr:.4f}  (A₀={np.degrees(A_fit):.1f}°)")
        print(f"  g teórica       = 9.8100 m/s²")
        print(f"  % Error         : {pct_fit:.2f}%")

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
