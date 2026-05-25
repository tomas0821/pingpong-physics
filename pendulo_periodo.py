import cv2
from ultralytics import YOLO
import numpy as np
import time
import math
from collections import deque
import matplotlib.pyplot as plt
from scipy.stats import linregress
import argparse
import os
from utils import PerspectiveCalibration

# ---------------- Configuración ----------------
DEFAULT_CALIB_WIDTH = 40.0
DEFAULT_CALIB_HEIGHT = 20.0
DEFAULT_MODEL = "best_openvino_model" if os.path.exists("best_openvino_model") else "best.pt"
CAMERA_INDEX = 0
TARGET_WIDTH, TARGET_HEIGHT = 1280, 720
CONFIDENCE_THRESHOLD = 0.6
POINT_HISTORY_LENGTH = 500
TARGET_CYCLES = 10          # períodos completos a promediar por medición
L_LOCK_FRAMES = 10          # fotogramas para fijar la longitud

# --- Variables de estado ---
calib = PerspectiveCalibration(DEFAULT_CALIB_WIDTH, DEFAULT_CALIB_HEIGHT)
pivot_point_px = None
pivot_point_cm = None
tracking_active = False
angle_history = deque(maxlen=POINT_HISTORY_LENGTH)

# Estado por medición (se reinicia cada vez que se presiona S)
pendulum_length_cm = None
last_cross_time = None
last_cross_side = None
period_samples = []
run_complete = False
run_result = None       # (L_cm, T_mean, T_std)

# Mediciones acumuladas en la sesión
measurements = []       # lista de (L_cm, T_mean, T_std)


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


def reset_run():
    global pendulum_length_cm, last_cross_time, last_cross_side
    global period_samples, run_complete, run_result
    angle_history.clear()
    pendulum_length_cm = None
    last_cross_time = None
    last_cross_side = None
    period_samples = []
    run_complete = False
    run_result = None


def detect_crossing(angle, current_time):
    global last_cross_time, last_cross_side, period_samples

    side = 1 if angle >= 0 else -1
    if last_cross_side is not None and side != last_cross_side and abs(angle) < 0.15:
        if last_cross_time is not None:
            T = 2.0 * (current_time - last_cross_time)
            if 0.3 < T < 15.0:
                period_samples.append(T)
        last_cross_time = current_time
    last_cross_side = side


def plot_results(data):
    if len(data) < 3:
        print("Se necesitan al menos 3 mediciones de longitud para graficar.")
        return

    L     = np.array([d[0] / 100.0 for d in data])   # cm → m
    T     = np.array([d[1]          for d in data])
    T_err = np.array([d[2]          for d in data])
    T2    = T ** 2
    T2_err = 2 * T * T_err

    slope, intercept, r, _, _ = linregress(L, T2)
    g_medida  = 4 * np.pi**2 / slope
    pct_error = abs(g_medida - 9.81) / 9.81 * 100

    L_fit  = np.linspace(L.min() * 0.9, L.max() * 1.1, 200)
    T2_fit = slope * L_fit + intercept

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # --- Panel 1: T² vs L ---
    ax1.errorbar(L * 100, T2, yerr=T2_err, fmt='o', color='steelblue',
                 capsize=4, label='Mediciones')
    ax1.plot(L_fit * 100, T2_fit, 'r-',
             label=f'Ajuste lineal  (R²={r**2:.4f})')
    ax1.set_xlabel('Longitud del péndulo L (cm)')
    ax1.set_ylabel('T²  (s²)')
    ax1.set_title(f'T² vs L  →  g = {g_medida:.3f} m/s²  (error: {pct_error:.1f}%)')
    ax1.legend(); ax1.grid()

    # --- Panel 2: T vs L con curva teórica ---
    T_teorico       = 2 * np.pi * np.sqrt(L_fit / 9.81)
    T_medido_curva  = 2 * np.pi * np.sqrt(L_fit / g_medida)
    ax2.errorbar(L * 100, T, yerr=T_err, fmt='o', color='steelblue',
                 capsize=4, label='Mediciones')
    ax2.plot(L_fit * 100, T_teorico, 'k--', linewidth=1.2,
             label='Teórico  (g = 9.81 m/s²)')
    ax2.plot(L_fit * 100, T_medido_curva, 'r-', linewidth=1.2,
             label=f'Ajuste  (g = {g_medida:.3f} m/s²)')
    ax2.set_xlabel('Longitud del péndulo L (cm)')
    ax2.set_ylabel('Período T  (s)')
    ax2.set_title('T vs L')
    ax2.legend(); ax2.grid()

    plt.tight_layout()
    plt.savefig('pendulo_periodo.pdf', bbox_inches='tight')

    print("\n--- Resultados: Período vs. Longitud ---")
    print(f"  {'L (cm)':>8}  {'T (s)':>8}  {'σT (s)':>8}")
    print(f"  {'-'*28}")
    for L_cm, T_mean, T_std in data:
        print(f"  {L_cm:>8.1f}  {T_mean:>8.4f}  {T_std:>8.4f}")
    print(f"\n  Pendiente (4π²/g) = {slope:.5f} s²/m")
    print(f"  g medida          = {g_medida:.4f} m/s²")
    print(f"  g teórica         = 9.8100 m/s²")
    print(f"  % Error           = {pct_error:.2f}%")
    print(f"\nGráfica guardada en pendulo_periodo.pdf")
    plt.show()


def run_period(model_path):
    global tracking_active, pivot_point_px, pivot_point_cm
    global pendulum_length_cm, period_samples, run_complete, run_result

    model = YOLO(model_path)
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, TARGET_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_HEIGHT)

    cv2.namedWindow("Período vs. Longitud")
    cv2.setMouseCallback("Período vs. Longitud", onMouse)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        curr_t = time.time()
        det_px = None

        if tracking_active and not run_complete and pivot_point_cm is not None:
            results = model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False, imgsz=1024)
            if results and results[0].boxes:
                best = max(results[0].boxes, key=lambda b: b.conf[0])
                box = best.xyxy[0]
                center_px = (int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2))
                pos_cm = calib.map_point(center_px[0], center_px[1])
                if pos_cm is not None:
                    if pendulum_length_cm is None:
                        angle_history.append((pos_cm, curr_t, 0))
                        if len(angle_history) >= L_LOCK_FRAMES:
                            L_samples = [
                                math.dist(entry[0], pivot_point_cm)
                                for entry in angle_history
                            ]
                            pendulum_length_cm = float(np.mean(L_samples))
                            print(f"Longitud fijada: {pendulum_length_cm:.1f} cm")
                    else:
                        dx = pos_cm[0] - pivot_point_cm[0]
                        dy = pos_cm[1] - pivot_point_cm[1]
                        angle = math.atan2(dx, dy)
                        angle_history.append((pos_cm, curr_t, angle))
                        detect_crossing(angle, curr_t)
                        det_px = (center_px, int(box[2] - box[0]))

                        if len(period_samples) >= TARGET_CYCLES:
                            T_mean = float(np.mean(period_samples))
                            T_std  = float(np.std(period_samples))
                            run_result = (pendulum_length_cm, T_mean, T_std)
                            measurements.append(run_result)
                            run_complete = True
                            tracking_active = False
                            print(f"\n--- Medición completada ---")
                            print(f"  L = {pendulum_length_cm:.1f} cm")
                            print(f"  T = {T_mean:.4f} ± {T_std:.4f} s  (N={len(period_samples)})")
                            print(f"  Total de mediciones guardadas: {len(measurements)}")

        disp = frame.copy()

        pts = list(angle_history)
        for i in range(1, len(pts)):
            p0 = tuple(np.array(pts[i-1][0]).astype(int))
            p1 = tuple(np.array(pts[i][0]).astype(int))
            cv2.line(disp, p0, p1, (0, 200, 255), 2)

        if pivot_point_px:
            cv2.drawMarker(disp, pivot_point_px, (0, 255, 255), cv2.MARKER_CROSS, 20, 2)
        if det_px:
            c_px, w_px = det_px
            cv2.circle(disp, c_px, max(5, int(w_px / 2)), (0, 255, 0), 2)

        y = 60
        if tracking_active and pendulum_length_cm is not None:
            cv2.putText(disp, f"L = {pendulum_length_cm:.1f} cm  |  Ciclos: {len(period_samples)}/{TARGET_CYCLES}",
                        (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
            y += 30
        elif tracking_active:
            cv2.putText(disp, "Fijando longitud del péndulo...",
                        (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2)
            y += 30

        if run_result is not None:
            L_cm, T_mean, T_std = run_result
            cv2.putText(disp, f"Medicion {len(measurements)}: L={L_cm:.1f}cm  T={T_mean:.3f}+/-{T_std:.3f}s",
                        (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            y += 25
            cv2.putText(disp, "Cambie la longitud y presione S para la siguiente medicion  |  G para graficar",
                        (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        if not calib.is_calibrated():
            estado = "CALIBRAR"
        elif pivot_point_px is None:
            estado = "FIJAR PIVOTE"
        elif tracking_active:
            estado = "MIDIENDO"
        elif run_complete:
            estado = f"LISTO ({len(measurements)} mediciones)"
        else:
            estado = "LISTO"

        cv2.putText(disp, f"{estado} | S:Iniciar  G:Grafica  C:Limpiar todo  R:Reiniciar pivote  Q:Salir",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        calib.draw_info(disp, lang='es')
        cv2.imshow("Período vs. Longitud", disp)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s') and pivot_point_cm is not None and not tracking_active:
            reset_run()
            tracking_active = True
        elif key == ord('g'):
            plot_results(measurements)
        elif key == ord('c'):
            measurements.clear()
            reset_run()
            print("Todas las mediciones borradas.")
        elif key == ord('r'):
            pivot_point_px = None
            pivot_point_cm = None
            reset_run()
            tracking_active = False

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',  type=str, default=DEFAULT_MODEL)
    parser.add_argument('--ciclos', type=int, default=TARGET_CYCLES,
                        help='Número de períodos completos a promediar por medición (defecto: 10)')
    args = parser.parse_args()
    TARGET_CYCLES = args.ciclos
    run_period(args.model)
