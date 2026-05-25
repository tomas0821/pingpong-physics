# Real-Time Ping-Pong Ball Tracking for Physics Education

This repository provides a high-speed, high-precision computer vision system for tracking ping-pong balls in physics experiments. Using the **YOLOv11** architecture, this method allows students and educators to capture complex trajectories (projectile motion, collisions, and damped oscillations) using standard webcams and standard laptop hardware.

## 🚀 Overview

Traditional video analysis (like Tracker) requires manual frame-by-frame processing or high-contrast backgrounds. This method leverages **Deep Learning** to:
1. Detect the ball in varied lighting and backgrounds.
2. Track multiple objects simultaneously in real-time.
3. Perform physics-based regressions (Linear for velocity, Damped Sine for oscillations) live.

## 📁 Repository Structure

| Script | Language | Experiment |
|---|---|---|
| `collisions_v2.py` | English | Multi-ball tracking for momentum and collision experiments |
| `colisiones_v2.py` | Spanish | Same as above |
| `pendulum_v2.py` | English | Damped oscillation fitting (θ vs t) |
| `pendulo_v2.py` | Spanish | Same as above |
| `pendulum_energy.py` | English | Energy conservation — live KE/PE readout and g measurement |
| `pendulo_energia.py` | Spanish | Same as above |
| `pendulum_period.py` | English | Period vs. length — T² vs L linear fit and g extraction |
| `pendulo_periodo.py` | Spanish | Same as above |
| `track_ball.py` | English | General-purpose coordinate logger (CSV output for post-processing) |
| `train.py` | — | Model training script (for reproducibility, runs on GPU cluster) |
| `export_model.py` | — | Exports trained weights to OpenVINO and ONNX formats |

## 📦 Pre-trained Weights

Pre-trained weights for YOLO11n optimized at 1024px resolution are available in the [Releases](https://github.com/tomas0821/pingpong-physics/releases/tag/v1.0.0) section.
*   **PyTorch (`.pt`)**: Best for standard use.
*   **OpenVINO**: Recommended for high FPS on Intel-based laptops.
*   **ONNX**: Universal format.

## 🛠️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/tomas0821/pingpong-physics.git
   cd pingpong-physics
   ```

2. Install dependencies:
   ```bash
   pip install ultralytics opencv-python scipy matplotlib
   ```

3. (Optional) For maximum speed on Intel CPUs:
   ```bash
   pip install openvino
   ```

## 📈 Experiments

All experiments share the same calibration workflow: click 4 corners of a known reference area to establish the pixel→cm mapping, then set up the experiment and press **S** to start.

### 1. Collisions & Momentum
Run `python collisions_v2.py` (or `colisiones_v2.py` for Spanish).
*   **Calibration**: Click the 4 corners of a known reference area (default 40×20 cm).
*   **Physics**: Calculates $V_x$ and $V_y$ via linear regression on user-selected trajectory segments. Supports coefficient of restitution mode (**M**) and momentum conservation mode (**K**).
*   **Output**: `vectores_colision.pdf` with position vs. time plots and velocity fits.

### 2. Damped Harmonic Motion
Run `python pendulum_v2.py` (or `pendulo_v2.py` for Spanish).
*   **Setup**: Calibrate, then click to set the pivot point.
*   **Fitting**: Press **G** while paused to fit the angular data to:
    $$\theta(t) = A_0 e^{-\beta t} \cos(\omega t + \phi)$$
*   **Output**: `pendulo_ajuste.pdf` and console summary of $A_0$, $\beta$, $\omega$, $T$, $\phi$.

### 3. Energy Conservation
Run `python pendulum_energy.py` (or `pendulo_energia.py` for Spanish).
*   **Setup**: Calibrate, then click to set the pivot point.
*   **Live readout**: Linear speed $v$ (cm/s), height $h$ (cm), KE/m, PE/m, and % energy retained.
*   **Physics**: Press **G** while paused to generate a 3-panel plot:
    1. KE/m, PE/m, and total E/m vs. time.
    2. Energy retention E(t)/E₀ (%) with per-swing markers.
    3. Gravitational acceleration $g$ extracted per half-swing via $g = v_\text{bottom}^2 / (2h_\text{top})$, compared against 9.81 m/s².
*   **Output**: `conservacion_energia.pdf` and console summary of measured $g$, % error, and energy loss per cycle.

### 4. Period vs. Length
Run `python pendulum_period.py` (or `pendulo_periodo.py` for Spanish).
*   **Setup**: Calibrate, then click to set the pivot point.
*   **Workflow**: Hang the pendulum at length $L_1$, press **S** — the script auto-locks $L$ from the first detections, counts 10 full cycles, stores the result, and stops. Change the string length and press **S** again. Repeat for 5–6 lengths.
*   **Physics**: Press **G** (with ≥ 3 runs) to plot $T^2$ vs $L$. The slope gives $4\pi^2/g$:
    $$g = \frac{4\pi^2}{\text{slope}}$$
*   **Output**: `pendulum_period.pdf` (two-panel: $T^2$ vs $L$ with linear fit, and $T$ vs $L$ with theoretical curve) and a console results table. Use `--cycles N` to change the number of periods averaged per run.

## 🔬 Physics Education Context

This tool is designed to bridge the gap between "Black Box" technology and fundamental physics.
*   **Error Analysis**: Students can compare their measured $g$ across methods (energy conservation vs. period vs. length) and discuss sources of systematic error.
*   **High Sampling Rate**: Reach up to 60–120 FPS, providing significantly more data points than manual video analysis.

## 📜 License
MIT License
