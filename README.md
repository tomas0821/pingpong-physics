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

All experiments share the same workflow: **calibrate** (click 4 corners of a known reference area) → **set up** → press **S** to start tracking → press **P** to pause → press **G** to generate the plot.

### 1. Collisions & Momentum
Run `python collisions_v2.py` (or `colisiones_v2.py` for Spanish).
*   **Calibration**: Click the 4 corners of a known reference area (default 40×20 cm).
*   **Physics**: Calculates $V_x$ and $V_y$ via linear regression on user-selected trajectory segments. Supports coefficient of restitution mode (press **M**) and momentum conservation mode (press **K**).
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
*   **Live readout**: Linear speed $v$ (cm/s), height $h$ (cm), KE/m, PE/m, and % energy retained overlaid on the video.
*   **Physics**: Press **G** while paused to generate a 3-panel plot:
    1. KE/m, PE/m, and total E/m vs. time.
    2. Energy retention E(t)/E₀ (%) with per-swing markers.
    3. Gravitational acceleration $g$ extracted per half-swing via $g = v_\text{bottom}^2 / (2h_\text{top})$, compared against the theoretical 9.81 m/s².
*   **Output**: `conservacion_energia.pdf` and console summary of measured $g$, % error, and energy loss per cycle.

## 🔬 Physics Education Context

This tool is designed to bridge the gap between "Black Box" technology and fundamental physics.
*   **Error Analysis**: Students can analyze how the coefficient of restitution changes with different surfaces, or compare their measured $g$ against the theoretical value.
*   **High Sampling Rate**: Reach up to 60–120 FPS, providing significantly more data points than manual video analysis.

## 📜 License
MIT License
