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

## 📷 Camera Setup

Optimal positioning depends on pendulum length:

| Pendulum length | Camera distance | Ball size in frame |
|---|---|---|
| 10–20 cm | **0.5–0.75 m** | ~40–60 px — maximum accuracy |
| 20–60 cm | 1.0–1.5 m | ~20–30 px — good balance |
| 60–100 cm | 1.5–2.0 m | ~15–20 px — needed to fit full arc |

- Point the camera **perpendicular to the swing plane** and at **pivot height**.
- Place the calibration ruler in the **same plane as the swing** — a ruler on a wall behind the pendulum introduces a scale error of approximately $\delta / D \times 100\%$ where $\delta$ is the depth offset and $D$ is the camera distance.

### ⚠️ Calibration depth — the most common source of g error

If g is consistently off by a **stable factor** (e.g. always 2× too high), the calibration reference is almost certainly at a different depth than the pendulum. The error propagates directly: a reference that is 50% further from the camera than the pendulum inflates every distance measurement by 50%, and g by the same factor.

**Diagnosis:** check the `L` value printed when tracking starts. If it is larger than your physical string + ball radius, the reference is too far away.

**Fixes (in order of preference):**

1. **Same-plane reference** — tape a strip of paper with two marks (20 cm apart) directly to the wall or surface the string hangs from. The marks must be at the same depth as the ball's arc.
2. **`--length` flag** — measure the pendulum physically (pivot to ball center = string length + ball radius ≈ +2 cm) and pass it directly:
   ```bash
   python pendulo_energia.py --length 20
   ```
   The HUD shows the L label in **green** (manual) or **red** (pixel-estimated) so you always know which is in use.
3. **Period vs. length method** (`pendulo_periodo.py`) — uses the *slope* of T² vs L across multiple runs. Absolute L accuracy is irrelevant; only relative changes in L matter, so the depth offset cancels out.

## 📈 Experiments

### Calibration

Two calibration methods are used depending on the experiment:

- **Pendulum scripts** — 2-click line calibration: click both ends of a known reference. Default reference length is 30 cm for most scripts and 20 cm for `pendulo_energia.py` / `pendulum_energy.py`. Establishes a uniform px→cm scale.
- **Collision scripts** — 4-point perspective calibration: click the 4 corners of a known rectangle (default 40×20 cm). Corrects for camera angle and perspective.

After calibration, click to set the pivot (pendulum) or begin tracking (collisions), then press **S** to start.

### 1. Collisions & Momentum
Run `python collisions_v2.py` (or `colisiones_v2.py` for Spanish).
*   **Calibration**: Click the 4 corners of a known reference area (default 40×20 cm).
*   **Physics**: Calculates $V_x$ and $V_y$ via linear regression on user-selected trajectory segments. Supports coefficient of restitution mode (**M**) and momentum conservation mode (**K**).
*   **Output**: `vectores_colision.pdf` with position vs. time plots and velocity fits.

### 2. Damped Harmonic Motion
Run `python pendulum_v2.py` (or `pendulo_v2.py` for Spanish).
*   **Calibration**: Click both ends of a 30 cm reference to set scale.
*   **Setup**: Click to set the pivot point.
*   **Fitting**: Press **G** while paused to fit the angular data to:
    $$\theta(t) = A_0 e^{-\beta t} \cos(\omega t + \phi)$$
*   **Output**: `pendulo_ajuste.pdf` and console summary of $A_0$, $\beta$, $\omega$, $T$, $\phi$.

### 3. Energy Conservation & g Extraction
Run `python pendulum_energy.py` (or `pendulo_energia.py` for Spanish).
*   **Calibration**: Click both ends of a 30 cm reference to set scale.
*   **Setup**: Click to set the pivot point.
*   **Velocity method**: At each frame, a degree-3 polynomial is fitted to the angle history of the current half-swing; the analytical derivative gives $\omega(t)$, then $v = L \cdot |\omega|$. This correctly forces $v \to 0$ at the turning points even at low frame rates.
*   **Pendulum length**: Computed as the median pivot-to-ball distance over all collected frames (robust to outlier detections).
*   **Physics**: Press **G** while paused to generate a 2-panel plot:
    1. **θ(t) fit** — raw angle data overlaid with $\theta(t) = A_0 e^{-\beta t} \cos(\omega t + \phi)$. Fit parameters ($A_0$, $\beta$, $\omega$, $T$) are shown in the legend.
    2. **Energy from the fit** — KE/m = $\frac{1}{2}v^2$ and PE/m = $g \cdot h$ computed analytically from the fit, where $h = L(1 - \cos\theta)$ (exact formula). Gravitational acceleration is extracted as:
    $$g = \omega^2 L \left(1 + \frac{A_0^2}{16}\right)^2$$
    The correction factor $(1 + A_0^2/16)^2$ accounts for the period elongation at large amplitudes (~3% at 30°, ~8% at 45°). The measured $g$ and % error vs 9.81 m/s² appear in the panel title.
*   **Output**: `conservacion_energia.pdf` and console summary of corrected $g$, raw $g$, correction factor, and % error.

### 4. Period vs. Length
Run `python pendulum_period.py` (or `pendulo_periodo.py` for Spanish).
*   **Calibration**: Click both ends of a 30 cm reference to set scale.
*   **Setup**: Click to set the pivot point.
*   **Workflow**: Hang the pendulum at length $L_1$, press **S** — the script auto-locks $L$ from the first detections, counts 10 full cycles, stores the result, and stops. Change the string length and press **S** again. Repeat for 5–6 lengths.
*   **Physics**: Press **G** (with ≥ 3 runs) to plot $T^2$ vs $L$. The slope gives $4\pi^2/g$:
    $$g = \frac{4\pi^2}{\text{slope}}$$
*   **Output**: `pendulum_period.pdf` (two-panel: $T^2$ vs $L$ with linear fit, and $T$ vs $L$ with theoretical curve) and a console results table. Use `--cycles N` to change the number of periods averaged per run.

## 🔬 Physics Education Context

This tool is designed to bridge the gap between "Black Box" technology and fundamental physics.
*   **Multiple g extraction methods**: Students can compare $g$ measured via energy conservation ($\omega^2 L$ fit), period-vs-length ($T^2 \propto L$ slope), and damped oscillation fitting, and discuss sources of systematic error in each.
*   **Large-angle correction**: The energy script applies the first-order period correction for non-small amplitudes, making the result accurate even when students use 30–45° swings.
*   **High Sampling Rate**: Reach up to 60–120 FPS, providing significantly more data points than manual video analysis.

## 📜 License
MIT License
