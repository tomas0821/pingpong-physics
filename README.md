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

   **Dependency notes:**
   - `ultralytics` — YOLOv11 inference engine; requires ~2.5 GB disk space for model downloads
   - `opencv-python` — real-time video capture and image processing
   - `scipy` — curve fitting (damped oscillation, linear regression for velocity)
   - `matplotlib` — output plots and live visualization

3. (Optional) For maximum speed on Intel CPUs:
   ```bash
   pip install openvino
   ```
   If OpenVINO is installed, the scripts automatically use `best_openvino_model/` (FP16, 30–50% faster). Without it, they fall back to `best.pt` (PyTorch).

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

## 🎯 Tips for Best Results

### Amplitude and Oscillation
- **Recommended amplitudes**: 20–45° for damped oscillation fitting. Large angles (>50°) require the large-angle correction factor; very small angles (<10°) give noisy data.
- **Minimum swings**: 5–10 full cycles for reliable damping parameter extraction.
- **Detection confidence**: Default `conf=0.6` works well for most lighting. Increase to 0.8 if false detections are frequent; decrease to 0.4 only in very poor lighting.

### Data Quality
- **Frame rate matters**: Aim for 30+ FPS. Higher frame rates (60+ FPS) reduce noise in velocity estimates. Check with `print(cap.get(cv2.CAP_PROP_FPS))`.
- **Ball visibility**: Ensure the ball is always in-frame during the entire swing. For wide arcs (>60°), position the camera further back.
- **Lighting**: Avoid direct backlighting. Diffuse, even lighting reduces glint and shadows. Shadows on the background can confuse the detector.

### Calibration Workflow
1. **Set up the reference**: Mark two points 20–30 cm apart on the same plane as the pendulum swing (use tape on a flat surface or card held at swing depth).
2. **Calibrate**: Click both endpoints of the reference. A horizontal or vertical line works equally well.
3. **Set pivot**: Click the fixed pivot point (where the string is attached).
4. **Test detection**: Press **S** to start tracking. Watch the first 20 frames to verify the ball is detected consistently (cyan bounding box).
5. **Collect data**: Let the pendulum swing freely, then press **P** to pause and **G** to analyze.

### Physics Accuracy
- **Energy conservation**: Damp oscillations naturally lose ~2–5% of energy per swing due to air resistance and friction. If energy loss is >10%, check for:
  - Touching the pivot (creates friction)
  - Air drafts or fan-induced motion
  - Incorrect amplitude measurement (re-calibrate)
- **g extraction**: All three methods (damped fit, energy variance, T² vs L) should agree within ±5% under ideal conditions. If results differ:
  - Check calibration depth first (see ⚠️ section above)
  - Verify pendulum length matches physical measurement
  - Ensure amplitude is in the 20–45° range

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
*   **Calibration**: Click both ends of a 20 cm reference to set scale.
*   **Setup**: Click to set the pivot point.
*   **Velocity method**: At each frame, a degree-3 polynomial is fitted to the angle history of the current half-swing; the analytical derivative gives $\omega(t)$, then $v = L \cdot |\omega|$. This correctly forces $v \to 0$ at the turning points even at low frame rates.
*   **Pendulum length**: Computed as the median pivot-to-ball distance over all collected frames (robust to outlier detections). Pass `--length <L_cm>` to use a physical measurement instead — strongly recommended when the calibration ruler is not in the same plane as the pendulum swing (see ⚠️ above).
*   **Physics**: Press **G** while paused to generate a 2-panel plot:
    1. **θ(t) fit** — raw angle data overlaid with $\theta(t) = A_0 e^{-\beta t} \cos(\omega t + \phi)$. Fit parameters ($A_0$, $\beta$, $\omega$, $T$) are shown in the legend.
    2. **Energy from the fit** — KE/m = $\frac{1}{2}v^2$ and PE/m = $g \cdot h$ computed analytically from the fit, where $h = L(1 - \cos\theta)$ (exact formula). Gravitational acceleration is extracted by two independent methods: $\omega^2 L$ with large-angle correction $(1 + A_0^2/16)^2$, and energy-variance minimisation. Both results and their % error vs 9.81 m/s² appear in the panel title.
*   **Output**: `energy_conservation.pdf` (`conservacion_energia.pdf` for Spanish) and console summary of corrected $g$, raw $g$, correction factor, and % error.

### 4. Period vs. Length
Run `python pendulum_period.py` (or `pendulo_periodo.py` for Spanish).
*   **Calibration**: Click both ends of a 30 cm reference to set scale.
*   **Setup**: Click to set the pivot point.
*   **Workflow**: Hang the pendulum at length $L_1$, press **S** — the script auto-locks $L$ from the first detections, counts 10 full cycles, stores the result, and stops. Change the string length and press **S** again. Repeat for 5–6 lengths.
*   **Physics**: Press **G** (with ≥ 3 runs) to plot $T^2$ vs $L$. The slope gives $4\pi^2/g$:
    $$g = \frac{4\pi^2}{\text{slope}}$$
*   **Output**: `pendulum_period.pdf` (two-panel: $T^2$ vs $L$ with linear fit, and $T$ vs $L$ with theoretical curve) and a console results table. Use `--cycles N` to change the number of periods averaged per run.

## 📚 Dataset & Retraining

The model is trained on a custom **Roboflow dataset** with ~1500 labeled frames of ping-pong balls across diverse lighting and backgrounds.

**Dataset reference:**
- **Workspace**: `pingpong-ojuhj`
- **Project**: `ping-pong-detection-0guzq`
- **Version**: 3 (YOLO11 format)
- **Roboflow URL**: [Ping-Pong Detection](https://universe.roboflow.com/) (search for the project ID)

### Retraining the Model

To retrain on new data or fine-tune the model:

1. **Download the dataset**:
   ```bash
   # Store your Roboflow API key in a file named ROBOFLOW_API_KEY (not committed to git)
   python download_dataset.py
   ```
   This downloads the dataset to `Ping-Pong-Detection-3/` in YOLO11 format.

2. **Train on a GPU cluster** (recommended for faster convergence):
   ```bash
   python train.py
   ```
   Outputs are saved to `runs/detect/train/`. Typical training on an A100 GPU takes 1–2 hours for 100 epochs at 1024px input resolution.

3. **Export optimized weights**:
   ```bash
   python export_model.py
   ```
   Produces:
   - `best.pt` — PyTorch weights (used by default)
   - `best_openvino_model/` — Intel OpenVINO format (faster on Intel CPUs)
   - ONNX weights (if needed for cross-platform deployment)

4. **Verify the new model**:
   ```bash
   python test_model.py
   ```

**Training notes:**
- Default: YOLOv11n (nano) at 1024px input resolution for balance between speed and accuracy.
- Inference always runs at 1024px regardless of export format.
- OpenVINO export uses FP16 quantization for ~30–50% speed improvement on Intel CPUs.

## 🔬 Physics Education Context

This tool is designed to bridge the gap between "Black Box" technology and fundamental physics.
*   **Multiple g extraction methods**: Students can compare $g$ measured via energy conservation ($\omega^2 L$ fit), period-vs-length ($T^2 \propto L$ slope), and damped oscillation fitting, and discuss sources of systematic error in each.
*   **Large-angle correction**: The energy script applies the first-order period correction for non-small amplitudes, making the result accurate even when students use 30–45° swings.
*   **High Sampling Rate**: Reach up to 60–120 FPS, providing significantly more data points than manual video analysis.

## 🔧 Troubleshooting

### Detection failures (no bounding box around ball)

**Problem**: The ball is not being detected or detections are sporadic.

**Solutions**:
1. **Check lighting**: Ensure the ball is well-lit and not in shadows. Backlighting confuses the detector.
2. **Verify model load**: Run `python test_model.py` to confirm the model loads correctly.
3. **Lower confidence threshold**: Modify the script's `conf=0.6` to `conf=0.4` and retry.
4. **Check camera resolution**: Some cameras default to low resolution. Verify the frame is at least 640×480.
5. **Retrain on your ball**: If using a different ball color/material, consider collecting new training data and retraining.

### Oscillation doesn't fit well (G pressed, poor θ(t) curve match)

**Problem**: The damped sine fit is visibly off from the actual angle data.

**Solutions**:
1. **Ensure enough cycles**: Need ≥ 5 full oscillation cycles. Very damped pendulums may require 10+ cycles.
2. **Check amplitude**: Very small (<10°) or very large (>60°) amplitudes can confuse the fitting algorithm. Aim for 20–45°.
3. **Verify calibration**: Re-calibrate the reference. A miscalibrated scale inflates angle errors.
4. **Increase data collection time**: Let the pendulum swing longer to capture the full damping envelope.

### g value is consistently wrong (too high, too low, or wildly varying)

**Problem**: The measured gravitational acceleration doesn't match 9.81 m/s².

**Solutions** (in order of likelihood):
1. **Check calibration depth** (most common) — See ⚠️ section under Camera Setup. Use `--length` flag to bypass pixel-based estimation.
2. **Verify amplitude is 20–45°** — Large-angle corrections are calibrated for this range.
3. **Confirm pendulum length** — Use `--length` flag to input the true physical length (pivot to ball center).
4. **Check for air currents** — Fans, open doors, or breeze can dampen motion differently than gravity alone.
5. **Compare methods** — Run both `pendulo_energia.py` and `pendulo_periodo.py` on the same pendulum. Do they agree? If not, calibration is the issue.

### Program crashes or freezes

**Problem**: Script hangs or crashes during tracking.

**Solutions**:
1. **Reduce inference size**: Comment out `imgsz=1024` in the script and let YOLO default to 640px (slower but more compatible).
2. **Check GPU memory**: If using GPU, ensure sufficient VRAM. Fall back to CPU if needed.
3. **Verify OpenCV camera**: Run `python -c "import cv2; cap = cv2.VideoCapture(0); print(cap.read())"` to test camera initialization.
4. **Update dependencies**: Ensure `ultralytics`, `opencv-python`, and `scipy` are up-to-date: `pip install --upgrade ultralytics opencv-python scipy`.

### FPS is very slow (<10 FPS)

**Problem**: Real-time tracking is laggy.

**Solutions**:
1. **Install OpenVINO**: Running on Intel CPU? `pip install openvino` for 30–50% speedup.
2. **Reduce input resolution**: Change `imgsz=1024` to `imgsz=640` in the script.
3. **Disable visualization**: Plotting in real-time is expensive. Press **P** early to stop rendering.
4. **Use a dedicated GPU**: YOLO11 is ~5–10× faster on GPU than CPU.

### Keyboard shortcuts don't respond

**Problem**: Pressing S, P, G, etc. has no effect.

**Solutions**:
1. **Click on the video window**: The window must be in focus. Click on the display to activate it.
2. **Check for input mode**: Calibration is active if you see "Click to set..." messages. Press **Esc** or click away to exit calibration.
3. **Verify terminal output**: Check the terminal for error messages that might indicate a crash.

## 📖 Citation

If you use this software in your research, please cite the accompanying paper (citation details to be updated upon acceptance):

> [Author] 2026 Real-time high-speed tracking of ping-pong balls for physics education using YOLOv11 *Phys. Educ.* (submitted)

## 📜 License
MIT License
