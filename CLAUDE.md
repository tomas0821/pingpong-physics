# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

A computer vision system for tracking ping-pong balls in real-time to conduct physics experiments (collisions, pendulum motion). It uses a YOLOv11 model trained on a custom Roboflow dataset, running at 1024px input resolution for small-object accuracy.

## Setup

```bash
pip install ultralytics opencv-python scipy matplotlib
pip install openvino  # optional, for faster inference on Intel CPUs
```

The model auto-selects OpenVINO (`best_openvino_model/`) if present, else falls back to `best.pt`.

## Running the applications

```bash
# Collision/momentum experiment (tracks up to 2 balls)
python collisions_v2.py [--model best_openvino_model]

# Pendulum / damped oscillation experiment
python pendulum_v2.py [--model best_openvino_model]

# Energy conservation experiment (linear KE + PE, live readout)
python pendulum_energy.py [--model best_openvino_model]

# General-purpose CSV logger (post-processing workflow)
python track_ball.py [--model best_openvino_model] [--output pingpong_data.csv]

# Verify the model loads and runs inference
python test_model.py
```

## Data pipeline (for retraining)

1. Place Roboflow API key in a file named `ROBOFLOW_API_KEY` (not committed).
2. Download dataset: `python download_dataset.py` → creates `Ping-Pong-Detection-3/`
3. Train on cluster (A100 GPU): `python train.py` → weights saved to `runs/detect/train/weights/best.pt`
4. Export optimized weights: `python export_model.py` → produces `best_openvino_model/` and ONNX

## Architecture

### Shared calibration layer (`utils.py`)
`PerspectiveCalibration` is the foundation used by all three apps. The user clicks 4 corners of a known physical reference area; the class computes a perspective transform matrix mapping pixel coordinates to centimeter coordinates. All physics calculations operate in cm-space, not pixel-space.

### Application structure
Each main script (`collisions_v2.py`, `pendulum_v2.py`, `track_ball.py`) follows the same pattern:
- Module-level state variables (calibration object, tracking flags, position histories as `deque`)
- `onMouse` callback registered with OpenCV for calibration + user interaction
- A main loop: capture frame → YOLO inference (`imgsz=1024`) → map detections to cm via calibration → append to history deque → draw overlays → handle keypress
- Physics analysis triggered by keypresses while paused

### Physics modules
- **`collisions_v2.py`**: Tracks two balls simultaneously via a nearest-neighbor matcher (`match_detections`). Velocity calculated as linear regression over a user-selected trajectory segment. Supports both coefficient-of-restitution mode and momentum conservation mode (4-segment selection).
- **`pendulum_v2.py`**: Tracks a single ball relative to a user-set pivot point, computes angle via `atan2`, detects period by zero-crossings, and fits `θ(t) = A₀ e^{-βt} cos(ωt + φ)` using `scipy.optimize.curve_fit` on keypress `G`.
- **`pendulum_energy.py`**: Energy conservation experiment. Uses linear velocity (v = Δcm / Δt over a smoothing window) and height above the lowest point of the swing to compute KE/m = ½v² and PE/m = gh each frame. Pendulum length L is auto-locked from the first 10 detections. Live overlay shows v, h, and energy values; press `G` while paused to plot KE/m, PE/m, and total E/m vs time and save `energy_conservation.pdf`.
- **`track_ball.py`**: Thin logger — records `(timestamp, x_px, y_px, x_cm, y_cm, conf)` to CSV for offline analysis.

### Model details
- Architecture: YOLO11n, trained at 1024px resolution
- Inference always uses `imgsz=1024`, `conf=0.6`
- OpenVINO export uses 640px with FP16 (`half=True`) — note the resolution mismatch vs. training; inference overrides with `imgsz=1024` at runtime
