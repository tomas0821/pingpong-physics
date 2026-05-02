# Real-Time Ping-Pong Ball Tracking for Physics Education

This repository provides a high-speed, high-precision computer vision system for tracking ping-pong balls in physics experiments. Using the **YOLOv11** architecture, this method allows students and educators to capture complex trajectories (projectile motion, collisions, and damped oscillations) using standard webcams and standard laptop hardware.

## 🚀 Overview

Traditional video analysis (like Tracker) requires manual frame-by-frame processing or high-contrast backgrounds. This method leverages **Deep Learning** to:
1. Detect the ball in varied lighting and backgrounds.
2. Track multiple objects simultaneously in real-time.
3. Perform physics-based regressions (Linear for velocity, Damped Sine for oscillations) live.

## 📁 Repository Structure

*   `collisions_v2.py`: Multi-ball tracking for momentum and collision experiments.
*   `pendulum_v2.py`: Interactive pendulum analyzer with damped oscillation fitting.
*   `track_ball.py`: General-purpose coordinate logger (logs to CSV for post-processing).
*   `train.py`: The script used to train the model on the cluster (for reproducibility).
*   `export_model.py`: Utility to optimize the model for CPU using OpenVINO.

## 📦 Pre-trained Weights

Pre-trained weights for YOLO11n optimized at 1024px resolution are available in the [Releases](https://github.com/tomas0821/pingpong-physics/releases/tag/v1.0.0) section.
*   **PyTorch (`.pt`)**: Best for standard use.
*   **OpenVINO**: Recommended for high FPS on Intel-based laptops.
*   **ONNX**: Universal format.

## 🛠️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/[your-username]/pingpong-physics.git
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

### 1. Collisions & Momentum
Run `python collisions_v2.py`. 
*   **Calibration**: Click the two ends of a 10cm reference object.
*   **Physics**: The script calculates $V_x$ and $V_y$ using linear regression on trajectory segments.
*   **Suggested Figure**: *Screenshot of the "Collision Analyzer" window showing two intersecting trajectories with their respective velocity vectors.*

### 2. Damped Harmonic Motion
Run `python pendulum_v2.py`.
*   **Setup**: Set the pivot point with a click.
*   **Fitting**: Press 'G' while paused to fit the data to:
    $$\theta(t) = A_0 e^{-\beta t} \cos(\omega t + \phi)$$
*   **Suggested Figure**: *The generated Matplotlib plot showing the blue data points perfectly overlaid by the green damped-sine fit.*

## 🔬 Physics Education Context

This tool is designed to bridge the gap between "Black Box" technology and fundamental physics.
*   **Error Analysis**: Students can analyze how the coefficient of restitution changes with different surfaces.
*   **High Sampling Rate**: Reach up to 60-120 FPS, providing significantly more data points than manual video analysis.

## 📜 License
MIT License
