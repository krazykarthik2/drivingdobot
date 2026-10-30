# Dobot Magician Autonomous Robotic Manipulation Suite

A vision-language-action (VLA) manipulation and evaluation platform integrating the **Dobot Magician 4-DOF URDF Digital Twin** with state-of-the-art robotic foundational models:
- **HuggingFace LeRobot SmolVLA (450M Flow-Matching VLA)** (`lerobot/smolvla_base`)
- **Autonomous ArUco Perspective Calibration & Pixel-to-Robot Coordinate Mapping**

---

## 🚀 Key Modules & Capabilities

### 1. Zero Hand-Holding SmolVLA Policy (`run_dobot_smolvla_eval.py`)
- **End-to-End Visual Language-Action Execution**: Powered by pre-trained `lerobot/smolvla_base` (SmolVLM2-500M backbone + flow-matching action head).
- **Zero Hand-Holding Spatial Grounding**: Eliminates hardcoded coordinates and teleportation cheats. Target locations are computed dynamically from camera image tokens and natural language instructions.
- **Model-Governed Gripper Actuation**: Direct extraction of the SO-100 jaw dimension (`dim 5` of the 6-DoF action space). Automatically triggers grasp ($< 45^\circ$) and release ($< 30^\circ$) based on model perception without manual timers.
- **Multi-Camera Visual Feedback**: Synchronous multi-stream rendering featuring third-person agent-view, in-hand wrist camera, and orthographic side perspective.

### 2. ArUco Auto-Calibration (`aruco_calibrator.py`, `live_aruco_camera.py`)
- Automatically detects table ArUco markers (IDs 0–3) to compute perspective homography.
- Maps 2D camera pixels directly to physical Dobot coordinates in millimeters.

---

## 📂 Core File Structure

| File | Description |
| :--- | :--- |
| `dobot_vla_agent.py` | PyTorch agent wrapper loading pre-trained `lerobot/smolvla_base` and tokenizers. |
| `run_dobot_smolvla_eval.py` | Closed-loop evaluation runner for language-conditioned pick, transfer, and placement. |
| `dobot_color_box_env.py` | PyBullet simulation environment with multi-camera rendering and dynamic color customization. |
| `dobot_kinematics.py` | Analytical forward and inverse kinematics for the Dobot Magician arm. |
| `aruco_calibrator.py` | Real-time perspective rectification and physical coordinate calibration. |
| `live_aruco_camera.py` | Calibration tool for live USB / RealSense webcams. |
| `generate_aruco_sheet.py` | Utility to generate printable 4-in-1 marker sheets (`aruco_markers_printable.png`). |

---

## 🛠️ Quick Start

### 1. Run SmolVLA Evaluation (Model-Driven Pick & Place)
```bash
python run_dobot_smolvla_eval.py
```
Outputs high-resolution demonstration video: `dobot_smolvla_language_benchmark.mp4`.

### 2. Live Interactive Camera Alignment & 50/50 Simulation Twin Overlay
```bash
python interactive_aruco_alignment.py --cam 0
```
- Real-time on-screen HUD prompts guide your physical webcam height, distance, tilt, and centering.
- Once coarse alignment is achieved, homography rectifies residual micro-errors automatically.
- Blends your physical webcam feed directly over the PyBullet simulation robot with 0.5 opacity.

### 3. Prompt-Driven Dobot Manipulation with Live Video Overlay
```bash
python interactive_dobot_control.py --cam 0
```
- Takes natural language prompts dynamically from the console (e.g., `"pick up red box"`, `"place on green plate"`, `"home"`, `"close gripper"`, `"open gripper"`).
- Visually grounds commands using SmolVLA and animates the robot in real-time over your physical table stream.

---

## 📊 Verification & Video Artifacts
- **SmolVLA Benchmark**: `dobot_smolvla_language_benchmark.mp4` (Pick red box $\rightarrow$ Place green plate, 14.6 mm center-to-center accuracy).
