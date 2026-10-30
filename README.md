# Dobot Magician Autonomous Robotic Manipulation Suite

A multi-policy, vision-language-action (VLA) manipulation and evaluation platform integrating the **Dobot Magician 4-DOF URDF Digital Twin** with state-of-the-art robotic foundational models:
1. **HuggingFace LeRobot SmolVLA (450M Flow-Matching VLA)** (`lerobot/smolvla_base`)
2. **UC Berkeley Octo Generalist Robot Policy** (`octo-models/octo`)
3. **DP3 (3D Diffusion Policy)** integration suite
4. **Autonomous ArUco Perspective Calibration & Pixel-to-Robot Mapping**

---

## 🚀 Key Modules & Capabilities

### 1. Zero Hand-Holding SmolVLA Policy (`run_dobot_smolvla_eval.py`)
- **End-to-End Visual Language-Action Execution**: Powered by pre-trained `lerobot/smolvla_base` (SmolVLM2-500M backbone + flow-matching action head).
- **Zero Hand-Holding Spatial Grounding**: Eliminates hardcoded coordinates and teleportation cheats. Target locations are computed dynamically from camera image tokens and language instructions.
- **Model-Governed Gripper Actuation**: Direct extraction of the SO-100 jaw dimension (`dim 5` of the 6-DoF action space). Automatically triggers grasp ($< 45^\circ$) and release ($< 30^\circ$) based on model perception without manual timers.
- **Multi-Camera Visual Feedback**: Synchronous multi-stream rendering featuring third-person agent-view, in-hand wrist camera, and orthographic side perspective.

### 2. Generalist Octo Policy Suite (`run_octo_dobot.py`)
- Natural language-guided Pick-and-Place and Push policies.
- Multi-camera visual rendering and offscreen PyRender DAE mesh synthesis.

### 3. ArUco Auto-Calibration (`aruco_calibrator.py`, `live_aruco_camera.py`)
- Automatically detects table ArUco markers (IDs 0–3) to compute perspective homography.
- Maps 2D camera pixels directly to physical Dobot coordinates in millimeters.

---

## 📂 Core File Structure

| File | Description |
| :--- | :--- |
| `dobot_vla_agent.py` | PyTorch agent wrapper loading pre-trained `lerobot/smolvla_base` and tokenizers. |
| `run_dobot_smolvla_eval.py` | Closed-loop evaluation runner for language-conditioned pick, transfer, and placement. |
| `dobot_color_box_env.py` | PyBullet simulation environment with multi-camera rendering and dynamic color customization. |
| `run_octo_dobot.py` | Octo policy manipulation runner with video recording. |
| `dobot_urdf_renderer.py` | Offscreen 3D kinematic renderer with Collada DAE meshes. |
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

### 2. Run Octo Generalist Manipulation
```bash
python run_octo_dobot.py
```

### 3. Live ArUco Calibration
```bash
python live_aruco_camera.py --camera_id 0
```
Marker layout for table calibration:
- **ID 0 (Top-Left)**: $X = +300\text{ mm}, Y = -150\text{ mm}$
- **ID 1 (Top-Right)**: $X = +300\text{ mm}, Y = +150\text{ mm}$
- **ID 2 (Bottom-Right)**: $X = +150\text{ mm}, Y = +150\text{ mm}$
- **ID 3 (Bottom-Left)**: $X = +150\text{ mm}, Y = -150\text{ mm}$

---

## 📊 Verification & Video Artifacts
- **SmolVLA Benchmark**: `dobot_smolvla_language_benchmark.mp4` (Pick red box $\rightarrow$ Place green plate, 14.6 mm center-to-center accuracy).
- **Octo Benchmark**: `octo_dobot_manipulation.mp4` (Multi-view visual execution).
