# Dobot Magician URDF + Octo Robot Policy & ArUco Calibration

This repository integrates UC Berkeley's **Octo Generalist Robot Policy** (`octo-models/octo`) with the **Dobot Magician 4-DOF URDF Digital Twin** (`krazykarthik2/dobot_magician_ros2`), providing an autonomous vision-language manipulation suite and an automatic ArUco camera calibration pipeline.

---

## Features
- **Octo Policy Inference**: Directly drives the robot end-effector using natural language prompts (*"pick up the red cube and place it on the green platform"*, *"push the yellow cube onto the cyan target"*).
- **URDF 3D Kinematic Model**: Forward & inverse analytical kinematics linking `magician_joint_1`, `magician_joint_2`, `magician_joint_3`, parallel mimic linkages, roll wrist, and prismatic gripper jaws with Collada DAE meshes.
- **Table Clearance Calibration**: Calibrated tool-tip vertical offset ($10.5\text{ cm}$) to ensure gripper jaws operate strictly above the table surface with zero mesh collision/penetration.
- **Multi-Camera Visual Pipeline**:
  - Primary Overhead Observation Camera ($256 \times 256$)
  - Wrist POV Tool Camera ($128 \times 128$)
  - Cinematic Studio Third-Person 3D View ($640 \times 480$)
- **ArUco Auto-Calibration Pipeline**: Automatically detects 4 table markers (IDs 0, 1, 2, 3), rectifies camera tilt/distortion via perspective homography, and maps image pixels directly to physical Dobot coordinates in millimeters.
- **Demonstration Video**: Generates multi-view 20 FPS video (`octo_dobot_manipulation.mp4`).

---

## File Structure
- `run_octo_dobot.py`: Main manipulation runner for Pick-and-Place and Push tasks with video output.
- `dobot_urdf_renderer.py`: Offscreen 3D PyRender engine utilizing Dobot Magician URDF meshes.
- `octo_patch.py`: JAX compatibility bridge for modern JAX & Transformers runtimes.
- `aruco_calibrator.py`: ArUco perspective rectification and pixel-to-robot coordinate mapper.
- `live_aruco_camera.py`: Real-time camera calibration tool for live USB / RealSense webcams.
- `generate_aruco_sheet.py`: Script to generate printable ArUco markers.
- `aruco_markers_printable.png`: Ready-to-print 4-in-1 marker sheet.
- `octo_dobot_manipulation.mp4`: Rendered demonstration video showing dual-camera Octo inputs and URDF robot execution.

---

## Quick Start

### 1. Run Autonomous Manipulation & Video Generation
```bash
python run_octo_dobot.py
```

### 2. Live Camera ArUco Calibration
1. Print `aruco_markers_printable.png` and place the 4 markers on the workspace table:
   - **ID 0 (Top-Left)**: $X = +300\text{ mm}, Y = -150\text{ mm}$
   - **ID 1 (Top-Right)**: $X = +300\text{ mm}, Y = +150\text{ mm}$
   - **ID 2 (Bottom-Right)**: $X = +150\text{ mm}, Y = +150\text{ mm}$
   - **ID 3 (Bottom-Left)**: $X = +150\text{ mm}, Y = -150\text{ mm}$
2. Start the live camera feed:
```bash
python live_aruco_camera.py --camera_id 0
```
