"""
interactive_aruco_alignment.py

Interactive Live Camera Alignment, ArUco Rectification, and PyBullet Digital Twin Overlay.
- Configured specifically for 4 seamless 8x8 cm markers:
      Top Row:    [ ID 0 ] [ ID 1 ]
      Bottom Row: [ ID 2 ] [ ID 3 ]
- Guides your physical camera adjustment (Distance, Height, Yaw/Tilt/Centering) via real-time HUD prompts.
- Once coarse alignment is achieved, automatically computes homography to rectify residual micro-errors.
- Blends the live physical camera stream directly over the PyBullet simulation digital twin with 0.5 opacity
  so physical objects and the virtual robot match seamlessly.
"""

import sys
import os
import time
import argparse
import numpy as np
import cv2
import pybullet as p

from aruco_calibrator import ArucoAutoCalibrator
from dobot_color_box_env import DobotColorBoxEnv


def run_interactive_alignment(cam_id=0):
    print("=" * 70)
    print("DOBOT MAGICIAN ARUCO ALIGNMENT (8x8cm Seamless Grid: [0,1]/[2,3])")
    print("=" * 70)
    print(f"[INFO] Opening Webcam Device ID: {cam_id}...")

    cap = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(cam_id)
    if not cap.isOpened():
        print(f"[WARN] Camera {cam_id} failed. Trying camera 0...")
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[ERROR] No camera available!")
            return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Initialize Calibrator with seamless 8x8cm grid layout
    calibrator = ArucoAutoCalibrator(output_size=(640, 480))

    # Initialize PyBullet Digital Twin
    print("[INFO] Initializing PyBullet Digital Twin Environment...")
    env = DobotColorBoxEnv(render_gui=False, width=640, height=480)
    
    # Spawn virtual objects matching calibration setup
    task_config = [
        {"color": "red", "box_pos": [0.24, 0.08], "plate_pos": [0.22, -0.10], "plate_color": "green"}
    ]
    env.spawn_colored_boxes_and_plates(task_config)

    # Ideal target framing box for the 16x16 cm grid in camera view
    TARGET_BOX = {
        "u_left": 180,
        "u_right": 460,
        "v_top": 140,
        "v_bottom": 360,
    }

    cv2.namedWindow("Dobot Camera Alignment & Twin Overlay", cv2.WINDOW_AUTOSIZE)

    print("\n--- INSTRUCTIONS ---")
    print("Layout: Top Row = [0, 1] | Bottom Row = [2, 3] (Touching side-by-side, no gaps)")
    print("1. Follow on-screen HUD arrows to center and level your camera.")
    print("2. When alignment turns GREEN, homography rectifies residual micro-errors.")
    print("3. Press [SPACE] to toggle 0.5 Opacity Simulation Twin Overlay.")
    print("4. Press [c] to switch camera index.")
    print("5. Press [q] or [ESC] to exit.\n")

    overlay_enabled = True
    alpha = 0.50

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.03)
            continue

        h, w = frame.shape[:2]
        centers, corners_map, corners, ids = calibrator.detect_markers(frame)

        display = frame.copy()

        # Draw guideline reference box for the 2x2 grid
        cv2.rectangle(
            display,
            (TARGET_BOX["u_left"], TARGET_BOX["v_top"]),
            (TARGET_BOX["u_right"], TARGET_BOX["v_bottom"]),
            (100, 100, 100),
            1,
            cv2.LINE_AA
        )
        cv2.putText(display, "Target 8x8cm Grid Zone", (TARGET_BOX["u_left"] + 5, TARGET_BOX["v_top"] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (140, 140, 140), 1)

        detected_ids = list(centers.keys()) if ids is not None else []
        required = [0, 1, 2, 3]
        missing = [mid for mid in required if mid not in detected_ids]

        if ids is not None and len(ids) > 0:
            cv2.aruco.drawDetectedMarkers(display, corners, ids)

        guidance_msgs = []
        is_aligned = False

        if missing:
            guidance_msgs.append(f"MISSING MARKERS: IDs {missing}")
            guidance_msgs.append("-> Frame the 4 touching markers [0,1]/[2,3] in view.")
        else:
            # Layout:
            # Top: ID 0 (Left), ID 1 (Right)
            # Bottom: ID 2 (Left), ID 3 (Right)
            c0 = centers[0]  # Top-Left
            c1 = centers[1]  # Top-Right
            c2 = centers[2]  # Bottom-Left
            c3 = centers[3]  # Bottom-Right

            center_x = (c0[0] + c1[0] + c2[0] + c3[0]) / 4.0
            center_y = (c0[1] + c1[1] + c2[1] + c3[1]) / 4.0

            # 1. Roll check: horizon between Top-Left (0) and Top-Right (1)
            dx_top = c1[0] - c0[0]
            dy_top = c1[1] - c0[1]
            roll_angle_deg = np.degrees(np.arctan2(dy_top, dx_top))

            if abs(roll_angle_deg) > 3.0:
                direction = "CLOCKWISE" if roll_angle_deg < 0 else "COUNTER-CLOCKWISE"
                guidance_msgs.append(f"TILT: Camera rotated {roll_angle_deg:.1f} deg -> Rotate camera {direction}.")

            # 2. Pan & Pitch check
            frame_center_x = w / 2.0
            frame_center_y = h / 2.0
            dx_center = center_x - frame_center_x
            dy_center = center_y - frame_center_y

            if abs(dx_center) > 30:
                guidance_msgs.append(f"PAN: Shift camera {'LEFT' if dx_center > 0 else 'RIGHT'}")
            if abs(dy_center) > 30:
                guidance_msgs.append(f"TILT PITCH: Tilt camera {'DOWN' if dy_center < 0 else 'UP'}")

            # 3. Distance check (width of 0->1 and 2->3)
            span_w = (c1[0] - c0[0] + c3[0] - c2[0]) / 2.0
            if span_w < 120:
                guidance_msgs.append("DISTANCE: Camera too FAR -> Move camera CLOSER.")
            elif span_w > 320:
                guidance_msgs.append("DISTANCE: Camera too CLOSE -> Move camera BACK.")

            if not guidance_msgs:
                is_aligned = True
                guidance_msgs.append("PHYSICAL ALIGNMENT EXCELLENT!")
                guidance_msgs.append("Micro-calibration active: Homography warping residual distortion.")

            # Compute homography rectification
            calibrator.calibrate(frame)

        if calibrator.is_calibrated and overlay_enabled:
            rectified_cam = cv2.warpPerspective(frame, calibrator.homography_matrix, (w, h))
            sim_rgb = env.render_frame()
            sim_bgr = cv2.cvtColor(sim_rgb, cv2.COLOR_RGB2BGR)

            # 0.5 opacity overlay
            blended = cv2.addWeighted(rectified_cam, alpha, sim_bgr, 1.0 - alpha, 0.0)
            display = blended

            cv2.rectangle(display, (0, 0), (w, 55), (0, 0, 0), -1)
            cv2.putText(
                display,
                f"MODE: 50/50 SIM-REAL OVERLAY | Grid: [0,1]/[2,3] ACTIVE",
                (15, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2
            )
            cv2.putText(
                display,
                "[SPACE] Toggle Overlay | [c] Switch Cam | [q] Exit",
                (15, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (200, 200, 200),
                1
            )
        else:
            box_color = (0, 255, 0) if is_aligned else (0, 0, 255)
            cv2.rectangle(display, (0, 0), (w, 65), (20, 20, 20), -1)
            status_title = "STATUS: ALIGNED (READY TO OVERLAY)" if is_aligned else "STATUS: ADJUST CAMERA AS GUIDED BELOW"
            cv2.putText(display, status_title, (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, box_color, 2)

            for i, msg in enumerate(guidance_msgs[:2]):
                cv2.putText(display, msg, (15, 48 + i * 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

        cv2.imshow("Dobot Camera Alignment & Twin Overlay", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord(' '):
            overlay_enabled = not overlay_enabled
        elif key == ord('c'):
            cap.release()
            cam_id = 1 if cam_id == 0 else 0
            print(f"[INFO] Switching to Camera ID {cam_id}...")
            cap = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap = cv2.VideoCapture(cam_id)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive ArUco Alignment (8x8cm Grid [0,1]/[2,3])")
    parser.add_argument("--cam", type=int, default=0, help="Camera device index (default: 0)")
    args = parser.parse_args()
    run_interactive_alignment(cam_id=args.cam)
