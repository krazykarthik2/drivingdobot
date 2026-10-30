"""
interactive_aruco_alignment.py

Interactive Live Camera Alignment, ArUco Rectification, and PyBullet Digital Twin Overlay.
- Detects the 4 physical ArUco markers on your table (IDs 0, 1, 2, 3).
- Guides your physical camera adjustment (Distance, Height, Yaw/Tilt/Centering) via real-time HUD prompts.
- Once coarse alignment is achieved, automatically computes homography to rectify residual micro-errors.
- Blends the live physical camera stream directly over the PyBullet simulation digital twin with 0.5 opacity
  so physical objects and the virtual robot match seamlessly.
- Ready to receive interactive user prompts to drive the Dobot.
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
    print("DOBOT MAGICIAN ARUCO ALIGNMENT & DIGITAL TWIN OVERLAY")
    print("=" * 70)
    print(f"[INFO] Opening Webcam Device ID: {cam_id}...")

    cap = cv2.VideoCapture(cam_id)
    if not cap.isOpened():
        print(f"[ERROR] Could not open camera {cam_id}. Trying camera 1...")
        cap = cv2.VideoCapture(1)
        if not cap.isOpened():
            print("[ERROR] No camera available!")
            return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Initialize Calibrator
    calibrator = ArucoAutoCalibrator(output_size=(640, 480), layout="perimeter")

    # Initialize PyBullet Digital Twin (Direct Mode for background rendering)
    print("[INFO] Initializing PyBullet Digital Twin Environment...")
    env = DobotColorBoxEnv(render_gui=False, width=640, height=480)
    
    # Spawn virtual objects matching physical calibration markers
    # Red block at [0.24, 0.08] and Green plate at [0.22, -0.10]
    task_config = [
        {"color": "red", "box_pos": [0.24, 0.08], "plate_pos": [0.22, -0.10], "plate_color": "green"}
    ]
    env.spawn_colored_boxes_and_plates(task_config)

    # Ideal target positions in camera frame (640x480) for well-aligned camera
    # Top markers ID 0 & 1 should be around v = 120
    # Bottom markers ID 3 & 2 should be around v = 380
    # Left markers ID 0 & 3 should be around u = 140
    # Right markers ID 1 & 2 should be around u = 500
    TARGET_BOX = {
        "u_left": 140,
        "u_right": 500,
        "v_top": 120,
        "v_bottom": 380,
    }

    cv2.namedWindow("Dobot Camera Alignment & Twin Overlay", cv2.WINDOW_AUTOSIZE)

    print("\n--- INSTRUCTIONS ---")
    print("1. Follow the on-screen arrows and guidance to adjust your camera height & angle.")
    print("2. When alignment turns GREEN, homography rectifies residual micro-errors.")
    print("3. Press [SPACE] to toggle 0.5 Opacity Simulation Twin Overlay.")
    print("4. Press [c] to switch camera index.")
    print("5. Press [q] or [ESC] to exit.\n")

    overlay_enabled = True
    alpha = 0.50

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Failed to grab frame from camera.")
            time.sleep(0.05)
            continue

        h, w = frame.shape[:2]
        centers, corners_map, corners, ids = calibrator.detect_markers(frame)

        # Base display canvas
        display = frame.copy()

        # Draw guideline reference box (where the 4 markers ideally sit for perfect perspective)
        cv2.rectangle(
            display,
            (TARGET_BOX["u_left"], TARGET_BOX["v_top"]),
            (TARGET_BOX["u_right"], TARGET_BOX["v_bottom"]),
            (80, 80, 80),
            1,
            cv2.LINE_AA
        )
        cv2.putText(display, "Target Framing Zone", (TARGET_BOX["u_left"] + 5, TARGET_BOX["v_top"] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1)

        # Analyze detected markers
        detected_ids = list(centers.keys()) if ids is not None else []
        required = [0, 1, 2, 3]
        missing = [mid for mid in required if mid not in detected_ids]

        # Draw detected ArUco tags
        if ids is not None and len(ids) > 0:
            cv2.aruco.drawDetectedMarkers(display, corners, ids)

        # Guidance logic
        guidance_msgs = []
        is_aligned = False

        if missing:
            guidance_msgs.append(f"MISSING MARKERS: IDs {missing}")
            guidance_msgs.append("-> Rotate/zoom camera so all 4 ArUco markers are visible on table.")
        else:
            # All 4 markers detected! Check geometric balance & framing
            c0 = centers[0]  # Top-Left
            c1 = centers[1]  # Top-Right
            c2 = centers[2]  # Bottom-Right
            c3 = centers[3]  # Bottom-Left

            # Center of the 4 markers
            center_x = (c0[0] + c1[0] + c2[0] + c3[0]) / 4.0
            center_y = (c0[1] + c1[1] + c2[1] + c3[1]) / 4.0

            # Tilt / Roll angle (horizon between ID 0 and ID 1)
            dx_top = c1[0] - c0[0]
            dy_top = c1[1] - c0[1]
            roll_angle_deg = np.degrees(np.arctan2(dy_top, dx_top))

            # Span width and height
            span_w = (c1[0] - c0[0] + c2[0] - c3[0]) / 2.0
            span_h = (c3[1] - c0[1] + c2[1] - c1[1]) / 2.0

            # 1. Roll check (camera tilt)
            if abs(roll_angle_deg) > 3.0:
                direction = "CLOCKWISE" if roll_angle_deg < 0 else "COUNTER-CLOCKWISE"
                guidance_msgs.append(f"TILT: Camera is rotated {roll_angle_deg:.1f} deg -> Rotate camera {direction}.")

            # 2. Centering check (Pan)
            frame_center_x = w / 2.0
            frame_center_y = h / 2.0
            dx_center = center_x - frame_center_x
            dy_center = center_y - frame_center_y

            if abs(dx_center) > 35:
                guidance_msgs.append(f"PAN: Shift camera {'LEFT' if dx_center > 0 else 'RIGHT'}")
            if abs(dy_center) > 35:
                guidance_msgs.append(f"TILT PITCH: Tilt camera {'DOWN' if dy_center < 0 else 'UP'}")

            # 3. Distance / Zoom check
            if span_w < 260:
                guidance_msgs.append("DISTANCE: Camera too FAR -> Move camera CLOSER to table.")
            elif span_w > 480:
                guidance_msgs.append("DISTANCE: Camera too CLOSE -> Move camera BACK.")

            if not guidance_msgs:
                is_aligned = True
                guidance_msgs.append("PHYSICAL ALIGNMENT EXCELLENT!")
                guidance_msgs.append("Micro-calibration active: Homography warping residual distortion.")

            # Compute Homography rectification for micro-errors
            calibrator.calibrate(frame)

        # If calibrated and overlay is enabled, blend simulation twin
        if calibrator.is_calibrated and overlay_enabled:
            # Warp the live camera frame to canonical workspace coordinates
            rectified_cam = cv2.warpPerspective(frame, calibrator.homography_matrix, (w, h))

            # Render matching PyBullet simulation camera
            sim_rgb = env.render_frame()
            sim_bgr = cv2.cvtColor(sim_rgb, cv2.COLOR_RGB2BGR)

            # Overlay with exact 0.5 opacity
            blended = cv2.addWeighted(rectified_cam, alpha, sim_bgr, 1.0 - alpha, 0.0)
            
            # Use blended image as main view
            display = blended

            # Draw HUD status
            cv2.rectangle(display, (0, 0), (w, 55), (0, 0, 0), -1)
            cv2.putText(
                display,
                f"MODE: 50/50 SIM-REAL OVERLAY | Calibration: ACTIVE",
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
            # Draw HUD guidance box on camera view
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
            print(f"[INFO] Overlay toggled: {overlay_enabled}")
        elif key == ord('c'):
            cap.release()
            cam_id = 1 if cam_id == 0 else 0
            print(f"[INFO] Switching to Camera ID {cam_id}...")
            cap = cv2.VideoCapture(cam_id)

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Alignment tool closed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive ArUco Camera Alignment & PyBullet Overlay")
    parser.add_argument("--cam", type=int, default=0, help="Camera device index (default: 0)")
    args = parser.parse_args()
    run_interactive_alignment(cam_id=args.cam)
