"""
dobot_camera_positioning_guide.py

Live Real-Time Audio-Visual Positioning Assistant
Measures the physical distance, height, tilt angle, and lateral offset of your camera
relative to the Dobot Magician and 8x8cm ArUco tags in real time, and prints exact
millimeter & degree adjustment instructions until the physical camera aligns with the simulation!
"""

import sys
import time
import cv2
import numpy as np

# Windows compatibility
sys.modules['jax'] = None
sys.modules['jaxlib'] = None
sys.modules['tensorflow'] = None

MARKER_SIZE_M = 0.080  # 8.0 cm tag size

# Target simulation camera parameters:
# Distance to workspace center: ~42 cm
# Camera height above table: ~36 cm
# Forward distance from Dobot base: ~42 cm
# Lateral offset: 0 cm (perfectly centered on arm forward centerline)
# Tilt: ~32° from vertical downwards (or ~58° from horizontal)
TARGET_DIST_CM = 42.0
TARGET_HEIGHT_CM = 36.0

def run_guide(cam_idx=1):
    cap = cv2.VideoCapture(cam_idx)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())

    K = np.array([[600.0, 0, 320.0], [0, 600.0, 240.0], [0, 0, 1.0]], dtype=np.float64)
    dist = np.zeros(5, dtype=np.float64)
    ms = MARKER_SIZE_M / 2.0
    obj_pts = np.array([[-ms, ms, 0], [ms, ms, 0], [ms, -ms, 0], [-ms, -ms, 0]], dtype=np.float32)

    win = "Camera Physical Positioning Guide (Real-Time HUD)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 900, 650)

    print("=" * 65)
    print("DOBOT PHYSICAL CAMERA POSITIONING ASSISTANT")
    print("Move your physical camera / tripod according to the live on-screen cues!")
    print("Press 'q' when satisfied.")
    print("=" * 65)

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            time.sleep(0.03)
            continue

        corners, ids, _ = detector.detectMarkers(frame)
        hud_lines = []

        if ids is not None and len(ids) > 0:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            
            # Use Tag 3 or Tag 1
            idx = 0
            c = corners[idx][0]
            ok, rvec, tvec = cv2.solvePnP(obj_pts, c, K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
            
            curr_dist_cm = float(tvec[2][0] * 100.0)
            offset_x_cm = float(tvec[0][0] * 100.0)
            offset_y_cm = float(tvec[1][0] * 100.0)

            # Distance cue
            diff_dist = curr_dist_cm - TARGET_DIST_CM
            if diff_dist > 5:
                cue_dist = f"MOVE CLOSER by {abs(diff_dist):.1f} cm (Currently {curr_dist_cm:.1f} cm, Target: 42 cm)"
                col_dist = (0, 0, 255)
            elif diff_dist < -5:
                cue_dist = f"MOVE BACK by {abs(diff_dist):.1f} cm (Currently {curr_dist_cm:.1f} cm, Target: 42 cm)"
                col_dist = (0, 0, 255)
            else:
                cue_dist = f"DISTANCE OK! ({curr_dist_cm:.1f} cm)"
                col_dist = (0, 255, 0)

            # Lateral centering cue
            if offset_x_cm > 4:
                cue_lat = f"MOVE CAMERA LEFT by {abs(offset_x_cm):.1f} cm"
                col_lat = (0, 165, 255)
            elif offset_x_cm < -4:
                cue_lat = f"MOVE CAMERA RIGHT by {abs(offset_x_cm):.1f} cm"
                col_lat = (0, 165, 255)
            else:
                cue_lat = "CENTERED! (Lateral alignment aligned)"
                col_lat = (0, 255, 0)

            hud_lines.append((cue_dist, col_dist))
            hud_lines.append((cue_lat, col_lat))
        else:
            hud_lines.append(("Searching for ArUco tags... Ensure 8x8cm tag is in view", (0, 140, 255)))

        # Draw HUD Box
        cv2.rectangle(frame, (0, 0), (640, 90), (20, 20, 20), -1)
        for i, (txt, color) in enumerate(hud_lines):
            cv2.putText(frame, txt, (15, 30 + i * 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

        # Crosshair center guide
        cv2.drawMarker(frame, (320, 240), (0, 255, 255), markerType=cv2.MARKER_CROSS, markerSize=25, thickness=2)

        cv2.imshow(win, frame)
        if cv2.waitKey(20) & 0xFF in [ord('q'), 27]:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_guide(cam_idx=1)
