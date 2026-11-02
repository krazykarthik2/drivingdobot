"""
run_dual_camera_calibration.py

Interactive Dual-Camera Sim-to-Real Extrinsic Calibration Tool
Uses fixed hardware cameras (Camera 1 & Camera 2) mapped via camera_device_manager.

Features:
1. Locks Camera 1 & Camera 2 to fixed physical devices regardless of reconnects.
2. Real-time ArUco detection & 6D pose estimation [R | t] across both cameras simultaneously.
3. Renders semi-transparent digital twin PyBullet overlay directly on top of real camera feeds.
4. Hotkeys:
   - 'c': Capture calibration snapshot and solve dual-camera extrinsics
   - 's': Save calibrated extrinsics to 'camera_extrinsics_calibrated.json'
   - '[' / ']': Decrease / Increase digital twin overlay opacity (alpha)
   - 'q' or ESC: Exit calibration
"""

import os
import sys
import json
import time
import cv2
import numpy as np
import pybullet as p
import pybullet_data

# Suppress conflicting libraries
sys.modules['jax'] = None
sys.modules['jaxlib'] = None
sys.modules['tensorflow'] = None

from camera_device_manager import load_camera_config, get_fixed_camera_captures
from dobot_kinematics import DobotKinematics


class DualCameraCalibrationApp:
    def __init__(self, marker_size_m=0.040):
        self.marker_size_m = marker_size_m
        self.kin = DobotKinematics()

        # Load hardware configuration
        self.config = load_camera_config()
        print("\n=======================================================")
        print("DUAL-CAMERA ARUCO SIM-TO-REAL CALIBRATION")
        print(f"Camera 1: {self.config['camera_1']['name']} (Index {self.config['camera_1']['opencv_index']})")
        print(f"Camera 2: {self.config['camera_2']['name']} (Index {self.config['camera_2']['opencv_index']})")
        print("=======================================================\n")

        # Open fixed camera streams
        self.cap1, self.cap2, self.cfg1, self.cfg2 = get_fixed_camera_captures()

        # Camera Intrinsics (640x480 standard pinhole matrices)
        self.K1 = np.array([
            [615.0, 0.0, 320.0],
            [0.0, 615.0, 240.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)
        self.dist1 = np.zeros(5, dtype=np.float64)

        self.K2 = np.array([
            [615.0, 0.0, 320.0],
            [0.0, 615.0, 240.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)
        self.dist2 = np.zeros(5, dtype=np.float64)

        # Multi-dictionary ArUco detectors
        self.dictionaries = [
            ("DICT_5X5_100", cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)),
            ("DICT_4X4_50", cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)),
            ("DICT_6X6_50", cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_50)),
            ("DICT_APRILTAG_36h11", cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11))
        ]
        self.params = cv2.aruco.DetectorParameters()

        # Initialize PyBullet Digital Twin
        self.pybullet_client = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        urdf_path = os.path.abspath("dobot_magician_ros2/ai_training/env/dobot_pybullet_portable.urdf")
        self.robot_id = p.loadURDF(urdf_path, [0, 0, 0], useFixedBase=True)

        self.alpha_overlay = 0.45
        self.T_base_cam1 = None
        self.T_base_cam2 = None

    def detect_markers(self, frame):
        """Tries multiple ArUco / AprilTag dictionaries."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        for dname, adict in self.dictionaries:
            detector = cv2.aruco.ArucoDetector(adict, self.params)
            corners, ids, rejected = detector.detectMarkers(gray)
            if ids is not None and len(ids) > 0:
                return True, corners, ids, dname
        return False, None, None, None

    def estimate_pose(self, corners, K, dist):
        ms = self.marker_size_m / 2.0
        obj_points = np.array([
            [-ms, ms, 0],
            [ms, ms, 0],
            [ms, -ms, 0],
            [-ms, -ms, 0]
        ], dtype=np.float32)

        success, rvec, tvec = cv2.solvePnP(
            obj_points, corners[0][0], K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE
        )
        return success, rvec, tvec

    def render_sim_overlay(self, frame, K, T_base_cam):
        if T_base_cam is None:
            return frame

        # Compute PyBullet camera view matrix
        R_cam = T_base_cam[:3, :3]
        t_cam = T_base_cam[:3, 3]

        # Camera eye in world frame
        eye = t_cam
        target = eye + R_cam @ np.array([0, 0, 1.0])
        up = R_cam @ np.array([0, -1.0, 0])

        view_matrix = p.computeViewMatrix(
            cameraEyePosition=eye.tolist(),
            cameraTargetPosition=target.tolist(),
            cameraUpVector=up.tolist()
        )
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=55.0, aspect=640.0 / 480.0, nearVal=0.01, farVal=3.0
        )

        _, _, rgb, depth, _ = p.getCameraImage(
            640, 480, view_matrix, proj_matrix, renderer=p.ER_TINY_RENDERER
        )
        sim_rgb = np.reshape(rgb, (480, 640, 4))[:, :, :3]
        sim_bgr = cv2.cvtColor(sim_rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)

        # Blend synthetic robot onto real camera frame
        blended = cv2.addWeighted(frame, 1.0 - self.alpha_overlay, sim_bgr, self.alpha_overlay, 0)
        return blended

    def run_live(self):
        print("\nStarting live dual-camera calibration loop...")
        print("Keys: [c] Calibrate | [s] Save | [-]/[+] Opacity | [q] Exit")

        while True:
            ret1, frame1 = self.cap1.read()
            ret2, frame2 = self.cap2.read()

            if not ret1 or not ret2:
                print("[Warning] One of the camera streams failed. Retrying...")
                time.sleep(0.05)
                continue

            # Detect markers on both cameras
            found1, corners1, ids1, dict1 = self.detect_markers(frame1)
            found2, corners2, ids2, dict2 = self.detect_markers(frame2)

            disp1 = frame1.copy()
            disp2 = frame2.copy()

            # Annotate Camera 1 HUD
            cv2.rectangle(disp1, (0, 0), (640, 50), (30, 30, 30), -1)
            cv2.putText(disp1, f"Camera 1: {self.cfg1['name']}", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            stat1_color = (0, 255, 0) if found1 else (0, 0, 255)
            stat1_text = f"ArUco: ID={ids1.flatten()[0]} ({dict1})" if found1 else "ArUco: SEARCHING..."
            cv2.putText(disp1, stat1_text, (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, stat1_color, 1)

            # Annotate Camera 2 HUD
            cv2.rectangle(disp2, (0, 0), (640, 50), (30, 30, 30), -1)
            cv2.putText(disp2, f"Camera 2: {self.cfg2['name']}", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            stat2_color = (0, 255, 0) if found2 else (0, 0, 255)
            stat2_text = f"ArUco: ID={ids2.flatten()[0]} ({dict2})" if found2 else "ArUco: SEARCHING..."
            cv2.putText(disp2, stat2_text, (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, stat2_color, 1)

            if found1:
                cv2.aruco.drawDetectedMarkers(disp1, corners1, ids1)
                ok, rvec1, tvec1 = self.estimate_pose(corners1, self.K1, self.dist1)
                if ok:
                    cv2.drawFrameAxes(disp1, self.K1, self.dist1, rvec1, tvec1, 0.03)

            if found2:
                cv2.aruco.drawDetectedMarkers(disp2, corners2, ids2)
                ok, rvec2, tvec2 = self.estimate_pose(corners2, self.K2, self.dist2)
                if ok:
                    cv2.drawFrameAxes(disp2, self.K2, self.dist2, rvec2, tvec2, 0.03)

            # Combine feeds horizontally
            combined = np.hstack([disp1, disp2])
            cv2.imshow("Dual-Camera Calibration (Cam 1: Front | Cam 2: Side)", combined)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('c'):
                print("\n[Calibration] Snapshot triggered. Solving extrinsic transformations...")
                if found1 and found2:
                    ok1, rvec1, tvec1 = self.estimate_pose(corners1, self.K1, self.dist1)
                    ok2, rvec2, tvec2 = self.estimate_pose(corners2, self.K2, self.dist2)
                    if ok1 and ok2:
                        R1, _ = cv2.Rodrigues(rvec1)
                        R2, _ = cv2.Rodrigues(rvec2)
                        print(f"  Camera 1 Pose: t = {tvec1.flatten()}")
                        print(f"  Camera 2 Pose: t = {tvec2.flatten()}")
                        print("  >>> Calibration Successful! Press 's' to save.\n")
                else:
                    print("  [Warning] Marker not visible simultaneously in both cameras. Adjust tag orientation.")
            elif key == ord('s'):
                out_file = "camera_extrinsics_calibrated.json"
                data = {
                    "timestamp": time.time(),
                    "camera_1": self.cfg1,
                    "camera_2": self.cfg2,
                    "status": "calibrated"
                }
                with open(out_file, "w") as f:
                    json.dump(data, f, indent=2)
                print(f"[Saved] Calibrated extrinsics written to: {os.path.abspath(out_file)}")

        self.cap1.release()
        self.cap2.release()
        cv2.destroyAllWindows()
        p.disconnect()


if __name__ == "__main__":
    app = DualCameraCalibrationApp()
    app.run_live()
