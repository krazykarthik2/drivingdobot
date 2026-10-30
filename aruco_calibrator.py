"""
aruco_calibrator.py

Universal ArUco Camera-to-Robot Calibrator for Dobot Magician & SmolVLA.
Configured for:
Four 8cm x 8cm (80mm x 80mm) markers placed seamlessly side-by-side with NO GAPS in a 2x2 grid:
    Top Row:    [ ID 0 ] [ ID 1 ]
    Bottom Row: [ ID 2 ] [ ID 3 ]

Provides two rectification modes:
1. "sim_projection" (DEFAULT): Warps the real webcam feed into the EXACT 3D camera projection
   of the PyBullet simulation environment (45° elevated front view). This ensures that physical
   markers, tables, and objects align with 0.5 opacity directly onto the virtual 3D Dobot twin.
2. "orthographic": Warps into a top-down flattened grid view.
"""

import cv2
import numpy as np
import pybullet as p


class ArucoAutoCalibrator:
    def __init__(
        self,
        output_size=(640, 480),
        dict_type=cv2.aruco.DICT_4X4_50,
        mode="sim_projection",
        marker_size_mm=80.0
    ):
        """
        mode="sim_projection":
            Computes homography directly into the PyBullet simulation's camera projection coordinates.
            PyBullet Camera Setup:
                Eye: [0.42, 0.0, 0.32], Target: [0.20, 0.0, -0.04], Up: [0.0, 0.0, 1.0], FOV: 45.0
            Physical Table Surface: Z = -0.050 m
        """
        self.output_size = output_size  # (width, height)
        self.dict_type = dict_type
        self.dictionary = cv2.aruco.getPredefinedDictionary(dict_type)
        self.parameters = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.dictionary, self.parameters)
        self.mode = mode
        self.marker_size_mm = marker_size_mm

        w, h = self.output_size

        # Physical marker center coordinates in meters in Dobot / PyBullet coordinates:
        # 80mm x 80mm markers touching with no gaps:
        # ID 0 (Top-Left):  X = 0.260 m, Y = -0.040 m, Z = -0.050 m (tabletop)
        # ID 1 (Top-Right): X = 0.260 m, Y = +0.040 m, Z = -0.050 m
        # ID 2 (Bottom-Left):  X = 0.180 m, Y = -0.040 m, Z = -0.050 m
        # ID 3 (Bottom-Right): X = 0.180 m, Y = +0.040 m, Z = -0.050 m
        self.table_z = -0.050  # tabletop height in PyBullet
        self.marker_world_pts = {
            0: [0.260, -0.040, self.table_z],
            1: [0.260,  0.040, self.table_z],
            2: [0.180, -0.040, self.table_z],
            3: [0.180,  0.040, self.table_z],
        }

        if mode == "sim_projection":
            # Compute the exact 3D camera projection matrix from PyBullet
            cam_eye = [0.42, 0.0, 0.32]
            cam_target = [0.20, 0.0, -0.04]
            cam_up = [0.0, 0.0, 1.0]
            fov = 45.0

            view_matrix = p.computeViewMatrix(cam_eye, cam_target, cam_up)
            proj_matrix = p.computeProjectionMatrixFOV(fov, 1.0, 0.05, 2.5)

            V = np.array(view_matrix).reshape((4, 4), order='F')
            P = np.array(proj_matrix).reshape((4, 4), order='F')
            self.VP = P @ V

            # Project the 4 marker centers into the exact simulation camera pixel coordinates
            self.dst_center_pts = np.array([
                self.project_world_to_sim_pixel(self.marker_world_pts[0]),
                self.project_world_to_sim_pixel(self.marker_world_pts[1]),
                self.project_world_to_sim_pixel(self.marker_world_pts[2]),
                self.project_world_to_sim_pixel(self.marker_world_pts[3]),
            ], dtype=np.float32)

        else:
            # Orthographic top-down flattened grid
            self.x_min_mm, self.x_max_mm = 100.0, 340.0
            self.y_min_mm, self.y_max_mm = -150.0, 150.0
            self.dst_center_pts = np.array([
                self._ortho_to_pixel(260.0, -40.0),
                self._ortho_to_pixel(260.0,  40.0),
                self._ortho_to_pixel(180.0, -40.0),
                self._ortho_to_pixel(180.0,  40.0),
            ], dtype=np.float32)

        self.homography_matrix = None
        self.inv_homography_matrix = None
        self.is_calibrated = False

    def project_world_to_sim_pixel(self, pt_3d):
        """Projects a 3D world coordinate [x, y, z] into PyBullet simulation camera pixel (u, v)."""
        w, h = self.output_size
        clip = self.VP @ np.array([pt_3d[0], pt_3d[1], pt_3d[2], 1.0])
        ndc = clip[:3] / clip[3]
        u = ((ndc[0] + 1.0) / 2.0) * (w - 1.0)
        v = ((1.0 - ndc[1]) / 2.0) * (h - 1.0)
        return [float(u), float(v)]

    def _ortho_to_pixel(self, x_mm, y_mm):
        w, h = self.output_size
        u = ((y_mm - self.y_min_mm) / (self.y_max_mm - self.y_min_mm)) * (w - 1.0)
        v = ((self.x_max_mm - x_mm) / (self.x_max_mm - self.x_min_mm)) * (h - 1.0)
        return [float(u), float(v)]

    def reset_calibration(self):
        """Resets the computed homography matrix and forces recalibration."""
        self.homography_matrix = None
        self.inv_homography_matrix = None
        self.is_calibrated = False

    def detect_markers(self, frame: np.ndarray):
        """Detects ArUco markers and returns corner centers and boundaries."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        corners, ids, rejected = self.detector.detectMarkers(gray)

        marker_centers = {}
        marker_corners_map = {}
        if ids is not None and len(ids) > 0:
            for i, marker_id in enumerate(ids.flatten()):
                c = corners[i][0]
                cx = float(np.mean(c[:, 0]))
                cy = float(np.mean(c[:, 1]))
                marker_centers[int(marker_id)] = (cx, cy)
                marker_corners_map[int(marker_id)] = c
        return marker_centers, marker_corners_map, corners, ids

    def calibrate(self, frame: np.ndarray) -> tuple:
        """
        Computes 3x3 homography matrix H to match the simulation's 3D perspective camera projection.
        """
        centers, _, _, _ = self.detect_markers(frame)
        required_ids = [0, 1, 2, 3]

        missing = [mid for mid in required_ids if mid not in centers]
        if missing:
            return False, f"Missing ArUco markers: IDs {missing}. Detected: {list(centers.keys())}"

        src_pts = np.array([centers[mid] for mid in required_ids], dtype=np.float32)
        H, _ = cv2.findHomography(src_pts, self.dst_center_pts)

        if H is None:
            return False, "Failed to compute homography matrix (degenerate points)."

        self.homography_matrix = H
        self.inv_homography_matrix = np.linalg.inv(H)
        self.is_calibrated = True
        return True, "Calibration successful! Homography matrix computed."

    def process_frame(self, frame: np.ndarray, auto_calibrate: bool = True) -> tuple:
        """
        Warps input frame to canonical rectified VLA observation matching simulation view.
        """
        debug_view = frame.copy()
        centers, _, corners, ids = self.detect_markers(frame)

        if ids is not None and len(ids) > 0:
            cv2.aruco.drawDetectedMarkers(debug_view, corners, ids)
            for mid, (cx, cy) in centers.items():
                cv2.circle(debug_view, (int(cx), int(cy)), 5, (0, 0, 255), -1)
                cv2.putText(
                    debug_view,
                    f"ID {mid}",
                    (int(cx) + 8, int(cy) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    2
                )

        if auto_calibrate or not self.is_calibrated:
            if all(mid in centers for mid in [0, 1, 2, 3]):
                self.calibrate(frame)

        w, h = self.output_size
        if self.is_calibrated:
            rectified = cv2.warpPerspective(frame, self.homography_matrix, (w, h))
            return rectified, debug_view
        else:
            fallback = cv2.resize(frame, (w, h))
            return fallback, debug_view
