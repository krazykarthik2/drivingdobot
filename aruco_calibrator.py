"""
aruco_calibrator.py

Universal ArUco Camera-to-Robot Calibrator for Dobot Magician & SmolVLA.
- Detects 4 table markers (IDs 0, 1, 2, 3) anywhere in the camera view.
- Computes perspective homography to automatically rectify tilt, height, and angle variations.
- Warps arbitrary webcam views into canonical 256x256 RGB observations for SmolVLA.
- Provides direct bidirectional mapping:
    (camera pixel u, v) <---> (Dobot physical X_mm, Y_mm)
"""

import cv2
import numpy as np


class ArucoAutoCalibrator:
    def __init__(
        self,
        output_size=(256, 256),
        dict_type=cv2.aruco.DICT_4X4_50,
        layout="perimeter"
    ):
        """
        layout="perimeter":
            ID 0 (Top-Left):     X = +300 mm, Y = -150 mm
            ID 1 (Top-Right):    X = +300 mm, Y = +150 mm
            ID 2 (Bottom-Right): X = +150 mm, Y = +150 mm
            ID 3 (Bottom-Left):  X = +150 mm, Y = -150 mm

        layout="compact":
            4 touching 80mm markers near Dobot base:
            ID 0: X = 120 mm, Y = -40 mm
            ID 1: X = 120 mm, Y = +40 mm
            ID 2: X = 200 mm, Y = +40 mm
            ID 3: X = 200 mm, Y = -40 mm
        """
        self.output_size = output_size  # (width, height)
        self.dict_type = dict_type
        self.dictionary = cv2.aruco.getPredefinedDictionary(dict_type)
        self.parameters = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.dictionary, self.parameters)
        self.layout = layout

        # Physical Workspace Definition (in Dobot Millimeters)
        # Looking from front of table towards Dobot:
        # Screen Top (v=0): Far from camera, near Dobot base (e.g. X = +350 mm)
        # Screen Bottom (v=h-1): Close to camera/edge (e.g. X = +100 mm)
        # Screen Left (u=0): Y = -160 mm
        # Screen Right (u=w-1): Y = +160 mm
        self.x_min_mm = 100.0
        self.x_max_mm = 350.0
        self.y_min_mm = -160.0
        self.y_max_mm = 160.0

        w, h = self.output_size

        if layout == "perimeter":
            # Real table perimeter positions for IDs 0, 1, 2, 3
            self.marker_phys_coords = {
                0: (300.0, -150.0),
                1: (300.0, +150.0),
                2: (150.0, +150.0),
                3: (150.0, -150.0),
            }
        else:
            self.marker_phys_coords = {
                0: (120.0, -40.0),
                1: (120.0, +40.0),
                2: (200.0, +40.0),
                3: (200.0, -40.0),
            }

        # Convert physical (X_mm, Y_mm) to target rectified output pixels (u, v)
        self.dst_center_pts = np.array([
            self.dobot_to_pixel(*self.marker_phys_coords[0]),
            self.dobot_to_pixel(*self.marker_phys_coords[1]),
            self.dobot_to_pixel(*self.marker_phys_coords[2]),
            self.dobot_to_pixel(*self.marker_phys_coords[3]),
        ], dtype=np.float32)

        self.homography_matrix = None
        self.inv_homography_matrix = None
        self.is_calibrated = False

    def dobot_to_pixel(self, x_mm: float, y_mm: float) -> list:
        """Maps Dobot physical coordinates (mm) to rectified output pixel (u, v)."""
        w, h = self.output_size
        u = ((y_mm - self.y_min_mm) / (self.y_max_mm - self.y_min_mm)) * (w - 1.0)
        # X increases going towards base (screen top)
        v = ((self.x_max_mm - x_mm) / (self.x_max_mm - self.x_min_mm)) * (h - 1.0)
        return [float(u), float(v)]

    def pixel_to_dobot(self, u: float, v: float) -> tuple:
        """Maps rectified output pixel (u, v) back to Dobot physical coordinates (x_mm, y_mm)."""
        w, h = self.output_size
        y_mm = self.y_min_mm + (u / (w - 1.0)) * (self.y_max_mm - self.y_min_mm)
        x_mm = self.x_max_mm - (v / (h - 1.0)) * (self.x_max_mm - self.x_min_mm)
        return float(x_mm), float(y_mm)

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
        Computes the 3x3 homography matrix H from live camera frame to canonical output space.
        Returns: (success: bool, status_message: str)
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
        Warps input frame to canonical rectified VLA observation.
        Returns:
            rectified_rgb: (H, W, 3) undistorted, canonical workspace image.
            debug_view: Original image annotated with detection markers & rectified bounds.
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
            # Warp webcam frame to canonical coordinate plane
            rectified = cv2.warpPerspective(frame, self.homography_matrix, (w, h))

            # Project canonical bounding box back onto camera debug view
            rect_corners = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
            poly_pts = cv2.perspectiveTransform(rect_corners[None, :, :], self.inv_homography_matrix)[0].astype(np.int32)
            cv2.polylines(debug_view, [poly_pts], isClosed=True, color=(0, 255, 0), thickness=2)
            cv2.putText(
                debug_view,
                "CALIBRATED WORKSPACE",
                (poly_pts[0][0], max(poly_pts[0][1] - 10, 25)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2
            )
            return rectified, debug_view
        else:
            fallback = cv2.resize(frame, (w, h))
            cv2.putText(
                debug_view,
                "WAITING FOR 4 ARUCO (0, 1, 2, 3)...",
                (25, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 0, 255),
                2
            )
            return fallback, debug_view
