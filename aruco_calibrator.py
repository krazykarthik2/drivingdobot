"""
aruco_calibrator.py

Universal ArUco Camera-to-Robot Calibrator for Dobot Magician & SmolVLA.
Configured for:
Four 8cm x 8cm (80mm x 80mm) markers placed seamlessly side-by-side with NO GAPS in a 2x2 grid:
    Top Row:    [ ID 0 ] [ ID 1 ]
    Bottom Row: [ ID 2 ] [ ID 3 ]

Total Grid Footprint: 160 mm wide x 160 mm deep.
Center-to-Center Pitch between adjacent markers = exactly 80.0 mm.
"""

import cv2
import numpy as np


class ArucoAutoCalibrator:
    def __init__(
        self,
        output_size=(256, 256),
        dict_type=cv2.aruco.DICT_4X4_50,
        grid_base_x_mm=140.0,
        grid_center_y_mm=0.0,
        marker_size_mm=80.0
    ):
        """
        Marker Grid Geometry:
        Four 80mm x 80mm markers touching edge-to-edge with NO gaps:
            [ ID 0 ] (Top-Left)     |  [ ID 1 ] (Top-Right)
            ------------------------+------------------------
            [ ID 2 ] (Bottom-Left)  |  [ ID 3 ] (Bottom-Right)

        From the front camera perspective facing the Dobot base:
        - Top row (closer to Dobot base): X = grid_base_x_mm + 80 + 40 = 260 mm (or custom)
          ID 0 (Top-Left):  X = 260.0 mm, Y = -40.0 mm
          ID 1 (Top-Right): X = 260.0 mm, Y = +40.0 mm
        - Bottom row (closer to camera):
          ID 2 (Bottom-Left):  X = 180.0 mm, Y = -40.0 mm
          ID 3 (Bottom-Right): X = 180.0 mm, Y = +40.0 mm
        """
        self.output_size = output_size  # (width, height)
        self.dict_type = dict_type
        self.dictionary = cv2.aruco.getPredefinedDictionary(dict_type)
        self.parameters = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.dictionary, self.parameters)
        self.marker_size_mm = marker_size_mm

        # Physical Workspace Definition (in Dobot Millimeters)
        # Covers reach envelope: X in [100, 340], Y in [-150, 150]
        self.x_min_mm = 100.0
        self.x_max_mm = 340.0
        self.y_min_mm = -150.0
        self.y_max_mm = 150.0

        # Exact physical center coordinates for 80mm touching markers (NO GAPS):
        # Center of ID 0 to ID 1 distance = 80mm
        # Center of ID 2 to ID 3 distance = 80mm
        # Center of ID 0 to ID 2 distance = 80mm
        # Center of ID 1 to ID 3 distance = 80mm
        self.marker_phys_coords = {
            0: (260.0, -40.0),  # Top-Left (Near Base, Robot Right)
            1: (260.0, +40.0),  # Top-Right (Near Base, Robot Left)
            2: (180.0, -40.0),  # Bottom-Left (Front, Robot Right)
            3: (180.0, +40.0),  # Bottom-Right (Front, Robot Left)
        }

        # Precompute target pixel coordinates in rectified frame
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
        Computes the 3x3 homography matrix H using the exact 80x80mm seamless grid layout.
        Returns: (success: bool, status_message: str)
        """
        centers, _, _, _ = self.detect_markers(frame)
        required_ids = [0, 1, 2, 3]

        missing = [mid for mid in required_ids if mid not in centers]
        if missing:
            return False, f"Missing ArUco markers: IDs {missing}. Detected: {list(centers.keys())}"

        # Match detected points in exact order: ID 0, ID 1, ID 2, ID 3
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
                "CALIBRATED WORKSPACE (8x8cm Seamless Grid)",
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
