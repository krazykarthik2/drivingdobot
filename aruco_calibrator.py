"""
ArucoAutoCalibrator:
Updated to support 8cm x 8cm markers placed seamlessly side-by-side (2x2 grid or customizable bounds).
Automatically detects markers, computes perspective homography,
crops the workspace to an exact square (256x256 for Octo or custom size),
and maps pixel coordinates to physical Dobot coordinates.
"""
import cv2
import numpy as np

class ArucoAutoCalibrator:
    def __init__(self, output_size=(256, 256), dict_type=cv2.aruco.DICT_4X4_50, marker_size_mm=80.0, layout="2x2_grid"):
        """
        layout="2x2_grid":
            Four 80mm x 80mm markers touching side-by-side with zero space:
            Total width = 160 mm, Total height = 160 mm.
            Distance between center of ID 0 and ID 1 = 80 mm.
            
            ID 0 (Top-Left)    | ID 1 (Top-Right)
            -------------------+-------------------
            ID 3 (Bottom-Left) | ID 2 (Bottom-Right)

            Outer Corners / Extents:
                Top-Left (outer corner of ID 0): (0, 0) in output
                Top-Right (outer corner of ID 1): (w-1, 0)
                Bottom-Right (outer corner of ID 2): (w-1, h-1)
                Bottom-Left (outer corner of ID 3): (0, h-1)
        """
        self.output_size = output_size # (width, height)
        self.dict_type = dict_type
        self.dictionary = cv2.aruco.getPredefinedDictionary(dict_type)
        self.parameters = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.dictionary, self.parameters)
        
        self.marker_size_mm = marker_size_mm
        self.layout = layout
        # Full physical workspace size (table coverage)
        self.ws_w_mm = 360.0
        self.ws_h_mm = 360.0

        # Camera View Geometry (Looking from front of table towards Dobot):
        # - Camera Top (v=0): Near Dobot base (X = 80 mm)
        # - Camera Bottom (v=h-1): Far table edge (X = 360 mm)
        # - Camera Screen Left (u=0): Robot RIGHT (Y = -140 mm) [Viewer Left = Robot Right]
        # - Camera Screen Right (u=w-1): Robot LEFT (Y = +140 mm) [Viewer Right = Robot Left]
        #
        # Workspace Span:
        # X: [80 mm, 360 mm]  (Span = 280 mm)
        # Y: [-140 mm, +140 mm] (Span = 280 mm)
        #
        # 4 Markers (80mm x 80mm) placed touching each other right in front of the Dobot base:
        # - Row 1 (touching base): X = 80 + 40 = 120 mm
        # - Row 2 (further out):   X = 120 + 80 = 200 mm
        #
        # Physical to Pixel Conversion:
        # u = ((Y_phys - (-140.0)) / 280.0) * (w - 1)
        # v = ((X_phys - 80.0) / 280.0) * (h - 1)
        w, h = output_size
        ws_span_x = 280.0 # mm
        ws_span_y = 280.0 # mm
        x_min_mm = 80.0   # mm
        y_min_mm = -140.0 # mm

        def to_pixel(x_phys, y_phys):
            u = ((y_phys - y_min_mm) / ws_span_y) * (w - 1.0)
            v = ((x_phys - x_min_mm) / ws_span_x) * (h - 1.0)
            return [float(u), float(v)]

        # Marker layout:
        # ID 0 (Screen Left, near base -> Robot Right: Y=-40, X=120)
        # ID 1 (Screen Right, near base -> Robot Left: Y=+40, X=120)
        # ID 2 (Screen Right, further out -> Robot Left: Y=+40, X=200)
        # ID 3 (Screen Left, further out -> Robot Right: Y=-40, X=200)
        self.dst_center_pts = np.array([
            to_pixel(120.0, -40.0), # ID 0
            to_pixel(120.0, +40.0), # ID 1
            to_pixel(200.0, +40.0), # ID 2
            to_pixel(200.0, -40.0)  # ID 3
        ], dtype=np.float32)

        # Full table workspace outer corners (u, v)
        self.dst_outer_corners = np.array([
            [0, 0],         # Top-Left (Near base, Robot Right)
            [w - 1, 0],     # Top-Right (Near base, Robot Left)
            [w - 1, h - 1], # Bottom-Right (Far table, Robot Left)
            [0, h - 1]      # Bottom-Left (Far table, Robot Right)
        ], dtype=np.float32)

        self.homography_matrix = None
        self.is_calibrated = False

    def detect_markers(self, frame):
        """Detects ArUco markers and returns corner dict {id: center_point} and corner vertices."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        corners, ids, rejected = self.detector.detectMarkers(gray)
        
        marker_centers = {}
        marker_corners_map = {}
        if ids is not None and len(ids) > 0:
            for i, marker_id in enumerate(ids.flatten()):
                c = corners[i][0] # 4 corners: [TL, TR, BR, BL]
                cx = float(np.mean(c[:, 0]))
                cy = float(np.mean(c[:, 1]))
                marker_centers[int(marker_id)] = (cx, cy)
                marker_corners_map[int(marker_id)] = c
        return marker_centers, marker_corners_map, corners, ids

    def calibrate(self, frame):
        """Computes homography warp matrix using either the 4 centers or all 16 corner points."""
        centers, corners_map, _, _ = self.detect_markers(frame)
        required_ids = [0, 1, 2, 3]
        
        missing = [mid for mid in required_ids if mid not in centers]
        if missing:
            return False, f"Missing ArUco markers: IDs {missing}. Found: {list(centers.keys())}"

        # Method: Use the 4 marker centers mapped to exact 25% and 75% positions of the square
        src_pts = np.array([centers[mid] for mid in required_ids], dtype=np.float32)
        H, _ = cv2.findHomography(src_pts, self.dst_center_pts)
        self.homography_matrix = H
        self.is_calibrated = True
        return True, "Calibration successful! Homography computed."

    def process_frame(self, frame, auto_calibrate=True):
        """
        Crops, rectifies, and rotates the camera frame.
        Returns:
            calibrated_rgb: Exactly (H, W, 3) square crop.
            debug_view: Annotated view showing detections and bounding box.
        """
        debug_view = frame.copy()
        centers, corners_map, corners, ids = self.detect_markers(frame)
        
        if ids is not None and len(ids) > 0:
            cv2.aruco.drawDetectedMarkers(debug_view, corners, ids)
            for mid, (cx, cy) in centers.items():
                cv2.circle(debug_view, (int(cx), int(cy)), 5, (0, 0, 255), -1)
                cv2.putText(debug_view, f"ID {mid}", (int(cx) + 8, int(cy) - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        if auto_calibrate or not self.is_calibrated:
            if all(mid in centers for mid in [0, 1, 2, 3]):
                self.calibrate(frame)

        if self.is_calibrated:
            w, h = self.output_size
            calibrated = cv2.warpPerspective(frame, self.homography_matrix, (w, h))
            
            # Draw boundary polygon on debug view
            inv_H = np.linalg.inv(self.homography_matrix)
            poly_pts = cv2.perspectiveTransform(self.dst_outer_corners[None, :, :], inv_H)[0].astype(np.int32)
            cv2.polylines(debug_view, [poly_pts], isClosed=True, color=(0, 255, 0), thickness=2)
            cv2.putText(debug_view, "OCTO WORKSPACE (360x360mm Table)", (poly_pts[0][0], max(poly_pts[0][1] - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
            return calibrated, debug_view
        else:
            fallback = cv2.resize(frame, self.output_size)
            cv2.putText(debug_view, "WAITING FOR 4 ARUCO (0, 1, 2, 3)...", (25, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return fallback, debug_view
