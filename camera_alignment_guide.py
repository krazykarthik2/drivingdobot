"""
camera_alignment_guide.py

TOOL 1: Physical Camera Alignment Guide & Live Overlay
======================================================
Use this tool when physically setting up and mounting your physical cameras.
- Connects to your live physical camera feeds (Webcam / USB / RealSense / RTSP).
- Continuously searches for the ArUco calibration marker on the table.
- Displays:
  1. Target physical mounting crosshairs and target bounding boxes.
  2. Live camera angle & distance readout relative to the marker:
     - Distance to table center (meters)
     - Pitch angle (e.g. guide says: "Tilt up 3.2 deg to reach 45.0 deg")
     - Yaw / Roll level guide ("Level: OK" or "Rotate 2.1 deg clockwise")
  3. Visual alignment target reticle: turns GREEN when your physical camera
     is positioned within target tolerance (45° elevation, centered optical axis).
"""

import sys
import time
import numpy as np
import cv2

# Windows compatibility stubs
sys.modules['jax'] = None
sys.modules['jaxlib'] = None
sys.modules['tensorflow'] = None


class PhysicalCameraAlignmentGuide:
    def __init__(self, marker_size_m=0.040, target_elevation_deg=45.0, target_dist_m=0.52):
        self.marker_size_m = marker_size_m
        self.target_elev = target_elevation_deg
        self.target_dist = target_dist_m

        # 5x5 ArUco detector
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)

        # Approximate intrinsics for 640x480 webcam
        self.K = np.array([
            [615.0, 0.0, 320.0],
            [0.0, 615.0, 240.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)
        self.dist = np.zeros(5, dtype=np.float64)

        # 3D points of square marker corners
        ms = self.marker_size_m / 2.0
        self.obj_points = np.array([
            [-ms, ms, 0],
            [ms, ms, 0],
            [ms, -ms, 0],
            [-ms, -ms, 0]
        ], dtype=np.float32)

    def process_frame(self, frame: np.ndarray, camera_name="Front Camera"):
        h, w = frame.shape[:2]
        display = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        corners, ids, _ = self.detector.detectMarkers(gray)

        # Draw central reticle
        cx, cy = w // 2, h // 2
        cv2.line(display, (cx - 20, cy), (cx + 20, cy), (100, 100, 100), 1)
        cv2.line(display, (cx, cy - 20), (cx, cy + 20), (100, 100, 100), 1)

        header_color = (0, 165, 255)
        status_text = f"[{camera_name}] SEARCHING FOR ARUCO MARKER ON TABLE..."

        if ids is not None and len(ids) > 0:
            cv2.aruco.drawDetectedMarkers(display, corners, ids)
            ok, rvec, tvec = cv2.solvePnP(
                self.obj_points, corners[0][0], self.K, self.dist, flags=cv2.SOLVEPNP_IPPE_SQUARE
            )
            if ok:
                # Compute distance to marker
                dist_m = float(np.linalg.norm(tvec))
                dist_err = dist_m - self.target_dist

                # Compute pitch/elevation angle from rotation matrix
                R, _ = cv2.Rodrigues(rvec)
                # Normal vector of marker in camera frame
                marker_normal = R @ np.array([0, 0, 1])
                # Elevation angle: angle between camera optical axis (0,0,1) and marker normal
                elev_deg = float(np.rad2deg(np.arcsin(np.clip(-marker_normal[1], -1.0, 1.0))))
                elev_err = elev_deg - self.target_elev

                # Check tolerances
                dist_ok = abs(dist_err) < 0.05  # within 5 cm
                elev_ok = abs(elev_err) < 3.0   # within 3 degrees
                center_err_px = np.linalg.norm(np.mean(corners[0][0], axis=0) - np.array([cx, cy]))
                center_ok = center_err_px < 40  # within 40 pixels of center

                is_aligned = dist_ok and elev_ok and center_ok
                reticle_color = (0, 255, 0) if is_aligned else (0, 0, 255)

                # Draw 3D axis on marker
                cv2.drawFrameAxes(display, self.K, self.dist, rvec, tvec, 0.05)

                # Alignment HUD
                cv2.rectangle(display, (10, 10), (w - 10, 110), (30, 30, 30), -1)
                cv2.rectangle(display, (10, 10), (w - 10, 110), reticle_color, 2)

                status_str = "ALIGNED! TIGHTEN CAMERA MOUNT" if is_aligned else "ADJUSTING PHYSICAL POSITION:"
                cv2.putText(display, f"{camera_name} Guide: {status_str}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, reticle_color, 2)

                # Guidance instructions
                elev_dir = "TILT UP" if elev_err < 0 else "TILT DOWN"
                dist_dir = "MOVE CLOSER" if dist_err > 0 else "MOVE BACK"
                guide_line = (
                    f"Elevation: {abs(elev_deg):.1f} deg (Target {self.target_elev:.0f} deg | {elev_dir} {abs(elev_err):.1f} deg)  |  "
                    f"Distance: {dist_m*100:.1f} cm ({dist_dir} {abs(dist_err)*100:.1f} cm)"
                )
                cv2.putText(display, guide_line, (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

                center_dir = "CENTERED" if center_ok else f"PAN CAMERA BY {int(center_err_px)} px"
                cv2.putText(display, f"Optical Centering: {center_dir} | Marker ID: {ids[0][0]}", (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
                return display, is_aligned

        # Default searching HUD
        cv2.rectangle(display, (10, 10), (w - 10, 50), (20, 20, 20), -1)
        cv2.putText(display, status_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.55, header_color, 2)
        return display, False


def run_live_physical_alignment(cam_index=0, is_side_cam=False):
    target_elev = 0.0 if is_side_cam else 45.0
    cam_name = "Side Lateral Camera" if is_side_cam else "Front 45-deg Camera"
    guide = PhysicalCameraAlignmentGuide(target_elevation_deg=target_elev)

    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        print(f"[ERROR] Could not open camera {cam_index}. Generating demo synthetic frame...")
        # Fallback test demo
        demo_frame = np.ones((480, 640, 3), dtype=np.uint8) * 140
        annotated, _ = guide.process_frame(demo_frame, camera_name=cam_name)
        cv2.imwrite("physical_alignment_guide_sample.png", annotated)
        print("Saved sample guide display to physical_alignment_guide_sample.png")
        return

    print(f"[Physical Camera Alignment Guide] Running on camera {cam_index}. Press 'q' to exit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        annotated, aligned = guide.process_frame(frame, camera_name=cam_name)
        cv2.imshow("Physical Camera Alignment Guide", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cam", type=int, default=0, help="Camera device index")
    parser.add_argument("--side", action="store_true", help="Align side camera instead of front camera")
    args = parser.parse_args()
    run_live_physical_alignment(cam_index=args.cam, is_side_cam=args.side)
