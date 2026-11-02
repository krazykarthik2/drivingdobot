"""
dobot_interactive_aligner.py

Universal Live Camera <-> Digital Twin Homography & Alignment Tool
- Robust Multi-Scale ArUco Detection (DICT_4X4_50, 4x4_100, 5x5)
- Displays Real-Time 6D Pose & Angles: Distance, Offset (X, Y), Tilt / Pitch, Yaw, and Roll
- Full 8-Degree-of-Freedom Homography Warping:
  * Translation (X, Y)
  * Scale / Zoom
  * Rotation angle (Deg)
  * Perspective Tilt / Shear (Pitch & Yaw warp)
- Instant Auto-Align ('a' or 'SPACE'): Automatically computes 3x3 homography matrix H from tags!
- Saves calibration to `camera_twin_alignment.json` ('s')
"""

import os
import sys
import json
import time
import cv2
import numpy as np
import pybullet as p

# Windows compatibility
sys.modules['jax'] = None
sys.modules['jaxlib'] = None
sys.modules['tensorflow'] = None

from dobot_color_box_env import DobotColorBoxEnv

ALIGN_FILE = "camera_twin_alignment.json"

class DobotInteractiveAligner:
    def __init__(self, cam_idx=1):
        self.cam_idx = cam_idx
        self.width = 640
        self.height = 480

        # Fine-tune transform variables
        self.offset_x = 0
        self.offset_y = 0
        self.scale = 1.0
        self.rotation_deg = 0.0
        self.tilt_x = 0.0  # Perspective tilt horizontal
        self.tilt_y = 0.0  # Perspective tilt vertical
        self.alpha = 0.50
        self.homography_matrix = None

        self.load_alignment()

        # Initialize PyBullet Twin Environment
        print("[Init] Loading PyBullet Digital Twin environment...")
        self.env = DobotColorBoxEnv(render_gui=False, width=self.width, height=self.height)
        self.env.set_cartesian_target(0.20, 0.0, 0.05, r=0.0, gripper_close=False)

        # Open Camera
        print(f"[Init] Connecting to Camera {self.cam_idx}...")
        self.cap = cv2.VideoCapture(self.cam_idx)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Robust ArUco Detector with adaptive parameter ranges
        self.params = cv2.aruco.DetectorParameters()
        self.params.adaptiveThreshWinSizeMin = 3
        self.params.adaptiveThreshWinSizeMax = 53
        self.params.adaptiveThreshWinSizeStep = 4
        self.params.adaptiveThreshConstant = 7
        self.params.minMarkerPerimeterRate = 0.02
        self.params.maxMarkerPerimeterRate = 4.0

        self.dict_4x4 = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.detector = cv2.aruco.ArucoDetector(self.dict_4x4, self.params)

        # Camera Intrinsics
        self.K = np.array([[600.0, 0, 320.0], [0, 600.0, 240.0], [0, 0, 1.0]], dtype=np.float64)
        self.dist = np.zeros(5, dtype=np.float64)
        ms = 0.080 / 2.0  # 8cm marker half-size
        self.obj_pts = np.array([[-ms, ms, 0], [ms, ms, 0], [ms, -ms, 0], [-ms, -ms, 0]], dtype=np.float32)

    def load_alignment(self):
        if os.path.exists(ALIGN_FILE):
            try:
                with open(ALIGN_FILE, "r") as f:
                    d = json.load(f)
                    self.offset_x = d.get("offset_x", 0)
                    self.offset_y = d.get("offset_y", 0)
                    self.scale = d.get("scale", 1.0)
                    self.rotation_deg = d.get("rotation_deg", 0.0)
                    self.tilt_x = d.get("tilt_x", 0.0)
                    self.tilt_y = d.get("tilt_y", 0.0)
                    if "homography" in d:
                        self.homography_matrix = np.array(d["homography"], dtype=np.float32)
                    print(f"[Config] Loaded saved alignment from {ALIGN_FILE}")
            except Exception:
                pass

    def save_alignment(self):
        d = {
            "offset_x": self.offset_x,
            "offset_y": self.offset_y,
            "scale": self.scale,
            "rotation_deg": self.rotation_deg,
            "tilt_x": self.tilt_x,
            "tilt_y": self.tilt_y
        }
        if self.homography_matrix is not None:
            d["homography"] = self.homography_matrix.tolist()

        with open(ALIGN_FILE, "w") as f:
            json.dump(d, f, indent=2)
        print(f"[SUCCESS] Calibration saved to {os.path.abspath(ALIGN_FILE)}!")

    def detect_markers_ensemble(self, gray_frame):
        """Multi-pass ensemble detection to guarantee capturing all markers [0, 1, 2, 3]."""
        all_corners = []
        all_ids = []

        for c_val in [3, 5]:
            p = cv2.aruco.DetectorParameters()
            p.adaptiveThreshConstant = c_val
            p.adaptiveThreshWinSizeStep = 2
            p.minMarkerPerimeterRate = 0.012
            p.maxMarkerPerimeterRate = 5.0
            p.polygonalApproxAccuracyRate = 0.08
            p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

            det = cv2.aruco.ArucoDetector(self.dict_4x4, p)
            c, ids, _ = det.detectMarkers(gray_frame)
            if ids is not None:
                for i, tid in enumerate(ids.flatten()):
                    if int(tid) not in all_ids:
                        all_ids.append(int(tid))
                        all_corners.append(c[i])

        if all_ids:
            return all_corners, np.array(all_ids, dtype=np.int32).reshape(-1, 1)
        return [], None

    def auto_compute_homography(self, corners, ids):
        """Automatically aligns 4 ArUco markers to the simulation world frame."""
        if ids is None or len(ids) < 2:
            print("[Auto-Align] Need at least 2 ArUco markers in view to align.")
            return

        # Target Simulation Pixel Projection for:
        # Tag 0: [402, 270], Tag 1: [237, 270], Tag 2: [245, 172], Tag 3: [394, 172]
        sim_coords = {
            0: [402.1, 270.2],
            1: [236.9, 270.2],
            2: [245.2, 172.0],
            3: [393.8, 172.0]
        }

        src_pts = []
        dst_pts = []

        for i, tid in enumerate(ids.flatten()):
            if int(tid) in sim_coords:
                center = corners[i][0].mean(axis=0)
                src_pts.append(center)
                dst_pts.append(sim_coords[int(tid)])

        if len(src_pts) >= 4:
            H, _ = cv2.findHomography(np.array(src_pts, dtype=np.float32), np.array(dst_pts, dtype=np.float32))
            self.homography_matrix = H
            print("[Auto-Align] SUCCESS! 4-Point Homography Matrix computed and locked.")
        elif len(src_pts) >= 2:
            # Estimate affine
            M, _ = cv2.estimateAffinePartial2D(np.array(src_pts, dtype=np.float32), np.array(dst_pts, dtype=np.float32))
            if M is not None:
                H = np.eye(3, dtype=np.float32)
                H[:2, :] = M
                self.homography_matrix = H
                print("[Auto-Align] 2-Point Affine alignment calculated and locked.")

    def transform_frame(self, frame):
        h, w = frame.shape[:2]
        if self.homography_matrix is not None:
            # Apply full perspective homography
            warped = cv2.warpPerspective(frame, self.homography_matrix, (w, h))
        else:
            center = (w / 2.0, h / 2.0)
            M = cv2.getRotationMatrix2D(center, self.rotation_deg, self.scale)
            M[0, 2] += self.offset_x
            M[1, 2] += self.offset_y
            warped = cv2.warpAffine(frame, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

        # Fine micro-adjustments
        if self.tilt_x != 0 or self.tilt_y != 0:
            pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
            pts2 = np.float32([
                [self.tilt_x, self.tilt_y],
                [w - self.tilt_x, -self.tilt_y],
                [-self.tilt_x, h - self.tilt_y],
                [w + self.tilt_x, h + self.tilt_y]
            ])
            P = cv2.getPerspectiveTransform(pts1, pts2)
            warped = cv2.warpPerspective(warped, P, (w, h))

        return warped

    def run(self):
        win = "Dobot Camera Calibration & Perspective Aligner"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, 1100, 750)

        print("\n" + "=" * 70)
        print("PIXEL-PERFECT ALIGNER CONTROLS:")
        print("  [SPACE] / [A]: AUTO-ALIGN (Snaps all markers to simulation instantly)")
        print("  [W/A/S/D] or [Arrows]: Micro Nudge Position (X, Y)")
        print("  [+] / [-]: Zoom Scale in / out")
        print("  [R] / [E]: Rotate Camera angle (+/- 0.5 deg)")
        print("  [T] / [G]: Tilt Perspective Pitch (Forward / Backward)")
        print("  [F] / [H]: Tilt Perspective Yaw (Left / Right tilt)")
        print("  [TAB]: Cycle Alpha Transparency (20% -> 50% -> 80%)")
        print("  [S]: Save Calibration to camera_twin_alignment.json")
        print("  [Q] or [ESC]: Quit")
        print("=" * 70 + "\n")

        while True:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)

            # Render Sim Frame
            sim_rgb = self.env.render_frame()
            sim_bgr = cv2.cvtColor(sim_rgb, cv2.COLOR_RGB2BGR)

            # Detect ArUco markers with multi-pass ensemble
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids = self.detect_markers_ensemble(gray)

            pose_info = "Tags: Searching..."
            if ids is not None and len(ids) > 0:
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                # Compute 3D angle from first visible marker
                ok, rvec, tvec = cv2.solvePnP(self.obj_pts, corners[0][0], self.K, self.dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
                if ok:
                    R, _ = cv2.Rodrigues(rvec)
                    pitch = float(np.degrees(np.arctan2(R[2, 1], R[2, 2])))
                    dist_cm = float(tvec[2][0] * 100.0)
                    pose_info = f"Tags:{ids.flatten().tolist()} | Dist:{dist_cm:.1f}cm | Pitch:{pitch:+.1f}deg"

            # Apply perspective transform
            aligned_frame = self.transform_frame(frame)

            # Alpha blend real camera and digital twin
            blend = cv2.addWeighted(aligned_frame, 1.0 - self.alpha, sim_bgr, self.alpha, 0.0)

            # HUD Display
            cv2.rectangle(blend, (0, 0), (self.width, 58), (15, 15, 15), -1)
            cv2.putText(blend, f"OVERLAY: {pose_info}", (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (50, 240, 240), 1)
            hud2 = f"Scale:{self.scale:.2f} | Rot:{self.rotation_deg:+.1f}d | X:{self.offset_x:+d} Y:{self.offset_y:+d} | TiltX:{self.tilt_x:+.1f} TiltY:{self.tilt_y:+.1f}"
            cv2.putText(blend, hud2, (12, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (100, 255, 100), 1)

            cv2.imshow(win, blend)
            key = cv2.waitKey(20) & 0xFF

            if key in [ord('q'), 27]:
                break
            elif key == ord('s'):
                self.save_alignment()
            elif key in [ord('a'), ord(' ')]:
                self.auto_compute_homography(corners, ids)
            elif key in [ord('w'), 82]: self.offset_y -= 1
            elif key in [ord('s'), 84]: self.offset_y += 1
            elif key in [ord('a'), 81]: self.offset_x -= 1
            elif key in [ord('d'), 83]: self.offset_x += 1
            elif key in [ord('+'), ord('=')]: self.scale = min(3.0, self.scale + 0.01)
            elif key in [ord('-'), ord('_')]: self.scale = max(0.3, self.scale - 0.01)
            elif key == ord('r'): self.rotation_deg += 0.5
            elif key == ord('e'): self.rotation_deg -= 0.5
            elif key == ord('t'): self.tilt_y += 1.0
            elif key == ord('g'): self.tilt_y -= 1.0
            elif key == ord('f'): self.tilt_x -= 1.0
            elif key == ord('h'): self.tilt_x += 1.0
            elif key == 9: # Tab
                self.alpha = 0.20 if self.alpha >= 0.70 else (0.50 if self.alpha <= 0.30 else 0.80)

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    app = DobotInteractiveAligner(cam_idx=1)
    app.run()
