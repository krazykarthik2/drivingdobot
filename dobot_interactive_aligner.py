"""
dobot_interactive_aligner.py

Universal Live Camera <-> Digital Twin Homography & Alignment Tool
- Displays the PyBullet 3D Dobot Magician AND the exact 4 ArUco markers [0, 1] & [2, 3]
  placed 1cm in front of the Dobot base with 1cm gaps and 8x8cm dimensions.
- Multi-pass ensemble ArUco detection captures all physical markers.
- Auto-Align ('SPACE' or 'a') automatically calculates and locks the 4-point homography matrix!
- Allows visual fine-tuning so you can confirm that physical tags & robot match the simulation.
- Press 's' to save the calibrated transformation matrix to `camera_twin_alignment.json`.
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
        self.tilt_x = 0.0
        self.tilt_y = 0.0
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

        self.dict_4x4 = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

        # Setup 3D Projection Matrices
        self.V = np.array(self.env.view_matrix).reshape((4, 4), order='F')
        self.P = np.array(self.env.proj_matrix).reshape((4, 4), order='F')
        self.VP = self.P @ self.V

        # Define 4 Virtual Simulation ArUco Tags [0, 1] (row near Dobot) and [2, 3] (row far)
        # Dimensions: 8cm x 8cm, 1cm inter-tag gap, 1cm Dobot base gap
        z_table = -0.048
        self.sim_tags_def = {
            0: ([0.140, -0.045, z_table], (240, 200, 50)),   # Cyan/Yellow
            1: ([0.140, +0.045, z_table], (50, 240, 240)),   # Yellow/Cyan
            2: ([0.230, -0.045, z_table], (50, 240, 50)),    # Green
            3: ([0.230, +0.045, z_table], (240, 50, 240))    # Magenta
        }

    def sim_proj(self, pt_3d):
        clip = self.VP @ np.array([pt_3d[0], pt_3d[1], pt_3d[2], 1.0])
        ndc = clip[:3] / clip[3]
        u = ((ndc[0] + 1.0) / 2.0) * (self.width - 1.0)
        v = ((1.0 - ndc[1]) / 2.0) * (self.height - 1.0)
        return [int(u), int(v)]

    def render_sim_with_virtual_tags(self):
        """Renders PyBullet simulation Dobot and draws virtual 3D 8x8cm tags on the table."""
        sim_rgb = self.env.render_frame()
        sim_bgr = cv2.cvtColor(sim_rgb, cv2.COLOR_RGB2BGR)

        hw = 0.040 # 8cm marker half-size
        for tid, (center, color) in self.sim_tags_def.items():
            cx, cy, cz = center
            corners_3d = [
                [cx - hw, cy - hw, cz],
                [cx - hw, cy + hw, cz],
                [cx + hw, cy + hw, cz],
                [cx + hw, cy - hw, cz]
            ]
            poly = np.array([self.sim_proj(c) for c in corners_3d], dtype=np.int32)
            cv2.fillPoly(sim_bgr, [poly], (30, 30, 30))
            cv2.polylines(sim_bgr, [poly], isClosed=True, color=color, thickness=2)

            # Inner pattern
            hw_in = 0.024
            inner_3d = [
                [cx - hw_in, cy - hw_in, cz],
                [cx - hw_in, cy + hw_in, cz],
                [cx + hw_in, cy + hw_in, cz],
                [cx + hw_in, cy - hw_in, cz]
            ]
            poly_in = np.array([self.sim_proj(c) for c in inner_3d], dtype=np.int32)
            cv2.fillPoly(sim_bgr, [poly_in], (210, 210, 210))

            c_px = self.sim_proj([cx, cy, cz])
            cv2.putText(sim_bgr, f'{tid}', (c_px[0] - 6, c_px[1] + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
            cv2.putText(sim_bgr, f'SIM ID {tid}', (c_px[0] - 28, c_px[1] - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)

        return sim_bgr

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
        all_corners = {}
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
                    if int(tid) not in all_corners:
                        all_corners[int(tid)] = c[i]

        if all_corners:
            sorted_items = sorted(all_corners.items())
            c_list = [item[1] for item in sorted_items]
            ids_arr = np.array([item[0] for item in sorted_items], dtype=np.int32).reshape(-1, 1)
            return c_list, ids_arr
        return [], None

    def auto_compute_homography(self, corners, ids):
        """Automatically aligns 4 ArUco markers to the simulation world frame in standard orientation."""
        if ids is None or len(ids) < 2:
            print("[Auto-Align] Need at least 2 ArUco markers in view to align.")
            return

        sim_coords = {
            0: [244.3, 181.9],
            1: [394.7, 181.9],
            2: [235.8, 282.5],
            3: [403.2, 282.5]
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
            M, _ = cv2.estimateAffinePartial2D(np.array(src_pts, dtype=np.float32), np.array(dst_pts, dtype=np.float32))
            if M is not None:
                H = np.eye(3, dtype=np.float32)
                H[:2, :] = M
                self.homography_matrix = H
                print("[Auto-Align] 2-Point Affine alignment calculated and locked.")

    def transform_frame(self, frame):
        h, w = frame.shape[:2]
        if self.homography_matrix is not None:
            warped = cv2.warpPerspective(frame, self.homography_matrix, (w, h))
        else:
            center = (w / 2.0, h / 2.0)
            M = cv2.getRotationMatrix2D(center, self.rotation_deg, self.scale)
            M[0, 2] += self.offset_x
            M[1, 2] += self.offset_y
            warped = cv2.warpAffine(frame, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

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
        win = "Dobot Camera Calibration & Virtual Marker Overlay"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, 1100, 750)

        print("\n" + "=" * 70)
        print("PIXEL-PERFECT ALIGNER CONTROLS:")
        print("  [SPACE] / [A]: AUTO-ALIGN (Snaps physical tags to virtual sim tags!)")
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

            # Render Sim Frame with Dobot Robot AND Virtual ArUco markers [0, 1] & [2, 3]
            sim_view = self.render_sim_with_virtual_tags()

            # Detect physical markers
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids = self.detect_markers_ensemble(gray)

            pose_info = "Tags: Searching..."
            if ids is not None and len(ids) > 0:
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                pose_info = f"Physical Tags Detected: {ids.flatten().tolist()}"

            # Apply homography warp to camera frame
            aligned_frame = self.transform_frame(frame)

            # Blend real camera and simulation (Dobot arm + virtual tags)
            blend = cv2.addWeighted(aligned_frame, 1.0 - self.alpha, sim_view, self.alpha, 0.0)

            # HUD Display
            cv2.rectangle(blend, (0, 0), (self.width, 58), (15, 15, 15), -1)
            cv2.putText(blend, f"OVERLAY: {pose_info}", (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (50, 240, 240), 1)
            hud2 = f"Scale:{self.scale:.2f} | Rot:{self.rotation_deg:+.1f}d | X:{self.offset_x:+d} Y:{self.offset_y:+d} | [SPACE]: Auto-Align | [S]: Save"
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
