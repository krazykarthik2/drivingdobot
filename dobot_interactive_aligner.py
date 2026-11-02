"""
dobot_interactive_aligner.py

Interactive Live Camera <-> Digital Twin Homography & Alignment Tool
Allows you to:
1. View real-time 50% Camera + 50% PyBullet 3D Simulation Overlay.
2. Fine-tune Translation (X, Y), Scale / Zoom, and Rotation interactively via keys or trackbars
   so that the physical Dobot base, links, and markers match the simulation pixel-perfectly!
3. Automatic ArUco Alignment (press 'a'): automatically snaps camera markers to sim world markers.
4. Press 's' to save the calibrated transformation matrix to `camera_twin_alignment.json`.
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

        # Load existing alignment or defaults
        self.offset_x = 0
        self.offset_y = 0
        self.scale = 1.0
        self.rotation_deg = 0.0
        self.alpha = 0.50
        self.load_alignment()

        # PyBullet Twin Environment
        print("[Init] Spawning PyBullet Digital Twin...")
        self.env = DobotColorBoxEnv(render_gui=False, width=self.width, height=self.height)
        self.env.set_cartesian_target(0.20, 0.0, 0.05, r=0.0, gripper_close=False)

        # Camera
        print(f"[Init] Opening Camera {self.cam_idx}...")
        self.cap = cv2.VideoCapture(self.cam_idx)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # ArUco
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, cv2.aruco.DetectorParameters())

    def load_alignment(self):
        if os.path.exists(ALIGN_FILE):
            try:
                with open(ALIGN_FILE, "r") as f:
                    d = json.load(f)
                    self.offset_x = d.get("offset_x", 0)
                    self.offset_y = d.get("offset_y", 0)
                    self.scale = d.get("scale", 1.0)
                    self.rotation_deg = d.get("rotation_deg", 0.0)
                    print(f"[Config] Loaded saved alignment from {ALIGN_FILE}")
            except Exception:
                pass

    def save_alignment(self):
        d = {
            "offset_x": self.offset_x,
            "offset_y": self.offset_y,
            "scale": self.scale,
            "rotation_deg": self.rotation_deg
        }
        with open(ALIGN_FILE, "w") as f:
            json.dump(d, f, indent=2)
        print(f"[SUCCESS] Saved alignment configuration to {os.path.abspath(ALIGN_FILE)}!")

    def transform_frame(self, frame):
        """Applies interactive scale, rotation, and translation to align with simulation."""
        h, w = frame.shape[:2]
        center = (w / 2.0, h / 2.0)

        # 2D affine transformation matrix: Rotation + Scaling
        M = cv2.getRotationMatrix2D(center, self.rotation_deg, self.scale)
        # Apply Translation offsets
        M[0, 2] += self.offset_x
        M[1, 2] += self.offset_y

        warped = cv2.warpAffine(frame, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        return warped

    def auto_align_aruco(self, corners, ids):
        """Calculates best-fit translation and scale from detected ArUco markers."""
        if ids is None or len(ids) == 0:
            print("[Auto-Align] No markers detected to compute automatic alignment.")
            return

        centers = {}
        for i, tid in enumerate(ids.flatten()):
            centers[int(tid)] = corners[i][0].mean(axis=0)

        # Sim projection targets:
        # Tag 1: [404, 192], Tag 3: [209, 270]
        if 1 in centers and 3 in centers:
            c1_real = centers[1]
            c3_real = centers[3]
            c1_sim = np.array([404.0, 192.0])
            c3_sim = np.array([209.0, 270.0])

            dist_real = np.linalg.norm(c1_real - c3_real)
            dist_sim = np.linalg.norm(c1_sim - c3_sim)

            if dist_real > 10:
                self.scale = float(dist_sim / dist_real)
                vec_real = c3_real - c1_real
                vec_sim = c3_sim - c1_sim
                ang_real = np.arctan2(vec_real[1], vec_real[0])
                ang_sim = np.arctan2(vec_sim[1], vec_sim[0])
                self.rotation_deg = float(np.rad2deg(ang_sim - ang_real))

                mid_real = (c1_real + c3_real) / 2.0
                mid_sim = (c1_sim + c3_sim) / 2.0
                self.offset_x = int(mid_sim[0] - mid_real[0])
                self.offset_y = int(mid_sim[1] - mid_real[1])
                print(f"[Auto-Align] Snapped! Scale={self.scale:.2f}, Rot={self.rotation_deg:.1f}deg, Offset=({self.offset_x}, {self.offset_y})")

    def run(self):
        win = "Pixel-Perfect Dobot Camera-to-Twin Alignment"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, 1100, 750)

        print("\n" + "=" * 65)
        print("PIXEL-PERFECT ALIGNMENT CONTROLS:")
        print("  [Arrow Keys] / [W/A/S/D]: Nudge Position (Offset X, Y)")
        print("  [+] / [-]: Zoom Scale in / out")
        print("  [R] / [E]: Rotate Camera angle (+/- 1 deg)")
        print("  [SPACE] / [A]: Auto-align with ArUco tags")
        print("  [S]: Save Calibration to camera_twin_alignment.json")
        print("  [TAB]: Toggle Alpha (20% -> 50% -> 80%)")
        print("  [Q] or [ESC]: Quit")
        print("=" * 65 + "\n")

        while True:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)

            # Render Sim Frame
            sim_rgb = self.env.render_frame()
            sim_bgr = cv2.cvtColor(sim_rgb, cv2.COLOR_RGB2BGR)

            # Detect markers on raw camera frame
            corners, ids, _ = self.detector.detectMarkers(frame)
            if ids is not None and len(ids) > 0:
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)

            # Transform raw frame
            aligned_frame = self.transform_frame(frame)

            # Overlay blend
            blend = cv2.addWeighted(aligned_frame, 1.0 - self.alpha, sim_bgr, self.alpha, 0.0)

            # HUD
            cv2.rectangle(blend, (0, 0), (self.width, 50), (15, 15, 15), -1)
            hud = f"Scale:{self.scale:.2f} | Rot:{self.rotation_deg:+.1f}d | X:{self.offset_x:+d}, Y:{self.offset_y:+d} | Alpha:{int(self.alpha*100)}%"
            cv2.putText(blend, hud, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (50, 240, 240), 2)

            cv2.imshow(win, blend)
            key = cv2.waitKey(25) & 0xFF

            if key in [ord('q'), 27]:
                break
            elif key == ord('s'):
                self.save_alignment()
            elif key in [ord('a'), ord(' ')]:
                self.auto_align_aruco(corners, ids)
            elif key in [ord('w'), 82]: # Up
                self.offset_y -= 2
            elif key in [ord('s'), 84]: # Down
                self.offset_y += 2
            elif key in [ord('a'), 81]: # Left
                self.offset_x -= 2
            elif key in [ord('d'), 83]: # Right
                self.offset_x += 2
            elif key in [ord('+'), ord('=')]:
                self.scale = min(2.5, self.scale + 0.02)
            elif key in [ord('-'), ord('_')]:
                self.scale = max(0.4, self.scale - 0.02)
            elif key == ord('r'):
                self.rotation_deg += 1.0
            elif key == ord('e'):
                self.rotation_deg -= 1.0
            elif key == 9: # Tab
                self.alpha = 0.20 if self.alpha >= 0.70 else (0.50 if self.alpha <= 0.30 else 0.80)

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    app = DobotInteractiveAligner(cam_idx=1)
    app.run()
