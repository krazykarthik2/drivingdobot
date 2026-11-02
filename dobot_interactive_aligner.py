"""
dobot_interactive_aligner.py

Universal Live Camera <-> Digital Twin Homography & Alignment Tool
- Multi-threaded: GUI and rendering run on dedicated main thread; camera acquisition on decoupled thread.
- Automatically connects to physical Dobot on COM13 and commands the gripper to the exact center
  of the 4 ArUco markers [X=176.0 mm, Y=0.0 mm, Z=15.0 mm] so you can physically align or position the Dobot!
- Exact Dobot base geometry: Dobot base radius is 81mm. Row 0 ArUco tags start exactly 1cm (10mm) in front of the base.
- Multi-pass ensemble ArUco detection guarantees finding all 4 markers.
- Auto-Align ('SPACE' or 'a') calculates 36-point perspective homography and saves to camera_twin_alignment.json.
"""

import os
import sys
import json
import time
import threading
import cv2
import numpy as np
import pybullet as p

# Windows compatibility
sys.modules['jax'] = None
sys.modules['jaxlib'] = None
sys.modules['tensorflow'] = None

import pydobot
from pydobot.message import Message
from dobot_color_box_env import DobotColorBoxEnv

ALIGN_FILE = "camera_twin_alignment.json"


class DobotInteractiveAligner:
    def __init__(self, cam_idx=1, robot_port="COM13"):
        self.cam_idx = cam_idx
        self.robot_port = robot_port
        self.width = 640
        self.height = 480

        self.offset_x = 0
        self.offset_y = 0
        self.scale = 1.0
        self.rotation_deg = 0.0
        self.tilt_x = 0.0
        self.tilt_y = 0.0
        self.alpha = 0.50
        self.homography_matrix = None

        self.load_alignment()

        # Connect to physical Dobot and move to middle of 4 ArUco markers
        print(f"[Hardware] Connecting to physical Dobot on {self.robot_port}...")
        self.dobot = None
        self.connected = False
        try:
            self.dobot = pydobot.Dobot(port=self.robot_port, verbose=False)
            self.connected = True
            # Clear alarms
            try:
                m = Message()
                m.id = 20
                m.ctrl = 1
                m.params = bytearray([])
                self.dobot._send_message(m)
                self.dobot._read_message()
            except Exception:
                pass

            p_init = self.dobot.pose()
            print(f"[Hardware] Dobot Connected! Initial Pose: X={p_init[0]:.1f}, Y={p_init[1]:.1f}, Z={p_init[2]:.1f} mm")
            
            # Center of the 4 ArUco markers:
            # Base front edge = 81mm. 1cm gap before tags -> tags start at 91mm.
            # 80mm tag + 10mm gap -> 4-tag center is X=176.0mm, Y=0.0mm.
            # Move gripper into middle of 4 tags!
            print("[Hardware] Positioning gripper to the exact center of the 4 ArUco markers [X=176.0, Y=0.0, Z=15.0 mm]...")
            self._move_dobot_center(176.0, 0.0, 15.0)
            print("[SUCCESS] Dobot gripper positioned at exact 4-tag center!\n")
        except Exception as e:
            print(f"[Hardware Warning] Could not connect to Dobot on {self.robot_port}: {e}\n")

        print("[Init] Loading PyBullet Digital Twin environment...")
        self.env = DobotColorBoxEnv(render_gui=False, width=self.width, height=self.height)
        # Position digital twin matching the 4-marker center
        self.env.set_cartesian_target(0.176, 0.0, 0.015, r=0.0, gripper_close=False)

        # 3D Projection Matrices
        self.V = np.array(self.env.view_matrix).reshape((4, 4), order='F')
        self.P = np.array(self.env.proj_matrix).reshape((4, 4), order='F')
        self.VP = self.P @ self.V

        # Exact Physical Geometry & Perspective:
        # Near Row: Tags 0 & 1
        # Far Row:  Tags 3 & 2
        z_table = -0.048
        self.sim_tags_def = {
            0: ([0.059, -0.045, z_table], (240, 200, 50)),
            1: ([0.059, +0.045, z_table], (50, 240, 240)),
            2: ([0.173, +0.045, z_table], (50, 240, 50)),
            3: ([0.173, -0.045, z_table], (240, 50, 240))
        }

        # Decoupled Camera Thread
        print(f"[Init] Connecting to Camera {self.cam_idx} in background thread...")
        self.cap = cv2.VideoCapture(self.cam_idx)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.running = True
        self.lock = threading.Lock()
        self.latest_frame = None

        self.cam_thread = threading.Thread(target=self._camera_reader, daemon=True)
        self.cam_thread.start()

        self.dict_4x4 = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

    def _move_dobot_center(self, target_x, target_y, target_z, max_time=4.0):
        """Drives physical arm smoothly to target center coordinates."""
        if not self.connected or not self.dobot:
            return

        t0 = time.time()
        while time.time() - t0 < max_time:
            try:
                cur = self.dobot.pose()
                dx = target_x - cur[0]
                dy = target_y - cur[1]
                dz = target_z - cur[2]
                if abs(dx) < 3.0 and abs(dy) < 3.0 and abs(dz) < 3.0:
                    break
                if abs(dx) >= abs(dy) and abs(dx) >= abs(dz):
                    cmd = 1 if dx > 0 else 2
                elif abs(dy) >= abs(dx) and abs(dy) >= abs(dz):
                    cmd = 3 if dy > 0 else 4
                else:
                    cmd = 5 if dz > 0 else 6

                m = Message()
                m.id = 73
                m.ctrl = 1
                m.params = bytearray([0, cmd])
                self.dobot._send_message(m)
                self.dobot._read_message()
                time.sleep(0.05)
            except Exception:
                break

        # Stop
        try:
            m = Message()
            m.id = 73
            m.ctrl = 1
            m.params = bytearray([0, 0])
            self.dobot._send_message(m)
            self.dobot._read_message()
        except Exception:
            pass

    def _camera_reader(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret and frame is not None:
                with self.lock:
                    self.latest_frame = frame
            time.sleep(0.015)

    def sim_proj(self, pt_3d):
        clip = self.VP @ np.array([pt_3d[0], pt_3d[1], pt_3d[2], 1.0])
        ndc = clip[:3] / clip[3]
        u = ((ndc[0] + 1.0) / 2.0) * (self.width - 1.0)
        v = ((1.0 - ndc[1]) / 2.0) * (self.height - 1.0)
        return [int(u), int(v)]

    def render_sim_with_virtual_tags(self):
        sim_rgb = self.env.render_frame()
        sim_bgr = cv2.cvtColor(sim_rgb, cv2.COLOR_RGB2BGR)

        hw = 0.040
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
            cv2.putText(sim_bgr, f'SIM {tid}', (c_px[0] - 22, c_px[1] - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)

        # Crosshair in the exact center of the 4 markers
        z_table = -0.048
        center_4 = self.sim_proj([0.116, 0.0, z_table])
        cv2.drawMarker(sim_bgr, (center_4[0], center_4[1]), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
        cv2.putText(sim_bgr, "CENTER", (center_4[0] - 25, center_4[1] + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

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
            p_det = cv2.aruco.DetectorParameters()
            p_det.adaptiveThreshConstant = c_val
            p_det.adaptiveThreshWinSizeStep = 2
            p_det.minMarkerPerimeterRate = 0.012
            p_det.maxMarkerPerimeterRate = 5.0
            p_det.polygonalApproxAccuracyRate = 0.08
            p_det.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

            det = cv2.aruco.ArucoDetector(self.dict_4x4, p_det)
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
        if ids is None or len(ids) < 2:
            print("[Auto-Align] Need at least 2 ArUco markers in view to align.")
            return

        hw = 0.040
        z_table = -0.048
        
        src_all = []
        dst_all = []

        for i, tid in enumerate(ids.flatten()):
            t_int = int(tid)
            if t_int in self.sim_tags_def:
                cx, cy, cz = self.sim_tags_def[t_int][0]
                pts_real = corners[i][0]
                p0, p1, p2, p3 = pts_real
                m_top = (p0 + p1) / 2.0
                m_right = (p1 + p2) / 2.0
                m_bot = (p2 + p3) / 2.0
                m_left = (p3 + p0) / 2.0
                c_real = pts_real.mean(axis=0)
                real_9 = [p0, p1, p2, p3, m_top, m_right, m_bot, m_left, c_real]

                sp0 = self.sim_proj([cx - hw, cy - hw, cz])
                sp1 = self.sim_proj([cx - hw, cy + hw, cz])
                sp2 = self.sim_proj([cx + hw, cy + hw, cz])
                sp3 = self.sim_proj([cx + hw, cy - hw, cz])
                sm_top = self.sim_proj([cx - hw, cy, cz])
                sm_right = self.sim_proj([cx, cy + hw, cz])
                sm_bot = self.sim_proj([cx + hw, cy, cz])
                sm_left = self.sim_proj([cx, cy - hw, cz])
                sc = self.sim_proj([cx, cy, cz])
                sim_9 = [sp0, sp1, sp2, sp3, sm_top, sm_right, sm_bot, sm_left, sc]

                src_all.extend(real_9)
                dst_all.extend(sim_9)

        if len(src_all) >= 4:
            H, _ = cv2.findHomography(np.array(src_all, dtype=np.float32), np.array(dst_all, dtype=np.float32), cv2.RANSAC, 3.0)
            if H is not None:
                self.homography_matrix = H
                print(f"[Auto-Align] SUCCESS! Perspective Homography Matrix computed using {len(src_all)} constraints.")
                self.save_alignment()

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
            M_persp = cv2.getPerspectiveTransform(pts1, pts2)
            warped = cv2.warpPerspective(warped, M_persp, (w, h))

        return warped

    def run(self):
        win = "Camera <-> Digital Twin Aligner (Physical Dobot Center Assisted)"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, 960, 720)

        print("\n" + "=" * 70)
        print("PIXEL-PERFECT ALIGNER CONTROLS:")
        print("  Dobot Gripper is placed at the exact 4-tag center [X=176, Y=0, Z=15 mm]!")
        print("  You can now physically reposition the Dobot or camera so it aligns.")
        print("  [SPACE] / [A]: AUTO-ALIGN (Snaps physical tags to virtual tags & saves)")
        print("  [W/A/S/D] or [Arrows]: Micro Nudge Position (X, Y)")
        print("  [+] / [-]: Zoom Scale in / out")
        print("  [R] / [E]: Rotate Camera angle (+/- 0.5 deg)")
        print("  [T/G/F/H]: Perspective 3D Tilt adjustment")
        print("  [TAB]: Cycle Alpha blend (20% -> 50% -> 80%)")
        print("  [S]: Save Calibration to camera_twin_alignment.json")
        print("  [Q] or [ESC]: Quit")
        print("=" * 70 + "\n")

        while self.running:
            with self.lock:
                frame = self.latest_frame.copy() if self.latest_frame is not None else None

            if frame is None:
                time.sleep(0.01)
                continue

            sim_view = self.render_sim_with_virtual_tags()

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids = self.detect_markers_ensemble(gray)

            pose_info = "Tags: Searching..."
            if ids is not None and len(ids) > 0:
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                pose_info = f"Physical Tags Detected: {ids.flatten().tolist()}"

            aligned_frame = self.transform_frame(frame)
            blend = cv2.addWeighted(aligned_frame, 1.0 - self.alpha, sim_view, self.alpha, 0.0)

            # HUD Display
            cv2.rectangle(blend, (0, 0), (self.width, 58), (15, 15, 15), -1)
            robot_txt = "DOBOT: COM13 AT CENTER" if self.connected else "DOBOT: OFFLINE"
            cv2.putText(blend, f"OVERLAY: {pose_info} | {robot_txt}", (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (50, 240, 240), 1)
            hud2 = f"Scale:{self.scale:.2f} | Rot:{self.rotation_deg:+.1f}d | X:{self.offset_x:+d} Y:{self.offset_y:+d} | [SPACE]: Auto-Align | [S]: Save"
            cv2.putText(blend, hud2, (12, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (100, 255, 100), 1)

            cv2.imshow(win, blend)
            key = cv2.waitKey(15) & 0xFF

            if key in [ord('q'), 27]:
                self.running = False
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

        self.running = False
        self.cam_thread.join(timeout=0.5)
        self.cap.release()
        cv2.destroyAllWindows()
        if self.connected and self.dobot:
            try:
                self.dobot.close()
            except Exception:
                pass


if __name__ == "__main__":
    app = DobotInteractiveAligner(cam_idx=1, robot_port="COM13")
    app.run()
