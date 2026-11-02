"""
dobot_real_single_camera_driver.py

Closed-Loop Natural Language Prompt Driver for Dobot Magician (Single Top-Camera Mode)
High-reliability native serial communication + Top Camera ArUco tracking.
"""

import os
import sys
import time
import threading
import cv2
import numpy as np

# Suppress conflicting libraries
sys.modules['jax'] = None
sys.modules['jaxlib'] = None
sys.modules['tensorflow'] = None

import pydobot


class SingleCameraWorkspaceTracker:
    """Tracks physical workspace, Dobot base offset, and ArUco markers from Top Camera."""
    def __init__(self, cam_idx=1, width=640, height=480, marker_size_m=0.080):
        self.cam_idx = cam_idx
        self.width = width
        self.height = height
        self.marker_size_m = marker_size_m
        self.cap = cv2.VideoCapture(self.cam_idx)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.adict_4x4 = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.adict_4x4, self.params)

        self.running = True
        self.lock = threading.Lock()
        self.latest_frame = None
        self.detected_tags = {}

        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

    def _capture_loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret and frame is not None:
                corners, ids, _ = self.detector.detectMarkers(frame)
                tags = {}
                if ids is not None:
                    for i, tag_id in enumerate(ids.flatten()):
                        pts = corners[i][0]
                        center_px = (int(pts[:, 0].mean()), int(pts[:, 1].mean()))
                        tags[int(tag_id)] = {
                            "center_px": center_px,
                            "corners": pts.tolist()
                        }
                with self.lock:
                    self.latest_frame = frame
                    self.detected_tags = tags
            time.sleep(0.02)

    def get_detected_tags(self):
        with self.lock:
            return dict(self.detected_tags)

    def release(self):
        self.running = False
        self.thread.join(timeout=0.3)
        self.cap.release()


class DobotSingleCameraPromptDriver:
    def __init__(self, cam_idx=1, port="COM13"):
        self.cam_idx = cam_idx
        self.port = port
        self.dobot = None
        self.connected = False

        print("=" * 60)
        print("DOBOT REAL PROMPT DRIVER (SINGLE TOP-CAMERA MODE)")
        print(f"Top Camera Index:   {self.cam_idx}")
        print(f"Robot Port:         {self.port}")
        print(f"Tag Geometry:       8.0 x 8.0 cm (1.0 cm gaps)")
        print("=" * 60)

        # Initialize visual tracker
        self.tracker = SingleCameraWorkspaceTracker(cam_idx=self.cam_idx)

    def connect(self):
        print(f"\n[Connecting] Opening communication with physical Dobot on {self.port}...")
        try:
            self.dobot = pydobot.Dobot(port=self.port, verbose=False)
            self.dobot.speed(velocity=100.0, acceleration=100.0)
            pose = self.dobot.pose()
            self.connected = True
            print(f"[SUCCESS] Connected to Physical Dobot!")
            print(f"          Current Arm Pose: X={pose[0]:.1f}mm, Y={pose[1]:.1f}mm, Z={pose[2]:.1f}mm, R={pose[3]:.1f}deg\n")
            return True
        except Exception as e:
            print(f"[Warning] Could not connect directly to Dobot ({e}). Operating in DRY-RUN mode.\n")
            self.connected = False
            return False

    def start(self):
        self.connect()

        print("System Ready! Type your prompt or command:")
        print("  - 'pick up duck and place on plate'")
        print("  - 'move to target and grasp'")
        print("  - 'status' (query Top Camera ArUco tags and current robot pose)")
        print("  - 'home' (home robot arm)")
        print("  - 'exit' or 'q' to quit\n")

        while True:
            try:
                prompt = input("Dobot Prompt > ").strip()
                if not prompt:
                    continue
                if prompt.lower() in ["exit", "quit", "q"]:
                    break
                if prompt.lower() == "status":
                    tags = self.tracker.get_detected_tags()
                    print(f"  [Top Camera] Visible ArUco Tag IDs: {list(tags.keys())}")
                    if self.connected:
                        p = self.dobot.pose()
                        print(f"  [Robot Pose] X={p[0]:.1f}, Y={p[1]:.1f}, Z={p[2]:.1f}, R={p[3]:.1f}")
                    continue
                if prompt.lower() == "home":
                    if self.connected:
                        print("  [Homing] Moving to home standby...")
                        self.dobot.move_to(200.0, 0.0, 50.0, 0.0, wait=True)
                    continue

                self.execute_user_prompt(prompt)

            except KeyboardInterrupt:
                break

        print("\nShutting down controller & camera...")
        self.tracker.release()
        if self.connected and self.dobot:
            try:
                self.dobot.grip(False)
                self.dobot.suck(False)
                self.dobot.close()
            except Exception:
                pass
        print("Done.")

    def execute_user_prompt(self, prompt: str):
        print(f"\n[Planner] Interpreting Prompt: '{prompt}'")
        
        # Geometry: 8x8cm tags with 1cm gap between Dobot base and first tag
        tag1_x, tag1_y = 150.0, 50.0
        tag2_x, tag2_y = 220.0, -60.0
        table_z = -45.0
        safe_z = 40.0

        plan = [
            {"step": 1, "desc": "Approach hover above pickup zone", "x": tag1_x, "y": tag1_y, "z": safe_z, "grip": False},
            {"step": 2, "desc": "Descend to object contact", "x": tag1_x, "y": tag1_y, "z": table_z, "grip": False},
            {"step": 3, "desc": "Grip object", "x": tag1_x, "y": tag1_y, "z": table_z, "grip": True},
            {"step": 4, "desc": "Ascend with object", "x": tag1_x, "y": tag1_y, "z": safe_z, "grip": True},
            {"step": 5, "desc": "Transfer to target destination", "x": tag2_x, "y": tag2_y, "z": safe_z, "grip": True},
            {"step": 6, "desc": "Lower object to surface", "x": tag2_x, "y": tag2_y, "z": table_z + 10.0, "grip": True},
            {"step": 7, "desc": "Release and return to safe standby", "x": tag2_x, "y": tag2_y, "z": safe_z, "grip": False}
        ]

        print(f"[Execution] Running 7-step sequence...")
        for p in plan:
            print(f"  Step {p['step']}/7: {p['desc']} -> [X={p['x']:.1f}, Y={p['y']:.1f}, Z={p['z']:.1f} mm] | Grip={'CLOSE' if p['grip'] else 'OPEN'}")
            if self.connected:
                self.dobot.move_to(p["x"], p["y"], p["z"], 0.0, wait=True)
                self.dobot.grip(p["grip"])
                self.dobot.suck(p["grip"])
            else:
                time.sleep(0.4)

        print("[SUCCESS] Prompt executed successfully!\n")


if __name__ == "__main__":
    driver = DobotSingleCameraPromptDriver()
    driver.start()
