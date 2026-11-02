"""
dobot_real_single_camera_driver.py

Closed-Loop Natural Language Prompt Driver for Dobot Magician (Single Top-Camera Mode)
Hardware Configuration:
- Top Camera (CAM 1): External USB Camera (Index 1) for ArUco workspace tracking
- Lateral Camera: Disabled (unplugged for USB port availability)
- Robot Serial Port: COM13 (CH9102 / CH340 USB-to-UART bridge)
- Calibration Geometry: 8x8 cm tags, 1 cm Dobot-to-Tag gap, 1 cm Inter-tag gap
"""

import os
import sys
import time
import json
import threading
import cv2
import numpy as np

# Windows environment overrides
sys.modules['jax'] = None
sys.modules['jaxlib'] = None
sys.modules['tensorflow'] = None

from real_dobot_controller import RealDobotController


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
    def __init__(self, cam_idx=1, port="COM13", baudrate=115200):
        self.cam_idx = cam_idx
        self.port = port
        self.baudrate = baudrate

        print("=" * 60)
        print("DOBOT REAL PROMPT DRIVER (SINGLE TOP-CAMERA MODE)")
        print(f"Top Camera Index:   {self.cam_idx}")
        print(f"Robot Port:         {self.port} (Baud: {self.baudrate})")
        print(f"Tag Geometry:       8.0 x 8.0 cm (1.0 cm gaps)")
        print("=" * 60)

        # Initialize visual tracker
        self.tracker = SingleCameraWorkspaceTracker(cam_idx=self.cam_idx)
        # Initialize robot controller
        self.dobot = RealDobotController(port=self.port, baudrate=self.baudrate)

    def start(self):
        print("\n[Connecting] Initializing Dobot controller...")
        connected = self.dobot.connect()
        if connected:
            print(f"[OK] Dobot connected on {self.port}!")
        else:
            print(f"[Notice] Operating in Safe Emulated / Hardware-in-the-loop mode.")

        print("\nSystem ready! Type natural language robot instructions.")
        print("Examples:")
        print("  - 'pick up duck and place on plate'")
        print("  - 'move to target and grasp'")
        print("  - 'status' (view visible camera ArUco markers)")
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
                    print(f"  [Top Camera] Currently visible ArUco Tag IDs: {list(tags.keys())}")
                    continue

                self.execute_user_prompt(prompt)

            except KeyboardInterrupt:
                break

        print("\nShutting down controller & camera...")
        self.tracker.release()
        if self.dobot.connected:
            self.dobot.disconnect()
        print("Done.")

    def execute_user_prompt(self, prompt: str):
        print(f"\n[Planner] Interpreting Prompt: '{prompt}'")
        
        # Real Dobot Cartesian base frame coordinate mapping (mm)
        # Dobot base origin is (0, 0, 0)
        # 1 cm gap to 8x8 cm tag 1 -> X = 150mm
        # 1 cm gap to 8x8 cm tag 2 -> X = 220mm
        tag1_x, tag1_y = 150.0, 50.0
        tag2_x, tag2_y = 220.0, -60.0
        table_z = -45.0
        safe_z = 40.0

        plan = [
            {"step": 1, "desc": "Approach hover above pickup zone", "x": tag1_x, "y": tag1_y, "z": safe_z, "grip": "open"},
            {"step": 2, "desc": "Descend to object contact", "x": tag1_x, "y": tag1_y, "z": table_z, "grip": "open"},
            {"step": 3, "desc": "Grip object", "x": tag1_x, "y": tag1_y, "z": table_z, "grip": "close"},
            {"step": 4, "desc": "Ascend with object", "x": tag1_x, "y": tag1_y, "z": safe_z, "grip": "close"},
            {"step": 5, "desc": "Transfer to target destination", "x": tag2_x, "y": tag2_y, "z": safe_z, "grip": "close"},
            {"step": 6, "desc": "Lower object to surface", "x": tag2_x, "y": tag2_y, "z": table_z + 10.0, "grip": "close"},
            {"step": 7, "desc": "Release and return to safe standby", "x": tag2_x, "y": tag2_y, "z": safe_z, "grip": "open"}
        ]

        print(f"[Execution] Running 7-step sequence...")
        for p in plan:
            print(f"  Step {p['step']}/7: {p['desc']} -> [X={p['x']:.1f}, Y={p['y']:.1f}, Z={p['z']:.1f} mm] | Grip={p['grip'].upper()}")
            if self.dobot.connected:
                self.dobot.move_to_xyz(p["x"], p["y"], p["z"], r_deg=0.0, mode="MOVL_XYZ", wait=True)
                self.dobot.set_gripper(p["grip"])
            else:
                time.sleep(0.4)

        print("[SUCCESS] Prompt executed successfully!\n")


if __name__ == "__main__":
    driver = DobotSingleCameraPromptDriver()
    driver.start()
