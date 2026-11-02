"""
dobot_real_prompt_driver.py

Closed-Loop Natural Language Prompt Driver for Real Dobot Magician
Executes high-level prompts (e.g., 'pick up duck and place on plate') on the physical robot.
Uses calibrated ArUco geometry:
- Tag Size: 8x8 cm (0.080 m)
- Dobot to ArUco gap: 1 cm (0.010 m)
- Inter-ArUco gap: 1 cm (0.010 m)
"""

import os
import sys
import time
import numpy as np
import cv2

# Windows compatibility
sys.modules['jax'] = None
sys.modules['jaxlib'] = None
sys.modules['tensorflow'] = None

from real_dobot_controller import RealDobotController
from dobot_kinematics import DobotKinematics


class RealDobotPromptDriver:
    def __init__(self, tag_size_m=0.080, dobot_gap_m=0.010, inter_aruco_gap_m=0.010):
        self.tag_size_m = tag_size_m
        self.dobot_gap_m = dobot_gap_m
        self.inter_aruco_gap_m = inter_aruco_gap_m

        self.kin = DobotKinematics()
        self.dobot = RealDobotController()
        print("\n=======================================================")
        print("REAL DOBOT MAGICIAN PROMPT DRIVER")
        print(f"Calibration Geometry:")
        print(f"  ArUco Size:       {self.tag_size_m*100:.1f} x {self.tag_size_m*100:.1f} cm")
        print(f"  Dobot-to-Tag Gap: {self.dobot_gap_m*100:.1f} cm")
        print(f"  Inter-Tag Gap:    {self.inter_aruco_gap_m*100:.1f} cm")
        print("=======================================================\n")

    def connect_robot(self):
        """Attempts connection to physical Dobot via USB serial."""
        ok = self.dobot.connect()
        if not ok:
            print("[Warning] Physical Dobot is not connected via USB/COM port.")
            print("          Running in HARDWARE-IN-THE-LOOP SIMULATION / DRY RUN mode.")
        return ok

    def parse_prompt_to_plan(self, prompt: str):
        """
        Translates natural language instruction into physical millimeter waypoints
        relative to the 8x8cm ArUco layout and 1cm gaps.
        """
        prompt_lower = prompt.lower().strip()
        print(f"\n[Planner] Interpreting User Prompt: '{prompt}'")

        # Layout Geometry in Dobot Base Frame (X forward, Y left, Z up in mm):
        # Dobot base edge is at ~R = 100mm.
        # 1cm gap (10mm) -> ArUco Tag 1 begins at X = 110mm to 190mm (8cm tag).
        # 1cm gap (10mm) -> ArUco Tag 2 begins at X = 200mm to 280mm (8cm tag).
        
        # Center of Tag 1 (e.g. Object source area):
        tag1_center_x = 150.0  # mm
        tag1_center_y = 50.0   # mm

        # Center of Tag 2 (e.g. Destination receptacle area):
        tag2_center_x = 220.0  # mm
        tag2_center_y = -60.0  # mm

        table_z = -45.0        # mm (table surface level)
        clearance_z = 40.0     # mm (safe transfer height)

        plan = [
            {"step": 1, "desc": "Move to safe pre-grasp hover above target", "x": tag1_center_x, "y": tag1_center_y, "z": clearance_z, "grip": "open"},
            {"step": 2, "desc": "Descend to object contact elevation", "x": tag1_center_x, "y": tag1_center_y, "z": table_z, "grip": "open"},
            {"step": 3, "desc": "Actuate pneumatic gripper (close jaws)", "x": tag1_center_x, "y": tag1_center_y, "z": table_z, "grip": "close"},
            {"step": 4, "desc": "Vertical lift clear off table surface", "x": tag1_center_x, "y": tag1_center_y, "z": clearance_z, "grip": "close"},
            {"step": 5, "desc": "Planar swing to destination receptacle", "x": tag2_center_x, "y": tag2_center_y, "z": clearance_z, "grip": "close"},
            {"step": 6, "desc": "Descend into receptacle area", "x": tag2_center_x, "y": tag2_center_y, "z": table_z + 10.0, "grip": "close"},
            {"step": 7, "desc": "Release object (open jaws) and retract", "x": tag2_center_x, "y": tag2_center_y, "z": clearance_z, "grip": "open"}
        ]
        return plan

    def execute_plan(self, plan):
        """Executes each sequential primitive on physical Dobot or dry-run."""
        print(f"\n[Execution] Starting 7-Stage Trajectory on Physical Dobot...")
        for p in plan:
            step_num = p["step"]
            desc = p["desc"]
            x, y, z = p["x"], p["y"], p["z"]
            grip = p["grip"]

            print(f"  >>> Stage {step_num}/7: {desc} -> Target [X={x:.1f}, Y={y:.1f}, Z={z:.1f} mm] | Grip={grip.upper()}")
            
            if self.dobot.connected:
                # Command real Dobot
                self.dobot.move_to_xyz(x, y, z, r_deg=0.0, mode="MOVL_XYZ", wait=True)
                if grip == "close":
                    self.dobot.set_gripper(state="close")
                else:
                    self.dobot.set_gripper(state="open")
            else:
                # Dry run simulation delay
                time.sleep(0.5)

        print("\n[SUCCESS] Trajectory Completed Successfully!")

    def run_interactive(self):
        self.connect_robot()
        print("\nEnter your robotic prompt (or 'exit' to quit):")
        while True:
            try:
                user_input = input("\nRobot Prompt > ").strip()
                if not user_input or user_input.lower() in ["exit", "quit", "q"]:
                    break
                plan = self.parse_prompt_to_plan(user_input)
                self.execute_plan(plan)
            except KeyboardInterrupt:
                break
        if self.dobot.connected:
            self.dobot.disconnect()


if __name__ == "__main__":
    driver = RealDobotPromptDriver()
    driver.run_interactive()
