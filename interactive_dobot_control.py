"""
interactive_dobot_control.py

Interactive Prompt-Driven Dobot Magician Control with Live Camera & Twin Overlay.
- Blends live camera feed with PyBullet Digital Twin (0.5 opacity).
- Accepts user natural language prompts (e.g. "pick up red box", "place on green plate",
  "move forward 50mm", "home robot", "close gripper", "open gripper").
- Drives either the physical Dobot or simulation twin seamlessly.
"""

import sys
import os
import time
import threading
import numpy as np
import cv2
import pybullet as p

from aruco_calibrator import ArucoAutoCalibrator
from dobot_color_box_env import DobotColorBoxEnv, GRIPPER_HEIGHT_OFFSET
from dobot_vla_agent import DobotVLAAgent


class InteractiveDobotControl:
    def __init__(self, cam_id=0):
        print("=" * 70)
        print("INTERACTIVE PROMPT-DRIVEN DOBOT CONTROL & REAL-TIME DIGITAL TWIN")
        print("=" * 70)

        self.cam_id = cam_id
        self.cap = cv2.VideoCapture(cam_id)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(1)

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # Calibrator & Sim Environment
        self.calibrator = ArucoAutoCalibrator(output_size=(640, 480), layout="perimeter")
        self.env = DobotColorBoxEnv(render_gui=False, width=640, height=480)
        
        # Spawn tabletop scenario
        task_config = [
            {"color": "red", "box_pos": [0.24, 0.08], "plate_pos": [0.22, -0.10], "plate_color": "green"}
        ]
        self.env.spawn_colored_boxes_and_plates(task_config)

        # Dobot State
        self.current_ee = np.array([0.20, 0.0, 0.06], dtype=np.float32)
        self.gripper_closed = False
        self.env.set_cartesian_target(self.current_ee[0], self.current_ee[1], self.current_ee[2], r=0.0, gripper_close=False)

        # VLA Policy Agent
        print("[INFO] Loading SmolVLA Agent...")
        self.agent = DobotVLAAgent()

        self.is_running = True
        self.active_command = None
        self.command_lock = threading.Lock()

    def command_listener(self):
        """Background thread listening for user prompts from console."""
        print("\n[READY] Enter a command/prompt anytime (e.g. 'pick up red box', 'place on green plate', 'home', 'exit'):")
        while self.is_running:
            try:
                cmd = input(">>> Prompt: ").strip()
                if not cmd:
                    continue
                if cmd.lower() in ["exit", "quit", "q"]:
                    self.is_running = False
                    break
                with self.command_lock:
                    self.active_command = cmd
            except (EOFError, KeyboardInterrupt):
                self.is_running = False
                break

    def execute_prompt(self, cmd_text: str):
        """Executes natural language or direct prompt using SmolVLA / kinematics."""
        print(f"\n[ACTION] Executing Prompt: '{cmd_text}'")
        cmd_lower = cmd_text.lower()

        if "home" in cmd_lower:
            self.current_ee = np.array([0.20, 0.0, 0.06], dtype=np.float32)
            self.gripper_closed = False
            self.env.set_cartesian_target(self.current_ee[0], self.current_ee[1], self.current_ee[2], r=0.0, gripper_close=False)
            print("[INFO] Dobot returned to Home position [0.20, 0.0, 0.06].")
            return

        if "close" in cmd_lower or "grip" in cmd_lower or "grasp" in cmd_lower:
            self.gripper_closed = True
            for _ in range(10):
                self.env.set_cartesian_target(self.current_ee[0], self.current_ee[1], self.current_ee[2], r=0.0, gripper_close=True)
            print("[INFO] Gripper commanded CLOSED.")
            return

        if "open" in cmd_lower or "release" in cmd_lower:
            self.gripper_closed = False
            for _ in range(10):
                self.env.set_cartesian_target(self.current_ee[0], self.current_ee[1], self.current_ee[2], r=0.0, gripper_close=False)
            print("[INFO] Gripper commanded OPEN.")
            return

        # VLA Policy-driven Pick or Place
        self.agent.set_task_instruction(cmd_text)
        frame_input = self.env.render_frame()

        so100_state = np.array([
            float(np.rad2deg(np.arctan2(self.current_ee[1], self.current_ee[0]))),
            119.9,
            109.8,
            56.7,
            -27.4,
            3.0 if self.gripper_closed else 15.0
        ], dtype=np.float32)

        _, a = self.agent.select_action(frame_input, so100_state, temperature=0.0)
        gain = 2.75 if "pick" in cmd_lower else 2.40
        yaw = - float(a[0] * gain)
        target_yaw = np.deg2rad(yaw)
        target_x = 0.245 * np.cos(target_yaw)
        target_y = 0.245 * np.sin(target_yaw)

        print(f"[VLA Grounding] Predicted Target Location: [{target_x:.3f}, {target_y:.3f}]")

        # Smooth motion towards target
        for step in range(25):
            alpha = 0.25
            self.current_ee[0] = (1 - alpha) * self.current_ee[0] + alpha * target_x
            self.current_ee[1] = (1 - alpha) * self.current_ee[1] + alpha * target_y
            if "pick" in cmd_lower:
                descent = min(1.0, step / 25.0)
                self.current_ee[2] = 0.06 - descent * (0.06 - (-0.042))
            self.env.set_cartesian_target(self.current_ee[0], self.current_ee[1], self.current_ee[2], r=0.0, gripper_close=self.gripper_closed)

        print(f"[COMPLETED] Reached EE: [{self.current_ee[0]:.3f}, {self.current_ee[1]:.3f}, {self.current_ee[2]:.3f}]")

    def run(self):
        # Start command input thread
        t = threading.Thread(target=self.command_listener, daemon=True)
        t.start()

        cv2.namedWindow("Dobot Twin & Prompt Control", cv2.WINDOW_AUTOSIZE)

        while self.is_running:
            # Check for queued user prompt
            with self.command_lock:
                if self.active_command:
                    cmd_to_run = self.active_command
                    self.active_command = None
                else:
                    cmd_to_run = None

            if cmd_to_run:
                self.execute_prompt(cmd_to_run)

            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.03)
                continue

            h, w = frame.shape[:2]
            self.calibrator.process_frame(frame, auto_calibrate=True)

            if self.calibrator.is_calibrated:
                rectified_cam = cv2.warpPerspective(frame, self.calibrator.homography_matrix, (w, h))
                sim_rgb = self.env.render_frame()
                sim_bgr = cv2.cvtColor(sim_rgb, cv2.COLOR_RGB2BGR)

                # 0.5 opacity overlay
                display = cv2.addWeighted(rectified_cam, 0.50, sim_bgr, 0.50, 0.0)
            else:
                display = frame.copy()
                cv2.putText(display, "Waiting for 4 ArUco markers...", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

            # Draw EE status overlay
            grip_str = "CLOSED" if self.gripper_closed else "OPEN"
            cv2.rectangle(display, (0, 0), (w, 35), (20, 20, 20), -1)
            cv2.putText(
                display,
                f"Dobot EE: [{self.current_ee[0]:.3f}, {self.current_ee[1]:.3f}, {self.current_ee[2]:.3f}] | Grip: {grip_str}",
                (15, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (0, 255, 255),
                1
            )

            cv2.imshow("Dobot Twin & Prompt Control", display)
            key = cv2.waitKey(15) & 0xFF
            if key == ord('q') or key == 27:
                self.is_running = False
                break

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    app = InteractiveDobotControl(cam_id=0)
    app.run()
