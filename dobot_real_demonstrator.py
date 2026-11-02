"""
dobot_real_demonstrator.py

Interactive Real-Time Dobot Demonstration Recorder & On-the-Fly Fine-Tuner:
- Single Top Camera (Index 1) with DirectShow backend (zero frame drops).
- Keyboard teleoperation / manual arm teaching:
  * [W / S]: Move Forward / Backward (X)
  * [A / D]: Move Left / Right (Y)
  * [E / C]: Move Up / Down (Z)
  * [SPACE]: Toggle Gripper / Suction (Open / Close)
  * [R]: Start / Stop Recording Episode
  * [T]: Trigger Background Local Fine-Tuning of SmolVLA Action Heads
  * [Q / ESC]: Quit
- Live HUD displaying recording status, episode count, frame count, and Dobot telemetry.
- Auto-saves episodes to 'real_dobot_demonstrations.pkl'.
- Every batch of recorded episodes can be fine-tuned locally in minutes!
"""

import os
import sys
import time
import pickle
import threading
import queue
import cv2
import numpy as np
import torch

# Compatibility stubs
sys.modules['jax'] = None
sys.modules['jaxlib'] = None
sys.modules['tensorflow'] = None
os.environ['HF_HUB_OFFLINE'] = '1'

import pydobot
from pydobot.message import Message
from dobot_kinematics import DobotKinematics


class DobotRealDemonstrator:
    def __init__(self, cam_idx=1, robot_port="COM13", save_path="real_dobot_demonstrations.pkl"):
        self.cam_idx = cam_idx
        self.robot_port = robot_port
        self.save_path = save_path
        self.kin = DobotKinematics()

        print("=" * 70)
        print("DOBOT REAL-TIME DEMONSTRATION RECORDER & FAST LOCAL FINE-TUNER")
        print("=" * 70)

        # 1. Camera Initialization with DirectShow
        print(f"[1/3] Connecting to Top Camera (Index {self.cam_idx})...")
        self.cap = cv2.VideoCapture(self.cam_idx, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.cam_idx)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(0)

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.frame_lock = threading.Lock()
        self.latest_frame = None
        self.running = True

        self.cam_thread = threading.Thread(target=self._camera_loop, daemon=True)
        self.cam_thread.start()

        # 2. Dobot Magician Connection
        print(f"[2/3] Connecting to Dobot Magician on {self.robot_port}...")
        self.dobot = None
        self.connected = False
        self.robot_lock = threading.Lock()
        self.current_pose = [190.0, 0.0, 35.0, 0.0]
        self.gripper_closed = False

        try:
            self.dobot = pydobot.Dobot(port=self.robot_port, verbose=False)
            # Reliable alarm clearance on startup
            m = Message()
            m.id = 20
            m.ctrl = 1
            m.params = bytearray([])
            self.dobot._send_message(m)
            time.sleep(0.05)
            try:
                self.dobot._read_message()
            except Exception:
                pass

            p_init = self.dobot.pose()
            self.current_pose = list(p_init)
            self.connected = True
            print(f"[Dobot] Connected! Initial Pose: X={p_init[0]:.1f}, Y={p_init[1]:.1f}, Z={p_init[2]:.1f} mm")
            print("[Dobot] Alarms cleared successfully.\n")
        except Exception as e:
            print(f"[Dobot Warning] Could not connect to {self.robot_port} ({e}). Running in manual simulation mode.\n")

        # 3. Load or Initialize Demonstrations Storage
        self.episodes = []
        if os.path.exists(self.save_path):
            try:
                with open(self.save_path, "rb") as f:
                    self.episodes = pickle.load(f)
                print(f"[Storage] Loaded {len(self.episodes)} existing demonstration episodes from {self.save_path}.")
            except Exception:
                self.episodes = []

        # Current Recording State
        self.is_recording = False
        self.current_instruction = "approach the lock"
        self.curr_cam_frames = []
        self.curr_states = []
        self.curr_actions = []

        # Step displacement increment (mm)
        self.step_size = 5.0
        self.training_in_progress = False

    def _camera_loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret and frame is not None:
                with self.frame_lock:
                    self.latest_frame = frame
            else:
                time.sleep(0.005)

    def dobot_to_so100_state(self, x_mm, y_mm, z_mm, is_grip_closed):
        x_m, y_m, z_m = x_mm / 1000.0, y_mm / 1000.0, z_mm / 1000.0
        try:
            inv = self.kin.inverse(x_m, y_m, z_m)
            pan_so100 = float(- np.rad2deg(inv[0]) / 2.45)
        except Exception:
            pan_so100 = 0.0

        return np.array([
            pan_so100,
            119.9,
            109.8,
            56.7,
            -27.4,
            3.0 if is_grip_closed else 15.0
        ], dtype=np.float32)

    def step_arm(self, dx, dy, dz):
        """Moves arm by delta and records step if recording is active."""
        if not self.running:
            return

        # Fetch current frame before motion
        with self.frame_lock:
            frame = self.latest_frame.copy() if self.latest_frame is not None else None

        cur_x, cur_y, cur_z, _ = self.current_pose
        new_x = float(np.clip(cur_x + dx, 140.0, 270.0))
        new_y = float(np.clip(cur_y + dy, -130.0, 130.0))
        new_z = float(np.clip(cur_z + dz, -35.0, 80.0))

        # Record transition
        if self.is_recording and frame is not None:
            # Downsample and JPEG encode frame to keep dataset lean
            f_small = cv2.resize(frame, (256, 256))
            _, buf = cv2.imencode(".jpg", f_small, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            st = self.dobot_to_so100_state(cur_x, cur_y, cur_z, self.gripper_closed)

            # Compute delta action in SO-100 degrees
            d_pan = float(- dy / 1.5)
            d_shoulder = float(119.9 + dx * 2.0)
            d_elbow = float(109.8 + dz * 2.5)
            act = np.array([d_pan, d_shoulder, d_elbow, 56.7, -27.4, 3.0 if self.gripper_closed else 15.0], dtype=np.float32)

            self.curr_cam_frames.append(buf.tobytes())
            self.curr_states.append(st)
            self.curr_actions.append(act)

        # Move physical arm
        self.current_pose[0] = new_x
        self.current_pose[1] = new_y
        self.current_pose[2] = new_z

        if self.connected and self.dobot:
            with self.robot_lock:
                try:
                    self.dobot.move_to(new_x, new_y, new_z, 0.0, wait=False)
                except Exception:
                    pass

    def toggle_gripper(self):
        """Toggles end-effector gripper/suction."""
        self.gripper_closed = not self.gripper_closed
        print(f"[Gripper] Toggled -> {'CLOSED / SUCK ON' if self.gripper_closed else 'OPEN / SUCK OFF'}")
        if self.connected and self.dobot:
            with self.robot_lock:
                try:
                    self.dobot.grip(self.gripper_closed)
                    self.dobot.suck(self.gripper_closed)
                except Exception:
                    pass

    def toggle_recording(self):
        """Starts or stops recording the current demonstration episode."""
        if not self.is_recording:
            # Start recording
            self.is_recording = True
            self.curr_cam_frames = []
            self.curr_states = []
            self.curr_actions = []
            print(f"\n[REC START] Recording episode {len(self.episodes) + 1} for: '{self.current_instruction}'")
        else:
            # Stop and save episode
            self.is_recording = False
            if len(self.curr_actions) >= 3:
                ep = {
                    "cam1": self.curr_cam_frames,
                    "cam2": [b"" for _ in self.curr_cam_frames],
                    "cam3": [b"" for _ in self.curr_cam_frames],
                    "states": self.curr_states,
                    "actions": self.curr_actions,
                    "instructions": [self.current_instruction for _ in self.curr_actions]
                }
                self.episodes.append(ep)
                with open(self.save_path, "wb") as f:
                    pickle.dump(self.episodes, f)
                print(f"[REC SAVED] Episode {len(self.episodes)} saved! ({len(self.curr_actions)} steps). Saved to {self.save_path}\n")

                # Auto-prompt batch fine-tuning every 5 episodes
                if len(self.episodes) % 5 == 0:
                    print(f"*** Milestone: {len(self.episodes)} episodes captured! Press [T] to fine-tune now! ***")
            else:
                print("[REC DISCARDED] Episode had fewer than 3 steps.\n")

    def trigger_background_finetune(self):
        """Runs fast local fine-tuning of the 12M action heads in the background."""
        if self.training_in_progress:
            print("[Train Warning] Fine-tuning already running!")
            return
        if len(self.episodes) < 3:
            print(f"[Train Warning] Only {len(self.episodes)} episodes recorded. Record at least 3-5 episodes before training.")
            return

        def _train_worker():
            self.training_in_progress = True
            try:
                print("\n" + "=" * 65)
                print(f"[FINE-TUNING START] Training SmolVLA Action Heads on {len(self.episodes)} real demonstrations...")
                print("=" * 65)
                import train_smolvla_local
                # Run fast 3-epoch local adaptation
                train_smolvla_local.train_local(epochs=3, batch_size=4, grad_accum_steps=2, lr=2e-4)
                print("\n[FINE-TUNING COMPLETE] New fine-tuned heads saved to smolvla_no_wrist_dobot/smolvla_no_wrist_heads.pt!\n")
            except Exception as e:
                print(f"[Fine-Tuning Error]: {e}")
            finally:
                self.training_in_progress = False

        t = threading.Thread(target=_train_worker, daemon=True)
        t.start()

    def run(self):
        window_title = "Dobot Magician Demonstration Recorder"
        cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_title, 800, 600)

        print("\nCONTROLS:")
        print("  [W / S]:      Forward / Backward (X +/-)")
        print("  [A / D]:      Left / Right (Y +/-)")
        print("  [E / C]:      Up / Down (Z +/-)")
        print("  [SPACE]:      Toggle Gripper / Suction")
        print("  [R]:          Start / Stop Recording Episode")
        print("  [T]:          Trigger Fast Local Fine-Tuning")
        print("  [1/2/3/4]:    Change Instruction (1: approach lock, 2: pick lock, 3: push right, 4: home)")
        print("  [Q / ESC]:    Quit\n")

        while self.running:
            with self.frame_lock:
                frame = self.latest_frame.copy() if self.latest_frame is not None else None

            if frame is None:
                time.sleep(0.01)
                continue

            disp = frame.copy()
            h, w = disp.shape[:2]

            # HUD Telemetry Overlay
            # Top banner
            status_color = (0, 0, 255) if self.is_recording else (0, 200, 0)
            status_text = f"REC (Steps: {len(self.curr_actions)})" if self.is_recording else "STANDBY"
            cv2.rectangle(disp, (0, 0), (w, 40), (20, 20, 20), -1)
            cv2.putText(disp, f"STATUS: {status_text}", (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            cv2.putText(disp, f"Total Episodes: {len(self.episodes)}", (w - 240, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Instruction and Robot Pose
            cv2.putText(disp, f"Goal: '{self.current_instruction}'", (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
            x, y, z, _ = self.current_pose
            cv2.putText(disp, f"Pose: X={x:.1f} Y={y:.1f} Z={z:.1f} mm | Grip={'CLOSED' if self.gripper_closed else 'OPEN'}",
                        (15, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

            if self.training_in_progress:
                cv2.putText(disp, "*** LOCAL FINE-TUNING IN PROGRESS... ***", (15, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 140, 255), 2)

            cv2.imshow(window_title, disp)
            key = cv2.waitKey(20) & 0xFF

            if key in [27, ord('q')]:
                self.running = False
                break
            elif key == ord(' '):
                self.toggle_gripper()
            elif key in [ord('r'), ord('R')]:
                self.toggle_recording()
            elif key in [ord('t'), ord('T')]:
                self.trigger_background_finetune()
            elif key in [ord('w'), ord('W')]:
                self.step_arm(+self.step_size, 0.0, 0.0)
            elif key in [ord('s'), ord('S')]:
                self.step_arm(-self.step_size, 0.0, 0.0)
            elif key in [ord('a'), ord('A')]:
                self.step_arm(0.0, +self.step_size, 0.0)  # Left
            elif key in [ord('d'), ord('D')]:
                self.step_arm(0.0, -self.step_size, 0.0)  # Right
            elif key in [ord('e'), ord('E')]:
                self.step_arm(0.0, 0.0, +self.step_size)  # Up
            elif key in [ord('c'), ord('C')]:
                self.step_arm(0.0, 0.0, -self.step_size)  # Down
            elif key == ord('1'):
                self.current_instruction = "approach the lock"
                print(f"[Instruction] Changed to: '{self.current_instruction}'")
            elif key == ord('2'):
                self.current_instruction = "pick up the lock"
                print(f"[Instruction] Changed to: '{self.current_instruction}'")
            elif key == ord('3'):
                self.current_instruction = "push the lock to the right"
                print(f"[Instruction] Changed to: '{self.current_instruction}'")
            elif key == ord('4'):
                self.current_instruction = "home standby"
                print(f"[Instruction] Changed to: '{self.current_instruction}'")

        self.cap.release()
        cv2.destroyAllWindows()
        if self.connected and self.dobot:
            try:
                self.dobot.close()
            except Exception:
                pass
        print("[Exit] Recorder closed successfully.")


if __name__ == "__main__":
    app = DobotRealDemonstrator()
    app.run()
