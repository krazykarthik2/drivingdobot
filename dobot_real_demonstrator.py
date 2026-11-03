"""
dobot_real_demonstrator.py

Dobot Physical Hand-Teaching (Kinesthetic Demonstration) Recorder:
- Uses Dobot's physical hand-guiding capability (press the unlock key on Dobot forearm
  or move arm by hand smoothly while the motors comply / read joint encoders).
- Continuously polls Dobot `get_pose()` at safe, steady intervals (~10-15 Hz) without serial flooding.
- No W/A/S/D/E/C keyboard driving needed! You physically move the arm with your hands.
- [SPACE]: Toggle Gripper / Suction (Open / Close) anytime during demonstration.
- [R]: Start / Stop Recording Episode.
- [T]: Trigger Background Local Fine-Tuning of SmolVLA Action Heads.
- [1/2/3/4]: Switch Task Goal ('approach the lock', 'pick up lock', 'push lock right', 'home standby').
- [Q / ESC]: Quit.
- Auto-saves all episodes to 'real_dobot_demonstrations.pkl'.
"""

import os
import sys
import time
import pickle
import threading
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
from dobot_clean import clean_dobot, get_clean_dobot
from dobot_kinematics import DobotKinematics


class DobotHandTeachingDemonstrator:
    def __init__(self, cam_idx=1, robot_port="COM13", save_path="real_dobot_demonstrations.pkl"):
        self.cam_idx = cam_idx
        self.robot_port = robot_port
        self.save_path = save_path
        self.kin = DobotKinematics()

        print("=" * 70)
        print("DOBOT PHYSICAL HAND-TEACHING RECORDER (KINESTHETIC DEMONSTRATIONS)")
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

        self.dobot = get_clean_dobot(port=self.robot_port, verbose=True)
        if self.dobot:
            try:
                p_init = self.dobot.pose()
                self.current_pose = [p_init[0], p_init[1], p_init[2], p_init[3]]
                self.connected = True
                print(f"[Dobot] Connected! Initial Pose: X={p_init[0]:.1f}, Y={p_init[1]:.1f}, Z={p_init[2]:.1f} mm")
                print("[Dobot] Alarms cleared & queues started. Hand-teaching mode active!\n")
            except Exception as e:
                print(f"[Dobot Warning] Could not read pose: {e}")
        else:
            print(f"[Dobot Warning] Could not connect to {self.robot_port}. Running in offline preview mode.\n")

        # 3. Load Existing Demonstrations
        self.episodes = []
        if os.path.exists(self.save_path):
            try:
                with open(self.save_path, "rb") as f:
                    self.episodes = pickle.load(f)
                print(f"[Storage] Loaded {len(self.episodes)} existing episodes from {self.save_path}.")
            except Exception:
                self.episodes = []

        # Recording State
        self.is_recording = False
        self.current_instruction = "approach the lock"
        self.curr_cam_frames = []
        self.curr_states = []
        self.curr_raw_poses = []
        self.curr_actions = []

        # Target suggestion area & Start point for guided demo collection
        self.target_zones = [
            {"name": "Right Zone", "rect": (400, 200, 180, 180), "color": (0, 165, 255), "desc": "Lock Target Area (Right)"},
            {"name": "Center Zone", "rect": (230, 220, 180, 180), "color": (0, 255, 0), "desc": "Lock Target Area (Center)"},
            {"name": "Left Zone", "rect": (60, 200, 180, 180), "color": (255, 144, 30), "desc": "Lock Target Area (Left)"},
        ]
        self.current_zone_idx = 0
        self.pick_new_random_zone()

        self.training_in_progress = False

    def pick_new_random_zone(self):
        """Picks a random placement zone, biasing towards right/center if right is lacking."""
        import random
        # Random choice from available target zones
        self.current_zone_idx = random.randint(0, len(self.target_zones) - 1)
        # Add slight random jitter within the region
        base_x, base_y, bw, bh = self.target_zones[self.current_zone_idx]["rect"]
        jx = random.randint(-25, 25)
        jy = random.randint(-25, 25)
        self.active_target_box = (
            max(20, base_x + jx),
            max(100, base_y + jy),
            bw,
            bh
        )

        # Dedicated safe polling thread for Dobot pose & recording transitions at ~12 Hz
        self.record_thread = threading.Thread(target=self._pose_poller_and_recorder_loop, daemon=True)
        self.record_thread.start()

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

    def _pose_poller_and_recorder_loop(self):
        """Continuously polls pose at precise 10.0 Hz and records relative delta actions."""
        dt_target = 0.10  # 10.0 Hz precise robotic sampling rate

        while self.running:
            t_start = time.time()

            # Safe pose query
            if self.connected and self.dobot:
                with self.robot_lock:
                    try:
                        p = self.dobot.pose()
                        self.current_pose = [p[0], p[1], p[2], p[3]]
                    except Exception:
                        pass

            cur_x, cur_y, cur_z = self.current_pose[0], self.current_pose[1], self.current_pose[2]

            # If recording is active, sample step
            if self.is_recording:
                with self.frame_lock:
                    frame = self.latest_frame.copy() if self.latest_frame is not None else None

                if frame is not None:
                    f_small = cv2.resize(frame, (256, 256))
                    _, buf = cv2.imencode(".jpg", f_small, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                    st = self.dobot_to_so100_state(cur_x, cur_y, cur_z, self.gripper_closed)

                    self.curr_cam_frames.append(buf.tobytes())
                    self.curr_states.append(st)
                    self.curr_raw_poses.append((cur_x, cur_y, cur_z, self.gripper_closed))

            elapsed = time.time() - t_start
            time.sleep(max(0.005, dt_target - elapsed))  # Strict 10.0 Hz cadence

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
            self.curr_cam_frames = []
            self.curr_states = []
            self.curr_raw_poses = []
            self.is_recording = True
            print(f"\n>>> [REC START] Recording Episode {len(self.episodes) + 1} for '{self.current_instruction}'...")
            print("    Move Dobot smoothly by hand now. Press [SPACE] for gripper.")
        else:
            # Stop and save episode
            self.is_recording = False
            total_samples = len(self.curr_raw_poses)
            if total_samples >= 5:
                # Compute true step-by-step relative delta actions (dX, dY, dZ, grip)
                actions = []
                for i in range(total_samples):
                    if i < total_samples - 1:
                        p_next = self.curr_raw_poses[i + 1]
                        p_curr = self.curr_raw_poses[i]
                        dx = float(p_next[0] - p_curr[0])
                        dy = float(p_next[1] - p_curr[1])
                        dz = float(p_next[2] - p_curr[2])
                    else:
                        dx, dy, dz = 0.0, 0.0, 0.0
                    grip_val = 3.0 if self.curr_raw_poses[i][3] else 15.0
                    # Pure Cartesian Relative Action Vector: [dx, dy, dz, 0.0, 0.0, grip]
                    act = np.array([dx, dy, dz, 0.0, 0.0, grip_val], dtype=np.float32)
                    actions.append(act)

                ep = {
                    "cam1": self.curr_cam_frames,
                    "cam2": [b"" for _ in self.curr_cam_frames],
                    "cam3": [b"" for _ in self.curr_cam_frames],
                    "states": self.curr_states,
                    "actions": actions,
                    "instructions": [self.current_instruction for _ in actions]
                }
                self.episodes.append(ep)
                with open(self.save_path, "wb") as f:
                    pickle.dump(self.episodes, f)
                print(f">>> [REC SAVED] Episode {len(self.episodes)} saved! ({len(actions)} relative delta steps at 10 Hz).")
                print(f"    Saved to: {self.save_path}\n")

                # Pick a new randomized zone for the next demo
                self.pick_new_random_zone()

                if len(self.episodes) % 5 == 0:
                    print(f"*** Batch Milestone: {len(self.episodes)} episodes collected! Run modal trainer! ***\n")
            else:
                print(f"[REC DISCARDED] Episode had only {total_samples} steps (need at least 5).\n")

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
                train_smolvla_local.train_local(
                    data_path=self.save_path,
                    epochs=3,
                    batch_size=4,
                    grad_accum_steps=2,
                    lr=2e-4
                )
                print("\n[FINE-TUNING COMPLETE] New fine-tuned heads saved to smolvla_no_wrist_dobot/smolvla_no_wrist_heads.pt!\n")
            except Exception as e:
                print(f"[Fine-Tuning Error]: {e}")
            finally:
                self.training_in_progress = False

        t = threading.Thread(target=_train_worker, daemon=False)
        t.start()

    def run(self):
        window_title = "Dobot Hand-Teaching Recorder (Kinesthetic Guiding)"
        cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_title, 800, 600)

        print("\nHOW TO RECORD DEMOS:")
        print("  1. Press unlock button on Dobot arm (or guide arm by hand).")
        print("  2. [R]:       Start recording (REC indicator turns RED).")
        print("  3. Guide arm smoothly to target.")
        print("  4. [SPACE]:   Toggle Gripper / Suction.")
        print("  5. [R]:       Stop recording -> Automatically saved!")
        print("  6. [T]:       Fine-Tune locally whenever you want!")
        print("  7. [1/2/3/4]: Change Goal (1: approach lock, 2: pick lock, 3: push right, 4: home)")
        print("  8. [Q / ESC]: Quit\n")

        while self.running:
            with self.frame_lock:
                frame = self.latest_frame.copy() if self.latest_frame is not None else None

            if frame is None:
                time.sleep(0.01)
                continue

            disp = frame.copy()
            h, w = disp.shape[:2]

            # Top Telemetry Banner
            status_color = (0, 0, 255) if self.is_recording else (0, 200, 0)
            status_text = f"REC ACTIVE (Steps: {len(self.curr_actions)})" if self.is_recording else "STANDBY (Hand-Guide Ready)"
            cv2.rectangle(disp, (0, 0), (w, 42), (20, 20, 20), -1)
            cv2.putText(disp, f"{status_text}", (15, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.72, status_color, 2)
            cv2.putText(disp, f"Saved Demos: {len(self.episodes)}", (w - 230, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Instruction and Live Pose from Encoders
            cv2.putText(disp, f"Goal: '{self.current_instruction}'", (15, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
            x, y, z = self.current_pose[0], self.current_pose[1], self.current_pose[2]
            grip_txt = "CLOSED" if self.gripper_closed else "OPEN"
            cv2.putText(disp, f"Live Arm Pose: X={x:.1f} Y={y:.1f} Z={z:.1f} mm | Grip={grip_txt}",
                        (15, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)

            # Draw Recommended Start Point of Arm (e.g. Home Standby: X=190, Y=0, Z=40)
            start_px, start_py = int(w * 0.50), int(h * 0.82)
            cv2.circle(disp, (start_px, start_py), 18, (0, 255, 255), 2)
            cv2.circle(disp, (start_px, start_py), 4, (0, 255, 255), -1)
            cv2.putText(disp, "RECOMMENDED ARM START", (start_px - 95, start_py + 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

            # Draw Randomized Target Area for Lock Placement
            tx, ty, tw, th = self.active_target_box
            zone_info = self.target_zones[self.current_zone_idx]
            zone_color = zone_info["color"]
            # Semi-transparent overlay for the target box
            overlay = disp.copy()
            cv2.rectangle(overlay, (tx, ty), (tx + tw, ty + th), zone_color, -1)
            cv2.addWeighted(overlay, 0.22, disp, 0.78, 0, disp)
            cv2.rectangle(disp, (tx, ty), (tx + tw, ty + th), zone_color, 2)
            cv2.putText(disp, f"PLACE LOCK HERE ({zone_info['name']})", (tx + 8, ty + 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2)
            cv2.putText(disp, "[N]: New Random Area", (tx + 8, ty + 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, zone_color, 1)

            if self.training_in_progress:
                cv2.rectangle(disp, (10, h - 45), (w - 10, h - 10), (0, 80, 200), -1)
                cv2.putText(disp, "*** LOCAL FINE-TUNING IN PROGRESS... ***", (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow(window_title, disp)
            key = cv2.waitKey(20) & 0xFF

            if key in [27, ord('q')]:
                self.running = False
                break
            elif key == ord(' '):
                self.toggle_gripper()
            elif key in [ord('r'), ord('R')]:
                self.toggle_recording()
            elif key in [ord('n'), ord('N')]:
                self.pick_new_random_zone()
                print(f"[Zone] Switched to new random target zone: {self.target_zones[self.current_zone_idx]['name']}")
            elif key in [ord('t'), ord('T')]:
                self.trigger_background_finetune()
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
    app = DobotHandTeachingDemonstrator()
    app.run()
