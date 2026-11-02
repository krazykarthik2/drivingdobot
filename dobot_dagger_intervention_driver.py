"""
dobot_dagger_intervention_driver.py

Interactive DAgger (Dataset Aggregation) with Physical Human-in-the-Loop Interventions:
- Runs fine-tuned SmolVLA in closed-loop physical inference on the Dobot Magician.
- Human Intervention Detection:
  * Whenever the policy makes a mistake or drifts, press the teach/unlock button on the Dobot arm
    and physically guide it to the correct target!
  * The script automatically senses the discrepancy between the autonomous command and the real human
    encoder movement, switches to INTERVENTION MODE, and records your corrective action!
- Instant Dataset Persistence:
  * Corrective recovery steps are appended directly into 'real_dobot_demonstrations.pkl'.
  * [T]: Trigger fast Modal Cloud GPU retrain anytime!
- Single Overhead Camera (Index 1) with DirectShow backend.
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
from transformers import AutoTokenizer
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from dobot_kinematics import DobotKinematics


class DobotDAggerController:
    def __init__(self, cam_idx=1, robot_port="COM13", save_path="real_dobot_demonstrations.pkl", heads_path="smolvla_no_wrist_dobot/smolvla_no_wrist_heads.pt"):
        print("=" * 70)
        print("DOBOT SMOLVLA DAGGER CONTROLLER (INTERACTIVE HUMAN INTERVENTIONS)")
        print("=" * 70)

        self.device = torch.device("cpu")
        self.kin = DobotKinematics()
        self.save_path = save_path
        self.heads_path = heads_path

        # 1. Camera Initialization with DirectShow
        print(f"[1/3] Connecting to Top Camera (Index {cam_idx})...")
        self.cap = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(cam_idx)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(0)

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.running = True
        self.frame_lock = threading.Lock()
        self.latest_frame = None

        self.cam_thread = threading.Thread(target=self._camera_reader_loop, daemon=True)
        self.cam_thread.start()

        # 2. Dobot Hardware Connection
        print(f"[2/3] Connecting to Physical Dobot on {robot_port}...")
        self.dobot = None
        self.connected = False
        self.cur_x, self.cur_y, self.cur_z = 190.0, 0.0, 40.0
        self.gripper_closed = False
        self.robot_lock = threading.Lock()

        try:
            self.dobot = pydobot.Dobot(port=robot_port, verbose=False)
            # Clear alarms on startup
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
            self.cur_x, self.cur_y, self.cur_z = p_init[0], p_init[1], p_init[2]
            self.connected = True
            print(f"[Dobot] Connected! Initial Pose: [{self.cur_x:.1f}, {self.cur_y:.1f}, {self.cur_z:.1f}] mm\n")
        except Exception as e:
            print(f"[Dobot Warning] Could not connect to {robot_port} ({e}). Running in DRY-RUN mode.\n")

        # 3. Load SmolVLA Policy
        print(f"[3/3] Loading SmolVLA Policy from {heads_path}...")
        tok_path = r"C:\Users\karthikkrazy\.cache\huggingface\hub\models--HuggingFaceTB--SmolVLM2-500M-Video-Instruct\snapshots\7b375e1b73b11138ff12fe22c8f2822d8fe03467"
        pol_path = r"C:\Users\karthikkrazy\.cache\huggingface\hub\models--lerobot--smolvla_base\snapshots\c83c3163b8ca9b7e67c509fffd9121e66cb96205"

        self.tokenizer = AutoTokenizer.from_pretrained(tok_path, local_files_only=True)
        self.policy = SmolVLAPolicy.from_pretrained(pol_path, local_files_only=True)

        if os.path.exists(heads_path):
            heads = torch.load(heads_path, map_location=self.device)
            self.policy.model.state_proj.load_state_dict(heads["state_proj"])
            self.policy.model.action_in_proj.load_state_dict(heads["action_in_proj"])
            self.policy.model.action_out_proj.load_state_dict(heads["action_out_proj"])
            self.policy.model.action_time_mlp_in.load_state_dict(heads["action_time_mlp_in"])
            self.policy.model.action_time_mlp_out.load_state_dict(heads["action_time_mlp_out"])
            print("[SmolVLA] Fine-tuned flow heads loaded!")

        self.policy.to(self.device)
        self.policy.eval()

        self.instruction = "approach the lock"
        self.input_ids = None
        self.attention_mask = None
        self._encode_instruction(self.instruction)

        # DAgger Intervention Storage
        self.episodes = []
        if os.path.exists(self.save_path):
            try:
                with open(self.save_path, "rb") as f:
                    self.episodes = pickle.load(f)
                print(f"[Storage] Loaded {len(self.episodes)} existing demonstration episodes.")
            except Exception:
                self.episodes = []

        self.intervening = False
        self.intervention_frames = []
        self.intervention_states = []
        self.intervention_actions = []
        self.training_in_progress = False

    def _encode_instruction(self, text: str):
        self.instruction = text
        self.policy.reset()
        tokens = self.tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            max_length=48,
            truncation=True
        )
        self.input_ids = tokens["input_ids"].to(self.device)
        self.attention_mask = tokens["attention_mask"].bool().to(self.device)

    def _camera_reader_loop(self):
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

    @torch.no_grad()
    def predict_action_chunk(self, rgb_frame: np.ndarray, state_vector: np.ndarray):
        img_resized = cv2.resize(rgb_frame, (256, 256))
        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        img_tensor = img_tensor.to(self.device)

        zero_tensor = torch.zeros_like(img_tensor)
        state_tensor = torch.from_numpy(state_vector.astype(np.float32)).unsqueeze(0).to(self.device)

        batch = {
            "observation.state": state_tensor,
            "observation.images.camera1": img_tensor,
            "observation.images.camera2": zero_tensor,
            "observation.images.camera3": zero_tensor,
            "observation.language.tokens": self.input_ids,
            "observation.language.attention_mask": self.attention_mask,
        }

        proj_dtype = self.policy.model.action_in_proj.weight.dtype
        noise = torch.zeros(
            1, self.policy.config.chunk_size, self.policy.config.max_action_dim,
            device=self.device, dtype=proj_dtype
        )

        action_tensor = self.policy.select_action(batch, noise=noise)
        if isinstance(action_tensor, torch.Tensor):
            act = action_tensor.detach().cpu().numpy().squeeze()
        else:
            act = np.array(action_tensor).squeeze()

        if act.ndim == 1:
            act = np.expand_dims(act, axis=0)
        return act

    def toggle_gripper(self):
        self.gripper_closed = not self.gripper_closed
        print(f"[Gripper] Toggled -> {'CLOSED' if self.gripper_closed else 'OPEN'}")
        if self.connected and self.dobot:
            with self.robot_lock:
                try:
                    self.dobot.grip(self.gripper_closed)
                    self.dobot.suck(self.gripper_closed)
                except Exception:
                    pass

    def trigger_modal_retrain(self):
        """Runs fast 90-second Modal A10G cloud retraining in background thread."""
        if self.training_in_progress:
            print("[Train Warning] Retraining already in progress!")
            return

        def _worker():
            self.training_in_progress = True
            try:
                print("\n" + "=" * 65)
                print("[DAGGER RETRAIN] Triggering fast Modal Cloud GPU retrain on updated dataset...")
                print("=" * 65)
                os.system("modal run modal_train_real_smolvla.py")
                print("\n[DAGGER RETRAIN COMPLETE] New fine-tuned weights ready!\n")
            except Exception as e:
                print(f"[Retrain Error]: {e}")
            finally:
                self.training_in_progress = False

        t = threading.Thread(target=_worker, daemon=False)
        t.start()

    def run_dagger_loop(self, task="approach the lock", ensemble_steps=8):
        self._encode_instruction(task)

        window_title = "Dobot Magician DAgger Controller (Press Arm Button to Intervene)"
        cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_title, 760, 560)

        print("\n" + "=" * 65)
        print("DAGGER INTERACTIVE CONTROL ACTIVE:")
        print("  - Arm executes autonomous SmolVLA policy.")
        print("  - If arm drifts or makes a mistake:")
        print("    --> PRESS TEACH/UNLOCK BUTTON on Dobot arm & guide it by hand!")
        print("    --> Script detects human movement & logs corrective transitions!")
        print("  - Release button when on target -> autonomous policy resumes.")
        print("  - [SPACE]: Toggle Gripper/Suction.")
        print("  - [T]: Retrain on Modal GPU with new corrective data.")
        print("  - [Q / ESC]: Quit.")
        print("=" * 65 + "\n")

        last_commanded_target = [self.cur_x, self.cur_y, self.cur_z]

        try:
            while self.running:
                with self.frame_lock:
                    frame = self.latest_frame.copy() if self.latest_frame is not None else None

                if frame is None:
                    time.sleep(0.01)
                    continue

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Poll physical pose from Dobot encoders
                if self.connected and self.dobot:
                    with self.robot_lock:
                        try:
                            p = self.dobot.pose()
                            self.cur_x, self.cur_y, self.cur_z = p[0], p[1], p[2]
                        except Exception:
                            pass

                # Detect Human Intervention:
                # If physical arm position deviates by > 12mm from the last commanded autonomous waypoint,
                # the human is physically guiding the arm via teach button!
                dist_from_cmd = np.linalg.norm(np.array([self.cur_x, self.cur_y, self.cur_z]) - np.array(last_commanded_target))
                is_human_moving = (dist_from_cmd > 12.0)

                if is_human_moving:
                    if not self.intervening:
                        self.intervening = True
                        self.intervention_frames = []
                        self.intervention_states = []
                        self.intervention_actions = []
                        print("\n>>> [HUMAN INTERVENTION DETECTED] Recording corrective recovery trajectory...")

                    # Record human corrective demonstration step
                    f_small = cv2.resize(frame, (256, 256))
                    _, buf = cv2.imencode(".jpg", f_small, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                    st = self.dobot_to_so100_state(self.cur_x, self.cur_y, self.cur_z, self.gripper_closed)

                    d_pan = float(- self.cur_y / 1.5)
                    d_shoulder = float(119.9 + (self.cur_x - 190.0) * 1.5)
                    d_elbow = float(109.8 + (self.cur_z - 35.0) * 2.0)
                    act = np.array([d_pan, d_shoulder, d_elbow, 56.7, -27.4, 3.0 if self.gripper_closed else 15.0], dtype=np.float32)

                    self.intervention_frames.append(buf.tobytes())
                    self.intervention_states.append(st)
                    self.intervention_actions.append(act)

                    last_commanded_target = [self.cur_x, self.cur_y, self.cur_z]
                    time.sleep(0.08)  # ~12 Hz sampling while human guides
                else:
                    if self.intervening:
                        # Human released the arm; save corrective trajectory
                        self.intervening = False
                        if len(self.intervention_actions) >= 5:
                            ep = {
                                "cam1": self.intervention_frames,
                                "cam2": [b"" for _ in self.intervention_frames],
                                "cam3": [b"" for _ in self.intervention_frames],
                                "states": self.intervention_states,
                                "actions": self.intervention_actions,
                                "instructions": [self.instruction for _ in self.intervention_actions]
                            }
                            self.episodes.append(ep)
                            with open(self.save_path, "wb") as f:
                                pickle.dump(self.episodes, f)
                            print(f"\n>>> [CORRECTION SAVED] Saved {len(self.intervention_actions)} corrective steps to {self.save_path}!")
                            print(f"    Total Dataset Episodes: {len(self.episodes)} (Press [T] to retrain on Modal GPU!)\n")
                        else:
                            print("\n>>> [Correction Discarded] Too brief (<5 steps).\n")

                    # Autonomous Policy Execution: predict action chunk and ensemble 8 steps
                    st = self.dobot_to_so100_state(self.cur_x, self.cur_y, self.cur_z, self.gripper_closed)
                    action_chunk = self.predict_action_chunk(rgb, st)

                    steps_to_exec = min(ensemble_steps, len(action_chunk))
                    for s_idx in range(steps_to_exec):
                        # If human grabs the arm during ensembled execution, immediately yield!
                        if self.connected and self.dobot:
                            with self.robot_lock:
                                try:
                                    p_now = self.dobot.pose()
                                    self.cur_x, self.cur_y, self.cur_z = p_now[0], p_now[1], p_now[2]
                                except Exception:
                                    pass

                        act = action_chunk[s_idx]
                        d_pan = float(act[0])
                        d_shoulder = float(act[1])
                        d_elbow = float(act[2])
                        grip_cmd = float(act[5]) if len(act) > 5 else 15.0

                        dy = float(- d_pan * 1.5)
                        dx = float((d_shoulder - 119.9) * 0.6)
                        dz = float((d_elbow - 109.8) * 0.5)

                        dx = float(np.clip(dx, -12.0, 12.0))
                        dy = float(np.clip(dy, -15.0, 15.0))
                        dz = float(np.clip(dz, -8.0, 8.0))

                        target_x = float(np.clip(self.cur_x + dx, 150.0, 260.0))
                        target_y = float(np.clip(self.cur_y + dy, -120.0, 120.0))
                        target_z = float(np.clip(self.cur_z + dz, -30.0, 75.0))

                        last_commanded_target = [target_x, target_y, target_z]
                        should_grip = (grip_cmd < 8.0)

                        if self.connected and self.dobot:
                            with self.robot_lock:
                                try:
                                    self.dobot.move_to(target_x, target_y, target_z, 0.0, wait=False)
                                    if should_grip != self.gripper_closed:
                                        self.gripper_closed = should_grip
                                        self.dobot.grip(self.gripper_closed)
                                        self.dobot.suck(self.gripper_closed)
                                except Exception:
                                    pass
                        time.sleep(0.05)

                # Render HUD
                disp = frame.copy()
                h, w = disp.shape[:2]
                banner_color = (0, 0, 220) if self.intervening else (0, 180, 0)
                status_txt = f"HUMAN INTERVENTION (Steps: {len(self.intervention_actions)})" if self.intervening else "AUTONOMOUS SMOLVLA"

                cv2.rectangle(disp, (0, 0), (w, 45), (25, 25, 25), -1)
                cv2.putText(disp, f"MODE: {status_txt}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, banner_color, 2)
                cv2.putText(disp, f"Demos: {len(self.episodes)}", (w - 170, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                cv2.putText(disp, f"Goal: '{self.instruction}'", (15, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
                cv2.putText(disp, f"Pose: [{self.cur_x:.1f}, {self.cur_y:.1f}, {self.cur_z:.1f}] mm | Grip: {'CLOSED' if self.gripper_closed else 'OPEN'}",
                            (15, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

                if self.training_in_progress:
                    cv2.rectangle(disp, (10, h - 45), (w - 10, h - 10), (0, 80, 200), -1)
                    cv2.putText(disp, "*** MODAL CLOUD GPU RETRAINING IN PROGRESS... ***", (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                cv2.imshow(window_title, disp)
                key = cv2.waitKey(20) & 0xFF

                if key in [27, ord('q')]:
                    break
                elif key == ord(' '):
                    self.toggle_gripper()
                elif key in [ord('t'), ord('T')]:
                    self.trigger_modal_retrain()

        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            if self.connected and self.dobot:
                try:
                    self.dobot.close()
                except Exception:
                    pass
            print(f"\n[Done] DAgger controller closed. Total episodes in dataset: {len(self.episodes)}.")


if __name__ == "__main__":
    task_goal = "approach the lock"
    if len(sys.argv) > 1:
        task_goal = " ".join(sys.argv[1:])
    controller = DobotDAggerController()
    controller.run_dagger_loop(task=task_goal)
