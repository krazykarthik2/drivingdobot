"""
run_real_dobot_smolvla.py

End-to-End Real-World Closed-Loop Physical Dobot Magician Control:
- Uses the newly fine-tuned SmolVLA policy (smolvla_no_wrist_dobot/smolvla_no_wrist_heads.pt)
  trained directly on your physical tabletop demonstrations.
- ZERO synthetic visual judges, ZERO heuristic handholding, ZERO discrete bounding boxes.
- Direct closed-loop motor control:
  * Top Camera (Index 1) captures live workspace frames (DirectShow backend).
  * SmolVLA flow-matching predicts continuous 6D action chunks.
  * Real Dobot executes smooth trajectory towards the physical object.
- Safe 115200 baud serial communication with automatic startup alarm clearance.
"""

import os
import sys
import time
import cv2
import numpy as np
import torch

# Compatibility stubs
sys.modules['jax'] = None
sys.modules['jaxlib'] = None
sys.modules['tensorflow'] = None
os.environ['HF_HUB_OFFLINE'] = '1'

import huggingface_hub
if not hasattr(huggingface_hub, 'sync_bucket'):
    huggingface_hub.sync_bucket = None

import transformers.utils
if not hasattr(transformers.utils, 'torch_compilable_check'):
    transformers.utils.torch_compilable_check = lambda: False

import pydobot
from pydobot.message import Message
from dobot_clean import clean_dobot, get_clean_dobot
from transformers import AutoTokenizer
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from dobot_kinematics import DobotKinematics


class RealDobotSmolVLAController:
    def __init__(self, cam_idx=1, robot_port="COM13", heads_path="smolvla_no_wrist_dobot/smolvla_no_wrist_heads.pt"):
        print("=" * 70)
        print("REAL PHYSICAL DOBOT MAGICIAN CLOSED-LOOP SMOLVLA CONTROLLER")
        print("=" * 70)

        self.device = torch.device("cpu")
        # Utilize all CPU cores for maximum inference speed
        try:
            torch.set_num_threads(os.cpu_count() or 8)
        except Exception:
            pass
        self.kin = DobotKinematics()

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

        # 2. Dobot Hardware Connection
        print(f"[2/3] Connecting to Physical Dobot on {robot_port}...")
        self.dobot = None
        self.connected = False
        self.cur_x, self.cur_y, self.cur_z = 190.0, 0.0, 40.0
        self.gripper_closed = False

        self.dobot = get_clean_dobot(port=robot_port, verbose=True)
        if self.dobot:
            try:
                p_init = self.dobot.pose()
                self.cur_x, self.cur_y, self.cur_z = p_init[0], p_init[1], p_init[2]
                self.connected = True
                print(f"[Dobot] Connected successfully! Pose: [{self.cur_x:.1f}, {self.cur_y:.1f}, {self.cur_z:.1f}] mm\n")
            except Exception as e:
                print(f"[Dobot Warning] Could not read pose: {e}")
        else:
            print(f"[Dobot Warning] Could not connect to {robot_port}. Running in DRY-RUN mode.\n")

        # 3. Load Fine-Tuned SmolVLA Policy
        print(f"[3/3] Loading Fine-Tuned SmolVLA Policy from {heads_path}...")
        tok_path = r"C:\Users\karthikkrazy\.cache\huggingface\hub\models--HuggingFaceTB--SmolVLM2-500M-Video-Instruct\snapshots\7b375e1b73b11138ff12fe22c8f2822d8fe03467"
        pol_path = r"C:\Users\karthikkrazy\.cache\huggingface\hub\models--lerobot--smolvla_base\snapshots\c83c3163b8ca9b7e67c509fffd9121e66cb96205"

        self.tokenizer = AutoTokenizer.from_pretrained(tok_path, local_files_only=True)
        self.policy = SmolVLAPolicy.from_pretrained(pol_path, local_files_only=True)

        # Acceleration: Optimize Euler flow-matching integration steps (10 -> 2 steps)
        # Optimal Transport flow-matching follows straight trajectories; 2 steps yield 5x speedup
        self.policy.config.num_steps = 2

        if os.path.exists(heads_path):
            print(f"[SmolVLA] Injecting real-demonstration trained flow-matching heads from {heads_path}...")
            heads = torch.load(heads_path, map_location=self.device)
            self.policy.model.state_proj.load_state_dict(heads["state_proj"])
            self.policy.model.action_in_proj.load_state_dict(heads["action_in_proj"])
            self.policy.model.action_out_proj.load_state_dict(heads["action_out_proj"])
            self.policy.model.action_time_mlp_in.load_state_dict(heads["action_time_mlp_in"])
            self.policy.model.action_time_mlp_out.load_state_dict(heads["action_time_mlp_out"])
            print("[SmolVLA] Real-world trained heads loaded successfully!")

        self.policy.to(self.device)
        self.policy.eval()

        self.instruction = ""
        self.input_ids = None
        self.attention_mask = None

    def set_task(self, instruction_text: str):
        self.instruction = instruction_text
        self.policy.reset()
        tokens = self.tokenizer(
            instruction_text,
            return_tensors="pt",
            padding="max_length",
            max_length=48,
            truncation=True
        )
        self.input_ids = tokens["input_ids"].to(self.device)
        self.attention_mask = tokens["attention_mask"].bool().to(self.device)
        print(f"\n[Task Active]: '{self.instruction}'")

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
        """
        Predicts full continuous action chunk of 50 future steps from SmolVLA flow-matching.
        Returns shape (chunk_size, 6).
        """
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

    def run_closed_loop(self, task="approach the lock", ensemble_steps=1):
        """
        Closed-loop execution with Direct Single-Step Action Execution (Temporal Ensembling Disabled):
        Executes immediate single action step per visual observation frame.
        """
        self.set_task(task)

        window_title = "Real Dobot SmolVLA Closed-Loop Controller"
        cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_title, 720, 540)

        total_steps = 0
        cycle = 0
        print(f"\nStarting Continuous SmolVLA Closed-Loop Control! (Horizon={ensemble_steps} steps/cycle)")
        print("Runs continuously. Press [Q] or [ESC] in window anytime to stop.\n")

        try:
            while True:
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    time.sleep(0.01)
                    continue

                cycle += 1
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Query latest pose
                if self.connected and self.dobot:
                    try:
                        p = self.dobot.pose()
                        self.cur_x, self.cur_y, self.cur_z = p[0], p[1], p[2]
                    except Exception:
                        pass

                st = self.dobot_to_so100_state(self.cur_x, self.cur_y, self.cur_z, self.gripper_closed)

                # Predict full action chunk (50 steps)
                t0 = time.time()
                action_chunk = self.predict_action_chunk(rgb, st)
                dt = time.time() - t0

                # Temporal Ensembling: Execute the first `ensemble_steps` smoothly
                steps_to_exec = min(ensemble_steps, len(action_chunk))
                print(f"[Cycle {cycle:03d}] Predicted 50-step flow chunk (in {dt:.2f}s). Ensembling {steps_to_exec} steps:")

                for s_idx in range(steps_to_exec):
                    total_steps += 1
                    act = action_chunk[s_idx]

                    # Pure Cartesian Relative Delta Actions (dx, dy, dz, grip)
                    # Apply flow-velocity boost gain (2.5x) so continuous flow matches physical travel
                    gain = 2.5
                    dx = float(act[0]) * gain
                    dy = float(act[1]) * gain
                    dz = float(act[2]) * gain
                    grip_cmd = float(act[5]) if len(act) > 5 else 15.0

                    # Safe single-step velocity limits (mm per step at 10 Hz)
                    dx = float(np.clip(dx, -25.0, 25.0))
                    dy = float(np.clip(dy, -25.0, 25.0))
                    dz = float(np.clip(dz, -15.0, 15.0))

                    # Discrepancy & Jerk Detection (Human intervention / external bump):
                    # Check if actual physical pose deviates excessively (> 35mm) from expected trajectory
                    if self.connected and self.dobot:
                        try:
                            p_now = self.dobot.pose()
                            disp_dev = np.linalg.norm([p_now[0] - self.cur_x, p_now[1] - self.cur_y, p_now[2] - self.cur_z])
                            if disp_dev > 35.0:
                                # External jerk or human grabbed the arm -> abort stale ensembled sub-plan immediately!
                                print(f"    [Human Intervention/Bump] Displacement {disp_dev:.1f} mm > 35 mm! Recalculating next visual frame...")
                                self.cur_x, self.cur_y, self.cur_z = p_now[0], p_now[1], p_now[2]
                                break
                            self.cur_x, self.cur_y, self.cur_z = p_now[0], p_now[1], p_now[2]
                        except Exception:
                            pass

                    target_x = float(np.clip(self.cur_x + dx, 150.0, 260.0))
                    target_y = float(np.clip(self.cur_y + dy, -120.0, 120.0))
                    target_z = float(np.clip(self.cur_z + dz, -30.0, 75.0))

                    should_grip = (grip_cmd < 8.0)

                    print(f"  Step {s_idx + 1}/{steps_to_exec}: Delta=[dX={dx:+.1f}, dY={dy:+.1f}, dZ={dz:+.1f} mm] -> Target: [{target_x:.1f}, {target_y:.1f}, {target_z:.1f}] | Grip={should_grip}")

                    if self.connected and self.dobot:
                        try:
                            self.dobot.move_to(target_x, target_y, target_z, 0.0, wait=False)
                            if should_grip != self.gripper_closed:
                                self.gripper_closed = should_grip
                                self.dobot.grip(self.gripper_closed)
                                self.dobot.suck(self.gripper_closed)
                            # Update tracked pose to commanded
                            self.cur_x, self.cur_y, self.cur_z = target_x, target_y, target_z
                        except Exception as e:
                            print(f"    [Move Error]: {e}")
                    else:
                        self.cur_x, self.cur_y, self.cur_z = target_x, target_y, target_z

                    # Smooth 10 Hz temporal progression (matched with 10 Hz demonstration rate)
                    time.sleep(0.10)

                # Telemetry HUD
                disp = frame.copy()
                cv2.rectangle(disp, (0, 0), (640, 45), (20, 20, 20), -1)
                cv2.putText(disp, f"SmolVLA Goal: '{self.instruction}'", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(disp, f"Cycle {cycle} (Steps: {total_steps})", (430, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
                cv2.putText(disp, f"Pose: [{self.cur_x:.1f}, {self.cur_y:.1f}, {self.cur_z:.1f}] mm | Grip: {'CLOSED' if self.gripper_closed else 'OPEN'} | [C] Clean",
                            (15, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

                cv2.imshow(window_title, disp)
                key = cv2.waitKey(20) & 0xFF
                if key in [ord('c'), ord('C')]:
                    print("\n[User Command] 'C' pressed -> Running instant Dobot alarm & queue cleanup...")
                    if self.connected and self.dobot:
                        clean_dobot(self.dobot)
                        p_now = self.dobot.pose()
                        self.cur_x, self.cur_y, self.cur_z = p_now[0], p_now[1], p_now[2]
                        print(f"[Dobot Cleaned] Updated pose: [{self.cur_x:.1f}, {self.cur_y:.1f}, {self.cur_z:.1f}]\n")
                elif key in [27, ord('q')]:
                    break

        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            if self.connected and self.dobot:
                try:
                    self.dobot.close()
                except Exception:
                    pass
            print(f"\n[Done] Closed-loop execution completed ({total_steps} smooth steps executed).")


if __name__ == "__main__":
    task_goal = "approach the lock"
    if len(sys.argv) > 1:
        task_goal = " ".join(sys.argv[1:])
    controller = RealDobotSmolVLAController()
    controller.run_closed_loop(task=task_goal)
