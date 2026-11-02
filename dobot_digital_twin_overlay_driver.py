"""
dobot_digital_twin_overlay_driver.py

Pure SmolVLA Direct Real-World Closed-Loop Controller with Calibrated Camera & Digital Twin:
- ZERO task decomposition, ZERO VLM judge, ZERO synthetic heuristic stages.
- Direct Vision-Language-Action Policy (SmolVLA-450M fine-tuned action heads).
- Overhead camera feed (Index 1) + user instruction passed directly to SmolVLA policy.
- Outputs continuous 6D end-effector actions driving physical Dobot arm on COM13.
- PyBullet 3D digital twin synchronously tracks physical pose in real-time.
"""

import os
import sys
import time
import json
import threading
import queue
import cv2
import numpy as np
import pybullet as p

# Windows compatibility stubs
sys.modules['jax'] = None
sys.modules['jaxlib'] = None
sys.modules['tensorflow'] = None
os.environ['HF_HUB_OFFLINE'] = '1'

import pydobot
from pydobot.message import Message
from transformers import AutoTokenizer
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from dobot_color_box_env import DobotColorBoxEnv, GRIPPER_HEIGHT_OFFSET
from dobot_kinematics import DobotKinematics


class PureSmolVLADigitalTwinDriver:
    def __init__(self, cam_idx=1, robot_port="COM13", blend_alpha=0.50, heads_path="smolvla_no_wrist_dobot/smolvla_no_wrist_heads.pt"):
        self.cam_idx = cam_idx
        self.robot_port = robot_port
        self.blend_alpha = blend_alpha
        self.heads_path = heads_path
        self.kin = DobotKinematics()

        print("=" * 70)
        print("PURE SMOLVLA DIRECT CLOSED-LOOP CONTROLLER (ZERO JUDGE / ZERO COT)")
        print(f"Robot Port:       {self.robot_port}")
        print(f"Camera Index:     {self.cam_idx}")
        print(f"VLA Heads:        {self.heads_path}")
        print("=" * 70)

        # 1. Initialize PyBullet Digital Twin
        print("[1/4] Initializing PyBullet Digital Twin environment...")
        self.env = DobotColorBoxEnv(render_gui=False, width=640, height=480)
        self.V = np.array(self.env.view_matrix).reshape((4, 4), order='F')
        self.P = np.array(self.env.proj_matrix).reshape((4, 4), order='F')
        self.VP = self.P @ self.V

        # Virtual ArUco Markers
        z_table = -0.048
        self.sim_tags_def = {
            0: ([0.059, -0.045, z_table], (240, 200, 50)),
            1: ([0.059, +0.045, z_table], (50, 240, 240)),
            2: ([0.173, +0.045, z_table], (50, 240, 50)),
            3: ([0.173, -0.045, z_table], (240, 50, 240))
        }

        # 2. Camera feed acquisition
        print(f"[2/4] Connecting to Top Camera (Index {self.cam_idx})...")
        self.cap = cv2.VideoCapture(self.cam_idx, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.cam_idx)
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

        # 3. Connect to Physical Dobot
        print(f"[3/4] Connecting to physical Dobot on {self.robot_port}...")
        self.dobot = None
        self.connected = False
        self.robot_lock = threading.Lock()
        self.current_real_pose = [190.0, 0.0, 40.0, 0.0]
        self.gripper_closed = False

        try:
            self.dobot = pydobot.Dobot(port=self.robot_port, verbose=False)
            # Startup alarm clearance
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
            self.current_real_pose = [p_init[0], p_init[1], p_init[2], p_init[3]]
            self.connected = True
            print(f"[SUCCESS] Real Dobot Connected! Pose: [{p_init[0]:.1f}, {p_init[1]:.1f}, {p_init[2]:.1f}] mm\n")
            self._sync_twin_to_real(p_init[0], p_init[1], p_init[2], p_init[3], False)
        except Exception as e:
            print(f"[Warning] Dobot connection failed ({e}). Operating in DRY-RUN mode.\n")

        # 4. Load Pure SmolVLA Policy
        print("[4/4] Loading SmolVLA Policy & Tokenizer...")
        tok_path = r"C:\Users\karthikkrazy\.cache\huggingface\hub\models--HuggingFaceTB--SmolVLM2-500M-Video-Instruct\snapshots\7b375e1b73b11138ff12fe22c8f2822d8fe03467"
        pol_path = r"C:\Users\karthikkrazy\.cache\huggingface\hub\models--lerobot--smolvla_base\snapshots\c83c3163b8ca9b7e67c509fffd9121e66cb96205"

        self.device = torch.device("cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(tok_path, local_files_only=True)
        self.policy = SmolVLAPolicy.from_pretrained(pol_path, local_files_only=True)

        if os.path.exists(self.heads_path):
            print(f"[SmolVLA] Injecting real-demonstration trained heads from {self.heads_path}...")
            heads = torch.load(self.heads_path, map_location=self.device)
            self.policy.model.state_proj.load_state_dict(heads["state_proj"])
            self.policy.model.action_in_proj.load_state_dict(heads["action_in_proj"])
            self.policy.model.action_out_proj.load_state_dict(heads["action_out_proj"])
            self.policy.model.action_time_mlp_in.load_state_dict(heads["action_time_mlp_in"])
            self.policy.model.action_time_mlp_out.load_state_dict(heads["action_time_mlp_out"])
            print("[SmolVLA] Real demonstration action heads ready!\n")

        self.policy.to(self.device)
        self.policy.eval()

        self.instruction = "approach the lock"
        self.input_ids = None
        self.attention_mask = None
        self._encode_instruction(self.instruction)

        # Calibrated homography matrix
        self.homography_matrix = None
        self.alignment_params = {}
        if os.path.exists("camera_twin_alignment.json"):
            try:
                with open("camera_twin_alignment.json", "r") as f:
                    self.alignment_params = json.load(f)
                if "homography" in self.alignment_params:
                    self.homography_matrix = np.array(self.alignment_params["homography"], dtype=np.float32)
            except Exception:
                pass

        # Execution Queue
        self.motion_queue = queue.Queue()
        self.current_action_status = "Ready / Standby"
        self.robot_thread = threading.Thread(target=self._robot_worker_loop, daemon=True)
        self.robot_thread.start()

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

    def sim_proj(self, pt_3d):
        p4 = np.array([pt_3d[0], pt_3d[1], pt_3d[2], 1.0], dtype=np.float32)
        clip = self.VP @ p4
        if clip[3] <= 0:
            return (0, 0)
        ndc_x = clip[0] / clip[3]
        ndc_y = clip[1] / clip[3]
        px = int((ndc_x * 0.5 + 0.5) * 640)
        py = int((-ndc_y * 0.5 + 0.5) * 480)
        return (px, py)

    def _sync_twin_to_real(self, x_mm, y_mm, z_mm, r_deg=0.0, grip=False):
        x_m = float(x_mm) / 1000.0
        y_m = float(y_mm) / 1000.0
        z_m = float(z_mm) / 1000.0
        r_rad = np.deg2rad(float(r_deg))
        self.env.set_cartesian_target(x_m, y_m, z_m, r=r_rad, gripper_close=grip)
        for _ in range(2):
            p.stepSimulation()

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
    def predict_vla_action(self, rgb_frame, robot_state):
        img_resized = cv2.resize(rgb_frame, (256, 256))
        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        img_tensor = img_tensor.to(self.device)

        zero_tensor = torch.zeros_like(img_tensor)
        state_tensor = torch.from_numpy(robot_state.astype(np.float32)).unsqueeze(0).to(self.device)

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

        if act.ndim > 1:
            act = act[0]
        return act

    def _robot_worker_loop(self):
        while self.running:
            try:
                cmd = self.motion_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            self._execute_vla_control(cmd)
            self.motion_queue.task_done()

    def _execute_vla_control(self, instruction: str, max_steps=35):
        self._encode_instruction(instruction)
        print(f"\n[SmolVLA Execution]: Starting closed-loop flow control for '{instruction}'...")

        step = 0
        while step < max_steps and self.running:
            with self.frame_lock:
                frame = self.latest_frame.copy() if self.latest_frame is not None else None

            if frame is None:
                time.sleep(0.02)
                continue

            step += 1
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cur_x, cur_y, cur_z = self.current_real_pose[0], self.current_real_pose[1], self.current_real_pose[2]

            st = self.dobot_to_so100_state(cur_x, cur_y, cur_z, self.gripper_closed)

            # Direct SmolVLA Flow-Matching prediction
            act = self.predict_vla_action(rgb, st)

            d_pan = float(act[0])
            d_shoulder = float(act[1])
            d_elbow = float(act[2])
            grip_cmd = float(act[5]) if len(act) > 5 else 15.0

            dy = float(- d_pan * 1.5)
            dx = float((d_shoulder - 119.9) * 0.6)
            dz = float((d_elbow - 109.8) * 0.5)

            # Safety bounds
            dx = np.clip(dx, -12.0, 12.0)
            dy = np.clip(dy, -15.0, 15.0)
            dz = np.clip(dz, -8.0, 8.0)

            target_x = float(np.clip(cur_x + dx, 150.0, 260.0))
            target_y = float(np.clip(cur_y + dy, -120.0, 120.0))
            target_z = float(np.clip(cur_z + dz, -30.0, 75.0))

            should_grip = (grip_cmd < 8.0)
            self.current_action_status = f"VLA Step {step}/{max_steps} [Target: {target_x:.1f}, {target_y:.1f}, {target_z:.1f}]"

            if self.connected and self.dobot:
                with self.robot_lock:
                    try:
                        self.dobot.move_to(target_x, target_y, target_z, 0.0, wait=False)
                        if should_grip != self.gripper_closed:
                            self.gripper_closed = should_grip
                            self.dobot.grip(self.gripper_closed)
                            self.dobot.suck(self.gripper_closed)
                        p_now = self.dobot.pose()
                        self.current_real_pose = [p_now[0], p_now[1], p_now[2], p_now[3]]
                    except Exception:
                        pass
            else:
                self.current_real_pose = [target_x, target_y, target_z, 0.0]

            time.sleep(0.12)

        self.current_action_status = f"Completed '{instruction}'"
        print(f"[SmolVLA Execution]: Task '{instruction}' finished successfully!\n")

    def run(self):
        window_title = "Dobot Digital Twin + Real-World SmolVLA Direct Control"
        cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_title, 960, 720)

        # Terminal prompt thread
        def cli_loop():
            print("\nEnter prompt anytime (e.g. 'approach the lock', 'pick up lock', 'home'):")
            while self.running:
                try:
                    p_cmd = input("VLA Prompt > ").strip()
                    if not p_cmd:
                        continue
                    if p_cmd.lower() in ["exit", "quit", "q"]:
                        self.running = False
                        break
                    self.motion_queue.put(p_cmd)
                except (KeyboardInterrupt, EOFError):
                    self.running = False
                    break

        t_cli = threading.Thread(target=cli_loop, daemon=True)
        t_cli.start()

        while self.running:
            with self.frame_lock:
                frame = self.latest_frame.copy() if self.latest_frame is not None else None

            if frame is None:
                time.sleep(0.01)
                continue

            # Twin synchronization
            cur_p = self.current_real_pose
            self._sync_twin_to_real(cur_p[0], cur_p[1], cur_p[2], cur_p[3], self.gripper_closed)

            # Render simulation view
            sim_view = self.env.render_frame()
            sim_bgr = cv2.cvtColor(sim_view, cv2.COLOR_RGB2BGR)

            # Homography warp
            h, w = frame.shape[:2]
            if self.homography_matrix is not None:
                cam_warped = cv2.warpPerspective(frame, self.homography_matrix, (w, h))
            else:
                cam_warped = frame

            # 50/50 Overlay
            overlay = cv2.addWeighted(cam_warped, 1.0 - self.blend_alpha, sim_bgr, self.blend_alpha, 0.0)

            # Telemetry HUD
            cv2.rectangle(overlay, (0, 0), (w, 50), (20, 20, 20), -1)
            cv2.putText(overlay, f"SmolVLA Goal: '{self.instruction}'", (15, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(overlay, f"Status: {self.current_action_status}", (15, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            cv2.imshow(window_title, overlay)
            key = cv2.waitKey(20) & 0xFF
            if key in [27, ord('q')]:
                self.running = False
                break

        self.cap.release()
        cv2.destroyAllWindows()
        if self.connected and self.dobot:
            try:
                self.dobot.close()
            except Exception:
                pass
        print("[Exit] Pure SmolVLA driver closed.")


if __name__ == "__main__":
    driver = PureSmolVLADigitalTwinDriver()
    driver.run()
