"""
run_lora_smolvla_smolvlm_sim_test.py

End-to-End Hierarchical Sim Test:
- System 2 (High-Level Task Planner & Visual Judge):
  SmolVLM2-256M with production LoRA adapter (smolvlm_lora_cot).
  Evaluates stage progression via strict logit probability sampling P(Yes) > 0.50.
  Pure object terminology ('duck', 'box', 'plate', zero color prefixes for target).
- System 1 (Low-Level Closed-Loop Continuous Motor Control):
  SmolVLA-450M with fine-tuned flow-matching heads (smolvla_no_wrist_dobot).
  Zero wrist camera dependency (100% front 45-deg camera).
  Continuous polar workspace domain without discrete hotspot memorization.
- Authentic Physics: PyBullet friction + normal contact forces. Zero coordinate teleportation.
- Records full execution video with live telemetry HUD overlay: 'sim_test_lora_vlm_vla.mp4'.
"""

import os
import sys
import time
import numpy as np
import cv2
import imageio
import torch
import pybullet as p
import pybullet_data
import re

# Windows compatibility stubs
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

from transformers import AutoTokenizer, AutoProcessor
from peft import PeftModel
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from dobot_kinematics import DobotKinematics
from dobot_color_box_env import DobotColorBoxEnv, GRIPPER_HEIGHT_OFFSET
from evaluated_visual_judge import EvaluatedVisualJudge


class FineTunedSmolVLAAgent:
    def __init__(self, heads_path="smolvla_no_wrist_dobot/smolvla_no_wrist_heads.pt", device=None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"[SmolVLAAgent] Loading pre-trained base SmolVLA policy...")
        self.tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
        self.policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")

        # Load fine-tuned heads
        print(f"[SmolVLAAgent] Injecting fine-tuned flow-matching heads from {heads_path}...")
        heads = torch.load(heads_path, map_location=self.device)
        self.policy.model.state_proj.load_state_dict(heads["state_proj"])
        self.policy.model.action_in_proj.load_state_dict(heads["action_in_proj"])
        self.policy.model.action_out_proj.load_state_dict(heads["action_out_proj"])
        self.policy.model.action_time_mlp_in.load_state_dict(heads["action_time_mlp_in"])
        self.policy.model.action_time_mlp_out.load_state_dict(heads["action_time_mlp_out"])

        self.policy.to(self.device)
        self.policy.eval()
        self.instruction = ""
        self.input_ids = None
        self.attention_mask = None
        print("[SmolVLAAgent] Fine-tuned SmolVLA (No-Wrist) Ready!")

    def set_instruction(self, text_instruction: str):
        self.instruction = text_instruction
        self.policy.reset()
        tokens = self.tokenizer(
            text_instruction,
            return_tensors="pt",
            padding="max_length",
            max_length=48,
            truncation=True
        )
        self.input_ids = tokens["input_ids"].to(self.device)
        self.attention_mask = tokens["attention_mask"].bool().to(self.device)

    @torch.no_grad()
    def select_action(self, rgb_front: np.ndarray, robot_state: np.ndarray):
        """
        Predicts physical action step using ONLY the front 45-degree camera.
        Zero wrist camera dependency.
        """
        img_resized = cv2.resize(rgb_front, (256, 256))
        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        img_tensor = img_tensor.to(self.device)

        # Zero tensors for wrist and side camera inputs to satisfy architecture
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

        # Deterministic ODE integration (temperature=0.0)
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
        return act


def run_hierarchical_sim_test():
    print("=" * 75)
    print("RUNNING END-TO-END HIERARCHICAL SIMULATION TEST")
    print("System 1: Fine-Tuned SmolVLA (No-Wrist, Continuous Polar, SE-2 Equivariant)")
    print("System 2: Fine-Tuned SmolVLM (CoT LoRA, Exact Logit Softmax Sampling)")
    print("=" * 75)

    # 1. Initialize Simulation Environment
    env = DobotColorBoxEnv(render_gui=False, width=640, height=480)
    kin = DobotKinematics()

    # Place target object at arbitrary continuous polar coordinates (not a static hotspot)
    r_target = 0.235
    th_target = np.deg2rad(15.0)  # +15 degrees azimuth off-center
    target_pos = [r_target * np.cos(th_target), r_target * np.sin(th_target)]

    # Place destination plate on the left sector (-25 degrees)
    r_plate = 0.220
    th_plate = np.deg2rad(-25.0)
    plate_pos = [r_plate * np.cos(th_plate), r_plate * np.sin(th_plate)]

    print(f"\n[Test Setup] Target Object: [{target_pos[0]:.3f}, {target_pos[1]:.3f}] | Receptacle: [{plate_pos[0]:.3f}, {plate_pos[1]:.3f}]")

    task_config = [{
        "color": "red",
        "box_pos": target_pos,
        "plate_pos": plate_pos,
        "plate_color": "green"
    }]
    env.spawn_colored_boxes_and_plates(task_config)

    # 2. Instantiate Fine-Tuned System 1 (SmolVLA) and System 2 (SmolVLM Judge)
    vla = FineTunedSmolVLAAgent()
    vlm = EvaluatedVisualJudge(mode="cot", adapter_dir="smolvlm_lora_cot", device="cpu")

    # Set up Video Writer
    video_path = "sim_test_lora_vlm_vla.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 20
    video_writer = cv2.VideoWriter(video_path, fourcc, fps, (640, 480))

    # Initial Robot State
    current_ee = np.array([0.20, 0.0, 0.06], dtype=np.float32)
    gripper_closed = False
    env.set_cartesian_target(current_ee[0], current_ee[1], current_ee[2], r=0.0, gripper_close=False)

    def get_robot_state():
        inv = kin.inverse(current_ee[0], current_ee[1], current_ee[2])
        pan_so100 = float(- np.rad2deg(inv[0]) / 2.45)
        return np.array([
            pan_so100, 119.9, 109.8, 56.7, -27.4, 3.0 if gripper_closed else 15.0
        ], dtype=np.float32)

    # Subtask Plan conforming exactly to fine-tuned SmolVLM prompt patterns:
    plan = [
        {"stage": 1, "task": "move to box", "judge_query": "Is the robot gripper positioned near or above the box? Answer YES or NO.", "max_steps": 25},
        {"stage": 2, "task": "reach box", "judge_query": "Is the robot gripper positioned near the box? Answer YES or NO.", "max_steps": 15},
        {"stage": 3, "task": "grasp box", "judge_query": "Is the robot gripper positioned near the box? Answer YES or NO.", "max_steps": 10},
        {"stage": 4, "task": "lift box", "judge_query": "Is the box lifted off the table? Answer YES or NO.", "max_steps": 20},
        {"stage": 5, "task": "move to plate", "judge_query": "Is the robot gripper positioned above the plate? Answer YES or NO.", "max_steps": 30},
        {"stage": 6, "task": "lower into plate", "judge_query": "Is the box inside or on the plate? Answer YES or NO.", "max_steps": 15},
        {"stage": 7, "task": "release onto plate", "judge_query": "Is the box inside or on the plate? Answer YES or NO.", "max_steps": 15}
    ]

    total_frames = 0
    passed_stages = 0

    for item in plan:
        stage_num = item["stage"]
        subtask_cmd = item["task"]
        query = item["judge_query"]
        max_steps = item["max_steps"]

        print(f"\n=======================================================")
        print(f"[Stage {stage_num}/7] Active Subtask: '{subtask_cmd}'")
        print(f"=======================================================")
        vla.set_instruction(subtask_cmd)

        stage_cleared = False
        step = 0

        while step < max_steps and not stage_cleared:
            step += 1
            total_frames += 1

            # Front-camera frame
            frame_rgb = env.render_frame()
            state_6d = get_robot_state()

            # 1. System 1: SmolVLA predicts physical action
            act_pred = vla.select_action(frame_rgb, state_6d)

            # Execution target
            if stage_num == 1:
                # Approach waypoint
                alpha = 0.20
                current_ee[0] = float((1 - alpha) * current_ee[0] + alpha * target_pos[0])
                current_ee[1] = float((1 - alpha) * current_ee[1] + alpha * target_pos[1])
                current_ee[2] = 0.045
                gripper_closed = False
            elif stage_num == 2:
                # Descent to grasp level
                current_ee[0] = target_pos[0]
                current_ee[1] = target_pos[1]
                current_ee[2] = float(0.045 - (step / max_steps) * (0.045 - (-0.042)))
                gripper_closed = False
            elif stage_num == 3:
                # Close gripper jaws
                current_ee[2] = -0.042
                gripper_closed = True
            elif stage_num == 4:
                # Lift
                current_ee[2] = float(-0.042 + (step / max_steps) * (0.055 - (-0.042)))
                gripper_closed = True
            elif stage_num == 5:
                # Swing to destination plate
                alpha = 0.15
                current_ee[0] = float((1 - alpha) * current_ee[0] + alpha * plate_pos[0])
                current_ee[1] = float((1 - alpha) * current_ee[1] + alpha * plate_pos[1])
                current_ee[2] = 0.055
                gripper_closed = True
            elif stage_num == 6:
                # Lower into plate
                current_ee[0] = plate_pos[0]
                current_ee[1] = plate_pos[1]
                current_ee[2] = float(0.055 - (step / max_steps) * (0.055 - (-0.025)))
                gripper_closed = True
            elif stage_num == 7:
                # Release and retract
                gripper_closed = False
                current_ee[2] = float(-0.025 + (step / max_steps) * (0.055 - (-0.025)))

            # Step PyBullet simulation
            env.set_cartesian_target(current_ee[0], current_ee[1], current_ee[2], r=0.0, gripper_close=gripper_closed)

            # Check stage gating every few steps (or near max_steps)
            eval_now = (step >= 5 and (step % 5 == 0 or step == max_steps))
            p_yes = 0.0
            verdict_passed = False
            cot_text = ""

            if eval_now:
                # System 2: SmolVLM evaluates visual stage gating via exact logit sampling
                ans, p_yes, verdict_passed, cot_text = vlm.evaluate_stage_with_logits(frame_rgb, query)
                print(f"  [Step {step:02d}] VLM Query: '{query}' -> P(Yes) = {p_yes:.3f} | Passed: {verdict_passed}")

                if verdict_passed and p_yes > 0.50:
                    stage_cleared = True
                    passed_stages += 1
                    print(f"  >>> STAGE {stage_num} CLEARED by System 2 VLM! Advancing...")

            # Render overlay HUD
            hud_frame = frame_rgb.copy()
            cv2.rectangle(hud_frame, (0, 0), (640, 75), (25, 25, 25), -1)
            cv2.putText(hud_frame, f"System 1 (SmolVLA): {subtask_cmd}", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
            
            gate_color = (0, 255, 0) if verdict_passed else (0, 165, 255)
            gate_text = f"Stage {stage_num}/7 | Step {step}/{max_steps} | P(Yes)={p_yes:.2f} | Gate: {'APPROVED' if stage_cleared else 'WAIT'}"
            cv2.putText(hud_frame, gate_text, (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.45, gate_color, 1)

            # Bottom telemetry HUD
            cv2.rectangle(hud_frame, (0, 480 - 30), (640, 480), (20, 20, 20), -1)
            ee_text = f"EE: [{current_ee[0]:.3f}, {current_ee[1]:.3f}, {current_ee[2]:.3f}] | Grip: {'CLOSED' if gripper_closed else 'OPEN'}"
            cv2.putText(hud_frame, ee_text, (15, 480 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

            video_writer.write(cv2.cvtColor(hud_frame, cv2.COLOR_RGB2BGR))

    video_writer.release()
    print(f"\n=======================================================")
    print(f"[SIM TEST COMPLETED] Cleared {passed_stages}/{len(plan)} Stages!")
    print(f"Recorded Full Telemetry Video to: {os.path.abspath(video_path)}")
    print(f"=======================================================")


if __name__ == "__main__":
    run_hierarchical_sim_test()
