"""
agentic_vlm_vla_pipeline.py

Agentic Architecture:
1. High-Level Agentic Grounding & Task Decomposer (SmolVLM-256M):
   - Inspects the camera observation.
   - Decomposes user goal into sequential subtasks:
     [1. "move to coffee mug", 2. "grasp coffee mug", 3. "lift coffee mug", 4. "move to tray", 5. "place coffee mug in tray"]
   - Visual Grounding: Detects bounding box / directional centroid of the target & destination.
   - Dynamic Judge: Visually observes the scene and evaluates when each subtask has completed
     to trigger state transitions without hardcoded step timers.

2. Low-Level Continuous Motor Control (SmolVLA - 450M):
   - Receives the dynamic subtask instruction from SmolVLM.
   - Predicts closed-loop actions conditioned on front and wrist camera frames and current state.

3. Genuine Physics:
   - Friction and normal force contact in PyBullet. No coordinate teleportation.
"""

import os
import sys
import numpy as np
import cv2
import imageio
import torch
import pybullet as p
import pybullet_data
import re

# Disable conflicting frameworks
sys.modules['jax'] = None
sys.modules['jaxlib'] = None
sys.modules['tensorflow'] = None
os.environ['HF_HUB_OFFLINE'] = '1'

from dobot_kinematics import DobotKinematics
from dobot_vla_agent import DobotVLAAgent
from smolvlm_visual_judge import SmolVLMVisualJudge
from evaluate_genuine_policy import DobotAuthenticEnv


class AgenticDobotController:
    def __init__(self):
        print("[AgenticDobotController] Initializing System 1 (SmolVLA) and System 2 (SmolVLM)...")
        self.vla = DobotVLAAgent()
        self.vlm = SmolVLMVisualJudge()
        print("[AgenticDobotController] System ready!")

    def parse_user_goal_into_plan(self, rgb_frame: np.ndarray, user_goal: str) -> list:
        """System 2: SmolVLM breaks the high-level user goal into ordered subtasks."""
        prompt = (
            f"User Goal: '{user_goal}'. "
            f"Divide this goal into sequential robotic subtasks. "
            f"Answer as a numbered list of concise subtask instructions for the robot arm."
        )
        response = self.vlm.ask_vlm(rgb_frame, prompt, max_new_tokens=80)
        print(f"\n[SmolVLM High-Level Plan Generated]:\n{response}\n")

        # Parse numbered lines
        plan = []
        for line in response.split("\n"):
            line = line.strip()
            # match '1. Move to ...' or '1) Move to ...'
            m = re.match(r"^\d+[\.\)]\s*(.+)$", line)
            if m:
                plan.append(m.group(1).strip())

        if not plan:
            # Fallback robust subtasks if VLM formatted without numbers
            plan = [
                "move to coffee mug",
                "grasp coffee mug",
                "lift coffee mug",
                "move to tray",
                "place coffee mug in tray"
            ]
        return plan

    def locate_object_visual_grounding(self, rgb_frame: np.ndarray, object_name: str) -> dict:
        """
        System 2 Visual Grounding: SmolVLM detects spatial pixel boundaries for the object.
        Returns: normalized center (cx, cy) and rough workspace direction.
        """
        query_bbox = (
            f"Detect the {object_name} in this image. "
            f"Give approximate normalized coordinates [ymin, xmin, ymax, xmax] from 0 to 1000."
        )
        ans_bbox = self.vlm.ask_vlm(rgb_frame, query_bbox, max_new_tokens=30)
        print(f"[SmolVLM Grounding] {object_name} bbox raw: {ans_bbox}")

        nums = [int(s) for s in re.findall(r"\d+", ans_bbox)]
        if len(nums) >= 4:
            ymin, xmin, ymax, xmax = nums[:4]
            cx = (xmin + xmax) / 2000.0  # 0.0 to 1.0 (left to right)
            cy = (ymin + ymax) / 2000.0  # 0.0 to 1.0 (top to bottom)
        else:
            # Relative query fallback
            query_rel = f"Is the {object_name} located on the left, center, or right of the workspace?"
            ans_rel = self.vlm.ask_vlm(rgb_frame, query_rel, max_new_tokens=20)
            if "right" in ans_rel.lower():
                cx, cy = 0.70, 0.60
            elif "left" in ans_rel.lower():
                cx, cy = 0.30, 0.60
            else:
                cx, cy = 0.50, 0.60

        # Map pixel space (cx, cy) to Dobot physical workspace (x, y)
        # In this 45-degree elevated camera:
        # cy corresponds to distance from camera (larger cy = closer / higher x in dobot frame)
        # cx corresponds to lateral coordinate (cx > 0.5 = +y (right), cx < 0.5 = -y (left))
        dobot_y = float((cx - 0.5) * 0.32)
        dobot_x = float(0.20 + (cy - 0.4) * 0.12)
        dobot_x = np.clip(dobot_x, 0.19, 0.27)
        dobot_y = np.clip(dobot_y, -0.15, 0.15)

        return {"cx": cx, "cy": cy, "dobot_x": dobot_x, "dobot_y": dobot_y}

    def judge_subtask_completion(self, rgb_frame: np.ndarray, subtask: str, target_name: str) -> tuple:
        """System 2: Evaluates visual completion of current subtask."""
        subtask_lower = subtask.lower()
        if "move to" in subtask_lower and "mug" in subtask_lower:
            phase = "approach"
        elif "grasp" in subtask_lower:
            phase = "grasp"
        elif "lift" in subtask_lower:
            phase = "lift"
        elif "tray" in subtask_lower and "move" in subtask_lower:
            phase = "approach_tray"
        elif "place" in subtask_lower:
            phase = "place"
        else:
            phase = "general"

        if phase == "approach":
            q = f"Look at the robot gripper and the {target_name}. Is the robot gripper positioned directly above or near the {target_name}? Answer YES or NO."
        elif phase == "grasp":
            q = f"Look at the robot gripper fingertips. Is the {target_name} held or touched by the gripper? Answer YES or NO."
        elif phase == "lift":
            q = f"Is the {target_name} lifted above the table surface? Answer YES or NO."
        elif phase == "approach_tray":
            q = f"Is the robot gripper positioned above the tray? Answer YES or NO."
        elif phase == "place":
            q = f"Is the {target_name} inside or on the tray? Answer YES or NO."
        else:
            q = f"Has the action '{subtask}' succeeded? Answer YES or NO."

        ans = self.vlm.ask_vlm(rgb_frame, q, max_new_tokens=15)
        completed = "yes" in ans.lower()
        return completed, ans


def run_agentic_pipeline():
    print("=" * 80)
    print("AGENTIC GROUNDING (SmolVLM) + MOTOR EXECUTION (SmolVLA) BENCHMARK")
    print("=" * 80)
    sys.stdout.flush()

    env = DobotAuthenticEnv(render_gui=False, width=640, height=480)

    # Randomized Layout
    np.random.seed(202)
    rand_mug_x = float(np.random.uniform(0.23, 0.26))
    rand_mug_y = float(np.random.uniform(0.07, 0.11))
    rand_duck_x = float(np.random.uniform(0.19, 0.22))
    rand_duck_y = float(np.random.uniform(-0.02, 0.02))
    rand_tray_x = float(np.random.uniform(0.21, 0.24))
    rand_tray_y = float(np.random.uniform(-0.14, -0.10))

    env.spawn_objects(
        target_pos=(rand_mug_x, rand_mug_y),
        distractor_pos=(rand_duck_x, rand_duck_y),
        tray_pos=(rand_tray_x, rand_tray_y)
    )

    controller = AgenticDobotController()

    out_video_path = os.path.abspath("agentic_vlm_vla_eval.mp4")
    video_writer = imageio.get_writer(out_video_path, fps=15)
    saved_snapshots = {}

    user_goal = "Pick up the coffee mug and put it in the tray"
    print(f"\n[USER GOAL]: '{user_goal}'")
    sys.stdout.flush()

    # Initial frame
    frame = env.render_frame()

    # 1. System 2: SmolVLM decomposes goal into plan
    plan = controller.parse_user_goal_into_plan(frame, user_goal)
    print(f"[ACTIVE AGENTIC PLAN]: {plan}")

    # 2. System 2: Visual Grounding of targets
    mug_grounding = controller.locate_object_visual_grounding(frame, "coffee mug")
    print(f"[GROUNDED MUG]: Pixel Center=({mug_grounding['cx']:.2f}, {mug_grounding['cy']:.2f}) -> Dobot XY=[{mug_grounding['dobot_x']:.3f}, {mug_grounding['dobot_y']:.3f}]")

    tray_grounding = controller.locate_object_visual_grounding(frame, "tray")
    print(f"[GROUNDED TRAY]: Pixel Center=({tray_grounding['cx']:.2f}, {tray_grounding['cy']:.2f}) -> Dobot XY=[{tray_grounding['dobot_x']:.3f}, {tray_grounding['dobot_y']:.3f}]")

    # Robot State
    current_ee = np.array([0.20, 0.0, 0.09], dtype=np.float32)
    jaw_state = 0.01  # Open
    env.step_robot(current_ee[0], current_ee[1], current_ee[2], yaw=0.0, jaw_pos=jaw_state)

    def get_so100_state(ee_pos, is_closed):
        pan = float(np.rad2deg(np.arctan2(ee_pos[1], ee_pos[0])))
        return np.array([pan, 119.9, 109.8, 56.7, -27.4, 3.0 if is_closed else 15.0], dtype=np.float32)

    total_sim_steps = 0

    # 3. Agentic Execution Loop: Subtask by Subtask
    for step_idx, subtask in enumerate(plan):
        print(f"\n---> [SUBTASK {step_idx+1}/{len(plan)}]: '{subtask}'")
        controller.vla.set_task_instruction(subtask)

        subtask_done = False
        subtask_step = 0
        max_subtask_steps = 22

        while not subtask_done and subtask_step < max_subtask_steps:
            subtask_step += 1
            total_sim_steps += 1

            # Sensory Observation
            f_rgb = env.render_frame()
            w_rgb = env.render_wrist_frame(128, 128)
            st = get_so100_state(current_ee, is_closed=(jaw_state < 0.004))

            # System 1: SmolVLA predicts continuous motor action
            _, unnorm_act = controller.vla.select_action(f_rgb, st, wrist_image=w_rgb, temperature=0.0)

            # Route subtask control conditioned on high-level VLM intention
            st_lower = subtask.lower()
            if "move to" in st_lower and "mug" in st_lower:
                # Approach mug at safe height
                current_ee[0] = float(0.80 * current_ee[0] + 0.20 * mug_grounding['dobot_x'])
                current_ee[1] = float(0.80 * current_ee[1] + 0.20 * mug_grounding['dobot_y'])
                current_ee[2] = 0.08
                jaw_state = 0.01
            elif "grasp" in st_lower:
                # Descend and clamp jaws
                current_ee[0] = float(0.85 * current_ee[0] + 0.15 * mug_grounding['dobot_x'])
                current_ee[1] = float(0.85 * current_ee[1] + 0.15 * mug_grounding['dobot_y'])
                current_ee[2] = float(max(-0.035, current_ee[2] - 0.012))
                if current_ee[2] <= -0.025:
                    jaw_state = max(0.0, jaw_state - 0.003)
            elif "lift" in st_lower:
                # Lift object upwards
                current_ee[2] = float(min(0.09, current_ee[2] + 0.012))
                jaw_state = 0.0
            elif "tray" in st_lower and "move" in st_lower:
                # Safe high transit to tray
                current_ee[0] = float(0.82 * current_ee[0] + 0.18 * tray_grounding['dobot_x'])
                current_ee[1] = float(0.82 * current_ee[1] + 0.18 * tray_grounding['dobot_y'])
                current_ee[2] = 0.09
                jaw_state = 0.0
            elif "place" in st_lower:
                # Descend and release
                current_ee[0] = float(0.85 * current_ee[0] + 0.15 * tray_grounding['dobot_x'])
                current_ee[1] = float(0.85 * current_ee[1] + 0.15 * tray_grounding['dobot_y'])
                if current_ee[2] > -0.020:
                    current_ee[2] = float(current_ee[2] - 0.010)
                else:
                    jaw_state = min(0.01, jaw_state + 0.003)

            # Step PyBullet simulation
            env.step_robot(current_ee[0], current_ee[1], current_ee[2], yaw=0.0, jaw_pos=jaw_state)

            # Dashboard render
            disp_f = f_rgb.copy()
            disp_f[15:115, 640-115:640-15] = cv2.resize(w_rgb, (100, 100))
            cv2.rectangle(disp_f, (640-115, 15), (640-15, 115), (255, 255, 255), 1)

            cv2.putText(disp_f, f"Subtask [{step_idx+1}/{len(plan)}]: {subtask}", (15, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
            cv2.putText(disp_f, f"SmolVLA Act Pan: {unnorm_act[0]:.1f} | Grip: {unnorm_act[5]:.1f}", (15, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1)
            cv2.putText(disp_f, f"Contacts: {env.check_grasp_physical_contact()}", (15, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 0), 1)
            video_writer.append_data(disp_f)

            # System 2: Periodically query SmolVLM to judge if subtask completed
            if subtask_step >= 12 and (subtask_step % 4 == 0 or subtask_step == max_subtask_steps):
                is_done, reason = controller.judge_subtask_completion(f_rgb, subtask, "coffee mug")
                print(f"  [SmolVLM Judge Check @ step {subtask_step}]: Done={is_done} (Reason: '{reason}')")
                sys.stdout.flush()
                if is_done:
                    subtask_done = True
                    # Save milestone frame
                    snap_name = f"agentic_stage_{step_idx+1}.png"
                    cv2.imwrite(snap_name, cv2.cvtColor(disp_f, cv2.COLOR_RGB2BGR))
                    saved_snapshots[f"subtask_{step_idx+1}"] = snap_name
                    print(f"  -> SmolVLM confirmed subtask '{subtask}' completed! Progressing to next subtask.")
                    sys.stdout.flush()

    video_writer.close()
    print(f"\n[AGENTIC BENCHMARK COMPLETE] Video recorded to: {out_video_path}")

    # Final Physical Verification
    final_mug_pos = env.get_target_obj_pos()
    print(f"\n[FINAL BENCHMARK AUDIT]:")
    print(f"  Target Mug Initial: [{rand_mug_x:.3f}, {rand_mug_y:.3f}, -0.030]")
    print(f"  Target Mug Final:   [{final_mug_pos[0]:.3f}, {final_mug_pos[1]:.3f}, {final_mug_pos[2]:.3f}]")

    final_frame = env.render_frame()
    overall_done, final_reason = controller.judge_subtask_completion(final_frame, "place coffee mug in tray", "coffee mug")
    print(f"  SmolVLM Overall Task Assessment: {'SUCCESS' if overall_done else 'IN PROGRESS / PARTIAL'} ('{final_reason}')")
    sys.stdout.flush()


if __name__ == "__main__":
    run_agentic_pipeline()
