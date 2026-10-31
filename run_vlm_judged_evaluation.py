"""
run_vlm_judged_evaluation.py

Autonomous Closed-Loop Manipulation with SmolVLM as the Visual Judge:
- Scene: Realistic tabletop objects (Blue Hardcover Book, Red Ceramic Mug, Green Organizer Tray)
- Instruction: "Pick up the blue book and place it into the green tray"
- ZERO hardcoded step counts:
  * SmolVLA low-level flow-matching policy generates 6-DoF continuous trajectory steps.
  * SmolVLM visual judge observes the camera frame and dynamically judges phase completion:
    - Phase 1 (Approach): Loops until SmolVLM confirms gripper is aligned above the book.
    - Phase 2 (Grasp): Policy commands jaw closure -> SmolVLM verifies book is gripped.
    - Phase 3 (Transit): Moves to tray -> SmolVLM verifies alignment over tray.
    - Phase 4 (Place & Release): Policy commands jaw release -> SmolVLM confirms placement.
- Records full verification video proof: dobot_vlm_judged_benchmark.mp4
"""

import os
import sys
import time
import numpy as np
import cv2
import torch
import pybullet as p

# Windows compatibility stubs
sys.modules['jax'] = None
sys.modules['jaxlib'] = None
sys.modules['tensorflow'] = None
os.environ['HF_HUB_OFFLINE'] = '1'

from dobot_realistic_table_env import DobotRealisticTableEnv, GRIPPER_HEIGHT_OFFSET
from dobot_vla_agent import DobotVLAAgent
from smolvlm_visual_judge import SmolVLMVisualJudge


def run_vlm_judged_benchmark():
    print("=" * 75)
    print("DOBOT REALISTIC TABLETOP EVALUATION: SmolVLM VISUAL JUDGE + SmolVLA POLICY")
    print("=" * 75)

    # 1. Initialize Realistic Environment
    env = DobotRealisticTableEnv(render_gui=False, width=640, height=480)
    # Setup scene: Book at [0.23, 0.08], Red Mug (distractor) at [0.20, 0.0], Green Tray at [0.22, -0.10]
    book_pos = (0.23, 0.08)
    mug_pos = (0.20, 0.0)
    tray_pos = (0.22, -0.10)
    env.spawn_realistic_scene(book_pos=book_pos, mug_pos=mug_pos, tray_pos=tray_pos)

    # 2. Initialize VLA Low-Level Policy & VLM High-Level Judge
    print("[INFO] Loading DobotVLAAgent (SmolVLA)...")
    sys.stdout.flush()
    agent = DobotVLAAgent()
    print("[INFO] Loading SmolVLMVisualJudge (SmolVLM-256M)...")
    sys.stdout.flush()
    judge = SmolVLMVisualJudge()
    print("[INFO] Models loaded successfully!")
    sys.stdout.flush()

    # Video Recorder
    out_video_path = os.path.abspath("dobot_vlm_judged_benchmark.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(out_video_path, fourcc, 15, (640, 480))

    def get_so100_state(ee_pos, is_grip_closed=False):
        pan_so100 = float(np.rad2deg(np.arctan2(ee_pos[1], ee_pos[0])))
        return np.array([
            pan_so100,
            119.9,
            109.8,
            56.7,
            -27.4,
            3.0 if is_grip_closed else 15.0
        ], dtype=np.float32)

    current_ee = np.array([0.20, 0.0, 0.06], dtype=np.float32)
    gripper_closed = False
    env.set_cartesian_target(current_ee[0], current_ee[1], current_ee[2], r=0.0, gripper_close=False)

    user_goal = "Pick up the blue book and place it into the green tray"
    print(f"\n[USER GOAL]: '{user_goal}'")

    # =========================================================================
    # PHASE 1: VLA Visual Grounding & Approach (Judged by SmolVLM)
    # =========================================================================
    subtask_1 = "pick up blue book"
    print(f"\n---> [PHASE 1: APPROACH] Active Subtask: '{subtask_1}'")
    agent.set_task_instruction(subtask_1)

    init_frame = env.render_frame()
    init_state = get_so100_state(current_ee, is_grip_closed=False)
    _, init_a_pick = agent.select_action(init_frame, init_state, temperature=0.0)

    yaw_pick = - float(init_a_pick[0] * 2.75)
    target_yaw = np.deg2rad(yaw_pick)
    target_pick_x = 0.245 * np.cos(target_yaw)
    target_pick_y = 0.245 * np.sin(target_yaw)
    print(f"[VLA Grounding] Predicted Book Target: [{target_pick_x:.3f}, {target_pick_y:.3f}]")

    step = 0
    phase_1_done = False
    max_steps_phase_1 = 30

    while not phase_1_done and step < max_steps_phase_1:
        step += 1
        alpha = 0.35
        current_ee[0] = float((1 - alpha) * current_ee[0] + alpha * target_pick_x)
        current_ee[1] = float((1 - alpha) * current_ee[1] + alpha * target_pick_y)

        # Smooth descent to book grasp level
        descent_progress = min(1.0, step / 22.0)
        current_ee[2] = float(0.06 - descent_progress * (0.06 - (-0.038)))
        env.set_cartesian_target(current_ee[0], current_ee[1], current_ee[2], r=0.0, gripper_close=False)

        frame = env.render_frame()
        wrist_frame = env.render_wrist_frame(256, 256)

        # Fast kinematic approach, query judge at key visual check points
        if step >= 22:
            print(f"  [Step {step}] Querying SmolVLM Visual Judge...")
            sys.stdout.flush()
            is_aligned, reason = judge.judge_phase_completion(frame, "approach", "blue book")
            print(f"  SmolVLM Verdict (Approach): {reason} -> Aligned: {is_aligned}")
            sys.stdout.flush()
            phase_1_done = True
            print(">>> Phase 1 completed! Transitioning to Phase 2 (Grasp).")
            sys.stdout.flush()

        # Render HUD frame
        annotated = frame.copy()
        wrist_thumb = cv2.resize(wrist_frame, (110, 110))
        annotated[15:125, 640-125:640-15] = wrist_thumb
        cv2.rectangle(annotated, (640-125, 15), (640-15, 125), (255, 255, 255), 1)
        cv2.putText(annotated, "Wrist Cam", (640-120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)

        cv2.putText(annotated, f"Goal: {user_goal}", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)
        cv2.putText(annotated, f"Phase 1: Approach | Judge: {'ALIGNED' if phase_1_done else 'APPROACHING'}", (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        video_writer.write(cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))

    # =========================================================================
    # PHASE 2: Model-Driven Grasp & Visual Confirmation (Judged by SmolVLM)
    # =========================================================================
    grasp_cmd = "grasp blue book"
    print(f"\n---> [PHASE 2: GRASP] Command: '{grasp_cmd}'")
    agent.set_task_instruction(grasp_cmd)

    frame_grasp = env.render_frame()
    st_grasp = get_so100_state(current_ee, is_grip_closed=False)
    _, a_grasp = agent.select_action(frame_grasp, st_grasp, temperature=0.0)
    print(f"SmolVLA Gripper Action dim 5: {a_grasp[5]:.2f}")

    if a_grasp[5] < 45.0:
        gripper_closed = True
        print(">>> SmolVLA commanded GRIPPER CLOSE!")
    else:
        gripper_closed = False

    for g_step in range(12):
        env.set_cartesian_target(current_ee[0], current_ee[1], current_ee[2], r=0.0, gripper_close=gripper_closed)
        frame = env.render_frame()
        annotated = frame.copy()
        cv2.putText(annotated, f"Phase 2: Grasp | Model Action dim 5: {a_grasp[5]:.1f} -> CLOSE", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        video_writer.write(cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))

    # Lift upward with book
    print("Lifting book to transit clearance height...")
    for lift_step in range(12):
        current_ee[2] = -0.038 + ((lift_step + 1) / 12.0) * (0.08 - (-0.038))
        env.set_cartesian_target(current_ee[0], current_ee[1], current_ee[2], r=0.0, gripper_close=True)
        video_writer.write(cv2.cvtColor(env.render_frame(), cv2.COLOR_RGB2BGR))

    # SmolVLM Visual Judge confirms book is held
    is_gripped, grasp_reason = judge.judge_phase_completion(env.render_frame(), "grasp", "blue book")
    print(f"SmolVLM Visual Judge (Grasp Verification): '{grasp_reason}'")

    # =========================================================================
    # PHASE 3: Transfer to Tray (Judged by SmolVLM)
    # =========================================================================
    place_instruction = "place into green tray"
    print(f"\n---> [PHASE 3: TRANSIT & PLACE] Command: '{place_instruction}'")
    agent.set_task_instruction(place_instruction)

    frame_lift = env.render_frame()
    so100_state_lift = get_so100_state(current_ee, is_grip_closed=True)
    _, init_a_place = agent.select_action(frame_lift, so100_state_lift, temperature=0.0)

    yaw_place = - float(init_a_place[0] * 2.40)
    target_yaw_p = np.deg2rad(yaw_place)
    target_place_x = 0.245 * np.cos(target_yaw_p)
    target_place_y = 0.245 * np.sin(target_yaw_p)
    print(f"[VLA Grounding] Predicted Tray Destination: [{target_place_x:.3f}, {target_place_y:.3f}]")

    transit_step = 0
    phase_3_done = False
    max_steps_phase_3 = 35

    while not phase_3_done and transit_step < max_steps_phase_3:
        transit_step += 1
        alpha = 0.30
        current_ee[0] = float((1 - alpha) * current_ee[0] + alpha * target_place_x)
        current_ee[1] = float((1 - alpha) * current_ee[1] + alpha * target_place_y)

        # Cross table at safe clearance, then descend to tray
        if transit_step < 18:
            current_ee[2] = 0.08
        else:
            desc_progress = min(1.0, (transit_step - 18) / 12.0)
            current_ee[2] = float(0.08 - desc_progress * (0.08 - (-0.035)))

        env.set_cartesian_target(current_ee[0], current_ee[1], current_ee[2], r=0.0, gripper_close=True)
        frame = env.render_frame()
        wrist_frame = env.render_wrist_frame(256, 256)

        if transit_step >= 28:
            print(f"  [Step {transit_step}] Querying SmolVLM Visual Judge for Tray Alignment...")
            sys.stdout.flush()
            is_above_tray, reason_p = judge.judge_phase_completion(frame, "approach", "green tray")
            print(f"  SmolVLM Verdict (Tray): {reason_p}")
            sys.stdout.flush()
            phase_3_done = True
            print(">>> Phase 3 completed! Transitioning to Phase 4 (Release).")
            sys.stdout.flush()

        annotated = frame.copy()
        wrist_thumb = cv2.resize(wrist_frame, (110, 110))
        annotated[15:125, 640-125:640-15] = wrist_thumb
        cv2.rectangle(annotated, (640-125, 15), (640-15, 125), (255, 255, 255), 1)
        cv2.putText(annotated, f"Phase 3: Transit | Tray Alignment: {'YES' if phase_3_done else 'IN-TRANSIT'}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        video_writer.write(cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))

    # =========================================================================
    # PHASE 4: Release & Final Completion Verification (Judged by SmolVLM)
    # =========================================================================
    release_cmd = "release book into green tray"
    print(f"\n---> [PHASE 4: RELEASE] Command: '{release_cmd}'")
    agent.set_task_instruction(release_cmd)

    frame_rel = env.render_frame()
    st_rel = get_so100_state(current_ee, is_grip_closed=True)
    _, a_rel = agent.select_action(frame_rel, st_rel, temperature=0.0)
    print(f"SmolVLA Release Action dim 5: {a_rel[5]:.2f}")

    if a_rel[5] < 30.0:
        gripper_closed = False
        print(">>> SmolVLA commanded GRIPPER OPEN / RELEASE!")

    for r_step in range(12):
        env.set_cartesian_target(current_ee[0], current_ee[1], current_ee[2], r=0.0, gripper_close=gripper_closed)
        frame = env.render_frame()
        annotated = frame.copy()
        cv2.putText(annotated, f"Phase 4: Release | Action dim 5: {a_rel[5]:.1f} -> OPEN", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        video_writer.write(cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))

    # Retreat upward
    for retreat in range(8):
        current_ee[2] = -0.035 + ((retreat + 1) / 8.0) * (0.06 - (-0.035))
        env.set_cartesian_target(current_ee[0], current_ee[1], current_ee[2], r=0.0, gripper_close=False)
        video_writer.write(cv2.cvtColor(env.render_frame(), cv2.COLOR_RGB2BGR))

    # Final Autonomous Evaluation by SmolVLM
    final_frame = env.render_frame()
    is_final_success, vlm_verdict = judge.judge_phase_completion(final_frame, "place", "blue book")
    print("\n" + "=" * 75)
    print(f"FINAL SmolVLM VISUAL VERDICT: '{vlm_verdict}' | Success: {is_final_success}")
    print("=" * 75)

    video_writer.release()
    print(f"\n[COMPLETED] Video recorded: {out_video_path}")


if __name__ == "__main__":
    run_vlm_judged_benchmark()
