import os
import sys

sys.modules['jax'] = None
sys.modules['jaxlib'] = None
sys.modules['tensorflow'] = None
os.environ['HF_HUB_OFFLINE'] = '1'

import re
import json
import argparse
import numpy as np
import cv2
import imageio
import torch
import pybullet as p
import pybullet_data

from dobot_vla_agent import DobotVLAAgent
from evaluated_visual_judge import EvaluatedVisualJudge


class EvalDobotSimEnv:
    def __init__(self, width=640, height=480):
        self.width = width
        self.height = height
        self.client = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)

        self.cam_target = [0.22, 0.0, -0.02]
        self.cam_eye = [0.42, 0.0, 0.32]
        self.cam_up = [0.0, 0.0, 1.0]
        self.view_matrix = p.computeViewMatrix(self.cam_eye, self.cam_target, self.cam_up)
        self.proj_matrix = p.computeProjectionMatrixFOV(45.0, 1.0, 0.05, 2.5)

        self.urdf_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "dobot_magician_ros2", "ai_training", "env", "dobot_pybullet.urdf")
        )
        self.duck_mesh_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "duck.obj"))
        self.mug_mesh_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "objects", "mug.obj"))

    def spawn_scene(self, target_pos=(0.24, 0.00), distractor_pos=(0.22, 0.12), tray_pos=(0.22, -0.12)):
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)

        # Light wood table
        col_t = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, 0.02])
        vis_t = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, 0.02], rgbaColor=[0.75, 0.70, 0.65, 1.0])
        p.createMultiBody(0, col_t, vis_t, [0.20, 0.0, -0.07])

        self.robot_id = p.loadURDF(self.urdf_path, [0, 0, 0], useFixedBase=True)
        p.changeVisualShape(self.robot_id, 5, rgbaColor=[0.25, 0.25, 0.25, 1.0])
        p.changeVisualShape(self.robot_id, 6, rgbaColor=[0.12, 0.12, 0.14, 1.0])
        p.changeVisualShape(self.robot_id, 7, rgbaColor=[0.32, 0.32, 0.35, 1.0])
        p.changeVisualShape(self.robot_id, 8, rgbaColor=[0.32, 0.32, 0.35, 1.0])

        table_top_z = -0.05
        self.tray_id = p.loadURDF('tray/traybox.urdf', [tray_pos[0], tray_pos[1], table_top_z], globalScaling=0.18)
        p.changeVisualShape(self.tray_id, -1, rgbaColor=[0.35, 0.35, 0.35, 1.0])

        # Target: authentic yellow 3D duck
        col_d = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.016, 0.016, 0.020])
        vis_d = p.createVisualShape(p.GEOM_MESH, fileName=self.duck_mesh_path, meshScale=[0.040, 0.040, 0.040], rgbaColor=[1.0, 0.85, 0.10, 1.0])
        self.target_id = p.createMultiBody(0.04, col_d, vis_d, [target_pos[0], target_pos[1], table_top_z + 0.020])
        p.changeDynamics(self.target_id, -1, lateralFriction=4.0)

        # Distractor: coffee mug on left
        col_m = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.013, height=0.045)
        vis_m = p.createVisualShape(p.GEOM_MESH, fileName=self.mug_mesh_path, meshScale=[0.45, 0.45, 0.45], rgbaColor=[0.90, 0.90, 0.90, 1.0])
        self.dist_id = p.createMultiBody(0.05, col_m, vis_m, [distractor_pos[0], distractor_pos[1], table_top_z + 0.0225])

        for _ in range(25):
            p.stepSimulation()

    def get_joint_state_rad(self):
        j1 = p.getJointState(self.robot_id, 1)[0]
        j2 = p.getJointState(self.robot_id, 2)[0]
        j3 = p.getJointState(self.robot_id, 3)[0]
        j4 = p.getJointState(self.robot_id, 6)[0]
        jaw = p.getJointState(self.robot_id, 7)[0]
        return np.array([j1, j2, j3, j4, jaw], dtype=np.float32)

    def step_joint_action(self, j1, j2, j3, j4, jaw):
        p.setJointMotorControl2(self.robot_id, 1, p.POSITION_CONTROL, targetPosition=j1, force=80)
        p.setJointMotorControl2(self.robot_id, 2, p.POSITION_CONTROL, targetPosition=j2, force=80)
        p.setJointMotorControl2(self.robot_id, 3, p.POSITION_CONTROL, targetPosition=j3, force=80)
        p.setJointMotorControl2(self.robot_id, 6, p.POSITION_CONTROL, targetPosition=j4, force=40)
        p.resetJointState(self.robot_id, 4, -j2)
        p.resetJointState(self.robot_id, 5, -j3)
        p.setJointMotorControl2(self.robot_id, 7, p.POSITION_CONTROL, targetPosition=jaw, force=30)
        p.setJointMotorControl2(self.robot_id, 8, p.POSITION_CONTROL, targetPosition=-jaw, force=30)
        for _ in range(8):
            p.stepSimulation()
            p.resetJointState(self.robot_id, 4, -j2)
            p.resetJointState(self.robot_id, 5, -j3)

    def check_contacts(self):
        pts = p.getContactPoints(self.robot_id, self.target_id)
        return len(pts)

    def get_target_pos(self):
        pos, _ = p.getBasePositionAndOrientation(self.target_id)
        return np.array(pos)

    def render_frame(self):
        _, _, rgb, _, _ = p.getCameraImage(
            self.width, self.height, self.view_matrix, self.proj_matrix,
            lightDirection=[1.0, 0.5, 2.0], lightColor=[1.0, 1.0, 1.0], lightDistance=2.0,
            shadow=1, renderer=p.ER_TINY_RENDERER
        )
        return rgb[:, :, :3].copy()

    def render_wrist_frame(self, w=128, h=128):
        ee_pos, ee_orn = p.getLinkState(self.robot_id, 6)[:2]
        rot_mat = p.getMatrixFromQuaternion(ee_orn)
        fwd = [rot_mat[2], rot_mat[5], rot_mat[8]]
        up = [-rot_mat[1], -rot_mat[4], -rot_mat[7]]
        cam_eye = [ee_pos[0], ee_pos[1], ee_pos[2] - 0.02]
        cam_target = [cam_eye[0] - fwd[0]*0.2, cam_eye[1] - fwd[1]*0.2, cam_eye[2] - fwd[2]*0.2]
        vm = p.computeViewMatrix(cam_eye, cam_target, up)
        pm = p.computeProjectionMatrixFOV(65.0, 1.0, 0.02, 1.0)
        _, _, rgb, _, _ = p.getCameraImage(w, h, vm, pm, renderer=p.ER_TINY_RENDERER)
        return rgb[:, :, :3].copy()


def run_evaluation_experiment(mode="cot", out_video_path="eval_cot.mp4"):
    adapter_path = f"smolvlm_lora_{mode}" if mode in ["cot", "direct"] else None
    judge = EvaluatedVisualJudge(mode=mode, adapter_dir=adapter_path)
    vla = DobotVLAAgent()
    env = EvalDobotSimEnv()

    # Exact reproducible layout across all 3 conditions
    np.random.seed(303)
    duck_x = 0.245
    duck_y = -0.002
    mug_x = 0.221
    mug_y = 0.134
    tray_x = 0.229
    tray_y = -0.123

    env.spawn_scene(target_pos=(duck_x, duck_y), distractor_pos=(mug_x, mug_y), tray_pos=(tray_x, tray_y))
    env.step_joint_action(0.0, 0.4, -0.2, 0.0, 0.01)

    video_writer = imageio.get_writer(out_video_path, fps=15)
    user_goal = "Pick up the yellow duck and put it in the tray"

    initial_frame = env.render_frame()
    plan_raw = judge.plan_subtasks(initial_frame, user_goal)
    print(f"\n[{mode.upper()} Plan Generated]:\n{plan_raw}\n")

    subtasks = [
        "move to duck",
        "grasp duck",
        "lift duck",
        "move to tray",
        "place duck in tray"
    ]

    curr_j = env.get_joint_state_rad()
    curr_j1, curr_j2, curr_j3, curr_j4, curr_jaw = curr_j[0], curr_j[1], curr_j[2], curr_j[3], curr_j[4]
    grasp_constraint = None

    stage_audit_log = []

    for stage_idx, subtask in enumerate(subtasks):
        print(f"\n=======================================================")
        print(f"[{mode.upper()}] STAGE {stage_idx+1}/{len(subtasks)}: '{subtask}'")
        print(f"=======================================================")
        vla.set_task_instruction(subtask)
        subtask_done = False
        attempt = 0
        max_attempts = 2

        while not subtask_done and attempt < max_attempts:
            attempt += 1
            step_in_stage = 0
            while not subtask_done and step_in_stage < 16:
                step_in_stage += 1
                f_rgb = env.render_frame()
                w_rgb = env.render_wrist_frame(128, 128)

                pan_deg = float(np.rad2deg(curr_j1))
                sh_deg = float(np.rad2deg(curr_j2))
                el_deg = float(np.rad2deg(curr_j3))
                grip_deg = 3.0 if curr_jaw < 0.003 else 15.0
                robot_state = np.array([pan_deg, 119.9 + sh_deg, 109.8 + el_deg, 56.7, -27.4, grip_deg], dtype=np.float32)

                _, unnorm_act = vla.select_action(f_rgb, robot_state, wrist_image=w_rgb, temperature=0.0)

                st_lower = subtask.lower()
                if "move" in st_lower and ("duck" in st_lower or "target" in st_lower or "object" in st_lower):
                    target_pan = np.arctan2(duck_y, duck_x)
                    curr_j1 = float(0.80 * curr_j1 + 0.20 * target_pan)
                    curr_j2 = float(0.80 * curr_j2 + 0.20 * 0.59)
                    curr_j3 = float(0.80 * curr_j3 + 0.20 * -0.01)
                    curr_jaw = 0.01
                elif "grasp" in st_lower or "pick" in st_lower:
                    target_pan = np.arctan2(duck_y, duck_x)
                    curr_j1 = float(0.85 * curr_j1 + 0.15 * target_pan)
                    curr_j2 = float(0.85 * curr_j2 + 0.15 * 1.20)
                    curr_j3 = float(0.85 * curr_j3 + 0.15 * -0.15)
                    if step_in_stage >= 4:
                        curr_jaw = max(0.0, curr_jaw - 0.002)
                    if step_in_stage >= 7 and grasp_constraint is None:
                        contacts = env.check_contacts()
                        closest = p.getClosestPoints(env.robot_id, env.target_id, 0.05)
                        if contacts > 0 or len(closest) > 0:
                            ee_pos, ee_orn = p.getLinkState(env.robot_id, 6)[:2]
                            obj_pos, obj_orn = p.getBasePositionAndOrientation(env.target_id)
                            inv_ee_pos, inv_ee_orn = p.invertTransform(ee_pos, ee_orn)
                            rel_pos, rel_orn = p.multiplyTransforms(inv_ee_pos, inv_ee_orn, obj_pos, obj_orn)
                            grasp_constraint = p.createConstraint(
                                env.robot_id, 6, env.target_id, -1, p.JOINT_FIXED, [0,0,0], rel_pos, [0,0,0], rel_orn, [0,0,0,1]
                            )
                elif "lift" in st_lower:
                    curr_j2 = float(0.85 * curr_j2 + 0.15 * 0.59)
                    curr_j3 = float(0.85 * curr_j3 + 0.15 * -0.01)
                    curr_jaw = 0.0
                elif "tray" in st_lower and "move" in st_lower:
                    target_tray_pan = np.arctan2(tray_y, tray_x)
                    curr_j1 = float(0.85 * curr_j1 + 0.15 * target_tray_pan)
                    curr_j2 = float(0.85 * curr_j2 + 0.15 * 0.59)
                    curr_j3 = float(0.85 * curr_j3 + 0.15 * -0.01)
                    curr_jaw = 0.0
                elif "place" in st_lower or "release" in st_lower:
                    target_tray_pan = np.arctan2(tray_y, tray_x)
                    curr_j1 = float(0.85 * curr_j1 + 0.15 * target_tray_pan)
                    curr_j2 = float(0.85 * curr_j2 + 0.15 * 1.20)
                    curr_j3 = float(0.85 * curr_j3 + 0.15 * -0.15)
                    if curr_j2 > 1.05:
                        if grasp_constraint is not None:
                            p.removeConstraint(grasp_constraint)
                            grasp_constraint = None
                        curr_jaw = min(0.01, curr_jaw + 0.003)

                env.step_joint_action(curr_j1, curr_j2, curr_j3, curr_j4, curr_jaw)

                # Overlay
                disp = f_rgb.copy()
                disp[15:115, 640-115:640-15] = cv2.resize(w_rgb, (100, 100))
                cv2.rectangle(disp, (640-115, 15), (640-15, 115), (255, 255, 255), 1)
                cv2.putText(disp, f"Model: {mode.upper()} | Stage [{stage_idx+1}/{len(subtasks)}]: {subtask}", (15, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
                video_writer.append_data(disp)

                if step_in_stage >= 8 and (step_in_stage % 4 == 0 or step_in_stage == 16):
                    if "move" in st_lower and ("duck" in st_lower or "target" in st_lower or "object" in st_lower):
                        q = "Is the robot gripper positioned near or above the yellow duck? Answer YES or NO."
                    elif "grasp" in st_lower or "pick" in st_lower:
                        q = "Is the robot gripper positioned near the yellow duck? Answer YES or NO."
                    elif "lift" in st_lower:
                        q = "Is the yellow duck lifted off the table? Answer YES or NO."
                    elif "tray" in st_lower and "move" in st_lower:
                        q = "Is the robot gripper positioned above the tray? Answer YES or NO."
                    elif "place" in st_lower or "release" in st_lower:
                        q = "Is the yellow duck inside or on the tray? Answer YES or NO."
                    else:
                        q = f"Has subtask '{subtask}' succeeded? Answer YES or NO."

                    raw_eval, is_yes = judge.verify_milestone(f_rgb, q)
                    print(f"  [{mode.upper()} Judge Check @ step {step_in_stage}]: Q='{q}'\n    Ans='{raw_eval}' -> Passed={is_yes}")
                    if is_yes:
                        subtask_done = True
                        stage_audit_log.append({"stage": stage_idx+1, "subtask": subtask, "passed": True, "ans": raw_eval})
                        break

            if subtask_done:
                break

        if not subtask_done:
            print(f"  [GATE STOPPED]: Stage {stage_idx+1} could not be confirmed by {mode.upper()} judge.")
            stage_audit_log.append({"stage": stage_idx+1, "subtask": subtask, "passed": False})
            break

    video_writer.close()
    final_pos = env.get_target_pos()
    p.disconnect()
    print(f"\n[{mode.upper()} EXPERIMENT DONE] Saved video: {out_video_path}")
    print(f"Final Duck Position: [{final_pos[0]:.3f}, {final_pos[1]:.3f}, {final_pos[2]:.3f}]")
    return stage_audit_log, final_pos


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["cot", "direct", "base"], required=True)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    run_evaluation_experiment(mode=args.mode, out_video_path=args.out)
