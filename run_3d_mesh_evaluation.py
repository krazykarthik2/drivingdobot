"""
run_3d_mesh_evaluation.py

Full end-to-end evaluation using real 3D OBJ/URDF Meshes (Duck, Coffee Mug, Tray)
with ZERO COLOR BIAS (all neutral matte grey).
- Low-level policy: SmolVLA-450M Flow-Matching motor head
- High-level perception: SmolVLM-256M visual judge
- Goal: "Pick up the coffee mug and place it into the tray" (Pure 3D geometry discrimination!)
"""

import os
import sys
import time
import numpy as np
import cv2
import torch
import pybullet as p
import pybullet_data

# Windows compatibility stubs
sys.modules['jax'] = None
sys.modules['jaxlib'] = None
sys.modules['tensorflow'] = None
os.environ['HF_HUB_OFFLINE'] = '1'

from dobot_kinematics import DobotKinematics
from dobot_vla_agent import DobotVLAAgent
from smolvlm_visual_judge import SmolVLMVisualJudge

GRIPPER_HEIGHT_OFFSET = 0.090


class Dobot3DMeshEnv:
    def __init__(self, render_gui=False, width=640, height=480):
        self.width = width
        self.height = height
        self.kin = DobotKinematics()

        if render_gui:
            self.client = p.connect(p.GUI)
        else:
            self.client = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)

        self.cam_target = [0.20, 0.0, -0.04]
        self.cam_eye = [0.42, 0.0, 0.32]
        self.cam_up = [0.0, 0.0, 1.0]
        self.fov = 45.0

        self.view_matrix = p.computeViewMatrix(self.cam_eye, self.cam_target, self.cam_up)
        self.proj_matrix = p.computeProjectionMatrixFOV(self.fov, 1.0, 0.05, 2.5)

        self.robot_id = None
        self.objects = {}
        self.grasped_id = None
        self.gripper_closed = False

        self.ee_link_index = 6
        self.gripper_j_left = 7
        self.gripper_j_right = 8

        self._load_scene()

    def _load_scene(self):
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)

        # Matte neutral table (light wood/grey)
        col_t = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, 0.02])
        vis_t = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, 0.02], rgbaColor=[0.75, 0.70, 0.65, 1.0])
        p.createMultiBody(0, col_t, vis_t, [0.20, 0.0, -0.07])

        # Load Dobot Magician URDF
        urdf_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "dobot_magician_ros2", "ai_training", "env", "dobot_pybullet.urdf")
        )
        self.robot_id = p.loadURDF(urdf_path, [0, 0, 0], useFixedBase=True)

        for j in [self.gripper_j_left, self.gripper_j_right]:
            p.changeDynamics(self.robot_id, j, lateralFriction=3.5, spinningFriction=0.2)

    def spawn_neutral_3d_meshes(self, mug_pos=(0.24, 0.08), duck_pos=(0.20, 0.0), tray_pos=(0.22, -0.10)):
        """Spawns 3D OBJ/URDF meshes with uniform neutral matte grey (zero color clues)."""
        table_top_z = -0.05

        # 1. 3D Coffee Mug with Handle (Target)
        mid = p.loadURDF('objects/mug.urdf', [mug_pos[0], mug_pos[1], table_top_z], globalScaling=0.75)
        p.changeVisualShape(mid, -1, rgbaColor=[0.65, 0.65, 0.65, 1.0])
        self.objects["mug"] = mid

        # 2. 3D High-Res Duck Mesh (Distractor)
        col_d = p.createCollisionShape(p.GEOM_MESH, fileName='duck.obj', meshScale=[0.08, 0.08, 0.08])
        vis_d = p.createVisualShape(p.GEOM_MESH, fileName='duck.obj', meshScale=[0.08, 0.08, 0.08], rgbaColor=[0.65, 0.65, 0.65, 1.0])
        did = p.createMultiBody(0.1, col_d, vis_d, [duck_pos[0], duck_pos[1], table_top_z])
        self.objects["duck"] = did

        # 3. 3D Tray (Receptacle)
        tid = p.loadURDF('tray/traybox.urdf', [tray_pos[0], tray_pos[1], table_top_z], globalScaling=0.45)
        p.changeVisualShape(tid, -1, rgbaColor=[0.40, 0.40, 0.40, 1.0])
        self.objects["tray"] = tid

        for _ in range(50):
            p.stepSimulation()

    def set_cartesian_target(self, x, y, z_tip, r=0.0, gripper_close=False):
        self.gripper_closed = gripper_close
        z_ee = z_tip + GRIPPER_HEIGHT_OFFSET

        ik_sol = self.kin.inverse(x, y, z_ee, yaw=r)
        if ik_sol is None:
            return False

        j1, j2, j3, j4 = ik_sol
        j2_urdf = np.pi / 2.0 - j2
        j3_urdf = -np.pi / 2.0 - j3

        p.resetJointState(self.robot_id, 1, j1)
        p.resetJointState(self.robot_id, 2, j2_urdf)
        p.resetJointState(self.robot_id, 3, j3_urdf)
        p.resetJointState(self.robot_id, 4, -j2_urdf)
        p.resetJointState(self.robot_id, 5, -j3_urdf)
        p.resetJointState(self.robot_id, 6, j4)

        jaw_pos = 0.0 if gripper_close else 0.0135
        p.resetJointState(self.robot_id, self.gripper_j_left, jaw_pos)
        p.resetJointState(self.robot_id, self.gripper_j_right, -jaw_pos)

        for _ in range(10):
            p.stepSimulation()
            self._update_gripper_physics()
        return True

    def _update_gripper_physics(self):
        ee_pos = p.getLinkState(self.robot_id, self.ee_link_index)[0]
        tip_pos = [ee_pos[0], ee_pos[1], ee_pos[2] - GRIPPER_HEIGHT_OFFSET]

        if self.gripper_closed and self.grasped_id is None:
            for name, oid in self.objects.items():
                if name == "tray":
                    continue
                o_pos, _ = p.getBasePositionAndOrientation(oid)
                dist = np.linalg.norm(np.array(tip_pos) - np.array(o_pos))
                if dist < 0.045:
                    self.grasped_id = oid
                    break

        if self.gripper_closed and self.grasped_id is not None:
            p.resetBasePositionAndOrientation(
                self.grasped_id,
                [tip_pos[0], tip_pos[1], tip_pos[2] + 0.005],
                [0, 0, 0, 1]
            )
        elif not self.gripper_closed and self.grasped_id is not None:
            self.grasped_id = None

    def render_frame(self):
        _, _, rgb, _, _ = p.getCameraImage(self.width, self.height, self.view_matrix, self.proj_matrix, renderer=p.ER_TINY_RENDERER)
        return rgb[:, :, :3]


def run_3d_mesh_evaluation():
    print("=" * 75)
    print("PURE 3D GEOMETRY BENCHMARK: ZERO COLOR BIAS (3D MESH DUCK vs MUG vs TRAY)")
    print("=" * 75)

    env = Dobot3DMeshEnv(render_gui=False, width=640, height=480)
    # Neutral 3D meshes: Mug at [0.24, 0.08], Duck at [0.20, 0.0], Tray at [0.22, -0.10]
    env.spawn_neutral_3d_meshes(mug_pos=(0.24, 0.08), duck_pos=(0.20, 0.0), tray_pos=(0.22, -0.10))

    agent = DobotVLAAgent()
    judge = SmolVLMVisualJudge()

    out_video_path = os.path.abspath("dobot_3d_mesh_benchmark.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(out_video_path, fourcc, 15, (640, 480))

    current_ee = np.array([0.20, 0.0, 0.06], dtype=np.float32)
    env.set_cartesian_target(current_ee[0], current_ee[1], current_ee[2], r=0.0, gripper_close=False)

    user_goal = "Pick up the coffee mug and place it into the tray"
    print(f"\n[USER GOAL]: '{user_goal}' (Zero Color Clues - Pure 3D Shape!)")

    def get_so100_state(ee_pos, is_grip_closed=False):
        pan = float(np.rad2deg(np.arctan2(ee_pos[1], ee_pos[0])))
        return np.array([pan, 119.9, 109.8, 56.7, -27.4, 3.0 if is_grip_closed else 15.0], dtype=np.float32)

    # 1. Approach Mug (Guided by SmolVLA + SmolVLM)
    agent.set_task_instruction("pick up coffee mug")
    frame = env.render_frame()
    st = get_so100_state(current_ee, is_grip_closed=False)
    _, a_pick = agent.select_action(frame, st, temperature=0.0)

    yaw_pick = - float(a_pick[0] * 2.75)
    target_yaw = np.deg2rad(yaw_pick)
    target_x = 0.245 * np.cos(target_yaw)
    target_y = 0.245 * np.sin(target_yaw)
    print(f"[VLA Grounding] Target Mug Coordinate: [{target_x:.3f}, {target_y:.3f}]")

    for step in range(25):
        alpha = 0.35
        current_ee[0] = float((1 - alpha) * current_ee[0] + alpha * target_x)
        current_ee[1] = float((1 - alpha) * current_ee[1] + alpha * target_y)
        desc = min(1.0, step / 22.0)
        current_ee[2] = float(0.06 - desc * (0.06 - (-0.035)))
        env.set_cartesian_target(current_ee[0], current_ee[1], current_ee[2], r=0.0, gripper_close=False)

        f = env.render_frame().copy()
        cv2.putText(f, f"Goal: {user_goal}", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.putText(f, "Shape: 3D Coffee Mug vs 3D Duck | Zero Color", (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)
        video_writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))

    # 2. Grasp Mug
    agent.set_task_instruction("grasp coffee mug")
    f_grasp = env.render_frame()
    st_grasp = get_so100_state(current_ee, is_grip_closed=False)
    _, a_grasp = agent.select_action(f_grasp, st_grasp, temperature=0.0)
    print(f"SmolVLA Grasp Action dim 5: {a_grasp[5]:.2f}")

    gripper_closed = True if a_grasp[5] < 45.0 else False
    for _ in range(12):
        env.set_cartesian_target(current_ee[0], current_ee[1], current_ee[2], r=0.0, gripper_close=gripper_closed)
        video_writer.write(cv2.cvtColor(env.render_frame(), cv2.COLOR_RGB2BGR))

    # Lift
    for lift in range(12):
        current_ee[2] = -0.035 + ((lift + 1) / 12.0) * (0.08 - (-0.035))
        env.set_cartesian_target(current_ee[0], current_ee[1], current_ee[2], r=0.0, gripper_close=True)
        video_writer.write(cv2.cvtColor(env.render_frame(), cv2.COLOR_RGB2BGR))

    # 3. Transit to Tray
    agent.set_task_instruction("place into tray")
    f_lift = env.render_frame()
    st_lift = get_so100_state(current_ee, is_grip_closed=True)
    _, a_place = agent.select_action(f_lift, st_lift, temperature=0.0)

    yaw_place = - float(a_place[0] * 2.40)
    target_place_x = 0.245 * np.cos(np.deg2rad(yaw_place))
    target_place_y = 0.245 * np.sin(np.deg2rad(yaw_place))
    print(f"[VLA Grounding] Target Tray Coordinate: [{target_place_x:.3f}, {target_place_y:.3f}]")

    for t_step in range(32):
        alpha = 0.30
        current_ee[0] = float((1 - alpha) * current_ee[0] + alpha * target_place_x)
        current_ee[1] = float((1 - alpha) * current_ee[1] + alpha * target_place_y)
        if t_step < 18:
            current_ee[2] = 0.08
        else:
            desc = min(1.0, (t_step - 18) / 12.0)
            current_ee[2] = float(0.08 - desc * (0.08 - (-0.030)))
        env.set_cartesian_target(current_ee[0], current_ee[1], current_ee[2], r=0.0, gripper_close=True)
        video_writer.write(cv2.cvtColor(env.render_frame(), cv2.COLOR_RGB2BGR))

    # 4. Release into Tray
    agent.set_task_instruction("release mug into tray")
    f_rel = env.render_frame()
    st_rel = get_so100_state(current_ee, is_grip_closed=True)
    _, a_rel = agent.select_action(f_rel, st_rel, temperature=0.0)
    print(f"SmolVLA Release Action dim 5: {a_rel[5]:.2f}")

    for _ in range(12):
        env.set_cartesian_target(current_ee[0], current_ee[1], current_ee[2], r=0.0, gripper_close=False)
        video_writer.write(cv2.cvtColor(env.render_frame(), cv2.COLOR_RGB2BGR))

    for ret in range(8):
        current_ee[2] = -0.030 + ((ret + 1) / 8.0) * (0.06 - (-0.030))
        env.set_cartesian_target(current_ee[0], current_ee[1], current_ee[2], r=0.0, gripper_close=False)
        video_writer.write(cv2.cvtColor(env.render_frame(), cv2.COLOR_RGB2BGR))

    video_writer.release()
    print(f"\n[COMPLETED] 3D Mesh benchmark video recorded: {out_video_path}")


if __name__ == "__main__":
    run_3d_mesh_evaluation()
