import sys, os
sys.modules['jax'] = None
sys.modules['jaxlib'] = None
sys.modules['tensorflow'] = None
os.environ['HF_HUB_OFFLINE'] = '1'

import json
import random
import numpy as np
import cv2
import pybullet as p
import pybullet_data

class SimDataHarvester:
    def __init__(self, output_dir="harvested_dataset", render_width=384, render_height=384):
        self.output_dir = os.path.abspath(output_dir)
        self.img_dir = os.path.join(self.output_dir, "images")
        os.makedirs(self.img_dir, exist_ok=True)

        self.width = render_width
        self.height = render_height

        self.client = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)

        # 45-degree elevated camera
        self.cam_target = [0.22, 0.0, -0.02]
        self.cam_eye = [0.42, 0.0, 0.32]
        self.cam_up = [0.0, 0.0, 1.0]
        self.view_matrix = p.computeViewMatrix(self.cam_eye, self.cam_target, self.cam_up)
        self.proj_matrix = p.computeProjectionMatrixFOV(45.0, 1.0, 0.05, 2.5)

        # Asset paths
        self.urdf_robot = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "dobot_magician_ros2", "ai_training", "env", "dobot_pybullet.urdf")
        )
        self.duck_mesh = os.path.abspath(os.path.join(os.path.dirname(__file__), "duck.obj"))
        self.bowl_mesh = os.path.abspath(os.path.join(os.path.dirname(__file__), "bowl.obj"))

        self.records = []
        self.frame_id = 0

    def reset_world(self, target_type="duck", container_type="tray"):
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)

        # Table
        col_t = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, 0.02])
        vis_t = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, 0.02], rgbaColor=[0.75, 0.70, 0.65, 1.0])
        p.createMultiBody(0, col_t, vis_t, [0.20, 0.0, -0.07])

        # Robot
        self.robot_id = p.loadURDF(self.urdf_robot, [0, 0, 0], useFixedBase=True)
        p.changeVisualShape(self.robot_id, 5, rgbaColor=[0.25, 0.25, 0.25, 1.0])
        p.changeVisualShape(self.robot_id, 6, rgbaColor=[0.12, 0.12, 0.14, 1.0])
        p.changeVisualShape(self.robot_id, 7, rgbaColor=[0.32, 0.32, 0.35, 1.0])
        p.changeVisualShape(self.robot_id, 8, rgbaColor=[0.32, 0.32, 0.35, 1.0])

        table_top_z = -0.05

        # Container (tray or bowl) on right
        cont_x = float(np.random.uniform(0.21, 0.24))
        cont_y = float(np.random.uniform(-0.14, -0.10))
        self.cont_pos = np.array([cont_x, cont_y, table_top_z])
        self.container_type = container_type

        if container_type == "tray":
            self.cont_id = p.loadURDF('tray/traybox.urdf', [cont_x, cont_y, table_top_z], globalScaling=0.18)
            p.changeVisualShape(self.cont_id, -1, rgbaColor=[0.35, 0.35, 0.35, 1.0])
        else: # bowl
            col_b = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.04, height=0.025)
            vis_b = p.createVisualShape(p.GEOM_MESH, fileName=self.bowl_mesh, meshScale=[1.0, 1.0, 1.0], rgbaColor=[0.25, 0.45, 0.65, 1.0])
            self.cont_id = p.createMultiBody(0, col_b, vis_b, [cont_x, cont_y, table_top_z + 0.0125])

        # Target (duck or cube) in center
        targ_x = float(np.random.uniform(0.22, 0.25))
        targ_y = float(np.random.uniform(-0.02, 0.02))
        self.targ_pos = np.array([targ_x, targ_y, table_top_z + 0.020])
        self.target_type = target_type

        if target_type == "duck":
            col_d = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.016, 0.016, 0.020])
            vis_d = p.createVisualShape(p.GEOM_MESH, fileName=self.duck_mesh, meshScale=[0.040, 0.040, 0.040], rgbaColor=[1.0, 0.85, 0.10, 1.0])
            self.target_id = p.createMultiBody(0.04, col_d, vis_d, self.targ_pos)
        else: # cube (red or green)
            col_c = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.015, 0.015, 0.015])
            vis_c = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.015, 0.015, 0.015], rgbaColor=[0.9, 0.2, 0.2, 1.0])
            self.target_id = p.createMultiBody(0.04, col_c, vis_c, self.targ_pos)

        # Distractor (e.g. coffee mug on left)
        mug_mesh_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "objects", "mug.obj"))
        col_m = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.013, height=0.045)
        vis_m = p.createVisualShape(p.GEOM_MESH, fileName=mug_mesh_path, meshScale=[0.45, 0.45, 0.45], rgbaColor=[0.9, 0.9, 0.9, 1.0])
        dist_x = float(np.random.uniform(0.21, 0.24))
        dist_y = float(np.random.uniform(0.10, 0.14))
        self.dist_pos = np.array([dist_x, dist_y, table_top_z + 0.0225])
        self.dist_id = p.createMultiBody(0.05, col_m, vis_m, self.dist_pos)

        p.changeDynamics(self.target_id, -1, lateralFriction=4.0)

        for _ in range(25):
            p.stepSimulation()

    def set_robot_joints(self, j1, j2, j3, jaw):
        p.setJointMotorControl2(self.robot_id, 1, p.POSITION_CONTROL, targetPosition=j1, force=80)
        p.setJointMotorControl2(self.robot_id, 2, p.POSITION_CONTROL, targetPosition=j2, force=80)
        p.setJointMotorControl2(self.robot_id, 3, p.POSITION_CONTROL, targetPosition=j3, force=80)
        p.resetJointState(self.robot_id, 4, -j2)
        p.resetJointState(self.robot_id, 5, -j3)
        p.setJointMotorControl2(self.robot_id, 7, p.POSITION_CONTROL, targetPosition=jaw, force=30)
        p.setJointMotorControl2(self.robot_id, 8, p.POSITION_CONTROL, targetPosition=-jaw, force=30)
        for _ in range(6):
            p.stepSimulation()
            p.resetJointState(self.robot_id, 4, -j2)
            p.resetJointState(self.robot_id, 5, -j3)

    def capture_image(self):
        _, _, rgb, _, _ = p.getCameraImage(
            self.width, self.height, self.view_matrix, self.proj_matrix,
            lightDirection=[1.0, 0.5, 2.0], lightColor=[1.0, 1.0, 1.0], lightDistance=2.0,
            shadow=1, renderer=p.ER_TINY_RENDERER
        )
        return rgb[:, :, :3].copy()

    def harvest_episode(self, ep_idx):
        t_type = "duck" if ep_idx % 2 == 0 else "cube"
        c_type = "tray" if (ep_idx // 2) % 2 == 0 else "bowl"
        self.reset_world(target_type=t_type, container_type=c_type)

        target_name = "yellow duck" if t_type == "duck" else "red cube"
        cont_name = "tray" if c_type == "tray" else "bowl"
        user_goal = f"Pick up the {target_name} and put it in the {cont_name}"

        # 1. INITIAL SCENE (Decomposition & Home Pose Verification)
        img = self.capture_image()
        img_fn = f"frame_{self.frame_id:05d}.jpg"
        cv2.imwrite(os.path.join(self.img_dir, img_fn), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        self.frame_id += 1

        # Ground truth observation
        obs_plan = f"I see a {target_name} in the center, a white mug on the left, and a {cont_name} on the right. Gripper is at home pose."
        plan_direct = f"1. move to {target_name}\n2. grasp {target_name}\n3. lift {target_name}\n4. move to {cont_name}\n5. place {target_name} in {cont_name}"
        plan_cot = f"<thought>{obs_plan}</thought>\n<plan>\n{plan_direct}\n</plan>"

        self.records.append({
            "image": img_fn,
            "task_type": "planning",
            "instruction": f"Goal: '{user_goal}'. Break into robotic subtasks.",
            "cot_response": plan_cot,
            "direct_response": plan_direct
        })

        # 2. SIMULATE APPROACH STAGE
        targ_pan = np.arctan2(self.targ_pos[1], self.targ_pos[0])
        self.set_robot_joints(targ_pan, 0.59, -0.01, 0.01) # above target
        img = self.capture_image()
        img_fn = f"frame_{self.frame_id:05d}.jpg"
        cv2.imwrite(os.path.join(self.img_dir, img_fn), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        self.frame_id += 1

        # Stage 1 positive gate
        q1 = f"Is the robot gripper positioned near or above the {target_name}? Answer YES or NO."
        self.records.append({
            "image": img_fn,
            "task_type": "stage_gate",
            "instruction": q1,
            "cot_response": f"<thought>Gripper is positioned directly over the {target_name}.</thought>\n<decision>YES</decision>",
            "direct_response": "YES"
        })

        # Negative gate check on destination
        q4 = f"Is the robot gripper positioned above the {cont_name}? Answer YES or NO."
        self.records.append({
            "image": img_fn,
            "task_type": "stage_gate",
            "instruction": q4,
            "cot_response": f"<thought>Gripper is over the {target_name}, not above the {cont_name}.</thought>\n<decision>NO</decision>",
            "direct_response": "NO"
        })

        # 3. SIMULATE GRASP STAGE
        self.set_robot_joints(targ_pan, 1.20, -0.15, 0.00) # plunge down & close jaws
        ee_pos = p.getLinkState(self.robot_id, 6)[0]
        obj_pos = p.getBasePositionAndOrientation(self.target_id)[0]
        inv_ee = p.invertTransform(*p.getLinkState(self.robot_id, 6)[:2])
        rel_pos, rel_orn = p.multiplyTransforms(inv_ee[0], inv_ee[1], *p.getBasePositionAndOrientation(self.target_id))
        
        cid = p.createConstraint(self.robot_id, 6, self.target_id, -1, p.JOINT_FIXED, [0,0,0], rel_pos, [0,0,0], rel_orn, [0,0,0,1])

        img = self.capture_image()
        img_fn = f"frame_{self.frame_id:05d}.jpg"
        cv2.imwrite(os.path.join(self.img_dir, img_fn), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        self.frame_id += 1

        q2 = f"Is the robot gripper positioned near the {target_name}? Answer YES or NO."
        self.records.append({
            "image": img_fn,
            "task_type": "stage_gate",
            "instruction": q2,
            "cot_response": f"<thought>Gripper jaws are clamped firmly around the {target_name}.</thought>\n<decision>YES</decision>",
            "direct_response": "YES"
        })

        # 4. SIMULATE LIFT STAGE
        self.set_robot_joints(targ_pan, 0.59, -0.01, 0.00) # lift
        for _ in range(10): p.stepSimulation()
        img = self.capture_image()
        img_fn = f"frame_{self.frame_id:05d}.jpg"
        cv2.imwrite(os.path.join(self.img_dir, img_fn), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        self.frame_id += 1

        q3 = f"Is the {target_name} lifted off the table? Answer YES or NO."
        self.records.append({
            "image": img_fn,
            "task_type": "stage_gate",
            "instruction": q3,
            "cot_response": f"<thought>The {target_name} is elevated in the air, clear of the table surface.</thought>\n<decision>YES</decision>",
            "direct_response": "YES"
        })

        # 5. SIMULATE TRANSIT TO CONTAINER
        cont_pan = np.arctan2(self.cont_pos[1], self.cont_pos[0])
        self.set_robot_joints(cont_pan, 0.59, -0.01, 0.00) # above container
        for _ in range(10): p.stepSimulation()
        img = self.capture_image()
        img_fn = f"frame_{self.frame_id:05d}.jpg"
        cv2.imwrite(os.path.join(self.img_dir, img_fn), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        self.frame_id += 1

        self.records.append({
            "image": img_fn,
            "task_type": "stage_gate",
            "instruction": q4,
            "cot_response": f"<thought>Gripper and held {target_name} are centered directly above the {cont_name}.</thought>\n<decision>YES</decision>",
            "direct_response": "YES"
        })

        # 6. SIMULATE RELEASE INTO CONTAINER
        self.set_robot_joints(cont_pan, 1.20, -0.15, 0.01) # lower and open
        p.removeConstraint(cid)
        for _ in range(25): p.stepSimulation()
        
        # Withdraw gripper slightly up so object in container is unobstructed
        self.set_robot_joints(cont_pan, 0.85, -0.08, 0.01)
        for _ in range(15): p.stepSimulation()

        img = self.capture_image()
        img_fn = f"frame_{self.frame_id:05d}.jpg"
        cv2.imwrite(os.path.join(self.img_dir, img_fn), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        self.frame_id += 1

        q5 = f"Is the {target_name} inside or on the {cont_name}? Answer YES or NO."
        self.records.append({
            "image": img_fn,
            "task_type": "stage_gate",
            "instruction": q5,
            "cot_response": f"<thought>The {target_name} is resting settled inside the {cont_name} basin.</thought>\n<decision>YES</decision>",
            "direct_response": "YES"
        })

    def run_harvest(self, num_episodes=20):
        print(f"[SimDataHarvester] Starting automated rollout for {num_episodes} episodes...")
        for ep in range(num_episodes):
            self.harvest_episode(ep)
            print(f"  Episode {ep+1}/{num_episodes} completed. Total frames collected: {self.frame_id}")
            sys.stdout.flush()

        ann_path = os.path.join(self.output_dir, "dataset_annotations.json")
        with open(ann_path, "w") as f:
            json.dump(self.records, f, indent=2)

        print(f"[SimDataHarvester] Finished! Saved {len(self.records)} samples to {ann_path}")
        p.disconnect()

if __name__ == "__main__":
    harvester = SimDataHarvester()
    harvester.run_harvest(num_episodes=20)
