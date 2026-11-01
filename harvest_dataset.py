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

class ProductionDataHarvester:
    def __init__(self, output_dir="harvested_dataset", render_width=384, render_height=384):
        self.output_dir = os.path.abspath(output_dir)
        self.img_dir = os.path.join(self.output_dir, "images")
        os.makedirs(self.img_dir, exist_ok=True)

        self.width = render_width
        self.height = render_height

        self.client = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)

        # Asset paths
        self.urdf_robot = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "dobot_magician_ros2", "ai_training", "env", "dobot_pybullet.urdf")
        )
        self.duck_mesh = os.path.abspath(os.path.join(os.path.dirname(__file__), "duck.obj"))
        self.bowl_mesh = os.path.abspath(os.path.join(os.path.dirname(__file__), "bowl.obj"))
        self.mug_mesh = os.path.abspath(os.path.join(os.path.dirname(__file__), "objects", "mug.obj"))

        self.records = []
        self.frame_id = 0

    def setup_camera(self):
        # Apply camera jitter (+/- 2cm) and FOV variations (+/- 2 deg)
        cam_target = [0.22 + np.random.uniform(-0.01, 0.01), 0.0, -0.02]
        cam_eye = [
            0.42 + np.random.uniform(-0.02, 0.02),
            0.00 + np.random.uniform(-0.02, 0.02),
            0.32 + np.random.uniform(-0.02, 0.02)
        ]
        cam_up = [0.0, 0.0, 1.0]
        fov = float(np.random.uniform(43.0, 47.0))
        self.view_matrix = p.computeViewMatrix(cam_eye, cam_target, cam_up)
        self.proj_matrix = p.computeProjectionMatrixFOV(fov, 1.0, 0.05, 2.5)

    def reset_world(self, target_type="duck", container_type="tray", table_style="wood"):
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        self.setup_camera()

        # Multi-table textures (wood, dark gray, white)
        table_colors = {
            "wood": [0.75, 0.70, 0.65, 1.0],
            "dark": [0.30, 0.30, 0.32, 1.0],
            "white": [0.88, 0.88, 0.88, 1.0]
        }
        col_t = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, 0.02])
        vis_t = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, 0.02], rgbaColor=table_colors[table_style])
        p.createMultiBody(0, col_t, vis_t, [0.20, 0.0, -0.07])

        # Robot
        self.robot_id = p.loadURDF(self.urdf_robot, [0, 0, 0], useFixedBase=True)
        p.changeVisualShape(self.robot_id, 5, rgbaColor=[0.25, 0.25, 0.25, 1.0])
        p.changeVisualShape(self.robot_id, 6, rgbaColor=[0.12, 0.12, 0.14, 1.0])
        p.changeVisualShape(self.robot_id, 7, rgbaColor=[0.32, 0.32, 0.35, 1.0])
        p.changeVisualShape(self.robot_id, 8, rgbaColor=[0.32, 0.32, 0.35, 1.0])

        table_top_z = -0.05

        # Container on right
        cont_x = float(np.random.uniform(0.21, 0.25))
        cont_y = float(np.random.uniform(-0.15, -0.10))
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
        targ_x = float(np.random.uniform(0.21, 0.26))
        targ_y = float(np.random.uniform(-0.03, 0.03))
        self.targ_pos = np.array([targ_x, targ_y, table_top_z + 0.020])
        self.target_type = target_type

        if target_type == "duck":
            col_d = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.016, 0.016, 0.020])
            vis_d = p.createVisualShape(p.GEOM_MESH, fileName=self.duck_mesh, meshScale=[0.040, 0.040, 0.040], rgbaColor=[1.0, 0.85, 0.10, 1.0])
            self.target_id = p.createMultiBody(0.04, col_d, vis_d, self.targ_pos)
        else: # cube
            col_c = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.015, 0.015, 0.015])
            vis_c = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.015, 0.015, 0.015], rgbaColor=[0.85, 0.2, 0.2, 1.0])
            self.target_id = p.createMultiBody(0.04, col_c, vis_c, self.targ_pos)

        # Distractor (coffee mug) on left
        col_m = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.013, height=0.045)
        vis_m = p.createVisualShape(p.GEOM_MESH, fileName=self.mug_mesh, meshScale=[0.45, 0.45, 0.45], rgbaColor=[0.9, 0.9, 0.9, 1.0])
        dist_x = float(np.random.uniform(0.20, 0.24))
        dist_y = float(np.random.uniform(0.10, 0.15))
        self.dist_pos = np.array([dist_x, dist_y, table_top_z + 0.0225])
        self.dist_id = p.createMultiBody(0.05, col_m, vis_m, self.dist_pos)

        p.changeDynamics(self.target_id, -1, lateralFriction=4.0)

        for _ in range(20):
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
        # Random light direction variations
        lx = float(np.random.uniform(0.8, 1.2))
        ly = float(np.random.uniform(0.3, 0.7))
        lz = float(np.random.uniform(1.8, 2.2))
        _, _, rgb, _, _ = p.getCameraImage(
            self.width, self.height, self.view_matrix, self.proj_matrix,
            lightDirection=[lx, ly, lz], lightColor=[1.0, 1.0, 1.0], lightDistance=2.0,
            shadow=1, renderer=p.ER_TINY_RENDERER
        )
        return rgb[:, :, :3].copy()

    def harvest_episode(self, ep_idx):
        t_type = "duck" if ep_idx % 2 == 0 else "cube"
        c_type = "tray" if (ep_idx // 2) % 2 == 0 else "bowl"
        t_style = ["wood", "dark", "white"][ep_idx % 3]
        self.reset_world(target_type=t_type, container_type=c_type, table_style=t_style)

        target_name = "duck" if t_type == "duck" else "cube"
        cont_name = "tray" if c_type == "tray" else "bowl"
        user_goal = f"Pick up the {target_name} and put it in the {cont_name}"

        # -------------------------------------------------------------
        # 1. INITIAL SCENE: Task Planning & Spatial VQA Regularization
        # -------------------------------------------------------------
        img = self.capture_image()
        img_fn = f"frame_{self.frame_id:05d}.jpg"
        cv2.imwrite(os.path.join(self.img_dir, img_fn), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        self.frame_id += 1

        obs_plan = f"I see a {target_name} in the center, a mug on the left, and a {cont_name} on the right. Gripper is at home pose."
        plan_direct = f"1. move to {target_name}\n2. grasp {target_name}\n3. lift {target_name}\n4. move to {cont_name}\n5. place {target_name} in {cont_name}"
        plan_cot = f"<thought>{obs_plan}</thought>\n<plan>\n{plan_direct}\n</plan>"

        self.records.append({
            "image": img_fn,
            "task_type": "planning",
            "instruction": f"Goal: '{user_goal}'. Break into robotic subtasks.",
            "cot_response": plan_cot,
            "direct_response": plan_direct
        })

        # Auxiliary Spatial VQA Sample (10% partition)
        self.records.append({
            "image": img_fn,
            "task_type": "spatial_vqa",
            "instruction": f"Where is the {target_name} located?",
            "cot_response": f"<thought>The {target_name} is situated in the central work area on the table.</thought>\n<answer>center of table</answer>",
            "direct_response": "center of table"
        })

        # -------------------------------------------------------------
        # 2. HARD NEGATIVE: Near-Miss Approach (Wrong azimuth / distractor)
        # -------------------------------------------------------------
        dist_pan = np.arctan2(self.dist_pos[1], self.dist_pos[0])
        self.set_robot_joints(dist_pan, 0.59, -0.01, 0.01) # over distractor mug
        img_neg1 = self.capture_image()
        img_fn_neg1 = f"frame_{self.frame_id:05d}.jpg"
        cv2.imwrite(os.path.join(self.img_dir, img_fn_neg1), cv2.cvtColor(img_neg1, cv2.COLOR_RGB2BGR))
        self.frame_id += 1

        self.records.append({
            "image": img_fn_neg1,
            "task_type": "hard_negative",
            "instruction": f"Is the robot gripper positioned near or above the {target_name}? Answer YES or NO.",
            "cot_response": f"<thought>The gripper is hovering above the mug on the left, not the {target_name}.</thought>\n<decision>NO</decision>",
            "direct_response": "NO"
        })

        # -------------------------------------------------------------
        # 3. POSITIVE: Approach Target
        # -------------------------------------------------------------
        targ_pan = np.arctan2(self.targ_pos[1], self.targ_pos[0])
        self.set_robot_joints(targ_pan, 0.59, -0.01, 0.01) # correctly above target
        img = self.capture_image()
        img_fn = f"frame_{self.frame_id:05d}.jpg"
        cv2.imwrite(os.path.join(self.img_dir, img_fn), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        self.frame_id += 1

        self.records.append({
            "image": img_fn,
            "task_type": "positive_gate",
            "instruction": f"Is the robot gripper positioned near or above the {target_name}? Answer YES or NO.",
            "cot_response": f"<thought>Gripper is positioned directly over the {target_name}.</thought>\n<decision>YES</decision>",
            "direct_response": "YES"
        })

        # -------------------------------------------------------------
        # 4. HARD NEGATIVE: Near-Miss Grasp (Jaws closed in empty air)
        # -------------------------------------------------------------
        self.set_robot_joints(targ_pan + 0.12, 1.20, -0.15, 0.00) # plunge 3cm away and close
        img_neg2 = self.capture_image()
        img_fn_neg2 = f"frame_{self.frame_id:05d}.jpg"
        cv2.imwrite(os.path.join(self.img_dir, img_fn_neg2), cv2.cvtColor(img_neg2, cv2.COLOR_RGB2BGR))
        self.frame_id += 1

        self.records.append({
            "image": img_fn_neg2,
            "task_type": "hard_negative",
            "instruction": f"Is the robot gripper positioned near the {target_name}? Answer YES or NO.",
            "cot_response": f"<thought>The gripper clamped empty air beside the {target_name}.</thought>\n<decision>NO</decision>",
            "direct_response": "NO"
        })

        # -------------------------------------------------------------
        # 5. POSITIVE: Authentic Grasp
        # -------------------------------------------------------------
        self.set_robot_joints(targ_pan, 1.20, -0.15, 0.00)
        inv_ee = p.invertTransform(*p.getLinkState(self.robot_id, 6)[:2])
        rel_pos, rel_orn = p.multiplyTransforms(inv_ee[0], inv_ee[1], *p.getBasePositionAndOrientation(self.target_id))
        cid = p.createConstraint(self.robot_id, 6, self.target_id, -1, p.JOINT_FIXED, [0,0,0], rel_pos, [0,0,0], rel_orn, [0,0,0,1])

        img = self.capture_image()
        img_fn = f"frame_{self.frame_id:05d}.jpg"
        cv2.imwrite(os.path.join(self.img_dir, img_fn), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        self.frame_id += 1

        self.records.append({
            "image": img_fn,
            "task_type": "positive_gate",
            "instruction": f"Is the robot gripper positioned near the {target_name}? Answer YES or NO.",
            "cot_response": f"<thought>Gripper jaws are firmly closed around the {target_name}.</thought>\n<decision>YES</decision>",
            "direct_response": "YES"
        })

        # -------------------------------------------------------------
        # 6. POSITIVE: Lift Object
        # -------------------------------------------------------------
        self.set_robot_joints(targ_pan, 0.59, -0.01, 0.00)
        for _ in range(8): p.stepSimulation()
        img = self.capture_image()
        img_fn = f"frame_{self.frame_id:05d}.jpg"
        cv2.imwrite(os.path.join(self.img_dir, img_fn), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        self.frame_id += 1

        self.records.append({
            "image": img_fn,
            "task_type": "positive_gate",
            "instruction": f"Is the {target_name} lifted off the table? Answer YES or NO.",
            "cot_response": f"<thought>The {target_name} is elevated and completely clear of the table.</thought>\n<decision>YES</decision>",
            "direct_response": "YES"
        })

        # -------------------------------------------------------------
        # 7. POSITIVE: Transit to Container
        # -------------------------------------------------------------
        cont_pan = np.arctan2(self.cont_pos[1], self.cont_pos[0])
        self.set_robot_joints(cont_pan, 0.59, -0.01, 0.00)
        for _ in range(8): p.stepSimulation()
        img = self.capture_image()
        img_fn = f"frame_{self.frame_id:05d}.jpg"
        cv2.imwrite(os.path.join(self.img_dir, img_fn), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        self.frame_id += 1

        self.records.append({
            "image": img_fn,
            "task_type": "positive_gate",
            "instruction": f"Is the robot gripper positioned above the {cont_name}? Answer YES or NO.",
            "cot_response": f"<thought>Gripper holding the {target_name} is centered directly above the {cont_name}.</thought>\n<decision>YES</decision>",
            "direct_response": "YES"
        })

        # -------------------------------------------------------------
        # 8. HARD NEGATIVE: Premature Release Check (Held above, not placed)
        # -------------------------------------------------------------
        self.records.append({
            "image": img_fn,
            "task_type": "hard_negative",
            "instruction": f"Is the {target_name} inside or on the {cont_name}? Answer YES or NO.",
            "cot_response": f"<thought>The {target_name} is still held in the gripper high above the {cont_name}, not yet placed.</thought>\n<decision>NO</decision>",
            "direct_response": "NO"
        })

        # -------------------------------------------------------------
        # 9. POSITIVE: Release & Settle in Container
        # -------------------------------------------------------------
        self.set_robot_joints(cont_pan, 1.20, -0.15, 0.01) # lower and release
        p.removeConstraint(cid)
        for _ in range(25): p.stepSimulation()
        self.set_robot_joints(cont_pan, 0.85, -0.08, 0.01) # withdraw gripper
        for _ in range(15): p.stepSimulation()

        img = self.capture_image()
        img_fn = f"frame_{self.frame_id:05d}.jpg"
        cv2.imwrite(os.path.join(self.img_dir, img_fn), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        self.frame_id += 1

        self.records.append({
            "image": img_fn,
            "task_type": "positive_gate",
            "instruction": f"Is the {target_name} inside or on the {cont_name}? Answer YES or NO.",
            "cot_response": f"<thought>The {target_name} is resting settled inside the {cont_name} basin.</thought>\n<decision>YES</decision>",
            "direct_response": "YES"
        })

    def run_harvest(self, num_episodes=150):
        print(f"[ProductionDataHarvester] Harvesting production mix across {num_episodes} episodes...")
        for ep in range(num_episodes):
            self.harvest_episode(ep)
            if (ep + 1) % 25 == 0:
                print(f"  Progress: {ep+1}/{num_episodes} episodes | Total Samples: {len(self.records)}")
                sys.stdout.flush()

        ann_path = os.path.join(self.output_dir, "dataset_annotations.json")
        with open(ann_path, "w") as f:
            json.dump(self.records, f, indent=2)

        print(f"[ProductionDataHarvester] Complete! Generated {len(self.records)} balanced samples at {ann_path}")
        p.disconnect()

if __name__ == "__main__":
    harvester = ProductionDataHarvester()
    harvester.run_harvest(num_episodes=150)
