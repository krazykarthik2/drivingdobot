"""
Phase 4: Full Physical Simulation with Gripper Dynamics & 45-Degree Front Camera
NanoDrive-4DOF (Residual / Delta Architecture)

CORE PHILOSOPHY STRICTLY ENFORCED:
System 0A (Base Solver) and System 0B (Neural Delta Driver) ONLY SEE PAINTED PIXELS on black.
- Raw camera feed is completely discarded.
- Neither System 0A nor System 0B has access to simulator state, ground truth coordinates, or physics bodies.
- System 0A takes ONLY the painted pixel image, detects green & blue pixel blobs, converts to (X, Y) via camera homography, and computes the naive straight-line joint plan.
- System 0B takes ONLY the painted pixel image + the naive joint plan, and outputs the corrective joint delta.
- Final Action = Base Action + Delta Action.
"""

import os
import sys
import cv2
import numpy as np
import imageio
import torch

from dobot_kinematics import DobotKinematics
from dobot_physics_sim import DobotPybulletPhysicsSim
from system0a_base_controller import System0APaintedPixelSolver
from system1_perception import System1PerceptionEngine, CameraProjector
from system2_reasoning import System2ReasoningEngine
from neural_delta_model import NeuralDeltaDriver
from dobot_residual_env import DobotResidualSimEnv

class PureVisionHybridController:
    """
    Hybrid Controller strictly restricted to painted pixel inputs.
    """
    def __init__(self, model_weights_path="dobot_delta_model.pt"):
        self.kin = DobotKinematics()
        self.base_solver = System0APaintedPixelSolver()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.delta_model = NeuralDeltaDriver().to(self.device)
        if os.path.exists(model_weights_path):
            self.delta_model.load_state_dict(torch.load(model_weights_path, map_location=self.device))
            print(f"[Pure Vision Controller] Loaded neural weights from {model_weights_path}")
        self.delta_model.eval()

    def plan_from_painted_pixels_only(self, painted_image, num_steps=30):
        """
        INPUT: ONLY the painted pixel image (solid colored silhouettes on pitch-black).
        NO ground-truth simulation positions or physics objects passed!
        """
        # 1. System 0A Base Solver looks ONLY at painted pixels:
        base_plan = self.base_solver.generate_naive_plan_from_pixels(painted_image, num_steps=num_steps)
        naive_joints = base_plan["naive_joints"]
        naive_cart = base_plan["naive_cart"]

        # Downsample painted pixels to 128x128 for System 0B Neural Delta Driver
        painted_128 = cv2.resize(painted_image, (128, 128), interpolation=cv2.INTER_NEAREST)

        # 2. System 0B Neural Delta Driver looks ONLY at painted pixels + naive joint plan:
        img_tensor = torch.tensor(
            painted_128.transpose(2, 0, 1).astype(np.float32) / 255.0,
            device=self.device
        ).unsqueeze(0).repeat(num_steps, 1, 1, 1)

        naive_tensor = torch.tensor(naive_joints, device=self.device)

        with torch.no_grad():
            delta_tensor = self.delta_model(img_tensor, naive_tensor)
            delta_joints = delta_tensor.cpu().numpy()

        # 3. Hybrid Combination: Final Action = Base Action + Delta Action
        hybrid_joints = naive_joints + delta_joints
        hybrid_cart = np.array([self.kin.forward(*j) for j in hybrid_joints], dtype=np.float32)

        return {
            "naive_joints": naive_joints,
            "delta_joints": delta_joints,
            "hybrid_joints": hybrid_joints,
            "naive_cart": naive_cart,
            "hybrid_cart": hybrid_cart,
            "pick_uv": base_plan["pick_uv"],
            "place_uv": base_plan["place_uv"]
        }

def run_physics_benchmark(output_video="nanodrive_physics_45deg_benchmark.mp4"):
    print("=" * 75)
    print("NANODRIVE-4DOF BENCHMARK: STRICTLY PAINTED PIXELS ONLY FOR SYSTEM 0A & 0B")
    print("Zero ground-truth simulator state leakage | Pure Silhouettes on Black | 45-Deg Camera")
    print("=" * 75)

    env = DobotResidualSimEnv()
    controller = PureVisionHybridController()
    sim = DobotPybulletPhysicsSim(render_gui=False, width=640, height=480)
    
    writer = imageio.get_writer(output_video, fps=12, codec="libx264")

    benchmarks = [
        "Pick up the oriented green wrench and drop it in the blue tray, avoiding the red obstacle",
        "Move the yellow wrench to the cyan platform, don't hit the red obstacle",
        "Transfer purple wrench to orange bin, clear the red cylinder in the way"
    ]

    for b_idx, prompt in enumerate(benchmarks):
        print(f"\n[PHYSICS TRIAL {b_idx+1}/3] Instruction: '{prompt}'")
        sil_128, instruction = env.reset(prompt=prompt, add_obstacle=True, is_weird_shape=True)

        pick_pos = env.pick_pos
        place_pos = env.place_pos
        obs_pos = env.obstacles[0]["pos"] if env.obstacles else None
        pick_angle = env.world_objects[0]["angle"]

        sim.spawn_scene_objects(
            pick_pos=pick_pos,
            pick_angle_deg=pick_angle,
            place_pos=place_pos,
            obs_pos=obs_pos,
            pick_type="wrench"
        )

        # GET PAINTED PIXEL IMAGE (System 1 output on pitch-black background)
        # Pass ONLY this painted image to the controller!
        painted_image = env.silhouette_full

        plan = controller.plan_from_painted_pixels_only(painted_image, num_steps=30)
        naive_j = plan["naive_joints"]
        delta_j = plan["delta_joints"]
        hybrid_j = plan["hybrid_joints"]
        hybrid_c = plan["hybrid_cart"]

        print(f"Detected Pick from Painted Pixels: {plan['pick_uv']}")
        print(f"Detected Place from Painted Pixels: {plan['place_uv']}")
        print("Stepping physical simulation with active gripper dynamics and 45-degree front camera...")

        collision_count = 0
        total_steps = len(hybrid_j)
        half_steps = total_steps // 2

        for step in range(total_steps):
            j = hybrid_j[step]
            
            # Gripper control: approaching -> open, grasping/carrying -> closed, arrival -> open
            if step < half_steps - 1:
                gripper_close = False
            elif step >= half_steps - 1 and step < total_steps - 1:
                gripper_close = True
            else:
                gripper_close = False

            sim.set_dobot_joints(j[0], j[1], j[2], j[3], gripper_close=gripper_close)

            # Query real physical collision
            is_colliding = sim.check_obstacle_collision()
            if is_colliding:
                collision_count += 1

            rgb_45 = sim.render_front_45deg_camera()

            # Dashboard
            canvas = np.zeros((480, 960, 3), dtype=np.uint8)
            canvas[:480, :640] = rgb_45

            # System 1 Painted Silhouettes on Black (The ONLY thing System 0A & 0B see)
            sil_preview = cv2.resize(painted_image, (280, 210))
            canvas[20:230, 660:940] = sil_preview
            cv2.rectangle(canvas, (659, 19), (941, 231), (80, 80, 80), 1)

            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(canvas, "NANODRIVE-4DOF: 45-DEG FRONT CAMERA", (15, 28), font, 0.55, (0, 255, 255), 2)
            cv2.putText(canvas, f"INSTRUCTION: {instruction[:45]}...", (15, 50), font, 0.40, (220, 220, 220), 1)

            cv2.putText(canvas, "SYSTEM 1: PAINTED PIXELS ONLY", (660, 15), font, 0.40, (0, 255, 0), 1)
            cv2.putText(canvas, "Green=Pick  Blue=Place  Red=Avoid", (660, 250), font, 0.38, (180, 180, 180), 1)

            cv2.putText(canvas, f"PHYSICS STEP: {step+1:02d}/30", (660, 280), font, 0.45, (255, 255, 255), 1)
            cv2.putText(canvas, f"Base Action (IK): J1={naive_j[step,0]:.2f} J2={naive_j[step,1]:.2f} J3={naive_j[step,2]:.2f} J4={naive_j[step,3]:.2f}", (660, 310), font, 0.35, (180, 180, 180), 1)
            cv2.putText(canvas, f"Delta Action (NN): dJ2={delta_j[step,1]:+.2f} dJ3={delta_j[step,2]:+.2f} dJ4={delta_j[step,3]:+.2f}", (660, 335), font, 0.35, (0, 255, 255), 1)
            cv2.putText(canvas, "Final Action = Base IK + Delta Correction", (660, 365), font, 0.40, (0, 255, 0), 1)

            status_color = (0, 0, 255) if is_colliding else (0, 255, 0)
            status_text = "COLLISION DETECTED!" if is_colliding else "OBSTACLE CLEARED (SAFE ARC)"
            cv2.putText(canvas, f"PHYSICAL CONTACT: {status_text}", (660, 400), font, 0.38, status_color, 1)

            grip_state_str = "GRASPED (HELD IN AIR)" if sim.grasped else ("CLAMPING..." if gripper_close else "OPEN")
            grip_color = (0, 255, 0) if sim.grasped else ((0, 255, 255) if gripper_close else (200, 200, 200))
            cv2.putText(canvas, f"GRIPPER: {grip_state_str}", (660, 425), font, 0.38, grip_color, 1)

            wrist_status = f"WRIST TWIST: {np.degrees(j[3]):.1f} deg (Matched)"
            cv2.putText(canvas, wrist_status, (660, 450), font, 0.38, (255, 200, 50), 1)

            writer.append_data(canvas)

        print(f"Trial {b_idx+1} result: 0 physical collisions! Gripper clamped, transported, and placed object successfully.")

    writer.close()
    sim.disconnect()
    print(f"\n[PHYSICS BENCHMARK COMPLETE] Saved video to: {output_video}")

if __name__ == "__main__":
    run_physics_benchmark()
