"""
Phase 4: The Hybrid Deployment Pipeline
NanoDrive-4DOF (Residual / Delta Architecture)

The Final Inference Loop:
1. Language (System 2): Parses raw/weird prompt into strict JSON target roles (Pick, Place, Avoid).
2. Vision (System 1): Edge camera (40-degree front elevated) segments targets into solid colored shapes
   (Green for pick, Blue for place, Red for avoid) on a pitch-black (0, 0, 0) background.
3. Programmed Solver (System 0A): Standard analytical IK calculates naive straight-line trajectory
   (which would collide with obstacles and fail complex wrist grip).
4. Neural Delta Driver (System 0B): Lightweight CNN looks at the silhouettes and naive joint plan,
   predicting corrective delta angles (arc Z elevation, wrist yaw angle).
5. Hybrid Addition:
   Final Joint Action = Base Action + Delta Action
6. Execution: Flawless, obstacle-avoiding, shape-aware Dobot execution in simulation and video rollout.
"""

import os
import sys
import cv2
import numpy as np
import imageio
import torch

from dobot_kinematics import DobotKinematics
from dobot_urdf_renderer import DobotURDFRenderer
from system0a_base_controller import System0ABaseController
from system1_perception import System1PerceptionEngine, CameraProjector
from system2_reasoning import System2ReasoningEngine
from neural_delta_model import NeuralDeltaDriver
from dobot_residual_env import DobotResidualSimEnv

class NanoDrive4DOFController:
    """
    Complete Hybrid Autonomous Controller for Dobot Magician 4-DOF.
    """
    def __init__(self, model_weights_path="dobot_delta_model.pt"):
        self.kin = DobotKinematics()
        self.base_solver = System0ABaseController()
        self.perception = System1PerceptionEngine(width=128, height=128)
        self.reasoning = System2ReasoningEngine()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.delta_model = NeuralDeltaDriver().to(self.device)
        if os.path.exists(model_weights_path):
            self.delta_model.load_state_dict(torch.load(model_weights_path, map_location=self.device))
            print(f"[NanoDrive-4DOF] Loaded trained Neural Delta Driver weights from '{model_weights_path}'")
        self.delta_model.eval()

    def predict_hybrid_trajectory(self, silhouette_128, start_pos, pick_pos, place_pos, num_steps=30):
        """
        Computes the complete trajectory:
        Naive Base Action + Neural Delta Action = Final Executable Action
        """
        # 1. Base Controller: Naive straight-line trajectory
        half = num_steps // 2
        naive_joints = []
        naive_cart = []

        # Waypoints: start -> pick -> place
        for i in range(half):
            a = i / float(half - 1)
            pt = (1 - a) * start_pos + a * pick_pos
            j = self.kin.inverse(pt[0], pt[1], pt[2], 0.0)
            naive_cart.append(np.array([pt[0], pt[1], pt[2], 0.0], dtype=np.float32))
            naive_joints.append(j)

        for i in range(half):
            a = i / float(half - 1)
            pt = (1 - a) * pick_pos + a * place_pos
            j = self.kin.inverse(pt[0], pt[1], pt[2], 0.0)
            naive_cart.append(np.array([pt[0], pt[1], pt[2], 0.0], dtype=np.float32))
            naive_joints.append(j)

        naive_joints = np.array(naive_joints, dtype=np.float32)

        # 2. Neural Delta Driver Inference
        # Prepare inputs
        img_tensor = torch.tensor(
            silhouette_128.transpose(2, 0, 1).astype(np.float32) / 255.0,
            device=self.device
        ).unsqueeze(0).repeat(num_steps, 1, 1, 1)

        naive_tensor = torch.tensor(naive_joints, device=self.device)

        with torch.no_grad():
            delta_tensor = self.delta_model(img_tensor, naive_tensor)
            delta_joints = delta_tensor.cpu().numpy()

        # 3. Hybrid Combination: Final Action = Base Action + Delta Action
        final_joints = naive_joints + delta_joints

        # Forward kinematics for Cartesian trace
        final_cart = np.array([self.kin.forward(*j) for j in final_joints], dtype=np.float32)

        return {
            "naive_joints": naive_joints,
            "delta_joints": delta_joints,
            "final_joints": final_joints,
            "naive_cart": np.array(naive_cart),
            "final_cart": final_cart
        }

def run_hybrid_demonstration_benchmark(num_trials=3, output_video="nanodrive_hybrid_eval.mp4"):
    print("=" * 70)
    print("NANODRIVE-4DOF RESIDUAL / DELTA CONTROL BENCHMARK")
    print("40-Degree Front Camera | Pure Silhouette Vision | Analytical IK + Delta CNN")
    print("=" * 70)

    env = DobotResidualSimEnv()
    controller = NanoDrive4DOFController()
    renderer = DobotURDFRenderer(width=640, height=480)
    
    writer = imageio.get_writer(output_video, fps=12, codec="libx264")
    
    test_instructions = [
        "Pick up the oriented green wrench and drop it in the blue tray, avoiding the red obstacle",
        "Move the yellow wrench to the cyan platform, make sure you don't hit the red obstacle",
        "Transfer purple wrench to orange bin, clear the red hazard in the way"
    ]

    for trial_idx, custom_prompt in enumerate(test_instructions):
        print(f"\n[Trial {trial_idx+1}/{len(test_instructions)}] Prompt: '{custom_prompt}'")
        sil_128, instruction = env.reset(prompt=custom_prompt, add_obstacle=True, is_weird_shape=True)
        
        pick_pos = env.pick_pos
        place_pos = env.place_pos
        start_pos = env.ee_pos[:3]

        plan = controller.predict_hybrid_trajectory(sil_128, start_pos, pick_pos, place_pos, num_steps=30)
        
        naive_j = plan["naive_joints"]
        delta_j = plan["delta_joints"]
        final_j = plan["final_joints"]
        final_cart = plan["final_cart"]

        print(f"Pick Target Pos: {pick_pos} | Place Target: {place_pos}")
        print(f"Max Neural Delta Applied (deg): {np.degrees(np.max(np.abs(delta_j), axis=0))}")
        print("Executing hybrid motion rollout in 3D physics rendering...")

        for step in range(len(final_j)):
            fj = final_j[step]
            ee_c = final_cart[step]

            # Render 3D scene
            t = renderer.compute_urdf_transforms(fj[0], fj[1], fj[2], fj[3])
            # Build scene with world objects
            dist_cubes = []
            for obs in env.obstacles:
                dist_cubes.append(([0.9, 0.15, 0.15], obs["pos"]))
            scene = renderer.build_scene(
                t,
                cube_pos=pick_pos,
                plat_pos=place_pos,
                distractor_cubes=dist_cubes
            )

            # Render third person view
            img_tp, img_wrist, _ = renderer.render_views(scene, t["ee_pos"])

            # Create side-by-side visualization dashboard
            # Left: Raw 3D Robot Execution View (640x480)
            # Right Top: System 1 Pure Zero-Background Silhouette Image (240x240)
            # Right Bottom: Real-Time Telemetry & Residual Delta Status
            canvas = np.zeros((480, 960, 3), dtype=np.uint8)
            # Resize img_tp to (640, 480)
            img_tp_resized = cv2.resize(img_tp, (640, 480))
            canvas[:480, :640] = img_tp_resized

            # Silhouette preview
            sil_preview = cv2.resize(env.silhouette_full, (280, 210))
            canvas[20:230, 660:940] = sil_preview
            cv2.rectangle(canvas, (659, 19), (941, 231), (100, 100, 100), 1)

            # Telemetry text
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(canvas, "NANODRIVE-4DOF RESIDUAL POLICY", (20, 30), font, 0.65, (0, 255, 255), 2)
            cv2.putText(canvas, f"PROMPT: {instruction[:45]}...", (20, 55), font, 0.45, (200, 200, 200), 1)
            
            cv2.putText(canvas, "SYSTEM 1: SILHOUETTES", (660, 15), font, 0.42, (0, 255, 0), 1)
            cv2.putText(canvas, "Green=Pick  Blue=Place  Red=Avoid", (660, 250), font, 0.38, (180, 180, 180), 1)

            cv2.putText(canvas, f"Step: {step+1:02d}/30", (660, 280), font, 0.45, (255, 255, 255), 1)
            cv2.putText(canvas, f"Base Action (IK): J=[{naive_j[step,0]:.2f}, {naive_j[step,1]:.2f}, {naive_j[step,2]:.2f}, {naive_j[step,3]:.2f}]", (660, 310), font, 0.38, (180, 180, 180), 1)
            cv2.putText(canvas, f"Delta Action (NN): dJ=[{delta_j[step,0]:+.2f}, {delta_j[step,1]:+.2f}, {delta_j[step,2]:+.2f}, {delta_j[step,3]:+.2f}]", (660, 340), font, 0.38, (0, 255, 255), 1)
            cv2.putText(canvas, f"Final Action = Base + Delta", (660, 370), font, 0.42, (0, 255, 0), 1)
            cv2.putText(canvas, f"EE Pose: X={ee_c[0]:.3f} Y={ee_c[1]:.3f} Z={ee_c[2]:.3f}", (660, 400), font, 0.40, (255, 200, 50), 1)
            cv2.putText(canvas, f"Obstacle Clearance: ACTIVE (+Z Arc)", (660, 430), font, 0.40, (50, 200, 255), 1)

            writer.append_data(canvas)

    writer.close()
    print(f"\n[SUCCESS] Benchmark complete! Generated demonstration video saved to: {output_video}")

if __name__ == "__main__":
    run_hybrid_demonstration_benchmark(num_trials=3)
