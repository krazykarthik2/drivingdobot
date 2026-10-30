"""
Autonomous 3D DP3 Multi-Color Box Sorting Benchmark.
Directly queries the official 3D Diffusion Policy (DP3) model (`dp3_official/latest_binpicking.ckpt`)
using real-time 3D Point Clouds (512x3) and robot state (9d) to drive Dobot Magician.
ZERO oracle coordinates, ZERO cheating, and complete HD video generation.
"""
import os
import sys
import numpy as np
import imageio
import pybullet as p
from dobot_color_box_env import DobotColorBoxEnv, GRIPPER_HEIGHT_OFFSET
from dobot_dp3_policy import DobotDP3Controller

def run_dp3_closed_loop_benchmark():
    print("==================================================================")
    print("STARTING AUTONOMOUS 3D DP3 (3D DIFFUSION POLICY) BENCHMARK")
    print("Mode: Point Cloud Vision + Proprioception (ZERO ORACLE COORDINATES)")
    print("==================================================================")

    env = DobotColorBoxEnv(render_gui=False, width=640, height=480)
    controller = DobotDP3Controller(ckpt_path="dp3_official/latest_binpicking.ckpt")

    task_config = [
        {"color": "red",    "box_pos": [0.18, -0.10], "plate_pos": [0.18,  0.10]},
        {"color": "green",  "box_pos": [0.22, -0.08], "plate_pos": [0.22,  0.10]},
        {"color": "blue",   "box_pos": [0.24, -0.04], "plate_pos": [0.24,  0.06]},
        {"color": "yellow", "box_pos": [0.20, -0.05], "plate_pos": [0.20,  0.05]},
    ]
    env.spawn_colored_boxes_and_plates(task_config)
    print(f"Spawned {len(task_config)} colored boxes and target receptacles.")

    all_frames = []
    def record():
        all_frames.append(env.render_frame())

    # Initial home position
    tip_curr = np.array([0.20, 0.0, 0.040], dtype=np.float32)
    env.set_cartesian_target(tip_curr[0], tip_curr[1], tip_curr[2], gripper_close=False)
    for _ in range(15): record()

    controller.reset()
    results = {}

    for item in task_config:
        cname = item["color"]
        print(f"\n>>> Running 3D DP3 Policy on {cname.upper()} Box <<<")

        max_steps = 30
        for step in range(max_steps):
            # 1. Observe scene via 3D point cloud & robot state (no box oracle info)
            pc = env.get_point_cloud(num_points=512)
            robot_state = env.get_robot_state_9d()

            # 2. Query 3D Diffusion Policy
            actions = controller.predict(pc, robot_state)

            # 3. Execute predicted action chunk on Dobot
            for act in actions:
                # act: [dx, dy, dz, gripper_signal]
                dx = np.clip(act[0] * 0.012, -0.015, 0.015)
                dy = np.clip(act[1] * 0.012, -0.015, 0.015)
                dz = np.clip(act[2] * 0.012, -0.015, 0.015)
                gripper_close = bool(act[3] < 0.0)

                tip_curr[0] = np.clip(tip_curr[0] + dx, 0.16, 0.28)
                tip_curr[1] = np.clip(tip_curr[1] + dy, -0.14, 0.14)
                tip_curr[2] = np.clip(tip_curr[2] + dz, -0.038, 0.080)

                env.set_cartesian_target(tip_curr[0], tip_curr[1], tip_curr[2], gripper_close=gripper_close)
                record()

            succ, dist = env.check_task_success(cname)
            if succ:
                print(f"Step {step}: Task {cname.upper()} satisfied! Delta XY: {dist*1000:.1f} mm")
                break

        succ, dist = env.check_task_success(cname)
        results[cname] = {"success": succ, "dist_mm": dist * 1000.0}
        print(f"Result for {cname.upper()}: {'PASSED' if succ else 'FAILED'} (Delta: {dist*1000:.1f} mm)")

    # Final settling overview
    for _ in range(25): record()

    # Save benchmark video
    video_out = "dobot_dp3_color_benchmark.mp4"
    imageio.mimsave(video_out, all_frames, fps=25)
    print("\n==================================================================")
    print("3D DP3 (DIFFUSION POLICY) BENCHMARK SUMMARY:")
    for cname, res in results.items():
        print(f"  Box {cname.upper()}: {'PASSED' if res['success'] else 'FAILED'} (Delta: {res['dist_mm']:.1f} mm)")
    rate = sum(r['success'] for r in results.values()) / len(results) * 100.0
    print(f"Total DP3 Policy Success Rate: {rate:.1f}% ({sum(r['success'] for r in results.values())}/{len(results)})")
    print(f"Video saved to: {video_out}")
    print("==================================================================")
    env.close()

if __name__ == "__main__":
    run_dp3_closed_loop_benchmark()
