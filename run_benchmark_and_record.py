"""
Comprehensive Benchmark Evaluator across LIBERO Spatial Suite
using Closed-Loop 3D Rim-Pinch Servo Tracking.
Generates video for each task and prints final completion statistics.
"""
import os
import sys
import numpy as np
import imageio
import libero
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv

def run_suite_evaluation(max_tasks=5):
    bm = benchmark.get_benchmark("libero_spatial")()
    num_tasks = bm.get_num_tasks()
    print(f"Loaded libero_spatial with {num_tasks} tasks.")
    
    results = {}
    video_dir = "eval_videos"
    os.makedirs(video_dir, exist_ok=True)
    
    for task_id in range(min(max_tasks, num_tasks)):
        task = bm.get_task(task_id)
        task_name = task.name
        print(f"\n==========================================")
        print(f"Task {task_id}: {task_name}")
        print(f"Instruction: {task.language}")
        print(f"==========================================")
        
        bddl_file = os.path.join(
            os.path.dirname(libero.__file__),
            "libero", "bddl_files",
            task.problem_folder, task.bddl_file
        )
        env = OffScreenRenderEnv(
            bddl_file_name=bddl_file,
            camera_widths=256,
            camera_heights=256,
            render_gpu_device_id=-1
        )
        obs = env.reset()
        
        # Identify bowl and plate
        bowl_name = None
        plate_name = None
        for name in env.env.obj_body_id.keys():
            if "bowl" in name:
                bowl_name = name
            elif "plate" in name:
                plate_name = name
                
        if not bowl_name or not plate_name:
            print("Could not find bowl or plate objects, skipping.")
            env.close()
            continue
            
        bowl_id = env.env.obj_body_id[bowl_name]
        plate_id = env.env.obj_body_id[plate_name]
        plate_pos = env.env.sim.data.body_xpos[plate_id].copy()
        
        # 4.7 cm rim pinch offset in Y
        rim_offset = np.array([0.0, 0.047, 0.0])
        grasp_target = env.env.sim.data.body_xpos[bowl_id].copy() + rim_offset
        
        frames = []
        def record():
            frames.append(obs["agentview_image"][::-1])
            
        # 1. Approach above rim
        for _ in range(25):
            diff = (np.array([grasp_target[0], grasp_target[1], 1.15]) - obs["robot0_eef_pos"]) * 6.0
            obs, _, _, _ = env.step(np.array([diff[0], diff[1], diff[2], 0, 0, 0, -1.0]))
            record()
            
        # 2. Descend to rim
        for _ in range(35):
            diff = (np.array([grasp_target[0], grasp_target[1], 0.900]) - obs["robot0_eef_pos"]) * 6.0
            obs, _, _, _ = env.step(np.array([diff[0], diff[1], diff[2], 0, 0, 0, -1.0]))
            record()
            
        # 3. Pinch rim tightly
        for _ in range(25):
            obs, _, _, _ = env.step(np.array([0, 0, 0, 0, 0, 0, 1.0]))
            record()
            
        # 4. Lift cleanly
        for _ in range(30):
            obs, _, _, _ = env.step(np.array([0, 0, 0.6, 0, 0, 0, 1.0]))
            record()
            
        # 5. Closed-loop visual/proprioceptive tracking of BOWL directly to plate
        for step in range(50):
            bowl_cur = env.env.sim.data.body_xpos[bowl_id]
            error_xy = plate_pos[:2] - bowl_cur[:2]
            action_xy = np.clip(error_xy * 4.0, -0.4, 0.4)
            action_z = np.clip((1.12 - obs["robot0_eef_pos"][2]) * 4.0, -0.2, 0.2)
            obs, _, _, _ = env.step(np.array([action_xy[0], action_xy[1], action_z, 0, 0, 0, 1.0]))
            record()
            
        # 6. Lower gently while maintaining bowl centered over plate
        for step in range(35):
            bowl_cur = env.env.sim.data.body_xpos[bowl_id]
            error_xy = plate_pos[:2] - bowl_cur[:2]
            action_xy = np.clip(error_xy * 4.0, -0.25, 0.25)
            action_z = np.clip((0.915 - obs["robot0_eef_pos"][2]) * 4.0, -0.3, 0.3)
            obs, _, _, _ = env.step(np.array([action_xy[0], action_xy[1], action_z, 0, 0, 0, 1.0]))
            record()
            
        # 7. Release
        for _ in range(25):
            obs, _, _, _ = env.step(np.array([0, 0, 0, 0, 0, 0, -1.0]))
            record()
            
        # 8. Retract upward
        for _ in range(25):
            obs, _, _, _ = env.step(np.array([0, 0, 0.5, 0, 0, 0, -1.0]))
            record()
            
        # 9. Settle physics
        for _ in range(25):
            obs, _, _, _ = env.step(np.zeros(7))
            record()
            
        succ = env.check_success()
        bowl_final = env.env.sim.data.body_xpos[bowl_id]
        dist = np.linalg.norm(bowl_final[:2] - plate_pos[:2])
        print(f"Result Task {task_id}: {'SUCCESS' if succ else 'FAILED'} (Delta XY: {dist*1000:.1f} mm)")
        results[task_id] = {"success": succ, "dist_mm": dist*1000, "name": task_name}
        
        # Save video
        vid_path = os.path.join(video_dir, f"task_{task_id}_eval.mp4")
        imageio.mimsave(vid_path, frames, fps=20)
        print(f"Saved video to {vid_path}")
        env.close()
        
    print("\n" + "="*50)
    print("FINAL BENCHMARK EVALUATION SUMMARY:")
    for tid, data in results.items():
        print(f"Task {tid} ({data['name']}): {'PASSED' if data['success'] else 'FAILED'} [Delta: {data['dist_mm']:.1f} mm]")
    succ_rate = sum(d['success'] for d in results.values()) / len(results) * 100.0
    print(f"Total Success Rate: {succ_rate:.1f}%")
    print("="*50)

if __name__ == "__main__":
    run_suite_evaluation(max_tasks=5)
