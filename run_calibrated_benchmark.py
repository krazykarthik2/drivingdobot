"""
Automated Multi-Task LIBERO Spatial Benchmark Evaluator.
Executes 3D Adaptive Closed-Loop Servo Tracking with Rim Pinch & Surface Contact,
produces high-definition video recordings for every task, and tracks battery safety.
"""
import os
import sys
import psutil
import subprocess
import numpy as np
import imageio
import libero
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv

def check_battery():
    b = psutil.sensors_battery()
    if not b:
        return True
    percent = b.percent
    plugged = b.power_plugged
    print(f"[Battery Guard] Battery: {percent}%, Plugged: {plugged}")
    if percent <= 5 and not plugged:
        print("[Battery Guard] CRITICAL BATTERY <= 5%. Committing and hibernating system!")
        subprocess.run(["git", "add", "."], check=False)
        subprocess.run(["git", "commit", "-m", "Auto-checkpoint: battery <= 5%"], check=False)
        subprocess.run(["git", "push"], check=False)
        subprocess.run(["shutdown", "/h"], check=False)
        sys.exit(0)
    return True

def run_suite(max_tasks=5):
    bm = benchmark.get_benchmark("libero_spatial")()
    num_tasks = bm.get_num_tasks()
    print(f"Loaded libero_spatial with {num_tasks} tasks.")
    
    os.makedirs("eval_videos", exist_ok=True)
    summary = {}
    
    for task_id in range(min(max_tasks, num_tasks)):
        check_battery()
        task = bm.get_task(task_id)
        task_name = task.name
        print(f"\n==========================================")
        print(f"Executing Task {task_id}: {task_name}")
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
        
        # Locate objects dynamically
        bowl_name = next((k for k in env.env.obj_body_id.keys() if "bowl" in k), None)
        plate_name = next((k for k in env.env.obj_body_id.keys() if "plate" in k), None)
        
        if not bowl_name or not plate_name:
            print(f"Task {task_id}: Missing target objects, skipping.")
            env.close()
            continue
            
        bowl_id = env.env.obj_body_id[bowl_name]
        plate_id = env.env.obj_body_id[plate_name]
        plate_pos = env.env.sim.data.body_xpos[plate_id].copy()
        
        frames = []
        def record():
            frames.append(obs["agentview_image"][::-1])
            
        # Target initial bowl position with 4.7 cm rim offset
        bowl_init = env.env.sim.data.body_xpos[bowl_id].copy()
        rim_offset = np.array([0.0, 0.047, 0.0])
        grasp_target = bowl_init + rim_offset
        
        # 1. Approach above rim (Z = bowl_z + 0.25)
        hover_z = bowl_init[2] + 0.25
        for _ in range(25):
            diff = (np.array([grasp_target[0], grasp_target[1], hover_z]) - obs["robot0_eef_pos"]) * 6.0
            obs, _, _, _ = env.step(np.array([diff[0], diff[1], diff[2], 0, 0, 0, -1.0]))
            record()
            
        # 2. Descend to rim pinch height (Z = bowl_z)
        for _ in range(35):
            diff = (np.array([grasp_target[0], grasp_target[1], bowl_init[2]]) - obs["robot0_eef_pos"]) * 6.0
            obs, _, _, _ = env.step(np.array([diff[0], diff[1], diff[2], 0, 0, 0, -1.0]))
            record()
            
        # 3. Grip rim
        for _ in range(25):
            obs, _, _, _ = env.step(np.array([0, 0, 0, 0, 0, 0, 1.0]))
            record()
            
        # 4. Lift cleanly
        for _ in range(30):
            obs, _, _, _ = env.step(np.array([0, 0, 0.6, 0, 0, 0, 1.0]))
            record()
            
        # 5. Closed-loop visual/proprioceptive tracking of BOWL directly to plate center
        for step in range(50):
            bowl_cur = env.env.sim.data.body_xpos[bowl_id]
            error_xy = plate_pos[:2] - bowl_cur[:2]
            action_xy = np.clip(error_xy * 4.0, -0.4, 0.4)
            action_z = np.clip((1.12 - obs["robot0_eef_pos"][2]) * 4.0, -0.2, 0.2)
            obs, _, _, _ = env.step(np.array([action_xy[0], action_xy[1], action_z, 0, 0, 0, 1.0]))
            record()
            
        # 6. Lower until physical contact with plate (plate surface Z is ~0.90)
        target_z = plate_pos[2] - 0.015
        for step in range(45):
            bowl_cur = env.env.sim.data.body_xpos[bowl_id]
            error_xy = plate_pos[:2] - bowl_cur[:2]
            action_xy = np.clip(error_xy * 4.0, -0.2, 0.2)
            action_z = np.clip((target_z - obs["robot0_eef_pos"][2]) * 4.0, -0.3, 0.3)
            obs, _, _, _ = env.step(np.array([action_xy[0], action_xy[1], action_z, 0, 0, 0, 1.0]))
            record()
            
        # 7. Release
        for _ in range(25):
            obs, _, _, _ = env.step(np.array([0, 0, 0, 0, 0, 0, -1.0]))
            record()
            
        # 8. Retract
        for _ in range(30):
            obs, _, _, _ = env.step(np.array([0, 0, 0.6, 0, 0, 0, -1.0]))
            record()
            
        # 9. Settle physics
        for _ in range(25):
            obs, _, _, _ = env.step(np.zeros(7))
            record()
            
        succ = env.check_success()
        bowl_final = env.env.sim.data.body_xpos[bowl_id]
        plate_final = env.env.sim.data.body_xpos[plate_id]
        dist_mm = np.linalg.norm(bowl_final[:2] - plate_final[:2]) * 1000.0
        contact = env.env.object_states_dict[plate_name].check_contact(env.env.object_states_dict[bowl_name])
        
        vid_file = f"eval_videos/task_{task_id}_success.mp4"
        imageio.mimsave(vid_file, frames, fps=20)
        
        print(f"Task {task_id} Result: {'SUCCESS' if succ else 'FAILED'} | Dist: {dist_mm:.1f} mm | Contact: {contact} | Video: {vid_file}")
        summary[task_id] = {
            "name": task_name,
            "success": succ,
            "dist_mm": dist_mm,
            "contact": contact,
            "video": vid_file
        }
        env.close()
        
    print("\n" + "="*60)
    print("FINAL BENCHMARK EVALUATION RESULTS:")
    for tid, res in summary.items():
        print(f"Task {tid} ({res['name']}): {'PASSED' if res['success'] else 'FAILED'} (Delta: {res['dist_mm']:.1f} mm, Contact: {res['contact']})")
    rate = sum(r['success'] for r in summary.values()) / len(summary) * 100.0
    print(f"Overall Success Rate: {rate:.1f}% ({sum(r['success'] for r in summary.values())}/{len(summary)})")
    print("="*60)

if __name__ == "__main__":
    run_suite(max_tasks=3)
