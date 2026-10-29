import os
import numpy as np
import libero
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv

libero_path = os.path.dirname(libero.__file__)
b = benchmark.get_benchmark("libero_spatial")()
task = b.get_task(0)
bddl_file = os.path.join(libero_path, "libero", "bddl_files", task.problem_folder, task.bddl_file)
env = OffScreenRenderEnv(bddl_file_name=bddl_file, camera_widths=128, camera_heights=128, render_gpu_device_id=-1)
obs = env.reset()

bowl_id = env.env.obj_body_id["akita_black_bowl_1"]
bowl_pos = env.env.sim.data.body_xpos[bowl_id].copy()

print("Initial bowl pos:", bowl_pos)

# Test descending step-by-step from 1.05 down to 0.90, checking when bowl moves:
for step in range(60):
    curr_eef = obs["robot0_eef_pos"]
    target_z = 1.05 - step * 0.0025
    diff = (np.array([bowl_pos[0], bowl_pos[1], target_z]) - curr_eef) * 4.0
    diff = np.clip(diff, -0.4, 0.4)
    obs, _, _, _ = env.step(np.array([diff[0], diff[1], diff[2], 0, 0, 0, -1.0]))
    curr_bowl = env.env.sim.data.body_xpos[bowl_id]
    if np.linalg.norm(curr_bowl[:2] - bowl_pos[:2]) > 0.01:
        print(f"Bowl was pushed at step {step}! Target Z = {target_z:.4f}, EEF Z = {obs['robot0_eef_pos'][2]:.4f}")
        break

env.close()
