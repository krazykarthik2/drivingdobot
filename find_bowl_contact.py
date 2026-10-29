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
plate_id = env.env.obj_body_id["plate_1"]
bowl_pos = env.env.sim.data.body_xpos[bowl_id].copy()
plate_pos = env.env.sim.data.body_xpos[plate_id].copy()

print("Initial Bowl pos:", bowl_pos)
print("Initial EEF pos:", obs["robot0_eef_pos"])

# Test step by step
for step in range(80):
    curr = obs["robot0_eef_pos"]
    target = np.array([bowl_pos[0], bowl_pos[1], bowl_pos[2] + 0.10 - step * 0.002])
    diff = (target - curr) * 5.0
    action = np.array([diff[0], diff[1], diff[2], 0, 0, 0, -1.0])
    obs, _, _, _ = env.step(action)
    has_contact = env.env.check_contact(env.env.get_object("akita_black_bowl_1"))
    if has_contact:
        print(f"Contact detected at step {step}: EEF Z = {obs['robot0_eef_pos'][2]:.4f}")
        break

env.close()
