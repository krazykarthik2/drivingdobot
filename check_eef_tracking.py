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

print("Testing direct end-effector pose tracking with proportional gains...")
bowl_id = env.env.obj_body_id["akita_black_bowl_1"]
plate_id = env.env.obj_body_id["plate_1"]

bowl_pos = env.env.sim.data.body_xpos[bowl_id].copy()
plate_pos = env.env.sim.data.body_xpos[plate_id].copy()

# Move EEF to [0.0, 0.20, 1.05]
for step in range(30):
    curr = obs["robot0_eef_pos"]
    action = np.zeros(7)
    action[:3] = np.clip((np.array([0.0, 0.20, 1.05]) - curr) * 5.0, -0.5, 0.5)
    action[6] = -1.0
    obs, _, _, _ = env.step(action)
    if step % 10 == 0:
        print(f"Step {step}: Target=[0.0, 0.20, 1.05] | Current EEF={obs['robot0_eef_pos']}")

env.close()
