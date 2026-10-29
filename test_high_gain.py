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

target = np.array([0.0, 0.20, 1.05])
for step in range(80):
    curr = obs["robot0_eef_pos"]
    action = np.zeros(7)
    # Higher proportional gain + velocity feedforward
    action[:3] = np.clip((target - curr) * 12.0, -1.0, 1.0)
    action[6] = -1.0
    obs, _, _, _ = env.step(action)
    if step % 15 == 0:
        err = np.linalg.norm(target - obs["robot0_eef_pos"])
        print(f"Step {step:2d}: EEF={obs['robot0_eef_pos']} | Error={err:.4f}m")

err_final = np.linalg.norm(target - obs["robot0_eef_pos"])
print(f"Final error at step 80: {err_final:.4f}m")

env.close()
