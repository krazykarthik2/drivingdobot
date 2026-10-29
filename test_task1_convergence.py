import os
import numpy as np
import libero
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv

libero_path = os.path.dirname(libero.__file__)
b = benchmark.get_benchmark("libero_spatial")()
task = b.get_task(1)
bddl_file = os.path.join(libero_path, "libero", "bddl_files", task.problem_folder, task.bddl_file)

env = OffScreenRenderEnv(bddl_file_name=bddl_file, camera_widths=128, camera_heights=128, render_gpu_device_id=-1)
obs = env.reset()

for _ in range(15):
    obs, _, _, _ = env.step(np.zeros(7))

b_id = env.env.obj_body_id["akita_black_bowl_1"]
b_pos = env.env.sim.data.body_xpos[b_id].copy()
target = b_pos + np.array([0.045, 0.0, 0.015])

print("Target waypoint:", target)
print("Start EEF:", obs["robot0_eef_pos"])

for step in range(80):
    curr = env.env._get_observations()["robot0_eef_pos"]
    err = target - curr
    act = np.zeros(7)
    act[:3] = np.clip(err * 15.0, -1.0, 1.0)
    act[6] = -1.0
    env.step(act)
    if np.linalg.norm(err) < 0.005:
        print(f"Converged at step {step}!")
        break

final_eef = env.env._get_observations()["robot0_eef_pos"]
print("Final EEF:", final_eef)
print("Final Error:", np.linalg.norm(target - final_eef))

env.close()
