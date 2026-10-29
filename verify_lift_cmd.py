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

f1_pad = env.env.sim.model.geom_name2id("gripper0_finger1_pad_collision")
f2_pad = env.env.sim.model.geom_name2id("gripper0_finger2_pad_collision")
env.env.sim.model.geom_friction[f1_pad][0] = 5.0
env.env.sim.model.geom_friction[f2_pad][0] = 5.0

bowl_id = env.env.obj_body_id["akita_black_bowl_1"]
bowl_pos = env.env.sim.data.body_xpos[bowl_id].copy()

# Descend to 0.920
for _ in range(40):
    curr = obs["robot0_eef_pos"]
    diff = (np.array([bowl_pos[0], bowl_pos[1], 0.920]) - curr) * 5.0
    diff = np.clip(diff, -0.5, 0.5)
    obs, _, _, _ = env.step(np.array([diff[0], diff[1], diff[2], 0, 0, 0, -1.0]))

for _ in range(30):
    obs, _, _, _ = env.step(np.array([0, 0, 0, 0, 0, 0, 1.0]))

# Check lift with [0, 0, 0.5] vs target_diff
for step in range(35):
    obs, _, _, _ = env.step(np.array([0, 0, 0.5, 0, 0, 0, 1.0]))
    if step % 10 == 0:
        b = env.env.sim.data.body_xpos[bowl_id]
        print(f"Lift step {step}: Bowl Z = {b[2]:.4f}")

curr_b = env.env.sim.data.body_xpos[bowl_id]
print("Final Bowl Z after pure [0, 0, 0.5]:", curr_b[2])

env.close()
