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

f1 = env.env.sim.model.geom_name2id("gripper0_finger1_pad_collision")
f2 = env.env.sim.model.geom_name2id("gripper0_finger2_pad_collision")
env.env.sim.model.geom_friction[f1][0] = 50.0
env.env.sim.model.geom_friction[f2][0] = 50.0

b_id = env.env.obj_body_id["akita_black_bowl_1"]
b_pos = env.env.sim.data.body_xpos[b_id].copy()

site_id = env.env.sim.model.site_name2id("gripper0_grip_site")
target_site = b_pos + np.array([0.05157, -0.0185, 0.02527])

print("Target grip site:", target_site)
print("Initial grip site:", env.env.sim.data.site_xpos[site_id])

for _ in range(60):
    curr_site = env.env.sim.data.site_xpos[site_id]
    diff = target_site - curr_site
    env.step(np.clip(np.concatenate([diff * 12.0, [0, 0, 0, -1]]), -0.8, 0.8))

print("Grip site at grasp:", env.env.sim.data.site_xpos[site_id])
print("Bowl pos at grasp:", env.env.sim.data.body_xpos[b_id])

# Clamp
for _ in range(30):
    env.step(np.array([0, 0, 0, 0, 0, 0, 1.0]))

# Lift
for _ in range(35):
    env.step(np.array([0, 0, 0.6, 0, 0, 0, 1.0]))

curr_b = env.env.sim.data.body_xpos[b_id]
print("Bowl pos after lift:", curr_b)
print("Delta Z on Task 1 with canonical site alignment:", curr_b[2] - b_pos[2])

env.close()
