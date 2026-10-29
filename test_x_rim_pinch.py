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

for _ in range(15):
    obs, _, _, _ = env.step(np.zeros(7))

bowl_id = env.env.obj_body_id["akita_black_bowl_1"]
plate_id = env.env.obj_body_id["plate_1"]

bowl_pos = env.env.sim.data.body_xpos[bowl_id].copy()
plate_pos = env.env.sim.data.body_xpos[plate_id].copy()

# Offset along X-axis: bowl_center_x + 0.045
rim_target = np.array([bowl_pos[0] + 0.045, bowl_pos[1], 0.940])
print("Approaching X-axis rim target:", rim_target)

# 1. Approach above X-axis rim
for _ in range(30):
    curr_eef = obs["robot0_eef_pos"]
    diff = (np.array([rim_target[0], rim_target[1], 1.15]) - curr_eef) * 4.0
    diff = np.clip(diff, -0.4, 0.4)
    obs, _, _, _ = env.step(np.array([diff[0], diff[1], diff[2], 0, 0, 0, -1.0]))

# 2. Descend straddling rim along Y axis
for _ in range(40):
    curr_eef = obs["robot0_eef_pos"]
    diff = (rim_target - curr_eef) * 4.0
    diff = np.clip(diff, -0.4, 0.4)
    obs, _, _, _ = env.step(np.array([diff[0], diff[1], diff[2], 0, 0, 0, -1.0]))

print("Gripper qpos before clamp:", obs["robot0_gripper_qpos"])

# 3. Clamp pinch on rim
for _ in range(30):
    obs, _, _, _ = env.step(np.array([0, 0, 0, 0, 0, 0, 1.0]))

print("Gripper qpos after clamp:", obs["robot0_gripper_qpos"])

# 4. Lift vertically
for _ in range(35):
    obs, _, _, _ = env.step(np.array([0, 0, 0.5, 0, 0, 0, 1.0]))

bowl_lift = env.env.sim.data.body_xpos[bowl_id]
print("Bowl pos after rim lift:", bowl_lift)
print("Delta Z after rim lift:", bowl_lift[2] - bowl_pos[2])

env.close()
