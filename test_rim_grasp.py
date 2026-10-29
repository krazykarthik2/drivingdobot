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

# Descend to rim height Z = 1.010 with fingers wide open
for _ in range(40):
    curr_eef = obs["robot0_eef_pos"]
    diff = (np.array([bowl_pos[0], bowl_pos[1], 1.010]) - curr_eef) * 4.0
    diff = np.clip(diff, -0.4, 0.4)
    obs, _, _, _ = env.step(np.array([diff[0], diff[1], diff[2], 0, 0, 0, -1.0]))

print("EEF pos at 1.010:", obs["robot0_eef_pos"])
print("Bowl pos at 1.010:", env.env.sim.data.body_xpos[bowl_id])
print("Gripper qpos:", obs["robot0_gripper_qpos"])

# Clamp gripper
for _ in range(30):
    obs, _, _, _ = env.step(np.array([0, 0, 0, 0, 0, 0, 1.0]))

print("Gripper qpos after clamp:", obs["robot0_gripper_qpos"])

# Lift upward
for _ in range(35):
    obs, _, _, _ = env.step(np.array([0, 0, 0.5, 0, 0, 0, 1.0]))

b_lift = env.env.sim.data.body_xpos[bowl_id]
print("Bowl pos after lift:", b_lift)
print("Delta Z after lift:", b_lift[2] - bowl_pos[2])

env.close()
