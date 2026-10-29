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
print("Initial EEF pos:", obs["robot0_eef_pos"])

# Test descending until EEF Z reaches 0.92m
for step in range(50):
    curr_eef = obs["robot0_eef_pos"]
    target = np.array([bowl_pos[0], bowl_pos[1], 0.92])
    diff = (target - curr_eef) * 4.0
    action = np.array([diff[0], diff[1], diff[2], 0, 0, 0, -1.0])
    obs, _, _, _ = env.step(action)
    if step % 10 == 0:
        print(f"Step {step}: EEF pos = {obs['robot0_eef_pos']}, Gripper qpos = {obs['robot0_gripper_qpos']}")

# Clamp gripper
print("Clamping gripper...")
for step in range(30):
    obs, _, _, _ = env.step(np.array([0, 0, 0, 0, 0, 0, 1.0]))
print("Gripper qpos after clamp:", obs['robot0_gripper_qpos'])

# Lift
print("Lifting...")
for step in range(40):
    obs, _, _, _ = env.step(np.array([0, 0, 0.5, 0, 0, 0, 1.0]))

bowl_lifted = env.env.sim.data.body_xpos[bowl_id]
print("Bowl pos after lift:", bowl_lifted)
print("Bowl Z lifted by:", bowl_lifted[2] - bowl_pos[2])
env.close()
