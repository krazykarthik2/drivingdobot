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
bowl_id = env.env.obj_body_id["akita_black_bowl_1"]
bowl_pos = env.env.sim.data.body_xpos[bowl_id].copy()

# Descend until finger pad Z reaches 0.990 (centered around bowl rim)
for step in range(50):
    curr_eef = obs["robot0_eef_pos"]
    pad_z = env.env.sim.data.geom_xpos[f1_pad][2]
    # We want pad_z -> 0.990, so diff_z = 0.990 - pad_z
    diff_xy = (bowl_pos[:2] - curr_eef[:2]) * 4.0
    diff_xy = np.clip(diff_xy, -0.4, 0.4)
    diff_z = (0.990 - pad_z) * 4.0
    diff_z = np.clip(diff_z, -0.3, 0.3)
    obs, _, _, _ = env.step(np.array([diff_xy[0], diff_xy[1], diff_z, 0, 0, 0, -1.0]))
    if step % 10 == 0:
        print(f"Step {step}: Pad Z = {pad_z:.4f}, Bowl Z = {env.env.sim.data.body_xpos[bowl_id][2]:.4f}")

print("Clamping at calibrated pad depth...")
for _ in range(30):
    obs, _, _, _ = env.step(np.array([0, 0, 0, 0, 0, 0, 1.0]))

print("Grip qpos after clamp:", obs["robot0_gripper_qpos"])

# Lift upward
for _ in range(35):
    obs, _, _, _ = env.step(np.array([0, 0, 0.5, 0, 0, 0, 1.0]))

b_lift = env.env.sim.data.body_xpos[bowl_id]
print("Bowl pos after lift:", b_lift)
print("Delta Z after lift:", b_lift[2] - bowl_pos[2])

env.close()
