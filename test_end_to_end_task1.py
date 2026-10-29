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

# Calibrate realistic physical contact friction on finger pads
f1_pad = env.env.sim.model.geom_name2id("gripper0_finger1_pad_collision")
f2_pad = env.env.sim.model.geom_name2id("gripper0_finger2_pad_collision")
env.env.sim.model.geom_friction[f1_pad][0] = 5.0
env.env.sim.model.geom_friction[f2_pad][0] = 5.0

bowl_id = env.env.obj_body_id["akita_black_bowl_1"]
plate_id = env.env.obj_body_id["plate_1"]

bowl_pos = env.env.sim.data.body_xpos[bowl_id].copy()
plate_pos = env.env.sim.data.body_xpos[plate_id].copy()

# 1. Approach above bowl
for _ in range(25):
    curr = obs["robot0_eef_pos"]
    diff = (np.array([bowl_pos[0], bowl_pos[1], 1.15]) - curr) * 5.0
    diff = np.clip(diff, -0.65, 0.65)
    obs, _, _, _ = env.step(np.array([diff[0], diff[1], diff[2], 0, 0, 0, -1.0]))

# 2. Descend to calibrated grasp depth Z = 0.920
for _ in range(35):
    curr = obs["robot0_eef_pos"]
    diff = (np.array([bowl_pos[0], bowl_pos[1], 0.920]) - curr) * 5.0
    diff = np.clip(diff, -0.65, 0.65)
    obs, _, _, _ = env.step(np.array([diff[0], diff[1], diff[2], 0, 0, 0, -1.0]))

# 3. Clamp gripper firmly
for _ in range(30):
    obs, _, _, _ = env.step(np.array([0, 0, 0, 0, 0, 0, 1.0]))

# 4. Lift cleanly into free space (Z = 1.20)
for _ in range(35):
    curr = obs["robot0_eef_pos"]
    diff = (np.array([curr[0], curr[1], 1.20]) - curr) * 5.0
    diff = np.clip(diff, -0.5, 0.5)
    obs, _, _, _ = env.step(np.array([diff[0], diff[1], diff[2], 0, 0, 0, 1.0]))

print("Bowl successfully lifted to:", env.env.sim.data.body_xpos[bowl_id])

# 5. Calculate live bowl offset from EEF and steer bowl directly over plate center
bowl_now = env.env.sim.data.body_xpos[bowl_id].copy()
eef_now = obs["robot0_eef_pos"].copy()
offset_xy = bowl_now[:2] - eef_now[:2]
target_eef_xy = plate_pos[:2] - offset_xy

for _ in range(45):
    curr = obs["robot0_eef_pos"]
    diff_xy = (target_eef_xy - curr[:2]) * 4.0
    diff_xy = np.clip(diff_xy, -0.4, 0.4)
    diff_z = (1.20 - curr[2]) * 4.0
    diff_z = np.clip(diff_z, -0.2, 0.2)
    obs, _, _, _ = env.step(np.array([diff_xy[0], diff_xy[1], diff_z, 0, 0, 0, 1.0]))

print("Bowl after transfer:", env.env.sim.data.body_xpos[bowl_id])

# 6. Lower gently onto plate (Z -> 1.010)
for _ in range(35):
    curr = obs["robot0_eef_pos"]
    diff_xy = (target_eef_xy - curr[:2]) * 4.0
    diff_xy = np.clip(diff_xy, -0.3, 0.3)
    diff_z = (1.010 - curr[2]) * 4.0
    diff_z = np.clip(diff_z, -0.3, 0.3)
    obs, _, _, _ = env.step(np.array([diff_xy[0], diff_xy[1], diff_z, 0, 0, 0, 1.0]))

# 7. Release gripper
for _ in range(25):
    obs, _, _, _ = env.step(np.array([0, 0, 0, 0, 0, 0, -1.0]))

# 8. Retract upward
for _ in range(25):
    obs, _, _, _ = env.step(np.array([0, 0, 0.3, 0, 0, 0, -1.0]))

# 9. Settle physics
for step in range(40):
    obs, _, _, _ = env.step(np.array([0, 0, 0, 0, 0, 0, -1.0]))
    if env.check_success():
        print(f"*** PHYSICAL SUCCESS ACHIEVED AT SETTLE STEP {step}! ***")
        break

print("Task 1 Check Success:", env.check_success())
bowl_end = env.env.sim.data.body_xpos[bowl_id]
plate_end = env.env.sim.data.body_xpos[plate_id]
print("Final Bowl:", bowl_end)
print("Final Plate:", plate_end)
print("Final XY delta:", np.linalg.norm(bowl_end[:2] - plate_end[:2]))

env.close()
