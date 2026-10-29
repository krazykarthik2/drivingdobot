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

# Approach from behind bowl (X - 0.05m, Z = table + 0.01m)
print("Initial bowl:", bowl_pos)
print("Initial plate:", plate_pos)

# 1. Position behind bowl
start_scoop = np.array([bowl_pos[0] - 0.06, bowl_pos[1], 0.905])
for _ in range(35):
    curr_eef = obs["robot0_eef_pos"]
    diff = (start_scoop - curr_eef) * 4.0
    diff = np.clip(diff, -0.4, 0.4)
    obs, _, _, _ = env.step(np.array([diff[0], diff[1], diff[2], 0, 0, 0, 1.0])) # keep closed

# 2. Push / sweep horizontally across the table directly toward plate
for step in range(70):
    curr_eef = obs["robot0_eef_pos"]
    curr_bowl = env.env.sim.data.body_xpos[bowl_id]
    # Steer from curr_bowl towards plate center
    diff_target = plate_pos[:2] - curr_bowl[:2]
    dist = np.linalg.norm(diff_target)
    if dist < 0.025:
        print(f"Bowl reached plate at push step {step}!")
        break
    push_dir = diff_target / (dist + 1e-6)
    push_target = curr_bowl[:2] - 0.04 * push_dir # stay behind bowl
    
    diff_xy = (push_target - curr_eef[:2]) * 4.0
    diff_xy = np.clip(diff_xy, -0.25, 0.25)
    # push along direction
    action_xy = diff_xy + 0.25 * push_dir
    obs, _, _, _ = env.step(np.array([action_xy[0], action_xy[1], 0.0, 0, 0, 0, 1.0]))

# 3. Retract upward and settle
for _ in range(30):
    obs, _, _, _ = env.step(np.array([0, 0, 0.3, 0, 0, 0, -1.0]))

for _ in range(30):
    obs, _, _, _ = env.step(np.zeros(7))
    if env.check_success():
        print("*** SUCCESS ACHIEVED VIA FLUSH SWEEP! ***")
        break

print("Task 1 Final Success:", env.check_success())
bowl_end = env.env.sim.data.body_xpos[bowl_id]
plate_end = env.env.sim.data.body_xpos[plate_id]
print("Final Bowl:", bowl_end)
print("Final Plate:", plate_end)
print("Final XY distance:", np.linalg.norm(bowl_end[:2] - plate_end[:2]))

env.close()
