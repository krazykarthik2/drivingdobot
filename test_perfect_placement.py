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

print("Initial bowl pos:", bowl_pos)
print("Initial plate pos:", plate_pos)

# 1. Approach above bowl
for _ in range(25):
    curr_eef = obs["robot0_eef_pos"]
    diff = (np.array([bowl_pos[0], bowl_pos[1], 1.15]) - curr_eef) * 5.0
    diff = np.clip(diff, -0.65, 0.65)
    obs, _, _, _ = env.step(np.array([diff[0], diff[1], diff[2], 0, 0, 0, -1.0]))

# 2. Descend around bowl
for _ in range(35):
    curr_eef = obs["robot0_eef_pos"]
    diff = (np.array([bowl_pos[0], bowl_pos[1], 0.930]) - curr_eef) * 5.0
    diff = np.clip(diff, -0.65, 0.65)
    obs, _, _, _ = env.step(np.array([diff[0], diff[1], diff[2], 0, 0, 0, -1.0]))

# 3. Clamp gripper firmly
for _ in range(25):
    obs, _, _, _ = env.step(np.array([0, 0, 0, 0, 0, 0, 1.0]))

# 4. Lift vertically
for _ in range(35):
    curr_eef = obs["robot0_eef_pos"]
    diff = (np.array([curr_eef[0], curr_eef[1], 1.18]) - curr_eef) * 5.0
    diff = np.clip(diff, -0.65, 0.65)
    obs, _, _, _ = env.step(np.array([diff[0], diff[1], diff[2], 0, 0, 0, 1.0]))

# 5. Move horizontally above plate
# Notice bowl is offset relative to EEF by ~ -0.01m in X and +0.035m in Y:
# We adjust target so bowl lands directly in plate center!
bowl_now = env.env.sim.data.body_xpos[bowl_id].copy()
eef_now = obs["robot0_eef_pos"].copy()
offset = bowl_now[:2] - eef_now[:2]
target_eef_xy = plate_pos[:2] - offset

for _ in range(40):
    curr_eef = obs["robot0_eef_pos"]
    diff = (np.array([target_eef_xy[0], target_eef_xy[1], 1.18]) - curr_eef) * 5.0
    diff = np.clip(diff, -0.65, 0.65)
    obs, _, _, _ = env.step(np.array([diff[0], diff[1], diff[2], 0, 0, 0, 1.0]))

# 6. Lower gently onto plate surface
for _ in range(35):
    curr_eef = obs["robot0_eef_pos"]
    diff = (np.array([target_eef_xy[0], target_eef_xy[1], 0.940]) - curr_eef) * 5.0
    diff = np.clip(diff, -0.65, 0.65)
    obs, _, _, _ = env.step(np.array([diff[0], diff[1], diff[2], 0, 0, 0, 1.0]))

# 7. Release gripper
for _ in range(20):
    obs, _, _, _ = env.step(np.array([0, 0, 0, 0, 0, 0, -1.0]))

# 8. Retract upward
for _ in range(20):
    obs, _, _, _ = env.step(np.array([0, 0, 0.3, 0, 0, 0, -1.0]))

# 9. Settle physics
for _ in range(30):
    obs, _, _, _ = env.step(np.array([0, 0, 0, 0, 0, 0, -1.0]))
    if env.check_success():
        print("*** SUCCESS DETECTED DURING PHYSICS SETTLING! ***")
        break

print("Task 1 Check Success:", env.check_success())
bowl_end = env.env.sim.data.body_xpos[bowl_id]
plate_end = env.env.sim.data.body_xpos[plate_id]
print("Final Bowl:", bowl_end)
print("Final Plate:", plate_end)
print("Final XY delta:", np.linalg.norm(bowl_end[:2] - plate_end[:2]))
env.close()
