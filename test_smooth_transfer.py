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

# 2. Descend to the exact calibrated grasp depth Z = 0.920
for _ in range(35):
    curr_eef = obs["robot0_eef_pos"]
    diff = (np.array([bowl_pos[0], bowl_pos[1], 0.920]) - curr_eef) * 5.0
    diff = np.clip(diff, -0.65, 0.65)
    obs, _, _, _ = env.step(np.array([diff[0], diff[1], diff[2], 0, 0, 0, -1.0]))

# 3. Clamp gripper firmly
for _ in range(30):
    obs, _, _, _ = env.step(np.array([0, 0, 0, 0, 0, 0, 1.0]))

# 4. Lift vertically by small increments while clamping
for _ in range(35):
    curr_eef = obs["robot0_eef_pos"]
    diff = (np.array([curr_eef[0], curr_eef[1], 1.15]) - curr_eef) * 4.0
    diff = np.clip(diff, -0.4, 0.4)
    obs, _, _, _ = env.step(np.array([diff[0], diff[1], diff[2], 0, 0, 0, 1.0]))

bowl_lift = env.env.sim.data.body_xpos[bowl_id]
print("Bowl pos after lift:", bowl_lift)
print("Delta Z after lift:", bowl_lift[2] - bowl_pos[2])

# 5. Move gently toward plate
for _ in range(50):
    curr_eef = obs["robot0_eef_pos"]
    diff = (np.array([plate_pos[0], plate_pos[1], 1.15]) - curr_eef) * 3.5
    diff = np.clip(diff, -0.35, 0.35)
    obs, _, _, _ = env.step(np.array([diff[0], diff[1], diff[2], 0, 0, 0, 1.0]))

bowl_mid = env.env.sim.data.body_xpos[bowl_id]
print("Bowl pos above plate:", bowl_mid)

# 6. Lower gently onto plate surface Z = 0.935
for _ in range(40):
    curr_eef = obs["robot0_eef_pos"]
    diff = (np.array([plate_pos[0], plate_pos[1], 0.935]) - curr_eef) * 3.5
    diff = np.clip(diff, -0.35, 0.35)
    obs, _, _, _ = env.step(np.array([diff[0], diff[1], diff[2], 0, 0, 0, 1.0]))

# 7. Release gripper
for _ in range(25):
    obs, _, _, _ = env.step(np.array([0, 0, 0, 0, 0, 0, -1.0]))

# 8. Retract upward
for _ in range(25):
    obs, _, _, _ = env.step(np.array([0, 0, 0.25, 0, 0, 0, -1.0]))

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
