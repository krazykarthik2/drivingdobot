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

# Complete pick and place pipeline:
# 1. Approach above bowl
# 2. Descend to EEF Z = 0.925
# 3. Clamp gripper (g = 1.0)
# 4. Lift to EEF Z = 1.18
# 5. Transfer above plate center (plate_x, plate_y, 1.18)
# 6. Lower to EEF Z = 0.950 (flush on plate)
# 7. Open gripper (g = -1.0)
# 8. Upward retract to EEF Z = 1.10
for step in range(240):
    curr_eef = obs["robot0_eef_pos"]
    curr_bowl = env.env.sim.data.body_xpos[bowl_id].copy()

    if step < 40: # hover above bowl
        target = np.array([bowl_pos[0], bowl_pos[1], 1.15])
        grasp = -1.0
        kp = 5.0
    elif step < 90: # descend onto bowl
        target = np.array([bowl_pos[0], bowl_pos[1], 0.925])
        grasp = -1.0
        kp = 5.0
    elif step < 125: # clamp gripper
        target = np.array([curr_bowl[0], curr_bowl[1], 0.925])
        grasp = 1.0
        kp = 6.0
    elif step < 160: # lift
        target = np.array([curr_bowl[0], curr_bowl[1], 1.20])
        grasp = 1.0
        kp = 5.5
    elif step < 195: # move above plate center
        target = np.array([plate_pos[0], plate_pos[1], 1.20])
        grasp = 1.0
        kp = 5.5
    elif step < 225: # lower flush onto plate
        target = np.array([plate_pos[0], plate_pos[1], 0.950])
        grasp = 1.0
        kp = 6.0
    else: # release and retract
        target = np.array([plate_pos[0], plate_pos[1], 1.10])
        grasp = -1.0
        kp = 4.5

    diff = (target - curr_eef) * kp
    diff = np.clip(diff, -0.65, 0.65)
    action = np.array([diff[0], diff[1], diff[2], 0, 0, 0, grasp])
    obs, r, d, info = env.step(action)
    if env.check_success():
        print(f"*** PHYSICAL SUCCESS ACHIEVED AT STEP {step}! ***")
        break

# Physics settlement
if not env.check_success():
    for _ in range(40):
        obs, _, _, _ = env.step(np.array([0, 0, 0.1, 0, 0, 0, -1.0]))
        if env.check_success():
            print("*** SUCCESS ACHIEVED DURING SETTLING! ***")
            break

print("Final success:", env.check_success())
bowl_end = env.env.sim.data.body_xpos[bowl_id]
plate_end = env.env.sim.data.body_xpos[plate_id]
print("Final bowl:", bowl_end, "Final plate:", plate_end)
print("Final XY dist:", np.linalg.norm(bowl_end[:2] - plate_end[:2]))
env.close()
