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

bowl_obj = env.env.get_object("akita_black_bowl_1")
plate_obj = env.env.get_object("plate_1")

bowl_id = env.env.obj_body_id["akita_black_bowl_1"]
plate_id = env.env.obj_body_id["plate_1"]

bowl_pos = env.env.sim.data.body_xpos[bowl_id].copy()
plate_pos = env.env.sim.data.body_xpos[plate_id].copy()

print("Initial bowl:", bowl_pos)
print("Initial plate:", plate_pos)

# Robot EEF site vs Gripper fingers:
# In Robosuite Panda, the EEF site is roughly 10-12cm above the fingertip contact pads.
# Table is at Z ~ 0.82-0.85m, bowl center is at Z=0.970m.
# To reach the bowl, EEF Z must descend to ~1.00 - 1.02m.
# Let's test a precise sequence:
for step in range(220):
    curr_eef = obs["robot0_eef_pos"]
    curr_bowl = env.env.sim.data.body_xpos[bowl_id].copy()

    if step < 35:
        # Hover directly above bowl center
        target = np.array([curr_bowl[0], curr_bowl[1], 1.15])
        grasp = -1.0
        kp = 5.0
    elif step < 75:
        # Descend around bowl
        target = np.array([curr_bowl[0], curr_bowl[1], 1.015])
        grasp = -1.0
        kp = 5.0
    elif step < 105:
        # Close gripper firmly
        target = np.array([curr_bowl[0], curr_bowl[1], 1.015])
        grasp = 1.0
        kp = 6.0
    elif step < 140:
        # Lift high vertically
        target = np.array([curr_bowl[0], curr_bowl[1], 1.25])
        grasp = 1.0
        kp = 5.5
    elif step < 175:
        # Move directly over plate center
        target = np.array([plate_pos[0], plate_pos[1], 1.25])
        grasp = 1.0
        kp = 5.5
    elif step < 200:
        # Descend flush into plate center
        target = np.array([plate_pos[0], plate_pos[1], 1.020])
        grasp = 1.0
        kp = 6.0
    else:
        # Open gripper and retract
        target = np.array([plate_pos[0], plate_pos[1], 1.15])
        grasp = -1.0
        kp = 4.5

    diff = (target - curr_eef) * kp
    diff = np.clip(diff, -0.65, 0.65)
    action = np.array([diff[0], diff[1], diff[2], 0, 0, 0, grasp])
    obs, r, d, info = env.step(action)
    if env.check_success():
        print(f"*** PHYSICAL SUCCESS ACHIEVED AT STEP {step}! ***")
        break

# Settle physics
if not env.check_success():
    for _ in range(30):
        obs, _, _, _ = env.step(np.array([0, 0, 0.1, 0, 0, 0, -1.0]))
        if env.check_success():
            print("*** SUCCESS ACHIEVED DURING SETTLING! ***")
            break

print("Final success check:", env.check_success())
bowl_end = env.env.sim.data.body_xpos[bowl_id]
plate_end = env.env.sim.data.body_xpos[plate_id]
print("Final bowl pos:", bowl_end)
print("Final plate pos:", plate_end)
print("Final XY distance:", np.linalg.norm(bowl_end[:2] - plate_end[:2]))
env.close()
