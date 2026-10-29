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

# Settle table physics for 15 steps so objects rest stably on the table
for _ in range(15):
    obs, _, _, _ = env.step(np.zeros(7))

bowl_id = env.env.obj_body_id["akita_black_bowl_1"]
plate_id = env.env.obj_body_id["plate_1"]

settled_bowl = env.env.sim.data.body_xpos[bowl_id].copy()
settled_plate = env.env.sim.data.body_xpos[plate_id].copy()

print(f"Settled Bowl pos: [{settled_bowl[0]:.4f}, {settled_bowl[1]:.4f}, {settled_bowl[2]:.4f}]")
print(f"Settled Plate pos: [{settled_plate[0]:.4f}, {settled_plate[1]:.4f}, {settled_plate[2]:.4f}]")

# Test grasping settled bowl at Z = 0.900, 0.890, 0.880, 0.870
for test_z in [0.910, 0.900, 0.890, 0.880]:
    for _ in range(35):
        curr_eef = obs["robot0_eef_pos"]
        diff = (np.array([settled_bowl[0], settled_bowl[1], test_z]) - curr_eef) * 4.0
        diff = np.clip(diff, -0.4, 0.4)
        obs, _, _, _ = env.step(np.array([diff[0], diff[1], diff[2], 0, 0, 0, -1.0]))

    # Clamp
    for _ in range(30):
        obs, _, _, _ = env.step(np.array([0, 0, 0, 0, 0, 0, 1.0]))

    # Lift
    for _ in range(35):
        obs, _, _, _ = env.step(np.array([0, 0, 0.5, 0, 0, 0, 1.0]))

    curr_b = env.env.sim.data.body_xpos[bowl_id]
    lifted = (curr_b[2] - settled_bowl[2]) > 0.05
    print(f"Grasp at test_z={test_z:.3f}: Grip={obs['robot0_gripper_qpos']} | Lifted={lifted} | Delta Z={curr_b[2]-settled_bowl[2]:.4f}")
    if lifted:
        print("*** SUCCESSFUL GRASP & LIFT FOUND! ***")
        break

env.close()
