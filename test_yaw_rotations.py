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

print("Testing gripper orientations around yaw axis:")
for yaw_angle in [0.0, np.pi/4, np.pi/2, 3*np.pi/4]:
    obs = env.reset()
    bowl_id = env.env.obj_body_id["akita_black_bowl_1"]
    bowl_pos = env.env.sim.data.body_xpos[bowl_id].copy()

    # Step to bowl with yaw_angle in action [dx, dy, dz, 0, 0, dyaw, -1.0]
    for _ in range(40):
        curr_eef = obs["robot0_eef_pos"]
        diff = (np.array([bowl_pos[0], bowl_pos[1], 0.925]) - curr_eef) * 4.0
        diff = np.clip(diff, -0.4, 0.4)
        obs, _, _, _ = env.step(np.array([diff[0], diff[1], diff[2], 0, 0, yaw_angle*0.1, -1.0]))

    # Clamp
    for _ in range(30):
        obs, _, _, _ = env.step(np.array([0, 0, 0, 0, 0, 0, 1.0]))

    # Lift
    for _ in range(35):
        obs, _, _, _ = env.step(np.array([0, 0, 0.5, 0, 0, 0, 1.0]))

    b_lift = env.env.sim.data.body_xpos[bowl_id]
    delta_z = b_lift[2] - bowl_pos[2]
    print(f"Yaw angle {np.degrees(yaw_angle):5.1f} deg -> Grip={obs['robot0_gripper_qpos']} -> Delta Z = {delta_z:.3f}")

env.close()
