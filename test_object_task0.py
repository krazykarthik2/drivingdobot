import os
import numpy as np
import libero
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv

libero_path = os.path.dirname(libero.__file__)
b = benchmark.get_benchmark("libero_object")()
task = b.get_task(0)
bddl_file = os.path.join(libero_path, "libero", "bddl_files", task.problem_folder, task.bddl_file)

env = OffScreenRenderEnv(bddl_file_name=bddl_file, camera_widths=128, camera_heights=128, render_gpu_device_id=-1)
obs = env.reset()

soup_id = env.env.obj_body_id["alphabet_soup_1"]
basket_id = env.env.obj_body_id["basket_1"]

# Let world settle
for _ in range(15):
    obs, _, _, _ = env.step(np.zeros(7))

soup_pos = env.env.sim.data.body_xpos[soup_id].copy()
basket_pos = env.env.sim.data.body_xpos[basket_id].copy()

print(f"Settled Soup Pos: {soup_pos}")
print(f"Settled Basket Pos: {basket_pos}")

# Test grasping alphabet soup at different heights:
grasp_z = soup_pos[2] + 0.05
print(f"Target grasp Z: {grasp_z:.4f}")

# 1. Approach above soup
approach = np.array([soup_pos[0], soup_pos[1], grasp_z + 0.15])
for _ in range(35):
    curr = obs["robot0_eef_pos"]
    action = np.zeros(7)
    action[:3] = np.clip((approach - curr) * 6.0, -0.6, 0.6)
    action[6] = -1.0 # open gripper
    obs, _, _, _ = env.step(action)

# 2. Descend to grasp
for _ in range(35):
    curr = obs["robot0_eef_pos"]
    target = np.array([soup_pos[0], soup_pos[1], grasp_z])
    action = np.zeros(7)
    action[:3] = np.clip((target - curr) * 6.0, -0.5, 0.5)
    action[6] = -1.0
    obs, _, _, _ = env.step(action)

# 3. Clamp
for _ in range(30):
    action = np.zeros(7)
    action[6] = 1.0 # close gripper
    obs, _, _, _ = env.step(action)

# 4. Lift
for _ in range(35):
    action = np.zeros(7)
    action[2] = 0.6
    action[6] = 1.0
    obs, _, _, _ = env.step(action)

soup_after_lift = env.env.sim.data.body_xpos[soup_id].copy()
delta_z = soup_after_lift[2] - soup_pos[2]
print(f"Soup pos after lift: {soup_after_lift} (Delta Z: {delta_z:+.4f} m)")
print(f"Gripper qpos after grasp: {obs['robot0_gripper_qpos']}")

if delta_z > 0.05:
    print("*** SOUP SUCCESSFULLY LIFTED! ***")
    # 5. Move above basket
    for _ in range(50):
        curr = obs["robot0_eef_pos"]
        target = np.array([basket_pos[0], basket_pos[1], grasp_z + 0.22])
        action = np.zeros(7)
        action[:3] = np.clip((target - curr) * 6.0, -0.6, 0.6)
        action[6] = 1.0
        obs, _, _, _ = env.step(action)

    # 6. Descend slightly into basket
    for _ in range(30):
        curr = obs["robot0_eef_pos"]
        target = np.array([basket_pos[0], basket_pos[1], basket_pos[2] + 0.12])
        action = np.zeros(7)
        action[:3] = np.clip((target - curr) * 6.0, -0.5, 0.5)
        action[6] = 1.0
        obs, _, _, _ = env.step(action)

    # 7. Release
    for _ in range(25):
        action = np.zeros(7)
        action[6] = -1.0
        obs, _, _, _ = env.step(action)

    # 8. Settle & check success
    for _ in range(25):
        obs, _, _, _ = env.step(np.zeros(7))

    succ = env.check_success()
    print(f"Task 0 Check Success: {succ}")
    print(f"Final Soup pos: {env.env.sim.data.body_xpos[soup_id]}")
    print(f"Final Basket pos: {env.env.sim.data.body_xpos[basket_id]}")

env.close()
