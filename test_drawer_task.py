import os
import numpy as np
import libero
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv

libero_path = os.path.dirname(libero.__file__)
b = benchmark.get_benchmark("libero_goal")()
task = b.get_task(0) # open_the_middle_drawer_of_the_cabinet
bddl_file = os.path.join(libero_path, "libero", "bddl_files", task.problem_folder, task.bddl_file)

env = OffScreenRenderEnv(bddl_file_name=bddl_file, camera_widths=128, camera_heights=128, render_gpu_device_id=-1)
obs = env.reset()

for _ in range(15):
    obs, _, _, _ = env.step(np.zeros(7))

# Handle center is around [0.039, -0.150, 1.015]
handle_pos = np.array([0.039, -0.150, 1.015])
print("Initial EEF pos:", obs["robot0_eef_pos"])
print("Target handle pos:", handle_pos)

# 1. Approach in front of the handle: [handle_x, handle_y + 0.12, handle_z]
pre_pos = np.array([handle_pos[0], handle_pos[1] + 0.12, handle_pos[2]])
for _ in range(35):
    curr = obs["robot0_eef_pos"]
    action = np.zeros(7)
    action[:3] = np.clip((pre_pos - curr) * 6.0, -0.6, 0.6)
    action[6] = -1.0 # open gripper
    obs, _, _, _ = env.step(action)

print("EEF at pre-position:", obs["robot0_eef_pos"])

# 2. Advance forward along Y axis to surround handle
grasp_pos = np.array([handle_pos[0], handle_pos[1] + 0.02, handle_pos[2]])
for _ in range(30):
    curr = obs["robot0_eef_pos"]
    action = np.zeros(7)
    action[:3] = np.clip((grasp_pos - curr) * 6.0, -0.4, 0.4)
    action[6] = -1.0
    obs, _, _, _ = env.step(action)

print("EEF at grasp pos:", obs["robot0_eef_pos"])

# 3. Clamp gripper tightly around handle
for _ in range(30):
    action = np.zeros(7)
    action[6] = 1.0
    obs, _, _, _ = env.step(action)

# 4. Pull backward along +Y direction
for step in range(50):
    action = np.zeros(7)
    action[1] = 0.5 # pull forward in robot Y (+Y is towards robot)
    action[6] = 1.0
    obs, _, _, _ = env.step(action)
    if env.check_success():
        print(f"*** SUCCESS ACHIEVED AT STEP {step}! ***")
        break

j_id = env.env.sim.model.joint_name2id("wooden_cabinet_1_middle_level")
qadr = env.env.sim.model.jnt_qposadr[j_id]
print("Final drawer joint qpos:", env.env.sim.data.qpos[qadr])
print("Final task success:", env.check_success())

env.close()
