import os
import numpy as np
import libero
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv

libero_path = os.path.dirname(libero.__file__)
b = benchmark.get_benchmark("libero_spatial")()
task = b.get_task(1)
bddl_file = os.path.join(libero_path, "libero", "bddl_files", task.problem_folder, task.bddl_file)

env = OffScreenRenderEnv(bddl_file_name=bddl_file, camera_widths=128, camera_heights=128, render_gpu_device_id=-1)
obs = env.reset()

for _ in range(15):
    obs, _, _, _ = env.step(np.zeros(7))

f1 = env.env.sim.model.geom_name2id("gripper0_finger1_pad_collision")
f2 = env.env.sim.model.geom_name2id("gripper0_finger2_pad_collision")
env.env.sim.model.geom_friction[f1][0] = 50.0
env.env.sim.model.geom_friction[f2][0] = 50.0

b_id = env.env.obj_body_id["akita_black_bowl_1"]
b_pos = env.env.sim.data.body_xpos[b_id].copy()

# Try -0.045 along X (left rim) rather than +0.045 (right rim)
for offset_x in [-0.045, 0.045]:
    target = b_pos + np.array([offset_x, 0.0, 0.035])
    print(f"\n--- Testing offset_x = {offset_x:+.3f} ---")
    
    # Reset pose
    obs = env.reset()
    for _ in range(15):
        obs, _, _, _ = env.step(np.zeros(7))
    b_pos = env.env.sim.data.body_xpos[b_id].copy()
    target = b_pos + np.array([offset_x, 0.0, 0.035])

    # Approach
    for _ in range(50):
        curr = env.env._get_observations()["robot0_eef_pos"]
        diff = target - curr
        env.step(np.clip(np.concatenate([diff * 12.0, [0, 0, 0, -1]]), -0.8, 0.8))

    print("EEF at grasp:", env.env._get_observations()["robot0_eef_pos"])
    print("Bowl at grasp:", env.env.sim.data.body_xpos[b_id])

    # Clamp
    for _ in range(30):
        env.step(np.array([0, 0, 0, 0, 0, 0, 1.0]))

    # Lift
    for _ in range(35):
        env.step(np.array([0, 0, 0.6, 0, 0, 0, 1.0]))

    curr_b = env.env.sim.data.body_xpos[b_id]
    delta_z = curr_b[2] - b_pos[2]
    print(f"Result for offset_x {offset_x:+.3f}: Lift Delta Z = {delta_z:+.4f}m")

env.close()
