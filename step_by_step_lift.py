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
bowl_pos = env.env.sim.data.body_xpos[bowl_id].copy()

# Descend to 0.920
for _ in range(40):
    curr_eef = obs["robot0_eef_pos"]
    diff = (np.array([bowl_pos[0], bowl_pos[1], 0.920]) - curr_eef) * 5.0
    diff = np.clip(diff, -0.65, 0.65)
    obs, _, _, _ = env.step(np.array([diff[0], diff[1], diff[2], 0, 0, 0, -1.0]))

print("Gripper qpos before clamp:", obs["robot0_gripper_qpos"])

# Clamp for 30 steps with grasp = 1.0
for _ in range(30):
    obs, _, _, _ = env.step(np.array([0, 0, 0, 0, 0, 0, 1.0]))

print("Gripper qpos after clamp:", obs["robot0_gripper_qpos"])
print("Bowl pos before lift:", env.env.sim.data.body_xpos[bowl_id])

# Lift 1 step at a time
for step in range(30):
    obs, _, _, _ = env.step(np.array([0, 0, 0.5, 0, 0, 0, 1.0]))
    b_pos = env.env.sim.data.body_xpos[bowl_id]
    e_pos = obs["robot0_eef_pos"]
    q_pos = obs["robot0_gripper_qpos"]
    print(f"Lift step {step:2d}: EEF Z={e_pos[2]:.4f}, Bowl Z={b_pos[2]:.4f}, Grip=[{q_pos[0]:.4f}, {q_pos[1]:.4f}]")
    if step == 5:
        break

env.close()
