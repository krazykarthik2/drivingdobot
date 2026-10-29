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

rel_vec = obs["akita_black_bowl_1_to_robot0_eef_pos"]
print("Observation: akita_black_bowl_1_to_robot0_eef_pos =", rel_vec)

# If this vector is (eef - bowl), then moving EEF by -rel_vec brings EEF directly onto bowl!
for step in range(30):
    rel_vec = obs["akita_black_bowl_1_to_robot0_eef_pos"]
    diff = -rel_vec[:2] * 4.0
    diff = np.clip(diff, -0.4, 0.4)
    # Descend until rel_vec[2] is small
    diff_z = (-rel_vec[2] + 0.05) * 4.0
    diff_z = np.clip(diff_z, -0.4, 0.4)
    obs, _, _, _ = env.step(np.array([diff[0], diff[1], diff_z, 0, 0, 0, -1.0]))

print("After relative centering, new rel_vec:", obs["akita_black_bowl_1_to_robot0_eef_pos"])
print("Gripper qpos:", obs["robot0_gripper_qpos"])

# Clamp
for _ in range(30):
    obs, _, _, _ = env.step(np.array([0, 0, 0, 0, 0, 0, 1.0]))

print("Gripper qpos after clamp:", obs["robot0_gripper_qpos"])

# Lift
for _ in range(35):
    obs, _, _, _ = env.step(np.array([0, 0, 0.5, 0, 0, 0, 1.0]))

bowl_id = env.env.obj_body_id["akita_black_bowl_1"]
b_pos = env.env.sim.data.body_xpos[bowl_id]
print("Bowl pos after relative grasp and lift:", b_pos)
print("Did it lift? Delta Z =", b_pos[2] - 0.970)

env.close()
