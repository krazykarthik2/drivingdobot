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

print("Initial EEF pos:", obs["robot0_eef_pos"])
print("Rel vector:", obs["akita_black_bowl_1_to_robot0_eef_pos"])

# In Franka Panda gripper frame:
# -Z points forward towards the fingertips, or +Z?
# Let's test stepping [0, 0, 0.2] vs [0, 0, -0.2] in action space:
print("Testing action delta...")
# Action space is in base frame (world frame)!
# So target_world_pos = obs['robot0_eef_pos'] + delta
# Bowl world position is known accurately from the geom center:
# akita_black_bowl_1_g0 is at Z ~ 0.998m, rim is at 1.00m, bottom is at 0.970m.
bowl_id = env.env.obj_body_id["akita_black_bowl_1"]
bowl_center = env.env.sim.data.body_xpos[bowl_id]
print("True body bowl center:", bowl_center)

env.close()
