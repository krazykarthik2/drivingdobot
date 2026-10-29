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

f1_pad = env.env.sim.model.geom_name2id("gripper0_finger1_pad_collision")
f2_pad = env.env.sim.model.geom_name2id("gripper0_finger2_pad_collision")

print("F1 Pad pos:", env.env.sim.data.geom_xpos[f1_pad])
print("F2 Pad pos:", env.env.sim.data.geom_xpos[f2_pad])
print("Pad midpoint:", (env.env.sim.data.geom_xpos[f1_pad] + env.env.sim.data.geom_xpos[f2_pad]) / 2.0)
print("EEF pos:", obs["robot0_eef_pos"])

# Test descending until pad midpoint XY is aligned with bowl center and pad midpoint Z is at 0.920
bowl_id = env.env.obj_body_id["akita_black_bowl_1"]
bowl_pos = env.env.sim.data.body_xpos[bowl_id].copy()

pad_mid = (env.env.sim.data.geom_xpos[f1_pad] + env.env.sim.data.geom_xpos[f2_pad]) / 2.0
offset_eef_to_pad = pad_mid - obs["robot0_eef_pos"]
print("Offset (Pad midpoint - EEF pos):", offset_eef_to_pad)

env.close()
