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

p1 = env.env.sim.data.geom_xpos[f1_pad]
p2 = env.env.sim.data.geom_xpos[f2_pad]
print("Pad 1:", p1)
print("Pad 2:", p2)
print("Vector from Pad 1 to Pad 2:", p2 - p1)

env.close()
