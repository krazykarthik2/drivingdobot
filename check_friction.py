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

print("F1 Pad geom friction:", env.env.sim.model.geom_friction[f1_pad])
print("F2 Pad geom friction:", env.env.sim.model.geom_friction[f2_pad])

bowl_id = env.env.obj_body_id["akita_black_bowl_1"]
bowl_geoms = [i for i in range(env.env.sim.model.ngeom) if "akita_black_bowl_1" in (env.env.sim.model.geom_id2name(i) or "")]
print("Bowl geom friction:", env.env.sim.model.geom_friction[bowl_geoms[0]])

# Increase friction in memory to test grasp adhesion
env.env.sim.model.geom_friction[f1_pad][0] = 5.0
env.env.sim.model.geom_friction[f2_pad][0] = 5.0

print("Friction updated in memory.")

env.close()
