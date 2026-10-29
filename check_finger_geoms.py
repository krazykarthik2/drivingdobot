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

f1_id = env.env.sim.model.geom_name2id("gripper0_finger1_collision") if "gripper0_finger1_collision" in env.env.sim.model.geom_names else -1
f2_id = env.env.sim.model.geom_name2id("gripper0_finger2_collision") if "gripper0_finger2_collision" in env.env.sim.model.geom_names else -1

print("Available gripper geoms:")
for i in range(env.env.sim.model.ngeom):
    name = env.env.sim.model.geom_id2name(i)
    if name and 'finger' in name:
        pos = env.env.sim.data.geom_xpos[i]
        print(f"Geom: {name:<30} | Pos: [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]")

bowl_id = env.env.obj_body_id["akita_black_bowl_1"]
print("Bowl pos:", env.env.sim.data.body_xpos[bowl_id])
print("EEF pos:", obs["robot0_eef_pos"])
print("Difference (EEF Z - finger Z):", obs["robot0_eef_pos"][2] - env.env.sim.data.geom_xpos[i][2])

env.close()
