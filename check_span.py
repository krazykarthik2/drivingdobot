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
f1_size = env.env.sim.model.geom_size[f1_pad]

# Finger opening at grasp = -1.0 vs +1.0
obs, _, _, _ = env.step(np.array([0, 0, 0, 0, 0, 0, -1.0]))
p1_open = env.env.sim.data.geom_xpos[f1_pad]
p2_open = env.env.sim.data.geom_xpos[f2_pad]
open_dist = np.linalg.norm(p1_open - p2_open)

obs, _, _, _ = env.step(np.array([0, 0, 0, 0, 0, 0, 1.0]))
for _ in range(20):
    obs, _, _, _ = env.step(np.array([0, 0, 0, 0, 0, 0, 1.0]))
p1_close = env.env.sim.data.geom_xpos[f1_pad]
p2_close = env.env.sim.data.geom_xpos[f2_pad]
close_dist = np.linalg.norm(p1_close - p2_close)

# Bowl outer diameter
bowl_id = env.env.obj_body_id["akita_black_bowl_1"]
bowl_geoms = [i for i in range(env.env.sim.model.ngeom) if "akita_black_bowl_1" in (env.env.sim.model.geom_id2name(i) or "")]
xs = [env.env.sim.data.geom_xpos[g][0] for g in bowl_geoms]
ys = [env.env.sim.data.geom_xpos[g][1] for g in bowl_geoms]
bowl_diam_x = max(xs) - min(xs)
bowl_diam_y = max(ys) - min(ys)

print(f"Gripper Opening Span: Fully Open = {open_dist*100:.2f} cm | Fully Closed = {close_dist*100:.2f} cm")
print(f"Bowl Outer Span: Diam X = {bowl_diam_x*100:.2f} cm | Diam Y = {bowl_diam_y*100:.2f} cm")

env.close()
